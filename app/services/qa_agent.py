"""LangGraph agent for document Q&A with corrective retrieval.

Plain RAG (`app.routers.query`) does one retrieve -> generate pass and takes
whatever the vector search returns, even if it's off-topic. This agent adds
the missing middle step: after retrieving, an LLM grades whether the
retrieved chunks actually address the question. If they don't, a second LLM
call rewrites the search query (drops conversational phrasing, surfaces the
concrete entities/terms the question is really asking about) and retries
retrieval before generating — a self-correcting flow ("Corrective RAG")
instead of confidently answering off a bad first retrieval.

A relevant retrieval doesn't guarantee a grounded answer — the generator can
still drift into wording that overstates or invents details the context
never said. So after generating, a groundedness grader checks the answer
against the context it was built from; an ungrounded answer is replaced with
the same abstention text the generator itself would use for missing info,
rather than letting a plausible-sounding hallucination reach the caller.

Graph:

    retrieve --> grade --[relevant, or out of retries]--> generate --> check_groundedness --[grounded]--> END
                   |                                                          |
                   `--[not relevant, retries left]--> rewrite_query           `--[not grounded]--> abstain --> END
                            |
                            `--> retrieve (loop)

Every node is a plain function over `QAState` so the control flow (who talks
to whom, and when) is explicit and independently testable — `grade_node`,
`rewrite_node`, and `check_groundedness_node` are the only nodes that make
LLM calls, and all are cheap to monkeypatch out in tests (see
tests/test_qa_agent.py) without touching the graph wiring itself.
"""
from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.config import settings
from app.services import llm_client, vector_store

# Cap on rewrite+re-retrieve cycles. Bounded on purpose — an ungrounded
# question (nothing in the document answers it) should fall through to
# "Not mentioned in the document." (see llm_client's prompt rules) rather
# than loop forever chasing a relevant chunk that doesn't exist.
MAX_RETRIES = 2

# Text substituted in for an answer the groundedness grader rejects — the
# same abstention wording llm_client's own prompt rules ask the generator to
# use for genuinely missing information, so a caller can't tell "the model
# said so itself" apart from "the agent caught it lying" from the text alone.
ABSTENTION_TEXT = "Not mentioned in the document."


class RelevanceGrade(BaseModel):
    """Structured grader output — a forced boolean beats parsing free text."""

    relevant: bool = Field(
        description="True if the retrieved context contains information that answers the question."
    )


class GroundednessGrade(BaseModel):
    """Structured grader output for the post-generation hallucination check."""

    grounded: bool = Field(
        description="True if every claim in the answer is directly supported by the context — "
        "no invented figures, entities, or specifics the context doesn't state."
    )


class QAState(TypedDict):
    original_question: str  # what's shown to the user / passed to generate_answer
    question: str  # current search query — may have been rewritten
    namespace: str
    context: str
    chunks: list[dict]  # source chunks backing `context` — id, text, score
    relevant: bool
    retries: int
    answer: str
    retrieval_attempts: int
    grounded: bool


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance grader for a document Q&A system. "
            "Given retrieved context and a question, decide whether the context "
            "actually contains information that answers the question. Be strict: "
            "tangentially related context that doesn't answer the question is not relevant.",
        ),
        ("human", "QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{context}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The search query below failed to retrieve context that answers the question. "
            "Rewrite it as a better search query for a semantic/vector search: drop "
            "conversational phrasing, and surface the concrete entities, terms, or clause "
            "topics the question is really asking about. Respond with only the rewritten "
            "query text — no explanation, no quotes.",
        ),
        (
            "human",
            "ORIGINAL QUESTION:\n{original_question}\n\n"
            "SEARCH QUERY THAT FAILED:\n{question}\n\n"
            "CONTEXT IT RETRIEVED (not relevant):\n{context}",
        ),
    ]
)

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict fact-checker for a document Q&A system. Given the context an "
            "answer was generated from and the answer itself, decide whether every claim in "
            "the answer is directly supported by the context. Be strict: a figure, date, "
            "entity, or clause that isn't stated in the context — even if plausible — is not "
            "grounded. An answer that says information is missing is always grounded.",
        ),
        ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
    ]
)

# Lazily built, module-level singletons — same reasoning as llm_client._get_chain:
# importing this module must never require live credentials (keeps unit tests
# and `alembic` runs import-safe), and there's no reason to rebuild the chain
# on every graph invocation.
_grader_chain = None
_rewriter_chain = None
_groundedness_chain = None


def _get_grader_chain():
    global _grader_chain
    if _grader_chain is None:
        llm = ChatGroq(model="llama3-8b-8192", temperature=0.0, api_key=settings.groq_api_key)
        _grader_chain = GRADE_PROMPT | llm.with_structured_output(RelevanceGrade)
    return _grader_chain


def _get_rewriter_chain():
    global _rewriter_chain
    if _rewriter_chain is None:
        llm = ChatGroq(model="llama3-8b-8192", temperature=0.0, max_tokens=60, api_key=settings.groq_api_key)
        _rewriter_chain = REWRITE_PROMPT | llm | StrOutputParser()
    return _rewriter_chain


def _get_groundedness_chain():
    global _groundedness_chain
    if _groundedness_chain is None:
        llm = ChatGroq(model="llama3-8b-8192", temperature=0.0, api_key=settings.groq_api_key)
        _groundedness_chain = GROUNDEDNESS_PROMPT | llm.with_structured_output(GroundednessGrade)
    return _groundedness_chain


def retrieve_node(state: QAState) -> dict:
    chunks = vector_store.retrieve_chunks(state["question"], state["namespace"])
    context = "\n".join(chunk["text"] for chunk in chunks)
    return {
        "context": context,
        "chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


def grade_node(state: QAState) -> dict:
    grade: RelevanceGrade = _get_grader_chain().invoke(
        {"question": state["original_question"], "context": state["context"]}
    )
    return {"relevant": grade.relevant}


def rewrite_node(state: QAState) -> dict:
    rewritten = _get_rewriter_chain().invoke(
        {
            "original_question": state["original_question"],
            "question": state["question"],
            "context": state["context"],
        }
    )
    return {"question": rewritten.strip(), "retries": state.get("retries", 0) + 1}


def generate_node(state: QAState) -> dict:
    # Answer is generated against the original question, not the (possibly
    # rewritten) search query — the rewrite only ever exists to improve
    # retrieval, the user never sees it.
    answer = llm_client.generate_answer(state["original_question"], state["context"])
    return {"answer": answer}


def check_groundedness_node(state: QAState) -> dict:
    # An answer that's already the abstention text has nothing to fact-check
    # against the context — skip the LLM call and just pass it through.
    if state["answer"].strip() == ABSTENTION_TEXT:
        return {"grounded": True}
    grade: GroundednessGrade = _get_groundedness_chain().invoke(
        {"context": state["context"], "answer": state["answer"]}
    )
    return {"grounded": grade.grounded}


def abstain_node(state: QAState) -> dict:
    return {"answer": ABSTENTION_TEXT}


def _route_after_grade(state: QAState) -> str:
    if state["relevant"] or state.get("retries", 0) >= MAX_RETRIES:
        return "generate"
    return "rewrite"


def _route_after_groundedness(state: QAState) -> str:
    return "done" if state["grounded"] else "abstain"


def build_graph():
    """Builds a fresh graph per call — compiling a 4-node graph is cheap, and
    it keeps tests free of singleton-cache staleness across monkeypatches."""
    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_groundedness", check_groundedness_node)
    graph.add_node("abstain", abstain_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", _route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "check_groundedness")
    graph.add_conditional_edges(
        "check_groundedness", _route_after_groundedness, {"done": END, "abstain": "abstain"}
    )
    graph.add_edge("abstain", END)

    return graph.compile()


def answer_question(question: str, namespace: str) -> dict:
    """Entry point used by the API layer. Synchronous by design — same
    reasoning as llm_client.generate_answer: called from FastAPI via a
    threadpool, so it shouldn't be async itself."""
    result = build_graph().invoke(
        {
            "original_question": question,
            "question": question,
            "namespace": namespace,
            "context": "",
            "chunks": [],
            "relevant": False,
            "retries": 0,
            "answer": "",
            "retrieval_attempts": 0,
            "grounded": False,
        }
    )
    return {
        "answer": result["answer"],
        "retrieval_attempts": result["retrieval_attempts"],
        "query_rewritten": result["question"] != question,
        "grounded": result["grounded"],
        "chunks": result["chunks"],
    }
