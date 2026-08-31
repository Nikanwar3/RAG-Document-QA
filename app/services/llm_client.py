from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import settings

_chain = None

SYSTEM_PROMPT = """You are a precise document assistant.

Answer the question using only the following context:

{context}

RESPONSE RULES:
- Answer in plain, clear English.
- Limit to 1-2 sentences and under 35 words.
- Include specific figures (e.g., "INR 5,000", "1% of SI", "24 months") if mentioned.
- Do NOT say "Based on the context", "I found", "Clause number", or similar.
- If the policy mentions any legal act or law, write it exactly as written.
- If the answer is missing, say exactly: "Not mentioned in the document."
"""


def _get_chain():
    """Builds the LangChain runnable once and reuses it across calls.

    ChatGroq gives LangChain's standard chat-model interface over Groq's
    hosted Llama3, so the retrieval-augmented prompt below is composed with
    LangChain's declarative `prompt | llm | parser` runnable syntax instead
    of hand-rolling the raw chat-completions payload.
    """
    global _chain
    if _chain is None:
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.0,
            max_tokens=80,
            api_key=settings.groq_api_key,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        _chain = prompt | llm | StrOutputParser()
    return _chain


def generate_answer(question: str, context: str) -> str:
    """Synchronous by design: called from FastAPI via a threadpool and directly
    from the Celery worker, neither of which wants an async client here."""
    chain = _get_chain()
    try:
        answer = chain.invoke({"context": context, "question": question})
        return answer.strip() if answer else "No response generated"
    except Exception as exc:
        return f"Error: {exc}"
