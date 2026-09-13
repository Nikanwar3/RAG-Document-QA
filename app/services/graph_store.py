"""Knowledge-graph layer: extracts (subject, predicate, object) relations from
a document's text and stores them in Neo4j, namespaced per document exactly
like `vector_store`'s Pinecone namespace - so a document's graph never bleeds
into another document's.

This complements vector search rather than replacing it: Pinecone answers
"which chunk of text is semantically closest to this question", while the
graph answers "what else in this document is this entity connected to" -
useful when a clause refers to a party or term ("the Policyholder", "the
Grace Period") whose defining details live in a different chunk than the one
vector search happened to retrieve. `app.services.qa_agent`'s
`graph_augment_node` is what actually bridges the two.

Import-safe without live credentials (same reasoning as vector_store._get_index):
the Neo4j driver and the LLM extraction chain are both built lazily.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from app.config import settings

_driver = None
_extraction_chain = None


class Relation(BaseModel):
    subject: str = Field(description="The entity the relationship starts from, e.g. 'the Policyholder'.")
    predicate: str = Field(description="A short label for the relationship, e.g. 'must pay'.")
    object: str = Field(description="The entity or value the relationship points to, e.g. 'the Premium'.")


class ExtractedRelations(BaseModel):
    relations: list[Relation] = Field(default_factory=list)


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Extract the key entities and relationships from this document as short "
                "(subject, predicate, object) triples - parties, terms, obligations, amounts, "
                "and how they relate (e.g. 'the Policyholder' -'must pay'-> 'the Premium'). "
                "Keep entity names short and reuse the same wording for the same entity across "
                "triples, so they connect into a graph rather than staying isolated. Extract at "
                "most {max_relations} of the most important relationships."
            ),
        ),
        ("human", "{text}"),
    ]
)

# Capped well under the model's context window - this runs once per document at
# ingestion time against the whole document, not per-chunk, so it only needs
# enough of the text to surface the entities that recur throughout it.
MAX_EXTRACTION_CHARS = 8000


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase

        _driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    return _driver


def _get_extraction_chain():
    global _extraction_chain
    if _extraction_chain is None:
        llm = ChatGroq(model="llama3-8b-8192", temperature=0.0, api_key=settings.groq_api_key)
        _extraction_chain = EXTRACTION_PROMPT | llm.with_structured_output(ExtractedRelations)
    return _extraction_chain


def extract_relations(text: str, max_relations: int = 15) -> list[Relation]:
    """LLM-based entity/relationship extraction over a document's text."""
    chain = _get_extraction_chain()
    result: ExtractedRelations = chain.invoke(
        {"text": text[:MAX_EXTRACTION_CHARS], "max_relations": max_relations}
    )
    return result.relations[:max_relations]


def store_relations(namespace: str, relations: list[Relation]) -> None:
    """Upsert relations into Neo4j, scoped to one document's namespace."""
    if not relations:
        return
    driver = _get_driver()
    with driver.session() as session:
        for relation in relations:
            session.run(
                """
                MERGE (a:Entity {name: $subject, namespace: $namespace})
                MERGE (b:Entity {name: $object, namespace: $namespace})
                MERGE (a)-[r:RELATION {type: $predicate}]->(b)
                """,
                subject=relation.subject,
                object=relation.object,
                predicate=relation.predicate,
                namespace=namespace,
            )


def get_relations(namespace: str) -> list[Relation]:
    """All relations extracted for one document - powers GET /documents/{id}/graph."""
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Entity {namespace: $namespace})-[r:RELATION]->(b:Entity {namespace: $namespace})
            RETURN a.name AS subject, r.type AS predicate, b.name AS object
            """,
            namespace=namespace,
        )
        return [Relation(subject=rec["subject"], predicate=rec["predicate"], object=rec["object"]) for rec in result]


def query_related(namespace: str, entity: str, limit: int = 5) -> list[Relation]:
    """1-hop neighbors of `entity` (case-insensitive substring match), in
    either direction - used by qa_agent's graph-lookup tool at query time."""
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Entity {namespace: $namespace})-[r:RELATION]-(b:Entity {namespace: $namespace})
            WHERE toLower(a.name) CONTAINS toLower($entity)
            RETURN a.name AS subject, r.type AS predicate, b.name AS object
            LIMIT $limit
            """,
            namespace=namespace,
            entity=entity,
            limit=limit,
        )
        return [Relation(subject=rec["subject"], predicate=rec["predicate"], object=rec["object"]) for rec in result]
