import asyncio
import os
import tempfile
from urllib.parse import urlparse

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.services import document_processor, llm_client, vector_store

router = APIRouter(tags=["legacy"])

# A fixed namespace keeps this endpoint's behavior identical to the pre-refactor
# single-tenant version: every call shares one Pinecone namespace.
_LEGACY_NAMESPACE = "hackrx-legacy"


class HackRxResponse(BaseModel):
    answers: list[str]


def _get_file_extension_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if ".pdf" in path:
        return ".pdf"
    if ".docx" in path:
        return ".docx"
    if ".doc" in path:
        return ".docx"
    if ".eml" in path or ".msg" in path:
        return ".eml"
    return ".pdf"


@router.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(request: Request):
    """Legacy single-shot endpoint kept for grader compatibility. Predates the
    Postgres/Redis/Celery pipeline, so it ingests synchronously inline rather
    than handing off to the background worker."""
    auth = request.headers.get("authorization", "")
    if not settings.hackrx_token or not auth.endswith(settings.hackrx_token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    try:
        blob_url = body["documents"]
        questions = body["questions"]

        response = await asyncio.to_thread(requests.get, blob_url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to download document. Status: {response.status_code}")

        suffix = _get_file_extension_from_url(blob_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        try:
            text = document_processor.extract_text_from_document(tmp_path)
            if not text.strip():
                raise Exception("No text could be extracted from the document")

            chunks = document_processor.chunk_text(text)
            if not chunks:
                raise Exception("No chunks created from document text")

            vector_store.embed_and_store_chunks(chunks, namespace=_LEGACY_NAMESPACE)

            async def answer_one(question: str) -> str:
                context = await asyncio.to_thread(
                    vector_store.query_top_chunks, question, _LEGACY_NAMESPACE
                )
                return await asyncio.to_thread(llm_client.generate_answer, question, context)

            answers = await asyncio.gather(*(answer_one(q) for q in questions))
            return {"answers": answers}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
