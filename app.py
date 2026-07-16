# app.py (rewritten for HackRx with Groq + FastAPI)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pipeline import process_document_and_answer
from fastapi.responses import JSONResponse
from config import HACKRX_TOKEN

app = FastAPI()


class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]


class HackRxResponse(BaseModel):
    answers: list[str]


@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.endswith(HACKRX_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    try:
        blob_url = body["documents"]
        questions = body["questions"]
        answers = await process_document_and_answer(blob_url, questions)
        return {"answers": answers}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
