from openai import OpenAI

from app.config import settings

_client = None

PROMPT_TEMPLATE = """
You are a precise document assistant.

Answer the question using only the following context:

{context}

QUESTION:
{question}

RESPONSE RULES:
- Answer in plain, clear English.
- Limit to 1-2 sentences and under 35 words.
- Include specific figures (e.g., "INR 5,000", "1% of SI", "24 months") if mentioned.
- Do NOT say "Based on the context", "I found", "Clause number", or similar.
- If the policy mentions any legal act or law, write it exactly as written.
- If the answer is missing, say exactly: "Not mentioned in the document."
"""


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        # Groq exposes an OpenAI-compatible API, so the official OpenAI SDK works
        # against it by pointing base_url at Groq's endpoint.
        _client = OpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def generate_answer(question: str, context: str) -> str:
    """Synchronous by design: called from FastAPI via a threadpool and directly
    from the Celery worker, neither of which wants an async client here."""
    client = _get_client()
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80,
        )
        answer = response.choices[0].message.content
        return answer.strip() if answer else "No response generated"
    except Exception as exc:
        return f"Error: {exc}"
