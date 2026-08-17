"""Manual latency smoke test for the legacy /hackrx/run endpoint.

Not part of the pytest suite (see tests/) — this hits a real running server
and real Groq/Pinecone credentials, so it's meant to be run by hand:

    uvicorn app.main:app --reload
    python scripts/manual_hackrx_smoke_test.py
"""

import time

import requests

from app.config import settings

QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?",
]

DOCUMENT_URL = (
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z"
    "&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
)


def main() -> None:
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": f"Bearer {settings.hackrx_token}",
        "Content-Type": "application/json",
    }
    payload = {"documents": DOCUMENT_URL, "questions": QUESTIONS}

    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload)
    total_time = time.time() - start_time

    if response.status_code == 200:
        answers = response.json().get("answers", [])
        print("Response:\n")
        for answer in answers:
            print(f"A -  {answer}\n\n")
        print(f"Total Latency for {len(QUESTIONS)} questions: {total_time:.2f} seconds")
        print(f"Average Latency per question: {total_time / len(QUESTIONS):.2f} seconds")
    else:
        print(f"Error {response.status_code}:")
        print(response.text)


if __name__ == "__main__":
    main()
