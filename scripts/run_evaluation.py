"""Offline evaluation sweep against a running instance + a ready document.

    python scripts/run_evaluation.py --document-id <uuid> --token <jwt>
    python scripts/run_evaluation.py --document-id <uuid> --token <jwt> --agent

Runs a small fixed set of questions through POST /query (or /query/agent
with --agent) and reports keyword-recall per question plus an aggregate
score - see app.services.evaluation for why keyword recall rather than an
LLM-judge metric. Not run by CI (same reasoning as
manual_hackrx_smoke_test.py: it needs a live server, a real document, and
real API keys behind it).

DEFAULT_CASES below are generic insurance/contract-style questions for demo
purposes - replace them with cases specific to the document under test for
a meaningful score.
"""
import argparse
import json

import requests

from app.services.evaluation import EvalCase, run_evaluation, summarize

DEFAULT_CASES = [
    EvalCase("What is the grace period for premium payment?", ["grace", "period"]),
    EvalCase("What is the notice period for policy cancellation?", ["notice", "period"]),
    EvalCase("Is there a waiting period for pre-existing conditions?", ["waiting", "period"]),
    EvalCase("What is not covered under this policy?", ["not", "cover"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--agent", action="store_true", help="Use /query/agent instead of /query")
    args = parser.parse_args()

    endpoint = "/query/agent" if args.agent else "/query"
    headers = {"Authorization": f"Bearer {args.token}"}

    def answer_fn(question: str) -> str:
        response = requests.post(
            f"{args.base_url}{endpoint}",
            json={"document_id": args.document_id, "question": question},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["answer"]

    results = run_evaluation(DEFAULT_CASES, answer_fn)
    for result in results:
        print(f"[{result.score:.2f}] {result.question}")
        print(f"  answer: {result.answer}")
        if result.missed_keywords:
            print(f"  missed: {result.missed_keywords}")
        print()

    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
