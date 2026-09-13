"""Offline evaluation harness for the RAG pipeline.

Scores answer quality against a small labeled dataset of
(question, expected_keywords) cases. Keyword recall against an expected
answer is a deliberately cheap, deterministic proxy metric - no LLM-judge
call, no extra API cost, and easy to unit test - rather than a full
RAGAS-style faithfulness/relevance evaluation. `answer_fn` is injected so the
same scoring logic runs against a live agent (see scripts/run_evaluation.py)
or a stub (tests), the same way `qa_agent`'s nodes are swapped out in tests.
"""
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]


@dataclass
class EvalResult:
    question: str
    answer: str
    matched_keywords: list[str] = field(default_factory=list)
    missed_keywords: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = len(self.matched_keywords) + len(self.missed_keywords)
        return len(self.matched_keywords) / total if total else 1.0


def score_answer(case: EvalCase, answer: str) -> EvalResult:
    lowered = answer.lower()
    matched = [kw for kw in case.expected_keywords if kw.lower() in lowered]
    missed = [kw for kw in case.expected_keywords if kw.lower() not in lowered]
    return EvalResult(question=case.question, answer=answer, matched_keywords=matched, missed_keywords=missed)


def run_evaluation(cases: list[EvalCase], answer_fn: Callable[[str], str]) -> list[EvalResult]:
    return [score_answer(case, answer_fn(case.question)) for case in cases]


def summarize(results: list[EvalResult]) -> dict:
    if not results:
        return {"cases": 0, "average_score": 0.0}
    average = sum(r.score for r in results) / len(results)
    return {
        "cases": len(results),
        "average_score": round(average, 4),
        "perfect": sum(1 for r in results if r.score == 1.0),
    }
