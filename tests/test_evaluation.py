from app.services.evaluation import EvalCase, run_evaluation, score_answer, summarize


def test_score_answer_all_keywords_present():
    case = EvalCase("q", ["grace", "period", "30 days"])
    result = score_answer(case, "The grace period is 30 days.")

    assert result.matched_keywords == ["grace", "period", "30 days"]
    assert result.missed_keywords == []
    assert result.score == 1.0


def test_score_answer_partial_match():
    case = EvalCase("q", ["grace", "60 days"])
    result = score_answer(case, "The grace period is 30 days.")

    assert result.matched_keywords == ["grace"]
    assert result.missed_keywords == ["60 days"]
    assert result.score == 0.5


def test_score_answer_is_case_insensitive():
    case = EvalCase("q", ["GRACE"])
    result = score_answer(case, "grace period applies")

    assert result.score == 1.0


def test_score_answer_with_no_expected_keywords_scores_perfect():
    case = EvalCase("q", [])
    result = score_answer(case, "anything")

    assert result.score == 1.0


def test_run_evaluation_uses_injected_answer_fn():
    cases = [EvalCase("q1", ["a"]), EvalCase("q2", ["b"])]

    results = run_evaluation(cases, lambda q: "a b" if q == "q1" else "nothing")

    assert [r.score for r in results] == [1.0, 0.0]


def test_summarize_empty_result_set():
    assert summarize([]) == {"cases": 0, "average_score": 0.0}


def test_summarize_aggregates_across_cases():
    cases = [EvalCase("q1", ["a"]), EvalCase("q2", ["b"])]
    results = run_evaluation(cases, lambda q: "a")

    summary = summarize(results)

    assert summary == {"cases": 2, "average_score": 0.5, "perfect": 1}
