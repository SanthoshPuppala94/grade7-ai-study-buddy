import os

import pytest

from evaluation.rag_evaluator import (
    build_deepeval_test_cases,
    build_deepeval_metrics,
    get_evaluation_judge_model,
    load_golden_questions,
    run_retrieval_golden_checks,
)


def test_golden_questions_are_defined():
    golden_questions = load_golden_questions()

    assert len(golden_questions) >= 3
    assert all(question.question for question in golden_questions)
    assert all(question.expected_context_terms for question in golden_questions)


def test_retrieval_golden_checks_pass_locally():
    results = run_retrieval_golden_checks()

    assert results
    assert all(result["passed"] for result in results)


def test_evaluation_uses_configured_rag_model_as_judge_model():
    assert get_evaluation_judge_model() == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_deepeval_rag_metrics_when_enabled():
    if os.getenv("RUN_DEEPEVAL") != "1":
        pytest.skip("Set RUN_DEEPEVAL=1 to run LLM-judge DeepEval metrics.")

    deepeval = pytest.importorskip("deepeval")
    metrics = pytest.importorskip("deepeval.metrics")

    test_cases = await build_deepeval_test_cases()
    assert test_cases

    deepeval_metrics = build_deepeval_metrics(metrics)

    for test_case in test_cases:
        deepeval.assert_test(
            test_case,
            deepeval_metrics,
        )
