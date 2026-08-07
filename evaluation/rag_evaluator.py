import json
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.agents.tutor_agent import TutorAgent
from app.services.vector_store import StudyVectorStore

GOLDEN_QUESTIONS_PATH = Path(__file__).with_name("golden_questions.json")
DEEPEVAL_METRIC_THRESHOLD = 0.7


@dataclass
class GoldenQuestion:
    id: str
    question: str
    subject: str
    expected_context_terms: list[str]
    expected_answer_terms: list[str]


def load_golden_questions(path: Path = GOLDEN_QUESTIONS_PATH) -> list[GoldenQuestion]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenQuestion(**row) for row in rows]


def run_retrieval_golden_checks(
    vector_store: StudyVectorStore | None = None,
    retrieval_mode: str = "hybrid",
) -> list[dict]:
    store = vector_store or StudyVectorStore()
    results = []
    for golden in load_golden_questions():
        retrieved = store.search(
            golden.question,
            k=4,
            subject=golden.subject,
            retrieval_mode=retrieval_mode,
        )
        context = "\n\n".join(result["text"] for result in retrieved).lower()
        matched_terms = [
            term for term in golden.expected_context_terms if term.lower() in context
        ]
        results.append(
            {
                "id": golden.id,
                "question": golden.question,
                "retrieval_mode": retrieval_mode,
                "matched_terms": matched_terms,
                "expected_terms": golden.expected_context_terms,
                "passed": bool(matched_terms),
                "sources": [result["source"] for result in retrieved],
            }
        )
    return results


def get_evaluation_judge_model() -> str:
    settings = get_settings()
    return settings.evaluation_judge_model or settings.openai_model


def build_deepeval_metrics(metrics_module):
    judge_model = get_evaluation_judge_model()
    return [
        metrics_module.AnswerRelevancyMetric(
            threshold=DEEPEVAL_METRIC_THRESHOLD,
            model=judge_model,
        ),
        metrics_module.FaithfulnessMetric(
            threshold=DEEPEVAL_METRIC_THRESHOLD,
            model=judge_model,
        ),
        metrics_module.ContextualRelevancyMetric(
            threshold=DEEPEVAL_METRIC_THRESHOLD,
            model=judge_model,
        ),
    ]


async def build_deepeval_test_cases(
    vector_store: StudyVectorStore | None = None,
    retrieval_mode: str = "hybrid",
):
    from deepeval.test_case import LLMTestCase

    store = vector_store or StudyVectorStore()
    agent = TutorAgent(vector_store=store)
    test_cases = []
    for golden in load_golden_questions():
        retrieved = store.search(
            golden.question,
            k=4,
            subject=golden.subject,
            retrieval_mode=retrieval_mode,
        )
        response = await agent.arun(
            {
                "question": golden.question,
                "subject": golden.subject,
            }
        )
        test_cases.append(
            LLMTestCase(
                input=golden.question,
                actual_output=response["answer"],
                retrieval_context=[result["text"] for result in retrieved],
                expected_output="; ".join(golden.expected_answer_terms),
            )
        )
    return test_cases
