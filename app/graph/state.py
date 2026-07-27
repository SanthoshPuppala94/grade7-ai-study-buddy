from typing import Any, TypedDict


class StudyState(TypedDict, total=False):
    question: str
    subject: str | None
    grade: int
    route: str
    agent_used: str
    answer: str
    citations: list[str]
    related_images: list[dict[str, Any]]
    practice_questions: list[str]

