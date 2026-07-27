BLOCKED_TERMS = {
    "exam answer key",
    "cheat",
    "copy homework",
}


def validate_student_question(question: str) -> str | None:
    lowered = question.lower()
    for term in BLOCKED_TERMS:
        if term in lowered:
            return (
                "I can help you learn the concept, but I cannot help with cheating "
                "or copying answers. Ask me to explain the topic step by step."
            )
    return None


def require_citations(answer: str, citations: list[str]) -> str:
    if citations:
        return answer
    return (
        answer
        + "\n\nI could not find a textbook citation for this answer, so please verify with your teacher."
    )

