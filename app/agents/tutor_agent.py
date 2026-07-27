from app.services.guardrails import require_citations, validate_student_question
from app.services.vector_store import StudyVectorStore


class TutorAgent:
    name = "tutor_agent"

    def __init__(self, vector_store: StudyVectorStore | None = None):
        self.vector_store = vector_store or StudyVectorStore()

    async def arun(self, state: dict) -> dict:
        blocked = validate_student_question(state["question"])
        if blocked:
            return {**state, "agent_used": self.name, "answer": blocked, "citations": []}

        subject = state.get("subject")
        results = self.vector_store.search(state["question"], k=4, subject=subject)
        citations = [str(result["source"]) for result in results]
        related_images = _collect_related_images(results)
        context = "\n\n".join(f"[{result['source']}]\n{result['text']}" for result in results)
        answer = _student_friendly_answer(state["question"], context, subject or "the subject")
        return {
            **state,
            "agent_used": self.name,
            "answer": require_citations(answer, citations),
            "citations": citations,
            "related_images": related_images,
        }


def _student_friendly_answer(question: str, context: str, subject: str) -> str:
    if not context:
        return "I could not find this topic in the indexed Grade 7 material."
    return (
        f"Here is a Grade 7 friendly explanation for your {subject} question:\n\n"
        f"Question: {question}\n\n"
        "From the textbook context, the key idea is:\n"
        f"{context[:900]}\n\n"
        "In simple words, focus on the main concept first, then try one small example."
    )


def _collect_related_images(results: list[dict]) -> list[dict]:
    images = []
    seen = set()
    for result in results:
        for image in result.get("related_images", []):
            path = image.get("artifact_path")
            if path in seen:
                continue
            seen.add(path)
            images.append(image)
    return images

