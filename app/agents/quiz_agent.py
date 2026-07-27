from app.services.vector_store import StudyVectorStore


class QuizAgent:
    name = "quiz_agent"

    def __init__(self, vector_store: StudyVectorStore | None = None):
        self.vector_store = vector_store or StudyVectorStore()

    async def arun(self, state: dict) -> dict:
        results = self.vector_store.search(state["question"], k=3, subject=state.get("subject"))
        topic = state.get("subject") or "this topic"
        practice_questions = [
            f"Explain one important idea from {topic} in your own words.",
            f"Give one real-life example related to {topic}.",
            f"Create a short answer question from the retrieved textbook section.",
        ]
        return {
            **state,
            "agent_used": self.name,
            "answer": "Here are practice questions based on the retrieved Grade 7 material.",
            "citations": [str(result["source"]) for result in results],
            "related_images": [],
            "practice_questions": practice_questions,
        }

