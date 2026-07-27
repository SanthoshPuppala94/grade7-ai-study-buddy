def route_question(state: dict) -> str:
    question = state["question"].lower()
    if any(word in question for word in ["quiz", "practice", "test me", "questions"]):
        return "quiz_agent"
    return "tutor_agent"

