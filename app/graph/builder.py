from langgraph.graph import END, StateGraph

from app.agents.quiz_agent import QuizAgent
from app.agents.tutor_agent import TutorAgent
from app.graph.state import StudyState
from app.graph.supervisor import route_question
from app.services.vector_store import StudyVectorStore


def build_graph():
    vector_store = StudyVectorStore()
    tutor_agent = TutorAgent(vector_store)
    quiz_agent = QuizAgent(vector_store)

    workflow = StateGraph(StudyState)

    async def supervisor(state: StudyState) -> StudyState:
        return {**state, "route": route_question(state)}

    async def run_tutor(state: StudyState) -> StudyState:
        return await tutor_agent.arun(state)

    async def run_quiz(state: StudyState) -> StudyState:
        return await quiz_agent.arun(state)

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("tutor_agent", run_tutor)
    workflow.add_node("quiz_agent", run_quiz)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {"tutor_agent": "tutor_agent", "quiz_agent": "quiz_agent"},
    )
    workflow.add_edge("tutor_agent", END)
    workflow.add_edge("quiz_agent", END)
    return workflow.compile()

