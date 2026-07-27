import pytest

from app.graph.builder import build_graph


@pytest.mark.asyncio
async def test_graph_routes_quiz_questions_to_quiz_agent():
    graph = build_graph()

    result = await graph.ainvoke(
        {
            "question": "Give me practice questions about one lakh",
            "subject": "mathematics",
            "grade": 7,
            "citations": [],
            "related_images": [],
            "practice_questions": [],
        }
    )

    assert result["agent_used"] == "quiz_agent"
    assert result["practice_questions"]

