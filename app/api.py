from fastapi import FastAPI, status

from app.config import get_settings
from app.graph.builder import build_graph
from app.schemas.chat import ChatRequest, ChatResponse, RelatedImage


settings = get_settings()
graph = build_graph()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "running", "docs": "/docs", "chat": "POST /chat"}


@app.get("/health/live", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    return {"status": "alive"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    result = await graph.ainvoke(
        {
            "question": request.question,
            "subject": request.subject,
            "grade": request.grade,
            "citations": [],
            "related_images": [],
            "practice_questions": [],
        }
    )
    return ChatResponse(
        answer=result["answer"],
        agent_used=result["agent_used"],
        citations=result.get("citations", []),
        related_images=[RelatedImage(**image) for image in result.get("related_images", [])],
        practice_questions=result.get("practice_questions", []),
    )

