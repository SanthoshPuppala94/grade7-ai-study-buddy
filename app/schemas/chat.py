from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    subject: str | None = Field(default=None, max_length=50)
    grade: int = Field(default=7, ge=1, le=12)


class RelatedImage(BaseModel):
    artifact_path: str
    image_base64: str
    media_type: str
    vision_caption: str
    page_number: int


class ChatResponse(BaseModel):
    answer: str
    agent_used: str
    citations: list[str] = Field(default_factory=list)
    related_images: list[RelatedImage] = Field(default_factory=list)
    practice_questions: list[str] = Field(default_factory=list)

