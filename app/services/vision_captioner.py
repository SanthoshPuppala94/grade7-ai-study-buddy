import base64
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageCaptionContext:
    subject: str
    grade: int
    source: str
    page_number: int
    nearby_text: str

    def as_prompt_context(self) -> str:
        return (
            f"Subject: {self.subject}\n"
            f"Grade: {self.grade}\n"
            f"Source: {self.source}\n"
            f"Page: {self.page_number}\n"
            f"Page Content:\n{self.nearby_text}"
        )


@dataclass(frozen=True)
class VisionCaption:
    image_base64: str
    media_type: str
    vision_caption: str
    vision_model: str
    embedding_text: str


class VisionCaptioner:
    """Vision-model boundary for context-aware textbook image captions.

    Local implementation is deterministic. Production can replace this with a
    multimodal LLM call using the same inputs: base64 image plus page context.
    """

    def __init__(self, model_name: str = "local-deterministic-vision-captioner"):
        self.model_name = model_name

    def caption_image(
        self,
        image_bytes: bytes,
        media_type: str,
        context: ImageCaptionContext,
    ) -> VisionCaption:
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        caption = self._local_caption(image_bytes, context)
        return VisionCaption(
            image_base64=image_base64,
            media_type=media_type,
            vision_caption=caption,
            vision_model=self.model_name,
            embedding_text=f"{caption}\n\nCaption context:\n{context.as_prompt_context()}",
        )

    def _local_caption(self, image_bytes: bytes, context: ImageCaptionContext) -> str:
        try:
            from PIL import Image, ImageStat
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required for local image captioning.") from exc

        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            width, height = image.size
            mean = ImageStat.Stat(image).mean
        color_family = _dominant_color_family(mean)
        nearby = " ".join(context.nearby_text.split())[:180]
        return (
            f"Grade {context.grade} {context.subject} textbook visual from page "
            f"{context.page_number}. It is a {width}x{height} image with a "
            f"{color_family} visual signal, related to: {nearby}."
        )


def _dominant_color_family(rgb_mean: list[float]) -> str:
    red, green, blue = rgb_mean[:3]
    if max(red, green, blue) - min(red, green, blue) < 20:
        return "neutral"
    if blue >= red and blue >= green:
        return "blue"
    if green >= red and green >= blue:
        return "green"
    return "warm"

