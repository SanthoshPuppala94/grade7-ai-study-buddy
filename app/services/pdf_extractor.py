import hashlib
import io
from pathlib import Path

from app.config import DATA_DIR
from app.services.vision_captioner import ImageCaptionContext, VisionCaptioner


IMAGE_ARTIFACTS_DIR = DATA_DIR / "rag_image_artifacts"
MAX_IMAGE_WIDTH = 1600
MAX_IMAGE_HEIGHT = 1600
MAX_IMAGE_PIXELS = 1_500_000
MIN_IMAGE_WIDTH = 120
MIN_IMAGE_HEIGHT = 120


def extract_pdf(path: Path, subject: str, grade: int) -> dict:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for PDF extraction.") from exc

    captioner = VisionCaptioner()
    doc = fitz.open(str(path))
    pages: list[dict] = []
    seen_hashes: set[str] = set()

    for page_index, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        images = []
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            image = doc.extract_image(image_info[0])
            keep, reason, image_meta = classify_image(image["image"], seen_hashes)
            if not keep:
                continue
            caption = captioner.caption_image(
                image_bytes=image["image"],
                media_type=f"image/{image_meta['extension']}",
                context=ImageCaptionContext(
                    subject=subject,
                    grade=grade,
                    source=str(path.relative_to(DATA_DIR)),
                    page_number=page_index,
                    nearby_text=_nearby_text(text),
                ),
            )
            artifact_path = save_image_artifact(path, image["image"], image_meta["hash"], image_meta["extension"])
            images.append(
                {
                    **image_meta,
                    "page_number": page_index,
                    "image_number": image_index,
                    "skip_reason": reason,
                    "artifact_path": artifact_path,
                    "image_base64": caption.image_base64,
                    "media_type": caption.media_type,
                    "vision_caption": caption.vision_caption,
                    "vision_model": caption.vision_model,
                    "embedding_text": caption.embedding_text,
                }
            )

        vector_drawings = page.get_drawings()
        pages.append(
            {
                "page_number": page_index,
                "text": text,
                "images": images,
                "vector_drawing_count": len(vector_drawings),
            }
        )
    return {"source": str(path.relative_to(DATA_DIR)), "pages": pages}


def classify_image(image_bytes: bytes, seen_hashes: set[str]) -> tuple[bool, str, dict]:
    try:
        from PIL import Image, ImageStat
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for image filtering.") from exc

    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest in seen_hashes:
        return False, "duplicate", {}
    seen_hashes.add(digest)

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGBA")
        width, height = image.size
        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            return False, "tiny_or_decorative", {}
        if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT or width * height > MAX_IMAGE_PIXELS:
            return False, "oversized_page_like_image", {}
        variance = ImageStat.Stat(image.convert("L")).var[0]
        if variance < 12:
            return False, "blank_or_low_contrast", {}
    return True, "kept", {"hash": digest[:16], "extension": "png", "width": width, "height": height}


def save_image_artifact(source_path: Path, image_bytes: bytes, image_hash: str, extension: str) -> str:
    artifact_dir = IMAGE_ARTIFACTS_DIR / source_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{source_path.stem}_{image_hash}.{extension}"
    artifact_path.write_bytes(image_bytes)
    return str(artifact_path.relative_to(DATA_DIR))


def _nearby_text(text: str, max_chars: int = 500) -> str:
    return " ".join(text.split())[:max_chars]

