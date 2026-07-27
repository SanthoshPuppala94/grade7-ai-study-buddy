import base64

from app.services.vision_captioner import ImageCaptionContext, VisionCaptioner


def test_vision_captioner_uses_image_and_context():
    image_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05"
        b"\xfe\x02\xfeA\xe2&\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    caption = VisionCaptioner().caption_image(
        image_bytes=image_bytes,
        media_type="image/png",
        context=ImageCaptionContext(
            subject="mathematics",
            grade=7,
            source="sample.pdf",
            page_number=1,
            nearby_text="A lakh varieties of rice and large numbers.",
        ),
    )

    assert base64.b64decode(caption.image_base64) == image_bytes
    assert "Grade 7 mathematics" in caption.vision_caption
    assert "lakh varieties" in caption.embedding_text

