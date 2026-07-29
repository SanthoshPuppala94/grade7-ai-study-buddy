from app.services.text_cleaning import clean_pdf_text


def test_clean_pdf_text_removes_common_noise_with_regex():
    text = """
    1.1   Large Numbers Around Us
    One lakh   is written as 100000.

    Reprint 2026-27
    Page 12
    12
    """

    cleaned = clean_pdf_text(text)

    assert "Large Numbers Around Us" in cleaned
    assert "One lakh is written as 100000." in cleaned
    assert "Reprint" not in cleaned
    assert "Page 12" not in cleaned
