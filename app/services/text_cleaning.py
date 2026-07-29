import re


NOISE_PATTERNS = [
    r"(?im)^\s*reprint\s+\d{4}(?:-\d{2,4})?\s*$",
    r"(?im)^\s*copyright\s+.*$",
    r"(?im)^\s*all\s+rights\s+reserved\.?\s*$",
    r"(?im)^\s*page\s+\d+\s*$",
    r"(?m)^\s*\d+\s*$",
]


def clean_pdf_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    for pattern in NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    return cleaned.strip()
