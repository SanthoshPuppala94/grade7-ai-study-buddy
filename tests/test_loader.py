from app.services.document_loader import load_and_split_documents


def test_loader_splits_sample_documents_with_metadata():
    chunks = load_and_split_documents()

    assert chunks
    assert all("subject" in chunk.metadata for chunk in chunks)
    assert any(chunk.metadata["subject"] == "mathematics" for chunk in chunks)

