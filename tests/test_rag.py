from app.services.vector_store import StudyVectorStore


def test_vector_store_retrieves_large_numbers_topic():
    results = StudyVectorStore().search("What is one lakh?", k=3, subject="mathematics")

    assert results
    assert any("lakh" in result["text"].lower() for result in results)

