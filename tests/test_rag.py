from app.services.vector_store import StudyVectorStore


def test_vector_store_retrieves_large_numbers_topic():
    results = StudyVectorStore().search("What is one lakh?", k=3, subject="mathematics")

    assert results
    assert any("lakh" in result["text"].lower() for result in results)
    assert all(result["retrieval_mode"] == "hybrid" for result in results)


def test_sparse_search_finds_exact_terms():
    results = StudyVectorStore().search("photosynthesis chlorophyll", k=2, retrieval_mode="sparse")

    assert results
    assert results[0]["retrieval_mode"] == "sparse"
    assert "photosynthesis" in results[0]["text"].lower()


def test_dense_and_hybrid_modes_are_available():
    store = StudyVectorStore()

    dense_results = store.search("plants make food from sunlight", k=2, retrieval_mode="dense")
    hybrid_results = store.search("plants make food from sunlight", k=2, retrieval_mode="hybrid")

    assert dense_results
    assert hybrid_results
    assert all(result["retrieval_mode"] == "dense" for result in dense_results)
    assert all(result["retrieval_mode"] == "hybrid" for result in hybrid_results)
    assert hybrid_results[0]["fusion"]["dense_weight"] == 0.6
    assert hybrid_results[0]["fusion"]["sparse_weight"] == 0.4
