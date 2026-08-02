import math
import re
from collections import Counter
from dataclasses import dataclass

from app.services.document_loader import load_and_split_documents
from app.services.embeddings import HashEmbedding


@dataclass
class Chunk:
    source: str
    text: str
    metadata: dict
    embedding: list[float]


class StudyVectorStore:
    def __init__(self):
        self.embedding_model = HashEmbedding()
        self._chunks: list[Chunk] = []

    def ensure_index(self) -> None:
        if self._chunks:
            return
        docs = load_and_split_documents()
        self._chunks = [
            Chunk(
                source=f"{doc.metadata['source']}#chunk-{doc.metadata['chunk_index']}",
                text=doc.page_content,
                metadata=doc.metadata,
                embedding=self.embedding_model.embed_query(doc.page_content),
            )
            for doc in docs
        ]

    def search(
        self,
        query: str,
        k: int = 4,
        subject: str | None = None,
        retrieval_mode: str = "hybrid",
    ) -> list[dict]:
        if retrieval_mode == "dense":
            return self.dense_search(query=query, k=k, subject=subject)
        if retrieval_mode == "sparse":
            return self.sparse_search(query=query, k=k, subject=subject)
        if retrieval_mode == "hybrid":
            return self.hybrid_search(query=query, k=k, subject=subject)
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    def dense_search(self, query: str, k: int = 4, subject: str | None = None) -> list[dict]:
        self.ensure_index()
        query_embedding = self.embedding_model.embed_query(query)
        scored = []
        for chunk in self._filtered_chunks(subject):
            scored.append(
                {
                    "source": chunk.source,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": _cosine(query_embedding, chunk.embedding),
                    "retrieval_mode": "dense",
                    "related_images": chunk.metadata.get("related_images", []),
                }
            )
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:k]

    def sparse_search(self, query: str, k: int = 4, subject: str | None = None) -> list[dict]:
        self.ensure_index()
        query_terms = _tokenize(query)
        scored = []
        for chunk in self._filtered_chunks(subject):
            score = _sparse_score(query_terms, chunk.text, chunk.metadata)
            if score <= 0:
                continue
            scored.append(
                {
                    "source": chunk.source,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": score,
                    "retrieval_mode": "sparse",
                    "related_images": chunk.metadata.get("related_images", []),
                }
            )
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:k]

    def hybrid_search(self, query: str, k: int = 4, subject: str | None = None) -> list[dict]:
        dense_results = self.dense_search(query=query, k=max(k * 2, 8), subject=subject)
        sparse_results = self.sparse_search(query=query, k=max(k * 2, 8), subject=subject)
        fused = _reciprocal_rank_fusion(
            ranked_result_sets=[dense_results, sparse_results],
            weights=[0.6, 0.4],
        )
        return fused[:k]

    def _filtered_chunks(self, subject: str | None) -> list[Chunk]:
        return [
            chunk
            for chunk in self._chunks
            if subject is None or chunk.metadata.get("subject") == subject
        ]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right)) / (
        math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)) or 1.0
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text.lower())


def _sparse_score(query_terms: list[str], text: str, metadata: dict) -> float:
    searchable_text = " ".join(
        [
            text,
            str(metadata.get("subject", "")),
            str(metadata.get("chapter", "")),
            str(metadata.get("section_title", "")),
            str(metadata.get("file_name", "")),
        ]
    )
    document_terms = Counter(_tokenize(searchable_text))
    score = 0.0
    for term in query_terms:
        exact_frequency = document_terms.get(term, 0)
        if exact_frequency:
            score += 1.0 + math.log1p(exact_frequency)
    return score


def _reciprocal_rank_fusion(ranked_result_sets: list[list[dict]], weights: list[float]) -> list[dict]:
    fused_scores: dict[str, float] = {}
    fused_results: dict[str, dict] = {}
    rank_constant = 60

    for results, weight in zip(ranked_result_sets, weights):
        for rank, result in enumerate(results, start=1):
            source = result["source"]
            fused_scores[source] = fused_scores.get(source, 0.0) + weight / (rank_constant + rank)
            fused_results[source] = {**result, "retrieval_mode": "hybrid"}

    for source, result in fused_results.items():
        result["score"] = fused_scores[source]
    return sorted(fused_results.values(), key=lambda item: item["score"], reverse=True)
