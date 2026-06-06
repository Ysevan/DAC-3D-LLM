"""Document retrieval for the DAC-3D assistant."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import AppConfig
from knowledge_base.build_kb import cosine_similarity, create_embedding_backend, tokenize


@dataclass(slots=True)
class RetrievalItem:
    """Structured retrieval result returned to the application layer."""

    text: str
    source: str
    title: str
    section: str
    document_type: str
    chunk_id: int
    score: float
    metadata: dict[str, Any]


class Retriever:
    """Retrieve the most relevant chunks from the persisted local store."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._manifest = self._load_manifest(config.vector_store_manifest_path)
        self._chunks: list[dict[str, Any]] = list(self._manifest.get("chunks", []))
        self._storage_backend = str(self._manifest.get("storage_backend", "manifest"))
        self._embedding_backend_name = str(self._manifest.get("embedding_backend", "hashing"))
        self._embedder = create_embedding_backend(
            config,
            preferred_backend=self._embedding_backend_name,
        )
        self._chroma_collection = self._load_chroma_collection()
        self._sparse_records = self._build_sparse_records()
        self._average_record_length = (
            sum(record["length"] for record in self._sparse_records) / len(self._sparse_records)
            if self._sparse_records
            else 1.0
        )
        self._document_frequency = self._build_document_frequency()

    @classmethod
    def from_config(cls, config: AppConfig) -> "Retriever":
        return cls(config)

    def _load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {manifest_path}. Run build_kb.py first."
            )
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _load_chroma_collection(self) -> Any | None:
        if self._storage_backend != "chroma":
            return None
        try:
            import chromadb
        except Exception:
            return None
        try:
            client = chromadb.PersistentClient(path=str(self.config.vector_store_path))
            return client.get_collection(self.config.vector_store_collection)
        except Exception:
            return None

    def _build_sparse_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for chunk in self._chunks:
            search_text = " ".join(
                [
                    str(chunk.get("title", "")),
                    str(chunk.get("section", "")),
                    str(chunk.get("text", "")),
                ]
            ).strip()
            token_counts = Counter(tokenize(search_text))
            records.append(
                {
                    "key": (str(chunk.get("source", "")), int(chunk.get("chunk_id", -1))),
                    "token_counts": token_counts,
                    "length": sum(token_counts.values()) or 1,
                    "document_type": str(chunk.get("document_type", "")),
                }
            )
        return records

    def _build_document_frequency(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for record in self._sparse_records:
            frequencies.update(record["token_counts"].keys())
        return frequencies

    def retrieve(
        self,
        query: str,
        document_type: str | None = None,
        top_k: int | None = None,
        *,
        include_all: bool = False,
    ) -> list[RetrievalItem]:
        """Return the best matching chunks with scores and metadata."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        limit = None if include_all else max(1, top_k or self.config.retrieval_top_k)
        candidate_limit = len(self._chunks) if include_all else max(limit or 1, self.config.retrieval_candidate_k)
        candidates = self._query_candidates(query, document_type, candidate_limit)
        ranked_items: list[RetrievalItem] = []
        query_text = query.strip().lower()
        for candidate in candidates:
            combined_text = " ".join(
                [
                    str(candidate.get("title", "")),
                    str(candidate.get("section", "")),
                    str(candidate.get("text", "")),
                ]
            )
            lexical_score = self._lexical_bonus(query_tokens, query, combined_text)
            title_text = str(candidate.get("title", "")).strip()
            section_text = str(candidate.get("section", "")).strip()
            title_token_counts = Counter(tokenize(title_text))
            section_token_counts = Counter(tokenize(section_text))
            title_overlap = sum(min(title_token_counts.get(token, 0), 1) for token in query_tokens)
            section_overlap = sum(min(section_token_counts.get(token, 0), 1) for token in query_tokens)
            title_phrase_bonus = 0.12 if title_text and title_text.lower() in query_text else 0.0
            section_phrase_bonus = 1.45 if section_text and section_text.lower() in query_text else 0.0
            overview_penalty = (
                0.32
                if title_text
                and section_text
                and title_text.lower() == section_text.lower()
                and section_text.lower() not in query_text
                else 0.0
            )
            dense_score = float(candidate.get("dense_score", candidate.get("base_score", 0.0)))
            sparse_score = float(candidate.get("sparse_score", 0.0))
            hybrid_score = (dense_score * 0.68) + (sparse_score * 0.32)
            rank_fusion_bonus = self._rank_fusion_bonus(
                candidate.get("dense_rank"),
                candidate.get("sparse_rank"),
            )
            score = (
                hybrid_score
                + rank_fusion_bonus
                + lexical_score
                + (title_overlap * 0.02)
                + (section_overlap * 0.12)
                + title_phrase_bonus
                + section_phrase_bonus
                - overview_penalty
            )
            if score < self.config.min_retrieval_score:
                continue
            metadata = dict(candidate.get("metadata", {}))
            ranked_items.append(
                RetrievalItem(
                    text=str(candidate.get("text", "")),
                    source=str(candidate.get("source", "")),
                    title=str(candidate.get("title", "")),
                    section=str(candidate.get("section", "")),
                    document_type=str(candidate.get("document_type", "")),
                    chunk_id=int(candidate.get("chunk_id", -1)),
                    score=score,
                    metadata=metadata,
                )
            )

        ranked_items.sort(key=lambda item: item.score, reverse=True)
        deduped_items: list[RetrievalItem] = []
        seen_sections: set[tuple[str, str]] = set()
        for item in ranked_items:
            dedupe_key = (item.source, item.section or item.title)
            if dedupe_key in seen_sections:
                continue
            seen_sections.add(dedupe_key)
            deduped_items.append(item)
            if limit is not None and len(deduped_items) >= limit:
                break
        return deduped_items

    def _query_candidates(
        self,
        query: str,
        document_type: str | None,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        fetch_limit = max(candidate_limit * 3, 12)
        dense_candidates = self._query_dense_candidates(query, document_type, fetch_limit)
        sparse_candidates = self._query_sparse_candidates(query, document_type, fetch_limit)
        dense_candidates = self._normalize_score_field(dense_candidates, "dense_score")
        sparse_candidates = self._normalize_score_field(sparse_candidates, "sparse_score")

        merged: dict[tuple[str, int], dict[str, Any]] = {}
        for dense_rank, item in enumerate(dense_candidates, start=1):
            key = (str(item.get("source", "")), int(item.get("chunk_id", -1)))
            merged[key] = {
                **item,
                "dense_rank": dense_rank,
            }

        for sparse_rank, item in enumerate(sparse_candidates, start=1):
            key = (str(item.get("source", "")), int(item.get("chunk_id", -1)))
            existing = merged.get(key)
            if existing is None:
                merged[key] = {
                    **item,
                    "sparse_rank": sparse_rank,
                }
                continue
            existing["sparse_score"] = max(
                float(existing.get("sparse_score", 0.0)),
                float(item.get("sparse_score", 0.0)),
            )
            existing["sparse_rank"] = sparse_rank

        return list(merged.values())

    def _query_dense_candidates(
        self,
        query: str,
        document_type: str | None,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, int], dict[str, Any]] = {}

        if self._chroma_collection is not None:
            for item in self._query_chroma(query, document_type, candidate_limit):
                key = (str(item.get("source", "")), int(item.get("chunk_id", -1)))
                merged[key] = item

        for item in self._query_manifest(query, document_type):
            key = (str(item.get("source", "")), int(item.get("chunk_id", -1)))
            existing = merged.get(key)
            if existing is None or float(item.get("dense_score", 0.0)) > float(
                existing.get("dense_score", 0.0)
            ):
                merged[key] = item

        dense_candidates = list(merged.values())
        dense_candidates.sort(key=lambda entry: float(entry.get("dense_score", 0.0)), reverse=True)
        return dense_candidates[:candidate_limit]

    def _query_chroma(
        self,
        query: str,
        document_type: str | None,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        query_embedding = self._embedder.embed_texts([query])[0]
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": candidate_limit,
        }
        if document_type:
            query_kwargs["where"] = {"document_type": document_type}
        include = ["documents", "metadatas", "distances"]
        try:
            query_kwargs["include"] = include
            payload = self._chroma_collection.query(**query_kwargs)
        except TypeError:
            payload = self._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_limit,
                where=query_kwargs.get("where"),
            )
        except Exception:
            return []

        documents = list((payload.get("documents") or [[]])[0])
        metadatas = list((payload.get("metadatas") or [[]])[0])
        distances = list((payload.get("distances") or [[]])[0])
        items: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            chunk_metadata = dict(metadata or {})
            dense_score = 1.0 / (1.0 + float(distance or 0.0))
            items.append(
                {
                    "text": str(document),
                    "source": str(chunk_metadata.get("source", "")),
                    "title": str(chunk_metadata.get("title", "")),
                    "section": str(chunk_metadata.get("section", "")),
                    "document_type": str(chunk_metadata.get("document_type", "")),
                    "chunk_id": int(chunk_metadata.get("chunk_id", -1)),
                    "metadata": chunk_metadata,
                    "dense_score": dense_score,
                    "base_score": dense_score,
                }
            )
        return items

    def _query_manifest(self, query: str, document_type: str | None) -> list[dict[str, Any]]:
        query_embedding = self._embedder.embed_texts([query])[0]
        candidates: list[dict[str, Any]] = []
        for chunk in self._chunks:
            metadata = dict(chunk.get("metadata", {}))
            if document_type and metadata.get("document_type") != document_type:
                continue
            dense_score = cosine_similarity(query_embedding, list(chunk.get("embedding", [])))
            candidates.append(
                {
                    "text": str(chunk.get("text", "")),
                    "source": str(chunk.get("source", "")),
                    "title": str(chunk.get("title", "")),
                    "section": str(chunk.get("section", "")),
                    "document_type": str(chunk.get("document_type", "")),
                    "chunk_id": int(chunk.get("chunk_id", -1)),
                    "metadata": metadata,
                    "dense_score": dense_score,
                    "base_score": dense_score,
                }
            )
        return candidates

    def _query_sparse_candidates(
        self,
        query: str,
        document_type: str | None,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        items: list[dict[str, Any]] = []
        unique_query_tokens = list(dict.fromkeys(query_tokens))
        total_documents = max(len(self._sparse_records), 1)
        for record, chunk in zip(self._sparse_records, self._chunks):
            if document_type and record.get("document_type") != document_type:
                continue
            token_counts: Counter[str] = record["token_counts"]
            sparse_score = 0.0
            for token in unique_query_tokens:
                term_frequency = token_counts.get(token, 0)
                if term_frequency <= 0:
                    continue
                document_frequency = self._document_frequency.get(token, 0)
                inverse_document_frequency = math.log(
                    1.0 + ((total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
                )
                denominator = term_frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * (record["length"] / max(self._average_record_length, 1.0))
                )
                sparse_score += inverse_document_frequency * (
                    (term_frequency * (1.5 + 1.0)) / max(denominator, 1e-8)
                )
            if sparse_score <= 0.0:
                continue
            metadata = dict(chunk.get("metadata", {}))
            items.append(
                {
                    "text": str(chunk.get("text", "")),
                    "source": str(chunk.get("source", "")),
                    "title": str(chunk.get("title", "")),
                    "section": str(chunk.get("section", "")),
                    "document_type": str(chunk.get("document_type", "")),
                    "chunk_id": int(chunk.get("chunk_id", -1)),
                    "metadata": metadata,
                    "sparse_score": sparse_score,
                }
            )
        items.sort(key=lambda entry: float(entry.get("sparse_score", 0.0)), reverse=True)
        return items[:candidate_limit]

    def _normalize_score_field(
        self,
        items: list[dict[str, Any]],
        field_name: str,
    ) -> list[dict[str, Any]]:
        if not items:
            return items
        max_value = max(float(item.get(field_name, 0.0)) for item in items)
        if max_value <= 0.0:
            return items
        normalized_items: list[dict[str, Any]] = []
        for item in items:
            normalized_item = dict(item)
            normalized_item[field_name] = float(item.get(field_name, 0.0)) / max_value
            normalized_items.append(normalized_item)
        normalized_items.sort(key=lambda entry: float(entry.get(field_name, 0.0)), reverse=True)
        return normalized_items

    def _rank_fusion_bonus(self, dense_rank: Any, sparse_rank: Any) -> float:
        bonus = 0.0
        for rank in (dense_rank, sparse_rank):
            if isinstance(rank, int) and rank > 0:
                bonus += 1.0 / (60.0 + rank)
        return bonus

    def _lexical_bonus(self, query_tokens: list[str], query: str, combined_text: str) -> float:
        token_counts = Counter(tokenize(combined_text))
        overlap = sum(min(token_counts.get(token, 0), 1) for token in query_tokens)
        heading_bonus = 0.12 if any(token in tokenize(combined_text[:240]) for token in query_tokens) else 0.0
        phrase_bonus = 0.18 if query.strip() and query.strip().lower() in combined_text.lower() else 0.0
        return (overlap * 0.05) + heading_bonus + phrase_bonus
