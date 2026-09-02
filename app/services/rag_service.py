import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from app.database import SessionLocal, engine
from app.models import ManualChunk, ManualDocument, utcnow


MANUAL_DIR = Path(__file__).resolve().parents[1] / "data" / "manuals"
MANIFEST_PATH = MANUAL_DIR / "manifest.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
logger = logging.getLogger(__name__)
_model = None
_ranker = None


class ManualIndexError(RuntimeError):
    """Raised when the HVAC manual index cannot be built or searched."""


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def _embed(texts: list[str]) -> np.ndarray:
    embeddings = np.asarray(list(_get_model().embed(texts)), dtype="float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def _extract_pdf_pages(pdf_path: Path) -> list[dict]:
    completed = subprocess.run(
        [sys.executable, "-m", "app.services.pdf_extract_worker", str(pdf_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown parser error"
        raise ManualIndexError(f"PDF extraction failed ({detail})")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ManualIndexError("PDF extractor returned invalid output") from exc


def _clean_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    printable = "".join(
        character if character in "\n\t" or 32 <= ord(character) <= 126 else " "
        for character in normalized
    )
    printable = re.sub(r"[^A-Za-z0-9\s]{4,}", " ", printable)
    return " ".join(token for token in printable.split() if len(token) <= 100)


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_./-]*", value.lower())


def _load_manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManualIndexError(f"Cannot load manual metadata manifest: {exc}") from exc


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    chunk_index = 0
    for page_data in pages:
        content = _clean_text(page_data["text"])
        words = content.split()
        for start in range(0, len(words), 200):
            chunk_words = words[start : start + 250]
            if len(chunk_words) <= 50:
                continue
            chunk_text = " ".join(chunk_words)[:2000]
            chunks.append(
                {
                    "page": page_data["page"],
                    "section": " ".join(chunk_words[:12])[:180],
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                }
            )
            chunk_index += 1
    return chunks


def ingest_manuals(db: Session | None = None, force: bool = False) -> dict:
    """Persist manual chunks, metadata, and embeddings in the configured database."""
    owns_session = db is None
    db = db or SessionLocal()
    manifest = _load_manifest()
    summary = {"indexed_documents": 0, "indexed_chunks": 0, "skipped_documents": [], "unchanged_documents": 0}

    try:
        for pdf_path in sorted(MANUAL_DIR.glob("*.pdf")):
            metadata = manifest.get(pdf_path.name)
            if metadata is None:
                summary["skipped_documents"].append({"document": pdf_path.name, "reason": "missing manifest metadata"})
                continue

            content_hash = _file_hash(pdf_path)
            existing = db.query(ManualDocument).filter(ManualDocument.filename == pdf_path.name).first()
            if existing and existing.content_hash == content_hash and existing.chunks and not force:
                summary["unchanged_documents"] += 1
                continue

            try:
                pages = _extract_pdf_pages(pdf_path)
                chunks = _chunk_pages(pages)
                if not chunks:
                    raise ManualIndexError("no extractable chunks")
                embeddings = _embed([chunk["content"] for chunk in chunks])
            except Exception as exc:
                logger.warning("Skipping unreadable manual %s: %s", pdf_path.name, exc)
                summary["skipped_documents"].append({"document": pdf_path.name, "reason": str(exc)})
                continue

            if existing:
                db.delete(existing)
                db.flush()

            document = ManualDocument(
                filename=pdf_path.name,
                title=metadata["title"],
                manufacturer=metadata.get("manufacturer", "unknown"),
                equipment_types=json.dumps(metadata.get("equipment_types", [])),
                model_families=json.dumps(metadata.get("model_families", [])),
                refrigerants=json.dumps(metadata.get("refrigerants", [])),
                document_version=metadata.get("document_version"),
                content_hash=content_hash,
                indexed_at=utcnow(),
            )
            db.add(document)
            db.flush()

            for chunk_data, embedding in zip(chunks, embeddings):
                chunk = ManualChunk(
                    document_id=document.id,
                    page=chunk_data["page"],
                    section=chunk_data["section"],
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    embedding_json=json.dumps(embedding.tolist()),
                )
                db.add(chunk)
                db.flush()
                if engine.dialect.name == "postgresql":
                    vector_literal = "[" + ",".join(str(float(value)) for value in embedding) + "]"
                    db.execute(
                        text("UPDATE manual_chunks SET embedding_vector = CAST(:embedding AS vector) WHERE id = :id"),
                        {"embedding": vector_literal, "id": chunk.id},
                    )

            summary["indexed_documents"] += 1
            summary["indexed_chunks"] += len(chunks)
            db.commit()
        return summary
    except Exception:
        db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


def _metadata_matches(document: ManualDocument, manufacturer: str | None, equipment_model: str | None, equipment_type: str | None) -> bool:
    if manufacturer and document.manufacturer.lower() not in {manufacturer.lower(), "unknown"}:
        return False
    equipment_types = [item.lower() for item in json.loads(document.equipment_types or "[]")]
    if equipment_type and equipment_types and equipment_type.lower() not in equipment_types:
        return False
    families = [item.upper() for item in json.loads(document.model_families or "[]")]
    if equipment_model and families:
        normalized_model = equipment_model.upper()
        if not any(family in normalized_model for family in families):
            return False
    return True


def _candidate_chunks(db: Session, manufacturer: str | None, equipment_model: str | None, equipment_type: str | None) -> list[ManualChunk]:
    query = db.query(ManualChunk).join(ManualDocument)
    if manufacturer:
        query = query.filter(
            or_(func.lower(ManualDocument.manufacturer) == manufacturer.lower(), ManualDocument.manufacturer == "unknown")
        )
    chunks = query.all()
    return [
        chunk for chunk in chunks
        if _metadata_matches(chunk.document, manufacturer, equipment_model, equipment_type)
    ]


def _rrf(keyword_ids: list[int], vector_ids: list[int]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked_ids in (keyword_ids, vector_ids):
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def _postgres_rankings(db: Session, query: str, query_embedding: np.ndarray, candidate_ids: list[int]):
    """Use PostgreSQL FTS and pgvector when the production backend is active."""
    if not candidate_ids:
        return [], []
    id_list = ",".join(str(int(chunk_id)) for chunk_id in candidate_ids)
    keyword_rows = db.execute(
        text(
            f"SELECT id, ts_rank_cd(to_tsvector('english', content), "
            f"websearch_to_tsquery('english', :query)) AS score "
            f"FROM manual_chunks WHERE id IN ({id_list}) "
            f"AND to_tsvector('english', content) @@ websearch_to_tsquery('english', :query) "
            f"ORDER BY score DESC LIMIT 20"
        ),
        {"query": query},
    ).all()
    vector_literal = "[" + ",".join(str(float(value)) for value in query_embedding) + "]"
    vector_rows = db.execute(
        text(
            f"SELECT id, 1 - (embedding_vector <=> CAST(:embedding AS vector)) AS score "
            f"FROM manual_chunks WHERE id IN ({id_list}) AND embedding_vector IS NOT NULL "
            f"ORDER BY embedding_vector <=> CAST(:embedding AS vector) LIMIT 20"
        ),
        {"embedding": vector_literal},
    ).all()
    return [(int(row.id), float(row.score)) for row in keyword_rows], [
        (int(row.id), float(row.score)) for row in vector_rows
    ]


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    if os.getenv("RERANK_ENABLED", "true").lower() != "true" or len(candidates) < 2:
        return candidates
    global _ranker
    try:
        from flashrank import Ranker, RerankRequest

        if _ranker is None:
            _ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir=".cache/flashrank")
        passages = [
            {"id": str(item["id"]), "text": item["text"], "meta": {"rrf_score": item["rrf_score"]}}
            for item in candidates
        ]
        reranked = _ranker.rerank(RerankRequest(query=query, passages=passages))
        by_id = {str(item["id"]): item for item in candidates}
        output = []
        for result in reranked:
            item = by_id[str(result["id"])]
            item["rerank_score"] = float(result["score"])
            output.append(item)
        return output
    except Exception as exc:
        logger.warning("Reranker unavailable; using RRF order: %s", exc)
        return candidates


def hybrid_search(
    query: str,
    *,
    top_k: int = 5,
    manufacturer: str | None = None,
    equipment_model: str | None = None,
    equipment_type: str | None = None,
    db: Session | None = None,
) -> list[dict]:
    if not query.strip():
        raise ValueError("Search query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    owns_session = db is None
    db = db or SessionLocal()
    try:
        if db.query(ManualChunk).count() == 0:
            ingest_manuals(db=db)

        chunks = _candidate_chunks(db, manufacturer, equipment_model, equipment_type)
        if not chunks:
            return []

        query_embedding = _embed([query])[0]
        if engine.dialect.name == "postgresql":
            keyword_pairs, vector_pairs = _postgres_rankings(
                db, query, query_embedding, [chunk.id for chunk in chunks]
            )
            by_id = {chunk.id: chunk for chunk in chunks}
            keyword_ranked = [(by_id[chunk_id], score) for chunk_id, score in keyword_pairs]
            dense_ranked = [(by_id[chunk_id], score) for chunk_id, score in vector_pairs]
        else:
            tokenized_query = _tokenize(query)
            corpus = [_tokenize(chunk.content) for chunk in chunks]
            bm25_scores = BM25Okapi(corpus).get_scores(tokenized_query)
            keyword_ranked = sorted(zip(chunks, bm25_scores), key=lambda pair: pair[1], reverse=True)[:20]

            dense_ranked = []
            for chunk in chunks:
                embedding = np.asarray(json.loads(chunk.embedding_json), dtype="float32")
                dense_ranked.append((chunk, float(np.dot(query_embedding, embedding))))
            dense_ranked.sort(key=lambda pair: pair[1], reverse=True)
            dense_ranked = dense_ranked[:20]

        rrf_scores = _rrf(
            [chunk.id for chunk, _score in keyword_ranked],
            [chunk.id for chunk, _score in dense_ranked],
        )
        keyword_scores = {chunk.id: float(score) for chunk, score in keyword_ranked}
        vector_scores = {chunk.id: float(score) for chunk, score in dense_ranked}
        by_id = {chunk.id: chunk for chunk in chunks}
        fused_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[: max(top_k * 3, 10)]

        candidates = []
        for chunk_id in fused_ids:
            chunk = by_id[chunk_id]
            candidates.append(
                {
                    "id": chunk.id,
                    "text": chunk.content,
                    "document": chunk.document.filename,
                    "title": chunk.document.title,
                    "manufacturer": chunk.document.manufacturer,
                    "page": chunk.page,
                    "section": chunk.section,
                    "keyword_score": keyword_scores.get(chunk_id, 0.0),
                    "vector_score": vector_scores.get(chunk_id, 0.0),
                    "rrf_score": rrf_scores[chunk_id],
                    "score": rrf_scores[chunk_id],
                }
            )

        reranked = _rerank(query, candidates)
        for item in reranked:
            item["score"] = item.get("rerank_score", item["rrf_score"])
        return reranked[:top_k]
    finally:
        if owns_session:
            db.close()


def search_manual(query: str, top_k: int = 3, **filters) -> list[dict]:
    """Backward-compatible facade used by the Stage 1/2 diagnosis service."""
    return hybrid_search(query, top_k=top_k, **filters)
