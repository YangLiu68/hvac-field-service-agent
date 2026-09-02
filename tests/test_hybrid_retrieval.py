import json

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import ManualChunk, ManualDocument
from app.services.rag_service import hybrid_search


def test_hybrid_search_filters_model_and_combines_scores(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'retrieval.db'}")
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    db = Session()
    try:
        carrier = ManualDocument(
            filename="carrier.pdf",
            title="Carrier Fan Coil Manual",
            manufacturer="Carrier",
            equipment_types=json.dumps(["fan_coil"]),
            model_families=json.dumps(["FK4B"]),
            refrigerants="[]",
            content_hash="a",
        )
        samsung = ManualDocument(
            filename="samsung.pdf",
            title="Samsung VRF Manual",
            manufacturer="Samsung",
            equipment_types=json.dumps(["VRF"]),
            model_families=json.dumps(["VRC"]),
            refrigerants=json.dumps(["R-410A"]),
            content_hash="b",
        )
        db.add_all([carrier, samsung])
        db.flush()
        db.add_all(
            [
                ManualChunk(
                    document_id=carrier.id,
                    page=7,
                    section="Airflow",
                    chunk_index=0,
                    content="Low airflow can be caused by a clogged return air filter.",
                    embedding_json=json.dumps([1.0, 0.0]),
                ),
                ManualChunk(
                    document_id=samsung.id,
                    page=12,
                    section="VRF",
                    chunk_index=0,
                    content="VRF communication error troubleshooting.",
                    embedding_json=json.dumps([0.0, 1.0]),
                ),
            ]
        )
        db.commit()
        monkeypatch.setenv("RERANK_ENABLED", "false")
        monkeypatch.setattr("app.services.rag_service._embed", lambda _texts: np.array([[1.0, 0.0]], dtype="float32"))

        results = hybrid_search(
            "low airflow clogged filter",
            manufacturer="Carrier",
            equipment_model="FK4B-001",
            equipment_type="fan_coil",
            top_k=5,
            db=db,
        )
        assert len(results) == 1
        assert results[0]["document"] == "carrier.pdf"
        assert results[0]["keyword_score"] != 0
        assert results[0]["vector_score"] == 1.0
        assert results[0]["rrf_score"] > 0
    finally:
        db.close()
        engine.dispose()
