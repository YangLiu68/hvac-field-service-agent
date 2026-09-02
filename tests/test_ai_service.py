from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.schemas import DiagnosticContent
from app.services.ai_service import analyze_job, answer_field_question


def test_analyze_job_attaches_retrieved_sources():
    retrieved = [
        {
            "document": "manual.pdf",
            "page": 7,
            "text": "Check airflow and evaporator coil condition.",
            "score": 0.82,
        }
    ]
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=DiagnosticContent(
            likely_cause="Restricted airflow",
            confidence=0.74,
            recommendation="Inspect the filter and evaporator coil.",
        )
    )

    with patch("app.services.ai_service.search_manual", return_value=retrieved):
        result = analyze_job(
            "AC is not cooling and airflow is weak",
            "MODEL-123",
            client=client,
        )

    assert result.likely_cause == "Restricted airflow"
    assert result.sources[0].document == "manual.pdf"
    assert result.sources[0].page == 7
    assert result.sources[0].score == 0.82


def test_analyze_job_rejects_blank_notes():
    try:
        analyze_job("   ", "MODEL-123", client=Mock())
    except Exception as exc:
        assert "notes" in str(exc).lower()
    else:
        raise AssertionError("Expected blank technician notes to be rejected")


def test_field_chat_uses_manual_evidence_and_returns_natural_reply():
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Check the filter first. [1]"))]
    )
    retrieved = [{
        "document": "carrier-manual.pdf",
        "page": 12,
        "text": "Inspect return-air filter for restriction.",
        "score": 0.91,
    }]

    with patch("app.services.ai_service.search_manual", return_value=retrieved):
        reply = answer_field_question(
            "Could a dirty filter cause this?",
            equipment_model="MODEL-123",
            manufacturer="Carrier",
            job_notes="Warm supply air",
            client=client,
        )

    assert reply == "Check the filter first. [1]"
    request = client.chat.completions.create.call_args.kwargs
    assert "carrier-manual.pdf" in request["messages"][-1]["content"]
