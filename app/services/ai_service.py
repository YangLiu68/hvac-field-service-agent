import os

from dotenv import load_dotenv
from openai import OpenAI

from app.schemas import AnalysisResult, DiagnosticContent, SourceItem
from app.services.rag_service import search_manual


load_dotenv()


class AIServiceError(RuntimeError):
    """Raised when diagnosis cannot be completed safely."""


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AIServiceError(
            "OPENAI_API_KEY is not configured. Copy .env.example to .env and add a valid key."
        )
    # Supports OpenAI itself and OpenAI-compatible providers. Do not append a
    # second /v1 here: the value in .env should already be the full API base.
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not base_url and os.getenv("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"
    return OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
        max_retries=0,
    )


def analyze_job(
    notes: str,
    equipment_model: str,
    *,
    manufacturer: str | None = None,
    equipment_type: str | None = None,
    client: OpenAI | None = None,
) -> AnalysisResult:
    if not notes.strip():
        raise AIServiceError("Technician notes cannot be empty")
    if not equipment_model.strip():
        raise AIServiceError("Equipment model cannot be empty")

    query = f"Equipment model: {equipment_model}\nTechnician notes: {notes}"
    rag_results = search_manual(
        query,
        top_k=5,
        manufacturer=manufacturer,
        equipment_model=equipment_model,
        equipment_type=equipment_type,
    )
    has_manual_evidence = bool(rag_results)
    context_parts = []
    for i, result in enumerate(rag_results, start=1):
        context_parts.append(
            f"Source {i}\n"
            f"Document: {result['document']}\n"
            f"Page: {result['page']}\n"
            f"Content: {result['text']}"
        )
    context = "\n\n".join(context_parts) if context_parts else "No matching manual excerpt is available for this equipment."

    try:
        response = (client or _get_client()).responses.parse(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative HVAC diagnostic assistant. When service-manual "
                        "evidence is supplied, use it. When it is not supplied, provide only a "
                        "preliminary hypothesis based on the technician notes, clearly recommend a "
                        "safe verification step, and keep confidence low. Do not invent citations. "
                        "Confidence is an estimate from 0 to 1, not a guarantee."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Equipment model:\n{equipment_model}\n\n"
                        f"Technician notes:\n{notes}\n\n"
                        f"Retrieved service-manual evidence:\n{context}\n\n"
                        "Return the likely cause, confidence, and safest next diagnostic action."
                    ),
                },
            ],
            text_format=DiagnosticContent,
        )
    except AIServiceError:
        raise
    except Exception as exc:
        raise AIServiceError(f"The AI diagnosis request failed: {exc}") from exc

    diagnostic = response.output_parsed
    if diagnostic is None:
        raise AIServiceError("The AI service returned no structured diagnosis")

    if not has_manual_evidence:
        diagnostic = diagnostic.model_copy(update={"confidence": min(diagnostic.confidence, 0.35)})

    sources = [
        SourceItem(
            document=item["document"],
            page=item["page"],
            text=item["text"],
            score=item["score"],
        )
        for item in rag_results
    ]
    return AnalysisResult(**diagnostic.model_dump(), sources=sources)


def answer_field_question(
    question: str,
    *,
    equipment_model: str,
    manufacturer: str | None = None,
    equipment_type: str | None = None,
    error_code: str | None = None,
    job_notes: str | None = None,
    history: list[dict[str, str]] | None = None,
    client: OpenAI | None = None,
) -> str:
    """Return a conversational, evidence-aware response for a technician.

    This is deliberately separate from the tool-only diagnostic orchestrator:
    questions can be answered naturally, while measurements and protected
    actions remain inside the audited guided workflow.
    """
    if not question.strip():
        raise AIServiceError("Assistant message cannot be empty")

    query = "\n".join(filter(None, [
        f"Model: {equipment_model}",
        f"Error code: {error_code}" if error_code else None,
        f"Reported issue: {job_notes}" if job_notes else None,
        f"Technician question: {question}",
    ]))
    results = search_manual(
        query,
        top_k=3,
        manufacturer=manufacturer,
        equipment_model=equipment_model,
        equipment_type=equipment_type,
    )
    evidence = "\n\n".join(
        f"[{index}] {item['document']}, p. {item['page']}: {item['text']}"
        for index, item in enumerate(results, start=1)
    ) or "No relevant local manual excerpt was found."
    conversation = (history or [])[-8:]
    messages = [
        {
            "role": "system",
            "content": (
                "You are Fieldwise, a careful HVAC assistant for trained field technicians. "
                "Answer the technician's question naturally and concisely. Use the supplied local-manual "
                "evidence when relevant and cite it as [1], [2], etc. Do not invent a manual citation. "
                "When evidence is missing, label your answer preliminary and state the safest next check. "
                "Never authorize hazardous electrical or refrigerant work; recommend qualified escalation instead. "
                "Do not call tools and do not claim that a diagnosis is complete."
            ),
        },
        *conversation,
        {
            "role": "user",
            "content": (
                f"Equipment: {(manufacturer + ' ') if manufacturer else ''}{equipment_model}\n"
                f"Type: {equipment_type or 'unknown'}\n"
                f"Error code: {error_code or 'not provided'}\n"
                f"Reported issue: {job_notes or 'not provided'}\n\n"
                f"Local manual evidence:\n{evidence}\n\n"
                f"Question: {question}"
            ),
        },
    ]
    try:
        response = (client or _get_client()).chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )
    except Exception as exc:
        raise AIServiceError(f"The assistant chat request failed: {exc}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise AIServiceError("The assistant returned an empty reply")
    return content.strip()
