# src/api/routers.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agents.final.auditor import CaseAuditReport, CaseAuditor
from src.contracts.case import CaseResult
from src.mcp.client import MCPToolClient

router = APIRouter()
FINAL_API_USE_LLM_MITIGATOR = True
FINAL_API_PERSIST_CASES = True
LEGACY_SUPPORTED_DATASETS = (
    "bot-iot",
    "bot_iot",
    "edge-iiotset",
    "edge_iiotset",
    "generic",
    "iot-23",
    "iot23",
    "ton-iot",
    "ton_iot",
    "unknown",
)
LEGACY_ANALYSIS_DETAIL = (
    "Endpoint retirado: usa /cases/analyze. El flujo final exige Mistral, "
    "abstencion controlada y revision por el juez."
)


class AnalyzeRequest(BaseModel):
    """Contrato conservado solo para responder 410 en endpoints retirados."""

    dataset: str = Field(default="generic")
    row: dict[str, Any] | None = None
    text: str | None = None
    raw: dict[str, Any] | str | None = None
    canonical_event: dict[str, Any] | None = None
    source_file: str = "api"
    row_id: str | int = 0
    use_llm_mitigator: bool | None = None
    persist: bool = False


class CaseAnalyzeRequest(BaseModel):
    """Entrada del flujo final; sus politicas operativas no son desactivables."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default="generic")
    row: dict[str, Any] | None = None
    text: str | None = None
    canonical_event: dict[str, Any] | None = None
    source_file: str = "api"
    row_id: str | int = 0


class AnalyzeFileRequest(BaseModel):
    path: str
    dataset: str = "generic"
    limit: int = Field(default=10, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    use_llm_mitigator: bool | None = None
    persist: bool = False


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _run_final_case(
    raw_input: dict[str, Any],
    *,
    use_llm_mitigator: bool = FINAL_API_USE_LLM_MITIGATOR,
    persist: bool = FINAL_API_PERSIST_CASES,
) -> dict[str, Any]:
    from src.orchestration.mcp_graph import run_case

    case = run_case(
        raw_input,
        use_llm_mitigator=use_llm_mitigator,
        persist=persist,
    )
    return case.model_dump(mode="json")


@router.post("/events/analyze", deprecated=True)
def analyze_event(request: AnalyzeRequest) -> dict[str, Any]:
    """Endpoint legacy cerrado para impedir rutas con adaptadores."""
    del request
    raise HTTPException(status_code=410, detail=LEGACY_ANALYSIS_DETAIL)


@router.post("/cases/analyze")
def analyze_case(request: CaseAnalyzeRequest) -> dict[str, Any]:
    """Analiza un evento con el grafo final MCP y devuelve el CaseResult con traza."""
    if not any(
        value is not None for value in (request.row, request.text, request.canonical_event)
    ):
        raise HTTPException(
            status_code=400,
            detail="Se requiere al menos uno de: row, text, canonical_event",
        )
    raw_input = request.model_dump(mode="json")
    return _run_final_case(
        raw_input,
        use_llm_mitigator=FINAL_API_USE_LLM_MITIGATOR,
        persist=FINAL_API_PERSIST_CASES,
    )


@router.get("/cases/{case_id}/audit", response_model=CaseAuditReport)
def audit_case(case_id: str) -> CaseAuditReport:
    """Audita de forma independiente un ``CaseResult`` ya persistido.

    El informe se recalcula en cada consulta y no modifica el caso, su traza
    ni la decision operacional tomada previamente por el juez.
    """
    try:
        stored = MCPToolClient(mode="inprocess").call(
            "case_memory",
            "get_case",
            case_id=case_id,
            include_trace=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="No se pudo acceder a la memoria de casos",
        ) from exc

    if not isinstance(stored, dict) or stored.get("ok") is not True:
        error = str(stored.get("error", "")) if isinstance(stored, dict) else ""
        if "Caso no encontrado:" in error:
            raise HTTPException(
                status_code=404,
                detail=f"Caso no encontrado: {case_id}",
            )
        raise HTTPException(
            status_code=503,
            detail="No se pudo acceder a la memoria de casos",
        )

    try:
        case = CaseResult.model_validate(stored.get("case"))
    except (ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="El caso persistido no cumple el contrato CaseResult",
        ) from exc

    if case.case_id != case_id:
        raise HTTPException(
            status_code=500,
            detail="El caso persistido no cumple el contrato CaseResult",
        )

    return CaseAuditor().audit(case)


@router.post("/datasets/adapt", deprecated=True)
def adapt_event(request: AnalyzeRequest) -> dict[str, Any]:
    """Endpoint legacy cerrado para impedir estandarizacion fuera del juez."""
    del request
    raise HTTPException(status_code=410, detail=LEGACY_ANALYSIS_DETAIL)


@router.get("/datasets/supported", deprecated=True)
def supported_datasets() -> dict[str, Any]:
    """Metadatos de formatos legacy; no son rutas del flujo final."""
    return {
        "datasets": list(LEGACY_SUPPORTED_DATASETS),
        "scope": "legacy_adapter_metadata_only",
        "analysis_endpoint": "/cases/analyze",
    }


@router.post("/datasets/analyze-file", deprecated=True)
def analyze_file(request: AnalyzeFileRequest) -> dict[str, Any]:
    """El batch legacy se retira para evitar llamadas Mistral masivas implícitas."""
    del request
    raise HTTPException(status_code=410, detail=LEGACY_ANALYSIS_DETAIL)
