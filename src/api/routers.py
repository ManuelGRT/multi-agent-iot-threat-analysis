# src/api/routers.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.ingest_parser import ADAPTERS


router = APIRouter()
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
    """Entrada del flujo final de casos (grafo MCP + agentes finales)."""

    dataset: str = Field(default="generic")
    row: dict[str, Any] | None = None
    text: str | None = None
    canonical_event: dict[str, Any] | None = None
    source_file: str = "api"
    row_id: str | int = 0
    use_llm_mitigator: bool | None = None
    persist: bool = False


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
    use_llm_mitigator: bool | None = None,
    persist: bool = False,
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
    raw_input = request.model_dump(
        mode="json", exclude={"use_llm_mitigator", "persist"}
    )
    return _run_final_case(
        raw_input,
        use_llm_mitigator=request.use_llm_mitigator,
        persist=request.persist,
    )


@router.post("/datasets/adapt", deprecated=True)
def adapt_event(request: AnalyzeRequest) -> dict[str, Any]:
    """Endpoint legacy cerrado para impedir estandarizacion fuera del juez."""
    del request
    raise HTTPException(status_code=410, detail=LEGACY_ANALYSIS_DETAIL)


@router.get("/datasets/supported", deprecated=True)
def supported_datasets() -> dict[str, Any]:
    """Metadatos de formatos legacy; no son rutas del flujo final."""
    return {
        "datasets": sorted(ADAPTERS),
        "scope": "legacy_adapter_metadata_only",
        "analysis_endpoint": "/cases/analyze",
    }


@router.post("/datasets/analyze-file", deprecated=True)
def analyze_file(request: AnalyzeFileRequest) -> dict[str, Any]:
    """El batch legacy se retira para evitar llamadas Mistral masivas implícitas."""
    del request
    raise HTTPException(status_code=410, detail=LEGACY_ANALYSIS_DETAIL)
