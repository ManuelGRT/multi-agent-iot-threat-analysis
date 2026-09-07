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


class CaseAnalyzeRequest(BaseModel):
    """Entrada del flujo final; sus politicas operativas no son desactivables."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default="generic")
    row: dict[str, Any] | None = None
    text: str | None = None
    source_file: str = "api"
    row_id: str | int = 0


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


@router.post("/cases/analyze")
def analyze_case(request: CaseAnalyzeRequest) -> dict[str, Any]:
    """Analiza una entrada cruda mediante Mistral y el grafo final MCP."""
    if not any(value is not None for value in (request.row, request.text)):
        raise HTTPException(
            status_code=400,
            detail="Se requiere al menos uno de: row, text",
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
        with MCPToolClient() as memory_client:
            stored = memory_client.call(
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
