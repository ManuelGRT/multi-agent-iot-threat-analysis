# src/api/routers.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.ingest_parser import ADAPTERS
from src.mcp.common import data_dir, resolve_confined_path
from src.orchestration.graph import AsyncGraphOrchestrator, default_agents


router = APIRouter()


class AnalyzeRequest(BaseModel):
    dataset: str = Field(default="generic")
    row: dict[str, Any] | None = None
    text: str | None = None
    raw: dict[str, Any] | str | None = None
    source_file: str = "api"
    row_id: str | int = 0
    use_llm: bool | None = None
    use_llm_detector: bool | None = None
    use_llm_classifier: bool | None = None
    use_llm_explainer: bool | None = None


class CaseAnalyzeRequest(BaseModel):
    """Entrada del flujo final de casos (grafo MCP + agentes finales)."""

    dataset: str = Field(default="generic")
    row: dict[str, Any] | None = None
    text: str | None = None
    canonical_event: dict[str, Any] | None = None
    source_file: str = "api"
    row_id: str | int = 0
    cache_key: str | None = None
    allow_llm: bool = False
    use_llm_mitigator: bool | None = None
    persist: bool = False


class AnalyzeFileRequest(BaseModel):
    path: str
    dataset: str = "generic"
    limit: int = Field(default=10, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    use_llm: bool = False
    use_llm_detector: bool | None = None
    use_llm_classifier: bool | None = None
    use_llm_explainer: bool | None = None


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/events/analyze")
async def analyze_event(request: AnalyzeRequest) -> dict[str, Any]:
    orchestrator = AsyncGraphOrchestrator(
        use_llm=request.use_llm is not False,
        use_llm_detector=request.use_llm_detector,
        use_llm_classifier=request.use_llm_classifier,
        use_llm_explainer=request.use_llm_explainer,
    )
    try:
        return await orchestrator.ainvoke({"raw_input": request.model_dump(mode="json")})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if request.use_llm is not False:
            raise HTTPException(status_code=502, detail=f"LLM ingestion failed: {exc}") from exc
        raise


@router.post("/cases/analyze")
def analyze_case(request: CaseAnalyzeRequest) -> dict[str, Any]:
    """Analiza un evento con el grafo final MCP y devuelve el CaseResult con traza."""
    from src.orchestration.mcp_graph import run_case

    if not any([request.row, request.text, request.canonical_event, request.cache_key]):
        raise HTTPException(
            status_code=400,
            detail="Se requiere al menos uno de: row, text, canonical_event, cache_key",
        )
    raw_input = request.model_dump(
        mode="json", exclude={"allow_llm", "use_llm_mitigator", "persist"}
    )
    case = run_case(
        raw_input,
        allow_llm=request.allow_llm,
        use_llm_mitigator=request.use_llm_mitigator,
        persist=request.persist,
    )
    return case.model_dump(mode="json")


@router.post("/datasets/adapt")
async def adapt_event(request: AnalyzeRequest) -> dict[str, Any]:
    parser = default_agents(use_llm=request.use_llm is not False).ingest
    try:
        event, ingest_output = await parser.ingest_async(request.model_dump(mode="json"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "canonical_event": event.model_dump(mode="json"),
        "ingest_output": ingest_output.model_dump(mode="json"),
    }


@router.get("/datasets/supported")
def supported_datasets() -> dict[str, list[str]]:
    return {"datasets": sorted(ADAPTERS)}


@router.post("/datasets/analyze-file")
async def analyze_file(request: AnalyzeFileRequest) -> dict[str, Any]:
    try:
        path = resolve_confined_path(request.path, data_dir())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {path}")

    rows = list(_read_rows(path, offset=request.offset, limit=request.limit))
    orchestrator = AsyncGraphOrchestrator(
        use_llm=request.use_llm,
        use_llm_detector=request.use_llm_detector,
        use_llm_classifier=request.use_llm_classifier,
        use_llm_explainer=request.use_llm_explainer,
    )
    results = []
    for index, row in enumerate(rows, start=request.offset):
        raw_input = {
            "dataset": request.dataset,
            "source_file": str(path),
            "row_id": index,
            "row": row,
            "use_llm": request.use_llm,
            "use_llm_detector": request.use_llm_detector,
            "use_llm_classifier": request.use_llm_classifier,
            "use_llm_explainer": request.use_llm_explainer,
        }
        results.append(await orchestrator.ainvoke({"raw_input": raw_input}))
    return {"path": str(path), "dataset": request.dataset, "count": len(results), "results": results}


def _read_rows(path: Path, offset: int, limit: int):
    if path.suffix.lower() == ".log":
        yield from _read_zeek_or_delimited(path, offset, limit)
        return
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|") if sample else csv.excel
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        for index, row in enumerate(reader):
            if index < offset:
                continue
            if index >= offset + limit:
                break
            yield dict(row)


def _read_zeek_or_delimited(path: Path, offset: int, limit: int):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fields = None
        emitted = 0
        seen = 0
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
                continue
            if line.startswith("#"):
                continue
            if fields is None:
                values = line.split("\t")
                fields = [f"col_{idx}" for idx in range(len(values))]
            else:
                values = line.split("\t")
            if seen < offset:
                seen += 1
                continue
            if emitted >= limit:
                break
            emitted += 1
            seen += 1
            yield dict(zip(fields, values))
