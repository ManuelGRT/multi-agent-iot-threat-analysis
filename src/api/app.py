# src/api/app.py
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.api.routers import router

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="TFM Multiagent API", version="0.1.0")
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        """Visor web del flujo multiagente (src/api/static/index.html)."""
        return FileResponse(_STATIC_DIR / "index.html")

    return app


app = create_app()
