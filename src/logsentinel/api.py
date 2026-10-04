"""REST API: score log lines on demand and browse detected incidents."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from . import __version__
from .bundle import ModelBundle
from .config import get_settings
from .detector import StreamingDetector
from .features import Window
from .parsing import parse_lines
from .storage import Repository, make_repository


class ScoreRequest(BaseModel):
    lines: list[str] = Field(..., min_length=1, max_length=10_000, description="Raw log lines forming one window")
    format: Literal["app", "bgl"] = "app"


class ModelStore:
    """Loads the model lazily and reloads it when the file on disk changes."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._bundle: ModelBundle | None = None
        self._mtime: float | None = None
        self._lock = threading.Lock()

    def get(self) -> ModelBundle | None:
        with self._lock:
            try:
                mtime = os.path.getmtime(self.path)
            except OSError:
                return self._bundle
            if mtime != self._mtime:
                self._bundle = ModelBundle.load(self.path)
                self._mtime = mtime
            return self._bundle


def create_app(repository: Repository | None = None, model_path: str | None = None) -> FastAPI:
    settings = get_settings()
    repo = repository if repository is not None else make_repository(settings.database_url)
    models = ModelStore(model_path or settings.model_path)

    app = FastAPI(
        title="LogSentinel",
        version=__version__,
        description="Real-time log anomaly detection. Scores log windows and exposes detected incidents.",
    )
    app.mount("/metrics", make_asgi_app())

    def require_model() -> ModelBundle:
        bundle = models.get()
        if bundle is None:
            raise HTTPException(status_code=503, detail="No trained model available yet")
        return bundle

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": models.get() is not None}

    @app.get("/v1/model")
    def model_info() -> dict:
        bundle = require_model()
        return {
            "detector": bundle.detector.name,
            "threshold": bundle.threshold,
            "window_seconds": bundle.window_seconds,
            "features": len(bundle.featurizer.feature_names),
            "metadata": bundle.metadata,
        }

    @app.post("/v1/score")
    def score(request: ScoreRequest) -> dict:
        """Treat the submitted lines as one window and return its anomaly score."""
        bundle = require_model()
        records = parse_lines(request.lines, request.format)
        if not records:
            raise HTTPException(status_code=422, detail="None of the lines could be parsed")
        window = Window(start=records[0].timestamp, end=records[-1].timestamp)
        for record in records:
            window.add(record)
        result = StreamingDetector(bundle).score_window(window)
        return {**result.to_dict(), "parsed_lines": len(records), "skipped_lines": len(request.lines) - len(records)}

    @app.get("/v1/incidents")
    def incidents(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
        return repo.recent_incidents(limit)

    @app.get("/v1/incidents/{incident_id}")
    def incident(incident_id: int) -> dict:
        found = repo.get_incident(incident_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Incident not found")
        return found

    @app.get("/v1/windows")
    def windows(limit: int = Query(60, ge=1, le=1440)) -> list[dict]:
        return repo.recent_windows(limit)

    return app
