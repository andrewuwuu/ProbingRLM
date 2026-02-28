from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router as api_router


def _frontend_dist_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="ProbingRLM Backend")
    app.include_router(api_router)
    app.mount(
        "/",
        StaticFiles(directory=str(_frontend_dist_path()), html=True, check_dir=False),
        name="frontend",
    )
    return app


app = create_app()
