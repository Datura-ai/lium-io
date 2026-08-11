"""Standalone sidecar shell for the vast_api module.

Temporary home: in the product these routes belong to the executor app itself;
this shell only exists so the experiment can iterate without touching the
executor release image.
"""

import logging

import uvicorn
from fastapi import FastAPI

from vast_api.config import VastSettings
from vast_api.docker_ops import DockerOps
from vast_api.errors import register_error_handlers
from vast_api.host_ops import HostOps
from vast_api.router import router
from vast_api.runs import RunStore
from vast_api.service import VastManager
from vast_api.vast_client import VastClient

logging.basicConfig(level=logging.INFO)


def build_app(settings: VastSettings | None = None) -> FastAPI:
    # construct-style wiring: one service = one class, injected here
    settings = settings or VastSettings()
    host = HostOps(settings)
    docker_ops = DockerOps(settings)
    vast = VastClient(settings)
    runs = RunStore(settings.RUNS_DIR)
    manager = VastManager(settings=settings, host=host, docker_ops=docker_ops, vast=vast, runs=runs)

    app = FastAPI(title="vast-manager", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.vast_settings = settings
    app.state.vast_manager = manager
    register_error_handlers(app)
    app.include_router(router)
    return app


if __name__ == "__main__":
    app = build_app()
    uvicorn.run(app, host="0.0.0.0", port=app.state.vast_settings.LISTEN_PORT)
