"""Wiring of the vast_api module into the executor FastAPI app.

Auth rides the executor's MinerMiddleware: every non-GET request must carry a
valid MinerAuthPayload signature in its body; GET routes are open like the rest
of the executor app.
"""

from fastapi import FastAPI

from vast_api.config import VastSettings
from vast_api.docker_ops import DockerOps
from vast_api.errors import register_error_handlers
from vast_api.host_ops import HostOps
from vast_api.router import router
from vast_api.runs import RunStore
from vast_api.service import VastManager
from vast_api.vast_client import VastClient


def attach_vast_api(app: FastAPI, settings: VastSettings | None = None) -> None:
    # construct-style wiring: one service = one class, injected here
    settings = settings or VastSettings()
    host = HostOps(settings)
    docker_ops = DockerOps(settings)
    vast = VastClient(settings, host)
    runs = RunStore(settings.RUNS_DIR)
    app.state.vast_manager = VastManager(
        settings=settings, host=host, docker_ops=docker_ops, vast=vast, runs=runs
    )
    register_error_handlers(app)
    app.include_router(router)
