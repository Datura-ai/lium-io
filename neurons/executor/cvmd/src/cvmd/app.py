"""Application factory.

Everything fail-closed happens here, at construction: the authorized-clients file is loaded and
every entry validated before the app exists. A config the daemon cannot fully understand is a
startup failure, not a daemon that authorizes a subset of what the operator wrote.
"""

import logging
import threading

from fastapi import FastAPI

from cvmd.auth.clients import load_authorized_clients
from cvmd.auth.middleware import AuthMiddleware
from cvmd.auth.replay import ReplayStore
from cvmd.config import Config
from cvmd.cvm.instance import InstanceStore
from cvmd.cvm.manager import CvmManager
from cvmd.routes.cvm import router
from cvmd.state.store import StateStore

logger = logging.getLogger(__name__)


def create_app(config: Config) -> FastAPI:
    clients = load_authorized_clients(config.authorized_clients)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    store = StateStore(config.state_dir)
    replay = ReplayStore(config.state_dir, skew_seconds=config.skew_seconds)
    instances = InstanceStore(config.state_dir)
    manager = CvmManager(config=config.launch, store=store, instances=instances)

    app = FastAPI(title="cvmd", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.config = config
    app.state.store = store
    app.state.replay = replay
    app.state.clients = clients
    app.state.instances = instances
    app.state.cvm = manager
    # One CVM operation at a time. Held by the route rather than inside the manager so a second
    # request is refused immediately instead of blocking a worker thread for the whole launch.
    app.state.cvm_lock = threading.Lock()

    app.add_middleware(
        AuthMiddleware,
        clients=clients,
        replay=replay,
        max_body_bytes=config.max_body_bytes,
    )
    app.include_router(router)

    # Before the first request, not on the first request: a node that came back under a running
    # CVM must report the truth to whoever asks first, including a health check.
    state = manager.reconcile()

    logger.info(
        "cvmd ready: %d authorized keys, state=%s, startup floor=%d, cvm=%s",
        len(clients),
        state,
        replay.startup_floor_ns,
        instances.current.instance_id if instances.current else "none",
    )
    return app
