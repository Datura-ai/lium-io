"""Application factory.

Everything fail-closed happens here, at construction: the authorized-clients file is loaded and
every entry validated before the app exists. A config the daemon cannot fully understand is a
startup failure, not a daemon that authorizes a subset of what the operator wrote.

The catalog is the exception, and deliberately so. A host with no usable manifest is a host that
launches nothing, which is the correct outcome — but it must still answer `/v1/state` and
`/v1/catalog`, because "why will this node not launch?" is exactly the question being asked of
it at that moment. So the catalog is loaded lazily and its problems are reported, never fatal.
"""

import asyncio
import contextlib
import logging
import threading

from fastapi import FastAPI

from cvmd.auth.clients import load_authorized_clients
from cvmd.auth.middleware import AuthMiddleware
from cvmd.auth.replay import ReplayStore
from cvmd.catalog import CatalogError, CatalogStore, refresh_once
from cvmd.config import Config
from cvmd.cvm.instance import InstanceStore
from cvmd.cvm.manager import CvmManager
from cvmd.cvm.switching import SwitchStore
from cvmd.routes.cvm import router
from cvmd.state.store import StateStore

logger = logging.getLogger(__name__)


async def _refresh_loop(app: FastAPI) -> None:
    """Ask the backend for the current catalog, forever, on the configured interval.

    `to_thread` because the fetch and the verification are both blocking, and a several-second
    stall in the event loop is a several-second stall in `/v1/state` — during a launch, the one
    endpoint anyone is watching.

    The first pass runs immediately: a daemon that has just restarted is the most likely one to
    be holding a stale catalog.
    """
    store: CatalogStore = app.state.catalog
    interval = store.config.refresh_seconds
    while True:
        try:
            await asyncio.to_thread(refresh_once, store)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a timer must not die of one bad cycle
            logger.exception("catalog refresh raised; the catalog in force is unchanged")
        await asyncio.sleep(interval)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    config = app.state.catalog.config
    task = None
    if config.manifest_url or config.seed_path:
        task = asyncio.create_task(_refresh_loop(app), name="catalog-refresh")
    else:
        logger.info(
            "this host has neither a catalog manifest URL nor a seed, so it never refreshes"
        )
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def create_app(config: Config) -> FastAPI:
    clients = load_authorized_clients(config.authorized_clients)
    config.state_dir.mkdir(parents=True, exist_ok=True)

    store = StateStore(config.state_dir)
    replay = ReplayStore(config.state_dir, skew_seconds=config.skew_seconds)
    instances = InstanceStore(config.state_dir)
    switches = SwitchStore(config.state_dir)
    catalog = CatalogStore(config.catalog)
    manager = CvmManager(
        config=config.launch,
        catalog_store=catalog,
        store=store,
        instances=instances,
        switches=switches,
    )

    app = FastAPI(title="cvmd", docs_url=None, redoc_url=None, openapi_url=None, lifespan=_lifespan)
    app.state.config = config
    app.state.store = store
    app.state.replay = replay
    app.state.clients = clients
    app.state.instances = instances
    app.state.switches = switches
    app.state.catalog = catalog
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
    _adopt_seed(catalog)

    logger.info(
        "cvmd ready: %d authorized keys, state=%s, startup floor=%d, cvm=%s, catalog=%s",
        len(clients),
        state,
        replay.startup_floor_ns,
        instances.current.instance_id if instances.current else "none",
        _catalog_summary(catalog),
    )
    return app


def _adopt_seed(catalog: CatalogStore) -> None:
    if catalog.config.seed_path is None:
        return
    try:
        catalog.install_seed_if_newer()
    except CatalogError as exc:
        # A seed that cannot be adopted leaves the previous catalog in force, which is a working
        # host. Refusing to start would turn an operator's stale file into an outage.
        logger.warning("the seed manifest was not adopted: %s", exc)


def _catalog_summary(catalog: CatalogStore) -> str:
    try:
        current = catalog.current(require_fresh=False)
    except CatalogError as exc:
        return f"none ({exc})"
    return (
        f"serial {current.serial}, {len(current.entries)} artifact(s), "
        f"expires {current.expires_at.isoformat()}"
    )
