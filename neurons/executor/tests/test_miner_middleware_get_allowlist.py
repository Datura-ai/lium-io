"""MinerMiddleware: a GET route is reachable only when it is on the allowlist.

The miner signature travels in the request body, which a GET does not have, so
the middleware used to wave every GET through. Both GET routes that exist today
are safe (`/version` is public; `/containers/{name}/logs` checks a validator
signature in headers), but the next GET route must not ship unauthenticated
because its author forgot — unknown GET paths now answer 401.
"""

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient
from middlewares.miner import MinerMiddleware
from routes.apis import apis_router


def _client(extra_get_route: str | None = None) -> TestClient:
    app = FastAPI()
    app.add_middleware(MinerMiddleware)
    app.include_router(apis_router)
    if extra_get_route:
        @app.get(extra_get_route)
        async def _unlisted():  # pragma: no cover - the body must never run
            return {"leaked": True}
    return TestClient(app)


def test_version_is_public():
    assert _client().get("/version").status_code == 200


def test_container_logs_reaches_its_own_signature_check():
    # Reaches the route: FastAPI rejects the missing signature headers (422),
    # not the middleware (401).
    response = _client().get("/containers/pod_x/logs")

    assert response.status_code == 422


def test_unlisted_get_route_is_denied_by_default(caplog):
    # core.logger may set propagate=False (then caplog's root handler sees nothing), so the
    # handler is attached to the module logger itself; a propagating logger records it twice.
    miner_logger = logging.getLogger("middlewares.miner")
    miner_logger.addHandler(caplog.handler)
    try:
        response = _client(extra_get_route="/future_route").get("/future_route")
    finally:
        miner_logger.removeHandler(caplog.handler)

    assert response.status_code == 401
    assert response.json() == "Unauthorized"
    # The denial is logged like the other ones here: which path, from which client.
    denied = {r.getMessage() for r in caplog.records if "GET route is not on the allowlist" in r.getMessage()}
    assert len(denied) == 1
    message = denied.pop()
    assert '"url": "/future_route"' in message
    assert '"client_host": "testclient"' in message


def test_unknown_get_path_is_denied_not_404():
    assert _client().get("/does_not_exist").status_code == 401
