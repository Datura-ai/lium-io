"""BodySizeLimitMiddleware: a body past the ceiling is refused before the auth check buffers it."""

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from middlewares.body_limit import MAX_REQUEST_BODY_BYTES, BodySizeLimitMiddleware
from middlewares.miner import MinerMiddleware

read_bytes: list[int] = []


def _client() -> TestClient:
    app = FastAPI()

    @app.post("/ping")  # self-authenticated: MinerMiddleware passes it through to the route
    async def _ping(request: Request):
        read_bytes.append(len(await request.body()))
        return {"ok": True}

    app.add_middleware(MinerMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    return TestClient(app)


def test_declared_oversize_body_is_refused_413():
    response = _client().post("/upload_ssh_key", content=b"x" * (MAX_REQUEST_BODY_BYTES + 1))

    assert response.status_code == 413


def test_chunked_oversize_body_is_cut_at_the_ceiling():
    read_bytes.clear()

    def chunks():
        for _ in range(MAX_REQUEST_BODY_BYTES // 1024 + 8):
            yield b"x" * 1024

    response = _client().post("/ping", content=chunks())

    assert response.status_code in (400, 413)
    assert read_bytes == []  # the route never got the body


def test_small_body_still_reaches_the_route():
    read_bytes.clear()

    response = _client().post("/ping", content=b"{}")

    assert response.status_code == 200
    assert read_bytes == [2]


def test_a_non_numeric_content_length_is_refused():
    response = _client().post("/ping", content=b"{}", headers={"content-length": "abc"})

    assert response.status_code in (400, 413)
