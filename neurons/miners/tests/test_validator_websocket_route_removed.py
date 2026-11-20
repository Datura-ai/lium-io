"""Validators talk to this miner over REST only; a WebSocket connect to the old /websocket/<key> path is refused."""

from fastapi import FastAPI

from routes.validator_interface import validator_router


async def test_a_websocket_connect_to_the_old_validator_path_is_refused():
    app = FastAPI()
    app.include_router(validator_router)
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "websocket",
        "path": "/websocket/5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
        "raw_path": b"/websocket/5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYjQX",
        "query_string": b"",
        "headers": [],
        "client": ("198.51.100.7", 40000),
        "server": ("192.0.2.1", 8000),
        "scheme": "ws",
        "root_path": "",
        "subprotocols": [],
        "asgi": {"version": "3.0"},
    }
    await app(scope, receive, send)

    assert [m["type"] for m in sent] == ["websocket.close"]
