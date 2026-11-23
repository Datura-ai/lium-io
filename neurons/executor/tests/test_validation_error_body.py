"""A request that fails validation answers 422 whatever its body held, never a 500 from encoding the error."""

import pytest
from executor import validation_exception_handler
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel


class _Body(BaseModel):
    name: str


def _client() -> TestClient:
    app = FastAPI()

    @app.post("/thing")
    async def _thing(body: _Body):
        return {"ok": True}

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("raw", [b"not json at all", b"\xff\xfe\x00binary"])
def test_a_non_json_body_answers_422(raw):
    # ticket 1328: the error echoes the raw body (bytes), which the JSON response could not encode
    response = _client().post("/thing", content=raw, headers={"content-type": "application/octet-stream"})

    assert response.status_code == 422
    assert response.json()["detail"]


def test_a_json_body_of_the_wrong_shape_still_names_the_field():
    response = _client().post("/thing", json={"other": 1})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "name"]
