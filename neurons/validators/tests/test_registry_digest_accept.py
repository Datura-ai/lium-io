import pytest

from services import default_docker_image_digest_service as module

OCI_INDEX = "application/vnd.oci.image.index.v1+json"


class _Response:
    def __init__(self, payload=None, headers=None):
        self._payload = payload or {}
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.accepts: list[str] = []

    def get(self, url, params=None):
        return _Response(payload={"token": "t"})

    def head(self, url, headers=None):
        self.accepts.append(headers["Accept"])
        return _Response(headers={"Docker-Content-Digest": "sha256:" + "a" * 64})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_template_digests_keep_the_docker_types_only():
    # DAH-5053 review: the executor's cache side resolves templates with the Docker types, so the
    # validator's template digest must not switch to an OCI index the other side never sees.
    session = _Session()
    assert await module.fetch_registry_digest(session, "daturaai/pytorch:latest")
    assert OCI_INDEX not in session.accepts[0]


@pytest.mark.asyncio
async def test_the_executor_image_digest_accepts_an_oci_index(monkeypatch):
    session = _Session()
    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda **kwargs: session)
    monkeypatch.setattr(module.settings, "EXECUTOR_IMAGE_REF", "daturaai/compute-subnet-executor:latest")
    assert await module.fetch_executor_image_digest() == "sha256:" + "a" * 64
    assert session.accepts[0].startswith(OCI_INDEX)
    assert "application/vnd.docker.distribution.manifest.v2+json" in session.accepts[0]
