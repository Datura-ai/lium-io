import pytest
from services import default_docker_image_digest_service as service

DIGEST = f"sha256:{'b' * 64}"


class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return {"token": "t"}


def fake_registry(monkeypatch: pytest.MonkeyPatch, head_status: int) -> list[str]:
    heads: list[str] = []

    class _Session:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            return None

        def get(self, *_args, **_kwargs) -> _Response:
            return _Response(200)

        def head(self, url: str, **_kwargs) -> _Response:
            heads.append(url)
            return _Response(head_status)

    monkeypatch.setattr(service.aiohttp, "ClientSession", _Session)
    monkeypatch.setattr(service, "_executor_build_cache", {})
    return heads


@pytest.mark.asyncio
@pytest.mark.parametrize(("head_status", "expected"), [(200, True), (404, False), (503, None)])
async def test_registry_answer_decides_whether_the_build_is_ours(monkeypatch, head_status, expected):
    heads = fake_registry(monkeypatch, head_status)

    assert await service.executor_build_published(DIGEST) is expected
    assert heads == [f"https://registry-1.docker.io/v2/daturaai/compute-subnet-executor/manifests/{DIGEST}"]


@pytest.mark.asyncio
async def test_no_digest_is_not_ours_without_asking(monkeypatch):
    heads = fake_registry(monkeypatch, 200)

    assert await service.executor_build_published(None) is False
    assert heads == []


@pytest.mark.asyncio
async def test_a_published_build_is_asked_once(monkeypatch):
    heads = fake_registry(monkeypatch, 200)

    await service.executor_build_published(DIGEST)
    await service.executor_build_published(DIGEST)

    assert len(heads) == 1
