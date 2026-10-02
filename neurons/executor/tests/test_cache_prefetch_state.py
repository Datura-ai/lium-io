"""DAH-2470 — the cache-prefetch loop's structured state file.

The validator reads this document when it zeroes a node for a bad image digest, and it
is the only thing a reader will have: the loop's log lines never leave the provider's
machine. So every exit path must be named, every error must be published as its class,
reason code, host and status and never as its text (the log line keeps the text, with any
credentials in it removed), and neither the document nor a broken write may ever disturb
the loop.
"""

import asyncio
import json
import time
from unittest.mock import MagicMock

import docker
import pytest

# conftest replaces the docker module with a MagicMock; except-clauses in the
# service need a real exception class to catch.
docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})

from services.cache_prefetch_state import (  # noqa: E402
    MAX_PAYLOAD_BYTES,
    SCHEMA_VERSION,
    CachePrefetchState,
    Failure,
    Outcome,
    Reason,
    _ImageRecord,
    describe_error,
    redact,
)

from services import cache_prefetch_state, cache_template_service  # noqa: E402

REPO = "daturaai/pytorch"
TAG = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind"
IMAGE_REF = f"{REPO}:{TAG}"
FRESH_DIGEST = "sha256:70bd5fa697877594b753a146e207ca4de66d9b875d606ae09e6ee7bac8f4f423"
STALE_DIGEST = "sha256:2d19c94ce8a37c6fa364f8a6211d8b6dc1a44ece574c4c22ab5579925ce7a4c8"
# A JWT's shape: a stub header, payload and signature.
STUB_JWT = "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJzdWIiOiJ4In0" + ".c2lnbmF0dXJl"


def _make_client(local_digests: list[str] | None = None, image_absent: bool = False) -> MagicMock:
    client = MagicMock()
    local = MagicMock()
    local.attrs = {"RepoDigests": [f"{REPO}@{digest}" for digest in (local_digests or [])]}
    # What the lookup after a pull returns; `image_absent` is about the tag before the pull.
    client.pulled_image = MagicMock()

    def get(ref):
        if "@" in ref:
            return client.pulled_image
        if image_absent:
            raise docker.errors.ImageNotFound("absent")
        return local

    client.images.get.side_effect = get
    client.api.pull.side_effect = lambda *args, **kwargs: iter([{"status": "Pull complete"}])
    client.images.list.return_value = []
    return client


def _template(digest: str | None = None, size: int = 0) -> dict:
    data = {"docker_image": REPO, "docker_image_tag": TAG, "docker_image_size": size}
    if digest is not None:
        data["docker_image_digest"] = digest
    return data


def _run(client, template, state):
    asyncio.run(cache_template_service._ensure_template(client, template, state))


def _image(state: CachePrefetchState) -> dict:
    return state.as_dict()["images"][IMAGE_REF]


def _error(code, cls=None, host=None, status=None) -> dict:
    return {"class": cls, "code": code, "host": host, "status": status}


# --- image-level outcomes ----------------------------------------------------------------


def test_up_to_date_at_backend_digest():
    state = CachePrefetchState(path=None)

    _run(_make_client(local_digests=[FRESH_DIGEST]), _template(FRESH_DIGEST), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.UP_TO_DATE
    assert record["expected_digest"] == FRESH_DIGEST
    assert record["local_digests"] == [f"{REPO}@{FRESH_DIGEST}"]


def test_up_to_date_at_registry_digest():
    # Legacy path: no backend digest, so the daemon's remote digest is the comparison.
    state = CachePrefetchState(path=None)
    client = _make_client(local_digests=[STALE_DIGEST])
    client.images.get_registry_data.return_value = MagicMock(id=STALE_DIGEST)

    _run(client, _template(), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.UP_TO_DATE
    assert record["remote_digest"] == STALE_DIGEST
    assert record["last_remote_read_ok_at"] is not None


def test_remote_digest_unreadable_names_the_registry_reason():
    # DAH-2470's leading hypothesis. The reason is the whole point: it says whether the
    # registry rate-limited us, rejected auth, or was simply unreachable.
    state = CachePrefetchState(path=None)
    client = _make_client(local_digests=[STALE_DIGEST])
    client.images.get_registry_data.side_effect = RuntimeError("429 Too Many Requests")

    _run(client, _template(), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.REMOTE_DIGEST_UNREADABLE
    assert record["last_error"] == _error(Reason.RATE_LIMITED, "RuntimeError")
    assert record["last_remote_error"] == _error(Reason.RATE_LIMITED, "RuntimeError")
    assert record["remote_digest"] is None
    client.api.pull.assert_not_called()


def test_repeated_unreadable_remote_accumulates_a_count():
    # One occurrence is noise; a count in the hundreds against an unchanged local digest
    # is the answer. That distinction only exists if the counter accumulates.
    state = CachePrefetchState(path=None)
    client = _make_client(local_digests=[STALE_DIGEST])
    client.images.get_registry_data.side_effect = RuntimeError("registry unreachable")

    for _ in range(3):
        _run(client, _template(), state)

    assert _image(state)["outcome_counts"][Outcome.REMOTE_DIGEST_UNREADABLE] == 3


def test_insufficient_disk_records_both_numbers(monkeypatch):
    state = CachePrefetchState(path=None)
    monkeypatch.setattr(
        cache_template_service.psutil, "disk_usage", lambda _: MagicMock(free=1_000)
    )
    client = _make_client(image_absent=True)

    _run(client, _template(size=10_000), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.INSUFFICIENT_DISK
    assert record["last_disk_required_bytes"] == 30_000
    assert record["last_disk_available_bytes"] == 1_000
    assert record["last_error"] == _error(Reason.DISK_FULL)
    client.api.pull.assert_not_called()


def test_lock_held(monkeypatch):
    state = CachePrefetchState(path=None)

    class _Denied:
        def __enter__(self):
            return False

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(cache_template_service, "cache_pull_lock", lambda: _Denied())
    client = _make_client(image_absent=True)

    _run(client, _template(FRESH_DIGEST), state)

    assert _image(state)["last_outcome"] == Outcome.LOCK_HELD
    client.api.pull.assert_not_called()


def test_pull_ok():
    state = CachePrefetchState(path=None)
    client = _make_client(image_absent=True)

    _run(client, _template(FRESH_DIGEST), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.PULL_OK
    assert record["last_pull_attempt_at"] is not None
    assert record["last_pull_ok_at"] is not None
    assert record["last_pull_error"] is None


def test_pull_failed_is_recorded_and_re_raised():
    # Today a failed pull aborts the sweep and backs off. Recording must not change that.
    state = CachePrefetchState(path=None)
    client = _make_client(image_absent=True)
    client.api.pull.side_effect = RuntimeError("manifest unknown")

    with pytest.raises(RuntimeError):
        _run(client, _template(FRESH_DIGEST), state)

    record = _image(state)
    assert record["last_outcome"] == Outcome.PULL_FAILED
    assert record["last_pull_error"] == _error(Reason.MANIFEST_UNKNOWN, "RuntimeError")
    assert record["last_error"] == _error(Reason.MANIFEST_UNKNOWN, "RuntimeError")


def test_local_read_error_is_kept():
    state = CachePrefetchState(path=None)
    client = _make_client()
    lookup = client.images.get.side_effect

    def get(ref):
        if "@" in ref:
            return lookup(ref)
        raise RuntimeError("daemon busy")

    client.images.get.side_effect = get

    _run(client, _template(FRESH_DIGEST), state)

    assert _image(state)["last_local_error"] == _error(Reason.DOCKER_ERROR, "RuntimeError")


def test_cleanup_error_is_kept():
    state = CachePrefetchState(path=None)
    client = _make_client(image_absent=True)
    client.images.list.side_effect = RuntimeError("cannot list")

    _run(client, _template(FRESH_DIGEST), state)

    assert _image(state)["last_cleanup_error"] == _error(Reason.DOCKER_ERROR, "RuntimeError")


def _docker_error(*args, **kwargs):
    raise RuntimeError("registry answered 401 to Authorization: Bearer s3cret")


def _assert_recorded(record, code):
    assert record == _error(code, "RuntimeError")


def _assert_logged(log_method):
    messages = [call.args[0] for call in log_method.call_args_list]
    assert any("RuntimeError: " in message for message in messages)
    assert not any("s3cret" in message for message in messages)


def _break_local_lookup(client):
    lookup = client.images.get.side_effect
    client.images.get.side_effect = lambda ref: lookup(ref) if "@" in ref else _docker_error()


@pytest.mark.parametrize(
    ("field", "break_client", "code"),
    [
        (
            "last_remote_error",
            lambda client: setattr(client.images.get_registry_data, "side_effect", _docker_error),
            Reason.REGISTRY_DENIED,
        ),
        ("last_local_error", _break_local_lookup, Reason.DOCKER_ERROR),
        (
            "last_cleanup_error",
            lambda client: setattr(client.images, "list", _docker_error),
            Reason.REGISTRY_DENIED,
        ),
    ],
)
def test_per_image_errors_are_logged_redacted_and_published_as_a_record(
    monkeypatch, field, break_client, code
):
    logger = MagicMock()
    monkeypatch.setattr(cache_template_service, "logger", logger)
    state = CachePrefetchState(path=None)
    client = _make_client(local_digests=[STALE_DIGEST], image_absent=field == "last_cleanup_error")
    break_client(client)

    _run(client, _template(None if field == "last_remote_error" else FRESH_DIGEST), state)

    _assert_recorded(_image(state)[field], code)
    _assert_logged(logger.warning)


def test_a_failed_image_removal_is_logged_redacted_and_published_as_a_record(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(cache_template_service, "logger", logger)
    state = CachePrefetchState(path=None)
    client = _make_client(image_absent=True)
    client.images.list.return_value = [MagicMock(tags=[f"{REPO}:old"])]
    client.images.remove.side_effect = _docker_error

    _run(client, _template(FRESH_DIGEST), state)

    _assert_recorded(_image(state)["last_cleanup_error"], Reason.REGISTRY_DENIED)
    _assert_logged(logger.warning)


def test_the_gpu_error_is_logged_redacted_and_returned_as_the_error(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(cache_template_service, "logger", logger)
    monkeypatch.setattr(cache_template_service.pynvml, "nvmlInit", _docker_error)

    gpu_model, _, error = cache_template_service._get_gpu_info()

    assert gpu_model == "unknown"
    assert isinstance(error, RuntimeError)
    state = CachePrefetchState(path=None)
    state.note_gpu(gpu_model, "unknown", error=error)
    _assert_recorded(state.as_dict()["gpu_error"], Reason.NVML_ERROR)
    (message,), _ = logger.error.call_args
    assert "RuntimeError: " in message
    assert "s3cret" not in message


def test_a_failed_sweep_is_logged_with_its_class(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(cache_template_service, "logger", logger)
    pre_puller = MagicMock()

    async def sweep(*args, **kwargs):
        _docker_error()

    pre_puller.sweep = sweep

    asyncio.run(cache_template_service._run_pre_pull_sweep(pre_puller, [], frozenset(), 0.0))

    (message,), _ = logger.warning.call_args
    assert message.startswith("pre-pull sweep failed: RuntimeError: ")
    assert "s3cret" not in message


def test_digest_change_is_timestamped():
    state = CachePrefetchState(path=None)

    state.note_local_digests(IMAGE_REF, [f"{REPO}@{STALE_DIGEST}"])
    assert _image(state)["local_digest_first_seen_at"] is not None
    assert _image(state)["digest_changed_at"] is None

    state.note_local_digests(IMAGE_REF, [f"{REPO}@{FRESH_DIGEST}"])
    assert _image(state)["digest_changed_at"] is not None


def test_misspelled_field_raises_instead_of_being_published():
    # Every writer runs under @_never_raises, so a bare dict would have absorbed the typo
    # and shipped a document quietly missing the field. `slots=True` is what stops that;
    # this test fails the moment someone drops it.
    record = _ImageRecord()

    with pytest.raises(AttributeError):
        record.last_pul_ok_at = "typo"


def test_published_images_are_a_copy_not_the_live_record():
    state = CachePrefetchState(path=None)
    state.note_local_digests(IMAGE_REF, [f"{REPO}@{FRESH_DIGEST}"])

    _image(state)["local_digests"].append("mutated by a reader")

    assert _image(state)["local_digests"] == [f"{REPO}@{FRESH_DIGEST}"]


def test_malformed_template_gets_its_own_field():
    # No usable image_ref exists, so it would only pollute the images map. It also cannot
    # live in the outcome slot: the sweep still finishes, and sweep_ok would bury it.
    state = CachePrefetchState(path=None)

    _run(_make_client(), {"docker_image": REPO}, state)

    doc = state.as_dict()
    assert doc["last_malformed_template"] == {"missing": ["docker_image_tag"]}
    assert doc["last_malformed_template_at"] is not None
    assert doc["outcome_counts"][Outcome.MALFORMED_TEMPLATE] == 1
    assert doc["images"] == {}


# --- loop-level outcomes -----------------------------------------------------------------


def _prefetch(state_path=None):
    asyncio.run(cache_template_service.run_cache_template_prefetch(state_path))


def test_prefetch_disabled_without_backend_url(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_template_service.settings, "COMPUTE_REST_API_URL", "")
    path = tmp_path / "state.json"

    _prefetch(str(path))

    doc = json.loads(path.read_text())
    assert doc["last_outcome"] == Outcome.PREFETCH_DISABLED_NO_BACKEND_URL
    assert doc["backend_url"] is None


def test_docker_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cache_template_service.settings, "COMPUTE_REST_API_URL", "https://lium.io/api"
    )
    monkeypatch.setattr(
        cache_template_service.docker,
        "from_env",
        MagicMock(side_effect=RuntimeError("no docker socket")),
    )
    path = tmp_path / "state.json"

    _prefetch(str(path))

    doc = json.loads(path.read_text())
    assert doc["last_outcome"] == Outcome.DOCKER_UNAVAILABLE
    assert doc["docker_available"] is False
    assert doc["docker_error"] == _error(Reason.DOCKER_ERROR, "RuntimeError")
    assert doc["last_error"] == _error(Reason.DOCKER_ERROR, "RuntimeError")
    assert doc["backend_url"] == "https://lium.io"


def test_docker_unavailable_is_logged_with_its_class(monkeypatch, tmp_path):
    logger = MagicMock()
    monkeypatch.setattr(cache_template_service, "logger", logger)
    monkeypatch.setattr(
        cache_template_service.settings, "COMPUTE_REST_API_URL", "https://lium.io/api"
    )
    monkeypatch.setattr(cache_template_service.docker, "from_env", _docker_error)

    _prefetch(str(tmp_path / "state.json"))

    (message,), _ = logger.error.call_args
    assert message.startswith("Cannot connect to docker; cache pre-pull disabled: RuntimeError: ")
    assert "s3cret" not in message


NVMLError_LibraryNotFound = type("NVMLError_LibraryNotFound", (Exception,), {})


def test_gpu_unknown():
    state = CachePrefetchState(path=None)
    error = NVMLError_LibraryNotFound("NVML Shared Library Not Found")

    state.note_gpu("unknown", "unknown", error=error)
    state.record_loop_outcome(Outcome.GPU_UNKNOWN, error=error)

    doc = state.as_dict()
    assert doc["last_outcome"] == Outcome.GPU_UNKNOWN
    assert doc["gpu_resolved_at"] is None
    assert doc["gpu_error"] == _error(Reason.NVML_ERROR, "NVMLError_LibraryNotFound")
    assert doc["last_error"] == doc["gpu_error"]


def test_backend_no_templates():
    state = CachePrefetchState(path=None, backend_url="https://u:s3cret@backend.example/api")
    failure = Failure(Reason.HTTP_ERROR, status=503)

    state.note_backend(status=503, template_count=0, error=failure)
    state.record_loop_outcome(Outcome.BACKEND_NO_TEMPLATES, error=failure)

    doc = state.as_dict()
    assert doc["last_outcome"] == Outcome.BACKEND_NO_TEMPLATES
    assert doc["last_backend_status"] == 503
    assert doc["last_backend_template_count"] == 0
    published = _error(Reason.HTTP_ERROR, host="backend.example", status=503)
    assert doc["last_backend_error"] == doc["last_error"] == published


def test_loop_error():
    state = CachePrefetchState(path=None)

    state.note_loop_error(ConnectionResetError("connection reset"))
    state.record_loop_outcome(Outcome.LOOP_ERROR, error=ConnectionResetError("connection reset"))

    doc = state.as_dict()
    assert doc["last_outcome"] == Outcome.LOOP_ERROR
    assert doc["last_loop_error"] == _error(Reason.CONNECT_ERROR, "ConnectionResetError")
    assert doc["last_loop_error_at"] is not None


def test_sweep_ok_clears_a_stale_loop_error():
    # Without this, a loop that erred once and then recovered would keep reporting
    # loop_error forever, and a reader would chase a problem that had already passed.
    state = CachePrefetchState(path=None)

    state.record_loop_outcome(Outcome.LOOP_ERROR, error="connection reset")
    state.record_loop_outcome(Outcome.SWEEP_OK)

    doc = state.as_dict()
    assert doc["last_outcome"] == Outcome.SWEEP_OK
    assert doc["last_error"] is None
    # The error is still on record, just no longer the headline.
    assert doc["outcome_counts"][Outcome.LOOP_ERROR] == 1


# --- the document itself -----------------------------------------------------------------


def test_header_is_always_published():
    # started_at + sweep_count are what stop a post-recreate reset reading as a healthy node.
    state = CachePrefetchState(
        path=None, backend_url="https://lium.io/api", refresh_interval_seconds=900
    )
    state.begin_sweep()
    state.begin_sweep()

    doc = state.as_dict()
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["executor_version"]
    assert doc["started_at"]
    assert doc["updated_at"]
    assert doc["sweep_count"] == 2
    assert doc["refresh_interval_seconds"] == 900


def test_error_text_handed_in_as_a_string_is_never_published():
    state = CachePrefetchState(path=None)

    state.record_image_outcome(IMAGE_REF, Outcome.PULL_FAILED, error="password=s3cret " * 300)
    state.note_loop_error("https://provider:s3cret@backend.example")

    assert _image(state)["last_error"] == _error(Reason.PULL_FAILED)
    assert state.as_dict()["last_loop_error"] == _error(Reason.UNKNOWN)
    assert "s3cret" not in state.render()


# Every credential shape the log redaction removes; the fuzz below also feeds each one to the
# document, which must hold none of it.
REDACTED_CASES = [
    (
        "Cannot connect to https://provider:s3cret@backend.example:8443/api",
        "Cannot connect to https://***@backend.example:8443/api",
    ),
    ("401, url='https://s3cret@backend.example/x'", "401, url='https://***@backend.example/x'"),
    ("Authorization: Bearer s3cret.value", "Authorization: Bearer ***"),
    ("sent token abcdefghijklmnop0123 to backend.example", "sent token *** to backend.example"),
    ("X-Api-Key: s3cret", "X-Api-Key: ***"),
    (
        "GET https://backend.example/x?gpu=H100&access_token=s3cret&sig=s3cret#top",
        "GET https://backend.example/x?***",
    ),
    ("GET https://backend.example/x#access_token=s3cret", "GET https://backend.example/x?***"),
    ("password=s3cret rejected", "password=*** rejected"),
    ("pushed with ghp_" + "a" * 36, "pushed with ***"),
    # A password holding an unencoded `@` or `/`: yarl and aiohttp carry such URLs as-is.
    (
        "Cannot connect to https://provider:p@ss@backend.example:8443/api",
        "Cannot connect to https://***@backend.example:8443/api",
    ),
    (
        "InvalidUrlClientError: https://provider:pa/ss@backend.example/executors/x",
        "InvalidUrlClientError: https://***@backend.example/executors/x",
    ),
    # Up to the last `@`, not the first: what follows an `@` inside a password is not a host.
    (
        "Cannot connect to https://provider:s3c@ret/x@backend.example/api",
        "Cannot connect to https://***@backend.example/api",
    ),
    # Whatever the host, port or length of the userinfo.
    (
        "InvalidURL: https://provider:s3cret@backend.example:bad/api",
        "InvalidURL: https://***@backend.example:bad/api",
    ),
    (
        "https://provider:s3cret@backend.example: connection refused",
        "https://***@backend.example: connection refused",
    ),
    (
        "InvalidURL: https://provider:s3c/ret@bäckend.example/api",
        "InvalidURL: https://***@bäckend.example/api",
    ),
    (
        "InvalidURL: https://provider:" + "s3cret/" * 43 + "@backend.example/api",
        "InvalidURL: https://***@backend.example/api",
    ),
    (
        "InvalidURL: https://provider:s3c<ret@backend.example/api",
        "InvalidURL: https://***@backend.example/api",
    ),
    # aiohttp quotes the request URL: its query is masked before any key=value rule runs.
    (
        "401, message='Unauthorized', url='https://backend.example/x?api_token=s3cret'",
        "401, message='Unauthorized', url='https://backend.example/x?***'",
    ),
    (
        "200, message='Attempt to decode JSON with unexpected mimetype: text/html', "
        "url='https://backend.example/executors/default-docker-image?token=s3cret&gpu=H100'",
        "200, message='Attempt to decode JSON with unexpected mimetype: text/html', "
        "url='https://backend.example/executors/default-docker-image?***'",
    ),
    # An `@` in the query leaves no telling where the userinfo ends: nothing after `://` stays.
    ("GET https://backend.example/x?to=a@b.example&token=s3cret", "GET https://***"),
    ("login with pass=s3cret and pw=s3cret", "login with pass=*** and pw=***"),
    ("{'password': 's3cret', 'user': 'provider'}", "{'password': '***', 'user': 'provider'}"),
    ('{"api_key": "s3cret", "gpu": "H100"}', '{"api_key": "***", "gpu": "H100"}'),
    ('{"password": "s3c\\"ret", "gpu": "H100"}', '{"password": "***", "gpu": "H100"}'),
    ('password="s3c ret" rejected', 'password="***" rejected'),
    ("password=b's3cret' rejected", "password=b'***' rejected"),
    ("Authorization: s3cret", "Authorization: ***"),
    ("Authorization: Token s3cret", "Authorization: Token ***"),
    ("X-Api-Token: s3cret", "X-Api-Token: ***"),
    ("X-Registry-Key: s3cret", "X-Registry-Key: ***"),
    ("X-Amz-Security-Token: s3cret", "X-Amz-Security-Token: ***"),
    ("Private-Token: s3cret", "Private-Token: ***"),
    ("Cookie: sessionid=s3cret; csrftoken=s3cret", "Cookie: ***"),
    ("Set-Cookie: sessionid=s3cret; Path=/; HttpOnly", "Set-Cookie: ***"),
    ("GET /x?apiKey=s3cret&clientSecret=s3cret", "GET /x?apiKey=***&clientSecret=***"),
    ("apikey=s3cret accesstoken=s3cret", "apikey=*** accesstoken=***"),
    ("mypassword=s3cret PGPASSWORD=s3cret", "mypassword=*** PGPASSWORD=***"),
    ("run with --password=s3cret", "run with --password=***"),
    ("password: s3cret", "password: ***"),
    ("password = s3cret", "password = ***"),
    ("client_secret: s3cret", "client_secret: ***"),
    # A non-secret pair does not hide a secret one that follows it.
    ("url=https://backend.example/x token=s3cret", "url=https://backend.example/x token=***"),
    ("GET url=/x?api_token=s3cret", "GET url=/x?api_token=***"),
    ("basic dXNlcjpzM2NyZXQtcGFzc3dvcmQ=", "basic ***"),
    ("Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJl", "Bearer ***"),
    # A JWT with nothing naming it, and a capitalised free-text Bearer with a non-JWT token.
    (f"rejected {STUB_JWT} upstream", "rejected *** upstream"),
    ("sent Bearer ABCDEFGHIJ0123456789 upstream", "sent Bearer *** upstream"),
    # A short value only when it is token-shaped: 8 or more characters, a letter and a digit.
    ("sent Token abc12345 upstream", "sent Token *** upstream"),
    # A fragment holding an `@`: what follows it is not a host, so nothing after `://` stays.
    ("GET https://backend.example/x#a@b.example/s3cret", "GET https://***"),
    # A password holding a whitespace, quote or `>` in free text: masked up to the `@` before
    # the host, not only up to the character that ends the span.
    (
        "Cannot connect to https://provider:s3c ret@backend.example/api",
        "Cannot connect to https://***@backend.example/api",
    ),
    (
        "Cannot connect to https://provider:s3c'ret@backend.example/api",
        "Cannot connect to https://***@backend.example/api",
    ),
    (
        'Cannot connect to https://provider:s3c"ret@backend.example/api',
        "Cannot connect to https://***@backend.example/api",
    ),
    (
        "Cannot connect to https://provider:s3c>ret@backend.example/api",
        "Cannot connect to https://***@backend.example/api",
    ),
    (
        "Cannot connect to https://provider:s3c\tret@backend.example/api",
        "Cannot connect to https://***@backend.example/api",
    ),
    (
        "Cannot connect to https://provider:s3c\xa0ret@backend.example/api",
        "Cannot connect to https://***@backend.example/api",
    ),
    (
        "url='https://provider:s3c r\"e't@backend.example:8443/api' refused",
        "url='https://***@backend.example:8443/api' refused",
    ),
    # A URL written without its scheme, as aiohttp's NonHttpUrlClientError quotes it.
    (
        "NonHttpUrlClientError: provider:s3cret@backend.example:8443/api",
        "NonHttpUrlClientError: ***@backend.example:8443/api",
    ),
    (
        "url='provider:s3cret@backend.example:bad/api?token=s3cret'",
        "url='***@backend.example:bad/api?***'",
    ),
    ("redirected to s3cret@backend.example/x", "redirected to ***@backend.example/x"),
    ("redirected to //s3cret@backend.example", "redirected to //***@backend.example"),
    ("provider:s3c'r>et@[2001:db8::1]:8443/x", "***@[2001:db8::1]:8443/x"),
    # A secret held in the path: after a secret-named segment, or behind an encoded `?` or `#`.
    (
        "GET https://backend.example/api/token/s3cret/templates",
        "GET https://backend.example/api/token/***/templates",
    ),
    ("GET https://backend.example/api-key/s3cret", "GET https://backend.example/api-key/***"),
    (
        "GET https://backend.example/api%3Ftoken%3Ds3cret",
        "GET https://backend.example/api%3F***",
    ),
    ("GET https://backend.example/api%23token=s3cret", "GET https://backend.example/api%23***"),
    (
        "GET https://backend.example/api;token=s3cret",
        "GET https://backend.example/api;token=***",
    ),
    # More names: session, csrf, jwt, hmac, an index or a bracket before the separator, a name
    # up to 128 characters, any number of spaces around the separator.
    ("session=s3cret csrf=s3cret", "session=*** csrf=***"),
    ("jwt: s3cret, hmac: s3cret", "jwt: ***, hmac: ***"),
    ("params[token]=s3cret&x[api_key]=s3cret", "params[token]=***&x[api_key]=***"),
    ("token[0]=s3cret", "token[0]=***"),
    ("[token]=s3cret", "[token]=***"),
    # The last host decides: a `?` before it leaves no telling where the userinfo ends.
    (
        "redirected to s3cret@a.example?x@backend.example:8443/api",
        "redirected to ***",
    ),
    # The prose reading of `key:` / `token:` is for an unquoted name and a `:` only.
    ("{'token': 'abcdefgh'}", "{'token': '***'}"),
    ("token='abcdefgh'", "token='***'"),
    ("a" * 120 + "_token=s3cret", "a" * 120 + "_token=***"),
    (
        "password" + " " * 12 + "=" + " " * 12 + "s3cret",
        "password" + " " * 12 + "=" + " " * 12 + "***",
    ),
    # Each secret word, prefix and token format on its own.
    ("X-Amz-Credential=AKIDEXAMPLE", "X-Amz-Credential=***"),
    ("sig=s3cret signature=s3cret", "sig=*** signature=***"),
    ("sessionid=s3cret", "sessionid=***"),
    ("privatekey=s3cret", "privatekey=***"),
    *(
        (f"pushed {prefix}{'a' * 36} upstream", "pushed *** upstream")
        for prefix in ("gho_", "ghs_", "ghu_", "ghr_", "github_pat_")
    ),
    # A quoted value holding letters and digits after `token:` is not an identifier.
    ("token: 'abc123def456'", "token: '***'"),
    # An upper-case scheme, and a lower-case encoded `?`.
    (
        "GET HTTPS://provider:s3cret@backend.example/x",
        "GET HTTPS://***@backend.example/x",
    ),
    (
        "GET https://backend.example/api%3ftoken%3Ds3cret",
        "GET https://backend.example/api%3f***",
    ),
    # A host-shaped `@` on the next line does not end the userinfo of the URL above it.
    (
        "GET https://provider\nrefused for admin@backend.example/x",
        "GET https://provider\nrefused for ***@backend.example/x",
    ),
    # The userinfo separator written as `%40`, or as a fullwidth or small at-sign.
    (
        "InvalidUrlClientError: https://provider:s3cret%40backend.example:8443/api",
        "InvalidUrlClientError: https://***%40backend.example:8443/api",
    ),
    (
        "InvalidUrlClientError: https://provider:s3cret\uff20backend.example/api",
        "InvalidUrlClientError: https://***\uff20backend.example/api",
    ),
    (
        "https://provider:s3c ret\ufe6bbackend.example/api refused",
        "https://***\ufe6bbackend.example/api refused",
    ),
    (
        "NonHttpUrlClientError: provider:s3cret%40backend.example:8443/api",
        "NonHttpUrlClientError: ***%40backend.example:8443/api",
    ),
    (
        "NonHttpUrlClientError: provider:s3cret%EF%BC%A0backend.example:8443/api",
        "NonHttpUrlClientError: ***%EF%BC%A0backend.example:8443/api",
    ),
    *(
        (
            f"redirected to provider:s3cret{sign}backend.example/x",
            f"redirected to ***{sign}backend.example/x",
        )
        for sign in ("%ef%bc%a0", "%EF%B9%AB", "%ef%b9%ab")
    ),
    (
        "Cannot connect to host s3cret%40backend.example:443",
        "Cannot connect to host ***%40backend.example:443",
    ),
    # A path secret behind an encoded `/`.
    (
        "GET https://backend.example/api/token%2Fs3cret/x",
        "GET https://backend.example/api/token%2F***/x",
    ),
    ("GET https://backend.example/token%2fs3cret", "GET https://backend.example/token%2f***"),
    # A span without a dotted host does not end the userinfo, even with a port-shaped part.
    (
        "Cannot connect to https://provider s3cret@backend.example/api",
        "Cannot connect to https://***@backend.example/api",
    ),
    (
        "Cannot connect to https://provider:12 34@backend.example/api",
        "Cannot connect to https://***@backend.example/api",
    ),
    # A `--name value` flag, `=>` and an encoded `=` separate a secret name from its value.
    ("run with --password s3cret --verbose", "run with --password *** --verbose"),
    ("run with --api-key 's3c ret'", "run with --api-key '***'"),
    (
        "{'password' => 's3cret', 'user' => 'provider'}",
        "{'password' => '***', 'user' => 'provider'}",
    ),
    ("password=>s3cret", "password=>***"),
    ("sent password%3Ds3cret%26gpu%3DH100", "sent password%3D***%26gpu%3DH100"),
    (
        "GET https://backend.example/x/token%3ds3cret",
        "GET https://backend.example/x/token%3d***",
    ),
    # Prose after `session:`, `credentials:` or `auth:` stays, but a value there is masked.
    ("session: abc123def456", "session: ***"),
    ("session: Abcdefgh more", "session: *** more"),
    ("session: abc123 more", "session: *** more"),
    ("auth: basic dXNlcjpwYXNz", "auth: basic ***"),
    ("auth: basic hunter", "auth: basic ***"),
    ("credentials: hunter", "credentials: ***"),
    # Only a name made secret by those words alone.
    ("password: hunter two", "password: *** two"),
    ("session_password: hunter two", "session_password: *** two"),
    # A `user@host` with a port after a URL is masked on its own; the URL and the words stay.
    (
        "Get https://registry.example/v2/: unauthorized for provider@host.example:22",
        "Get https://registry.example/v2/: unauthorized for ***@host.example:22",
    ),
]


@pytest.mark.parametrize(("text", "expected"), REDACTED_CASES)
def test_redact_removes_credentials_and_keeps_the_rest(text, expected):
    assert redact(text) == expected


def test_redact_is_linear_on_a_long_error():
    # The patterns run on the executor's event loop, three times per loop error.
    for text in (
        "a." * 100_000,
        "key" * 70_000,
        "https://" * 25_000,
        "a=" * 100_000,
        "a:" * 100_000,
        "@" * 200_000,
        "https://a@" * 20_000,
        "password='" * 20_000,
        "'password': '" * 15_000,
        "Bearer " * 30_000,
        "token=" + "a" * 200_000,
        "Cookie: " + "a" * 200_000,
        " eyJa" * 50_000,
        "eyJ" + "a." * 100_000,
        "'password': '" * 15_000 + "x" * 10_000,
        "https://a " + "b@ " * 60_000,
        "https://a " + "@a" * 100_000,
        "https://a'" + "a." * 100_000 + "@",
        "a:b@" * 50_000,
        "a:" + "a." * 100_000 + "@a",
        "/token" * 30_000,
        "https://a/" + "token/" * 30_000,
        "token" + " " * 200_000 + "=",
        "a[" * 100_000,
        "%3F" * 60_000,
        "https://a" + "%40sha256:" * 20_000,
        "https://a" + "\uff20" * 200_000,
        "a%40" * 50_000,
        "--password " * 20_000,
        "--" * 100_000,
        "session: " * 20_000,
        "password%3D" * 20_000,
    ):
        started = time.perf_counter()
        redacted = redact(text)
        assert time.perf_counter() - started < 0.05
        assert len(redacted) <= cache_prefetch_state.MAX_REDACTED_CHARS + 1


def test_a_credential_past_the_redacted_length_is_cut_not_published():
    text = "x" * (cache_prefetch_state.MAX_REDACTED_CHARS - 10) + " password=s3cret-and-more"

    assert "s3cret" not in redact(text)
    assert redact(text).endswith("…")


@pytest.mark.parametrize(
    "text",
    [
        # Earlier redactions shrink the text; what lay past the cut must still not be pulled in.
        "Bearer " + "a" * 2440 + " https://u:s3cret" + "x" * 50 + "@backend.example/x",
        "password=" + "p" * 1500 + " " + "q" * 600 + " s3cret",
        "Authorization: " + "a" * 1990 + " s3cret",
        # A URL the cut splits before its `@`, and a token the cut splits: no half is published.
        "x" * 1985 + " https://provider:s3cret@backend.example/api",
        "x" * 1990 + " Bearer s3cretabcdefghijklmnopqrstuvwxyz",
    ],
)
def test_nothing_past_the_cut_or_split_by_it_is_published(text):
    redacted = redact(text)

    assert "s3cret" not in redacted and "s3c" not in redacted
    assert len(redacted) <= cache_prefetch_state.MAX_REDACTED_CHARS + 1
    assert redacted.endswith("…")


@pytest.mark.parametrize(
    "text",
    [
        f"manifest for {REPO}@{FRESH_DIGEST} not found",
        f"manifest unknown: {REPO}@{FRESH_DIGEST}",
        "toomanyrequests: You have reached your pull rate limit",
        "no space left on device",
        "write /var/lib/docker/tmp/x: disk quota exceeded",
        "unauthorized: authentication required",
        "pull access denied for daturaai/pytorch, repository does not exist or may require "
        "'docker login': denied: requested access to the resource is denied",
        "403 Forbidden",
        "404 Not Found",
        "net/http: TLS handshake timeout",
        "read tcp 192.0.2.2:51234->203.0.113.7:443: read: connection reset by peer",
        "dial tcp: lookup registry-1.docker.io: no such host",
        'Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for '
        "connection (Client.Timeout exceeded while awaiting headers)",
        "Error while fetching server API version: ('Connection aborted.', "
        "FileNotFoundError(2, 'No such file or directory'))",
        "NVMLError_LibraryNotFound: NVML Shared Library Not Found",
        "No such image: daturaai/pytorch:latest",
        "failed to register layer: basic checks failed",
        "Bearer token expired",
        "token expired",
        "Cannot connect to host backend.example:443 ssl:default [Connection refused]",
        "https://backend.example:8443/executors/default-docker-image",
        "monkey=1 design=2 author=3 passenger=4 cache_keyring=5 tokenizer=6 bypass=7 passport=8",
        "{'monkey': 'banana', 'author': 'provider'}",
        "404 Client Error for http+docker://localhost/v1.44/images/"
        f"{REPO}@{FRESH_DIGEST}/json: Not Found",
        # `key` / `token` before a quoted identifier or an error phrase, and a class name before
        # its message, name no secret.
        "invalid key: 'gpu_model'",
        'missing key: "docker_image_tag"',
        "Token: unexpected EOF",
        "token: invalid character",
        "RuntimeError: KeyError: 'docker_image'",
        "TokenRefreshError: refresh failed",
        "sent Token abcdefgh upstream",
        # `>` ends a URL: what follows it is not the URL's query.
        "moved to <https://backend.example/x>?",
        # An `@` not followed by a host does not end a URL's userinfo, nor does the host name.
        "GET https://backend.example/x done @ 12:00",
        "GET https://backend done @ 12:00",
        'Head "https://auth.docker.io/token": unauthorized',
        "contact provider@example.com",
        f"image {REPO}:{TAG}@{FRESH_DIGEST} pulled",
        "Session is closed",
        # Prose after `session:`, `credentials:` or `auth:` names no value.
        "failed to create session: context deadline exceeded",
        'error getting credentials: exec: "docker-credential-desktop": executable file not found',
        "auth: token expired",
        "session: expired",
        # A flag followed by another flag has no value.
        "docker login --password-stdin --username provider",
        # An e-mail address after a URL on the same line: the URL's host and the words stay.
        "GET https://backend.example/x refused for admin@example.com",
        'Get "https://registry.example/v2/": unauthorized: denied for provider@example.com',
        "GET https://registry.example:5000/v2/ denied for provider@example.com",
        "GET https://backend.example refused for admin@example.com",
        "GET https://[2001:db8::1]:8443 refused for admin@example.com",
        "https://backend.example: unauthorized for admin@example.com",
        # An encoded `@` in a path or a query is not a userinfo separator.
        "GET https://backend.example/users/provider%40example.com/templates",
    ],
)
def test_redact_leaves_errors_without_credentials_unchanged(text):
    assert redact(text) == text


def test_an_error_is_described_by_its_class_and_redacted_text():
    assert describe_error(RuntimeError("Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.c2ln")) == (
        "RuntimeError: Bearer ***"
    )
    assert describe_error(TimeoutError()) == "TimeoutError"
    assert describe_error("HTTP 503") == "HTTP 503"


def test_an_unprintable_error_is_described_by_its_class():
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("str() failed")

    assert describe_error(Unprintable()) == "Unprintable"

    state = CachePrefetchState(path=None)
    state.note_loop_error(ValueError("old"))
    state.note_loop_error(Unprintable())
    assert state.as_dict()["last_loop_error"] == _error(Reason.UNKNOWN, "Unprintable")


def test_an_error_whose_text_or_class_name_misbehaves_is_still_described():
    class HostileText(str):
        def __getitem__(self, key):
            raise RuntimeError("no slicing")

        def replace(self, *args):
            raise RuntimeError("no replace")

    class HostileStr(Exception):
        def __str__(self):
            return HostileText("password=s3cret")

    class HostileName(type):
        @property
        def __name__(cls):
            raise RuntimeError("no name")

    class Nameless(Exception, metaclass=HostileName):
        pass

    assert describe_error(HostileStr()) == "HostileStr: password=***"
    assert describe_error(Nameless("password=s3cret")) == (
        f"{cache_prefetch_state.UNNAMED_ERROR}: password=***"
    )
    state = CachePrefetchState(path=None)
    state.note_loop_error(Nameless("password=s3cret"))
    assert state.as_dict()["last_loop_error"] == _error(
        Reason.UNKNOWN, cache_prefetch_state.UNNAMED_ERROR
    )
    state.note_loop_error(HostileStr())
    assert state.as_dict()["last_loop_error"] == _error(Reason.UNKNOWN, "HostileStr")


def test_an_error_carrying_its_url_is_masked_whatever_the_password_holds():
    # aiohttp's InvalidURL carries the URL it refused as `.url`, and its text is that URL.
    from aiohttp.client_exceptions import InvalidUrlClientError

    for password in ('s3c/r"et', "s3c/r et", "s3c/r>et", "s3c/r<et"):
        url = f"https://provider:{password}@backend.example/executors/x"

        described = describe_error(InvalidUrlClientError(url))

        assert described == "InvalidUrlClientError: https://***@backend.example/executors/x"


def test_an_error_carrying_a_url_without_a_scheme_is_masked():
    # NonHttpUrlClientError and NonHttpUrlRedirectClientError carry their URL in `args`, no `.url`.
    from aiohttp.client_exceptions import NonHttpUrlClientError, NonHttpUrlRedirectClientError
    from yarl import URL

    for error_class in (NonHttpUrlClientError, NonHttpUrlRedirectClientError):
        for url, masked in (
            ("provider:s3cret@backend.example:8443/api", "***@backend.example:8443/api"),
            ("provider:s3c r'et@backend.example/api", "***@backend.example/api"),
        ):
            for carried in (URL(url, encoded=True), url):
                described = describe_error(error_class(carried))

                assert described == f"{error_class.__name__}: {masked}"


def test_an_error_carrying_its_request_url_is_masked_as_one_url():
    # Only the carried URL tells where this one ends: its password holds a space, then `://`.
    from aiohttp import RequestInfo
    from aiohttp.client_exceptions import ClientResponseError
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    url = URL.build(
        scheme="https",
        user="provider",
        password="s3c r://et",
        host="backend.example",
        path="/x",
        encoded=True,
    )
    info = RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url)

    described = describe_error(ClientResponseError(info, (), status=401, message="Unauthorized"))

    assert described == (
        "ClientResponseError: 401, message='Unauthorized', url='https://***@backend.example/x'"
    )


# Described a second time, "AuthFailure: registry said no" reads as a secret named AuthFailure.
AuthFailure = type("AuthFailure", (Exception,), {})


@pytest.mark.parametrize("error", [KeyError("Descriptor"), AuthFailure("registry said no")])
def test_a_per_image_error_is_logged_described_once_and_published_as_a_record(
    monkeypatch, error
):
    logger = MagicMock()
    monkeypatch.setattr(cache_template_service, "logger", logger)
    state = CachePrefetchState(path=None)
    client = _make_client(local_digests=[STALE_DIGEST])
    client.images.get_registry_data.side_effect = error

    _run(client, _template(None), state)

    described = describe_error(error)
    assert described in ("KeyError: 'Descriptor'", "AuthFailure: registry said no")
    assert _image(state)["last_remote_error"] == _error(
        Reason.DOCKER_ERROR, type(error).__name__
    )
    (message,), _ = logger.warning.call_args
    assert message.endswith(described)


def test_a_carried_url_without_a_scheme_the_cut_splits_is_dropped_whole():
    from aiohttp.client_exceptions import NonHttpUrlRedirectClientError

    url = "provider:" + "p" * 1900 + " " + "q" * 200 + "s3cret@backend.example/x"

    described = describe_error(NonHttpUrlRedirectClientError(url))

    assert "ppp" not in described and "qqq" not in described and "s3cret" not in described
    assert described == "NonHttpUrlRedirectClientError: …"


def test_a_long_carried_url_is_described_in_bounded_time():
    from aiohttp.client_exceptions import InvalidUrlClientError

    url = "https://provider:s3cret@backend.example/" + "x/" * 2_500_000

    started = time.perf_counter()
    described = describe_error(InvalidUrlClientError(url))

    assert time.perf_counter() - started < 0.05
    assert "s3cret" not in described
    assert len(described) <= cache_prefetch_state.MAX_REDACTED_CHARS + 30


def test_a_carried_url_the_cut_splits_is_dropped_whole():
    from aiohttp.client_exceptions import InvalidUrlClientError

    url = "https://provider:" + "p" * 1900 + " " + "q" * 200 + "s3cret@backend.example/executors/x"

    described = describe_error(InvalidUrlClientError(url))

    assert "ppp" not in described and "qqq" not in described and "s3cret" not in described
    assert described == "InvalidUrlClientError: …"


def test_the_document_holds_no_error_text_and_only_the_backend_origin():
    state = CachePrefetchState(path=None, backend_url="https://provider:s3cret@backend.example")

    state.note_docker(available=False, error=RuntimeError("Authorization: Bearer s3cret"))
    state.note_loop_error(ConnectionError("https://provider:s3cret@backend.example refused"))
    state.record_loop_outcome(Outcome.LOOP_ERROR, error=ValueError("password=s3cret"))
    state.record_image_outcome(IMAGE_REF, Outcome.PULL_FAILED, error="?token=s3cret")

    payload = state.render()
    assert "s3cret" not in payload and "provider" not in payload
    doc = json.loads(payload)
    assert doc["backend_url"] == "https://backend.example"
    assert doc["docker_error"] == _error(Reason.DOCKER_ERROR, "RuntimeError")
    # Not an aiohttp error, so not known to be the backend's: no host.
    assert doc["last_loop_error"] == _error(Reason.CONNECT_ERROR, "ConnectionError")
    assert doc["last_error"] == _error(Reason.UNKNOWN, "ValueError")
    assert _image(state)["last_error"] == _error(Reason.PULL_FAILED)


@pytest.mark.parametrize(
    ("backend_url", "published"),
    [
        ("https://s3cret@backend.example/api?gpu=H100", "https://backend.example"),
        ("https://provider:s3cret@backend.example:8443/api", "https://backend.example"),
        ("https://backend.example:8443/api?token=s3cret#x", "https://backend.example"),
        ("https://bäckend.example/api", "https://xn--bckend-bua.example"),
        ("http://[::1]:8443/api", "http://[::1]"),
        ("HTTPS://Backend.Example/API", "https://backend.example"),
        # Nothing of the path is published, secret or not.
        ("https://backend.example/api;token=s3cret", "https://backend.example"),
        ("https://backend.example/api/token/s3cret", "https://backend.example"),
        ("https://backend.example/api%3Ftoken%3Ds3cret", "https://backend.example"),
        ("https://backend.example/api/token%2Fs3cret", "https://backend.example"),
    ],
)
def test_the_backend_url_is_published_as_its_scheme_and_host(backend_url, published):
    doc = CachePrefetchState(path=None, backend_url=backend_url).as_dict()

    assert doc["backend_url"] == published


@pytest.mark.parametrize(
    "backend_url",
    [
        # yarl refuses these; none of their text is published.
        "https://provider:pa/s3cret@backend.example/api",
        "https://provider:s3cret@backend.example:bad/api",
        'https://provider:s3c/r"et@backend.example/api',
        "https://provider:s3c/r et@backend.example/api",
        "https://provider:" + "s3cret/" * 43 + "@backend.example/api",
        "https://provider:s3c/ret@bäckend.example/api",
        # yarl parses these, but the password lands in the port and path.
        "https://provider:1234/s3cret@backend.example/api",
        "https://provider:1234?s3cret@backend.example/api",
        "https://provider:1234#s3cret@backend.example/api",
        # yarl reads the userinfo before a `%40` as part of the host.
        "https://s3cret%40backend.example/api",
        "https://provider:s3cret%40backend.example:8443/api",
        "https://provider:s3cret\uff20backend.example/api",
        "https://s3cret%EF%BC%A0backend.example/api",
        # No scheme, or no scheme and host.
        "//backend.example/api",
        "backend.example/api",
        "provider:s3cret@backend.example:8443/api",
        # An at-sign beyond the userinfo's, in any spelling: the host cannot be told.
        "https://provider:p@ss@backend.example/api",
        "https://provider:p%40ss@backend.example/api",
        "https://provider:p\uff20ss@backend.example/api",
        "https://backend.example/x?to=a@b.example",
        # A host yarl accepts that is not ASCII labels.
        "https://s3cret%2Fbackend.example/api",
        "https://s3c_ret.backend.example/api",
    ],
)
def test_a_backend_url_without_a_clean_parse_is_published_as_a_placeholder(backend_url):
    doc = CachePrefetchState(path=None, backend_url=backend_url).as_dict()

    assert doc["backend_url"] == cache_prefetch_state.UNPARSEABLE_URL


def test_document_is_capped_and_marked_truncated():
    state = CachePrefetchState(path=None)
    for index in range(40):
        ref = f"{REPO}-{index}:{TAG}"
        state.record_image_outcome(ref, Outcome.PULL_FAILED, error=RuntimeError("y" * 500))
        state.note_local_digests(ref, [f"{REPO}@{STALE_DIGEST}"])

    payload = state.render()

    assert len(payload.encode("utf-8")) <= MAX_PAYLOAD_BYTES
    assert json.loads(payload)["truncated"] is True


def test_flush_writes_atomically_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "nested" / "state.json"
    state = CachePrefetchState(path=str(path))
    state.begin_sweep()
    state.record_image_outcome(IMAGE_REF, Outcome.UP_TO_DATE)

    state.flush()

    assert json.loads(path.read_text())["images"][IMAGE_REF]["last_outcome"] == Outcome.UP_TO_DATE
    assert list(path.parent.iterdir()) == [path]


def test_flush_failure_is_swallowed(tmp_path):
    # A read-only filesystem must cost the loop nothing.
    path = tmp_path / "state.json"
    path.mkdir()  # a directory where the file should go: os.replace will refuse
    state = CachePrefetchState(path=str(path))

    state.flush()  # must not raise

    assert path.is_dir()


# --- the published error record ------------------------------------------------------------


def _connector_error(os_error, host="backend.example"):
    from aiohttp.client_exceptions import ClientConnectorError
    from aiohttp.client_reqrep import ConnectionKey

    return ClientConnectorError(ConnectionKey(host, 443, True, True, None, None, None), os_error)


def _response_error(url, status, error_class=None):
    from aiohttp import RequestInfo
    from aiohttp.client_exceptions import ClientResponseError
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    info = RequestInfo(URL(url), "GET", CIMultiDictProxy(CIMultiDict()), URL(url))
    return (error_class or ClientResponseError)(info, (), status=status, message="Unauthorized")


DockerException = type("DockerException", (Exception,), {})
APIError = type("APIError", (DockerException,), {})


def _reason_cases():
    import errno
    import socket

    from aiohttp.client_exceptions import (
        ContentTypeError,
        InvalidUrlClientError,
        NonHttpUrlClientError,
        ServerTimeoutError,
    )

    from services.cache_template_service import PullStreamError

    backend = "backend.example"
    return [
        (asyncio.TimeoutError(), _error(Reason.TIMEOUT, "TimeoutError", backend)),
        (ServerTimeoutError("read"), _error(Reason.TIMEOUT, "ServerTimeoutError", backend)),
        (
            _connector_error(ConnectionRefusedError("refused")),
            _error(Reason.CONNECT_ERROR, "ClientConnectorError", backend),
        ),
        (
            _connector_error(socket.gaierror("no such host")),
            _error(Reason.DNS_ERROR, "ClientConnectorError", backend),
        ),
        (
            _response_error("https://u:s3cret@registry.example/v2/?token=s3cret", 401),
            _error(Reason.HTTP_ERROR, "ClientResponseError", "registry.example", 401),
        ),
        (
            _response_error("https://backend.example/x", 200, ContentTypeError),
            _error(Reason.BAD_RESPONSE, "ContentTypeError", backend, 200),
        ),
        (
            InvalidUrlClientError("https://u:s3c/ret@backend.example/x"),
            _error(Reason.INVALID_URL, "InvalidUrlClientError", backend),
        ),
        (
            NonHttpUrlClientError("u:s3cret@backend.example/x"),
            _error(Reason.INVALID_URL, "NonHttpUrlClientError", backend),
        ),
        (OSError(errno.ENOSPC, "No space left"), _error(Reason.DISK_FULL, "OSError")),
        (NVMLError_LibraryNotFound(), _error(Reason.NVML_ERROR, "NVMLError_LibraryNotFound")),
        (
            APIError("toomanyrequests: rate limit, token=s3cret"),
            _error(Reason.RATE_LIMITED, "APIError"),
        ),
        (APIError("daemon hiccup"), _error(Reason.DOCKER_ERROR, "APIError")),
        (
            PullStreamError("unauthorized: authentication required"),
            _error(Reason.REGISTRY_DENIED, "PullStreamError"),
        ),
        (PullStreamError("something new"), _error(Reason.PULL_FAILED, "PullStreamError")),
        (KeyError("password=s3cret"), _error(Reason.UNKNOWN, "KeyError")),
    ]


def test_a_loop_error_is_published_as_its_class_reason_host_and_status():
    for error, published in _reason_cases():
        state = CachePrefetchState(path=None, backend_url="https://u:s3cret@backend.example/api")
        state.note_loop_error(error)

        assert state.as_dict()["last_loop_error"] == published, error
        assert "s3cret" not in state.render()


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ('404 Client Error: Not Found ("No such image: daturaai/torch@sha256:aaa")', "image_missing"),
        ("toomanyrequests: You have reached your pull rate limit", "rate_limited"),
        ("429 Client Error: Too Many Requests", "rate_limited"),
        ("write /var/lib/docker/tmp/x: no space left on device", "disk_full"),
        ("unauthorized: authentication required", "registry_denied"),
        ("403 Forbidden", "registry_denied"),
        ("manifest for daturaai/torch@sha256:aaa not found: manifest unknown", "manifest_unknown"),
        # A status code inside a digest is not the registry's answer.
        ("pull daturaai/torch@sha256:c404e failed: connection reset by peer", "registry_unreachable"),
        ("net/http: TLS handshake timeout", "registry_unreachable"),
        ("something new", "pull_failed"),
    ],
)
def test_a_pull_error_is_published_as_the_reason_its_text_names(text, code):
    state = CachePrefetchState(path=None)

    state.note_pull_error(IMAGE_REF, RuntimeError(text))

    assert _image(state)["last_pull_error"] == _error(code, "RuntimeError")


def test_a_malformed_template_publishes_only_the_fields_it_lacks():
    state = CachePrefetchState(path=None)

    state.note_malformed_template({"docker_image": "https://u:s3cret@registry.example/x"})
    assert state.as_dict()["last_malformed_template"] == {"missing": ["docker_image_tag"]}
    state.note_malformed_template("password=s3cret")
    assert state.as_dict()["last_malformed_template"] == {
        "missing": ["docker_image", "docker_image_tag"]
    }
    assert "s3cret" not in state.render()


# --- fuzz: no text of an error reaches the document ---------------------------------------

# Credentials a 4-character window of which never occurs in a document by chance.
FUZZ_CREDENTIALS = (
    "Zq9Xw7Yv",
    "pQ@z/ Kx'J\"m>W#?Tq",
    "Vb%40Nm\uff20Rt\ufe6bHy",
    STUB_JWT,
    "ghp_" + "Kq7Zw" * 8,
    "Jk Lm\tNp\xa0Qr",
)
# Shapes the log redaction does not cover, and ones that cut a URL at the 2000-character limit.
FUZZ_EXTRA_TEMPLATES = (
    "{c}",
    "https://{c}@backend.example",
    "https://provider:{c}@backend.example:8443/api",
    "provider:{c}@backend.example:8443/api",
    "https://{c}:1234/x@backend.example/api",
    "https://provider:1234?{c}@backend.example/api",
    "https://{c}%40backend.example/api",
    "https://{c}\uff20backend.example/api",
    "https://first.last:12 {c}@host.example",
    "https:\\/\\/user:{c}@host\\/api",
    "x" * 1985 + " https://provider:{c}@backend.example/api",
    "Bearer " + "a" * 2440 + " https://u:{c}@backend.example/x",
    "<password>{c}</password>",
    "('password', '{c}')",
    'Digest username="u", response="{c}"',
    "userpass={c} creds={c} otp={c} api%5Ftoken={c}",
    "passwd {c} -p {c}",
    "session: {c} auth: basic {c}",
    "https://backend.example/api/{c}",
    "https://backend.example/x/token%2F{c}",
)
FUZZ_TEMPLATES = tuple(
    dict.fromkeys(
        [text.replace("s3cret", "{c}") for text, _ in REDACTED_CASES if "s3cret" in text]
        + list(FUZZ_EXTRA_TEMPLATES)
    )
)


def _fuzz_errors(text):
    """Every kind of error the loop meets, carrying ``text`` in its message and attributes."""
    from aiohttp import ClientError
    from aiohttp.client_exceptions import (
        InvalidUrlClientError,
        NonHttpUrlClientError,
        NonHttpUrlRedirectClientError,
    )

    from services.cache_template_service import PullStreamError

    errors = [
        RuntimeError(text),
        OSError(text),
        KeyError(text),
        asyncio.TimeoutError(text),
        ClientError(text),
        InvalidUrlClientError(text),
        NonHttpUrlClientError(text),
        NonHttpUrlRedirectClientError(text),
        PullStreamError(text),
        APIError(text),
        NVMLError_LibraryNotFound(text),
        _connector_error(OSError(text), host=text),
    ]
    try:
        errors.append(_response_error(text, 401))
    except Exception:
        pass
    return errors


def _fill(state, error, text):
    state.note_docker(available=False, error=error)
    state.note_gpu("unknown", "unknown", error=error)
    state.note_backend(status=None, template_count=0, error=error)
    state.note_malformed_template({"docker_image": text})
    state.note_loop_error(error)
    state.record_loop_outcome(Outcome.LOOP_ERROR, error=error)
    state.note_local_digests(IMAGE_REF, [], error=error)
    state.note_remote_digest(IMAGE_REF, None, error=error)
    state.note_pull_error(IMAGE_REF, error)
    state.note_cleanup_error(IMAGE_REF, error)
    state.record_image_outcome(IMAGE_REF, Outcome.PULL_FAILED, error=error)


def _windows(credential, size=4):
    return {credential[i : i + size] for i in range(len(credential) - size + 1)}


def test_no_substring_of_a_credential_in_any_error_reaches_the_document():
    checked = 0
    for credential in FUZZ_CREDENTIALS:
        windows = _windows(credential)
        for template in FUZZ_TEMPLATES:
            text = template.replace("{c}", credential)
            for error in _fuzz_errors(text):
                # The backend URL is the text too, so a credential in COMPUTE_REST_API_URL is fed
                # to backend_url and to the host every aiohttp error falls back to.
                state = CachePrefetchState(path=None, backend_url=text)
                _fill(state, error, text)
                payload = state.render()
                readable = json.dumps(json.loads(payload), ensure_ascii=False)
                leaked = [w for w in windows if w in payload or w in readable]
                assert not leaked, (template, type(error).__name__, leaked, payload)
                checked += 1
    assert checked > 5_000
