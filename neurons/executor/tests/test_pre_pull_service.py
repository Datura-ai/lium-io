"""DAH-2977 — idle-time pre-pull of the top-N official templates.

On by default since DAH-3604 (off restores the pre-DAH-2977 behaviour byte-for-byte); on, at most one digest-pinned pull per
sweep, never while a rental exists or is starting, never below the disk floor (LRU
pre-pulled images are evicted first), its budget cut at the per-image timeout and short of
the loop's refresh deadline; the loop starts it as a task and never waits for it.
"""

import asyncio
import logging
import socket
import threading
import time
from unittest.mock import MagicMock

import docker
import pytest
import urllib3

# conftest replaces the docker module with a MagicMock; except-clauses in the
# services need a real exception class to catch, and the pull helper unpacks
# docker.auth's registry resolution.
docker.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
docker.auth.resolve_repository_name = lambda repo: ("docker.io", repo)
docker.auth.get_config_header = lambda api, registry: None

from services import cache_template_service, pre_pull_service  # noqa: E402
from services.pre_pull_service import PrePuller, _pull_pinned, rental_activity  # noqa: E402

REPO = "daturaai/pytorch"
DEFAULT_TAG = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind"
CU128_TAG = "2.12.0-py3.12-cuda12.8-devel-ubuntu24.04-dind"
CU128_REF = f"{REPO}:{CU128_TAG}"
DIGEST_CU128 = "sha256:2d19c94ce8a37c6fa364f8a6211d8b6dc1a44ece574c4c22ab5579925ce7a4c8"
GIB = pre_pull_service.GIB


def _entry(repo: str, tag: str, digest: str | None, size: int = 4_300_000_000, pre_pull: bool = True) -> dict:
    data = {"docker_image": repo, "docker_image_tag": tag, "docker_image_size": size, "pre_pull": pre_pull}
    if digest:
        data["docker_image_digest"] = digest
    return data


def _client(present_digests: set[str] = frozenset(), containers: list = ()) -> MagicMock:
    client = MagicMock()

    def images_get(ref: str):
        if any(ref.endswith(f"@{digest}") for digest in present_digests):
            return MagicMock()
        raise docker.errors.ImageNotFound(ref)

    client.images.get.side_effect = images_get
    client.containers.list.return_value = list(containers)
    return client


def _container(name: str, image: str = CU128_REF, status: str = "running") -> MagicMock:
    container = MagicMock()
    container.name = name
    container.status = status
    container.attrs = {"Config": {"Image": image}}
    return container


def _proc(cmdline: list[str]) -> MagicMock:
    proc = MagicMock()
    proc.info = {"cmdline": cmdline}
    return proc


@pytest.fixture(autouse=True)
def quiet_node(monkeypatch):
    """Idle host with plenty of disk, no start jitter, and an instant fake pull."""
    monkeypatch.setattr(pre_pull_service.psutil, "process_iter", lambda *_: [])
    monkeypatch.setattr(pre_pull_service.psutil, "disk_usage", lambda _: MagicMock(free=1000 * GIB))
    monkeypatch.setattr(pre_pull_service.settings, "PRE_PULL_START_JITTER_SECONDS", 0)
    monkeypatch.setattr(pre_pull_service.settings, "PRE_PULL_MIN_FREE_GB", 200)
    pulls: list[tuple] = []

    def fake_pull(client, repo, tag, digest, timeout_seconds):
        pulls.append((repo, tag, digest, timeout_seconds))
        return "pull_ok", None

    monkeypatch.setattr(pre_pull_service, "_pull_pinned", fake_pull)
    return pulls


def _sweep(puller: PrePuller, entries: list[dict]) -> None:
    asyncio.run(puller.sweep(entries))


def _one_loop_iteration(
    monkeypatch, templates: list[dict], puller: type | None = None, iterations: int = 1
) -> dict:
    """Run run_cache_template_prefetch through ``iterations`` refreshes and record what it did.

    The loop is stopped inside its sleep after the last refresh. The sweep is a task the loop
    does not await, so the fake sleep yields once before stopping to let it run."""
    seen: dict = {"params": None, "ensured": [], "swept": [], "protected": [], "pullers": 0, "sleeps": []}
    seen["protected_at_ensure"] = []  # what the puller's `protected` was while the mandatory pass ran
    real_sleep = asyncio.sleep
    pullers: list = []

    async def fetch(session, url, params):
        seen["params"] = dict(params)
        return templates, 200, None

    async def ensure(client, data, state, keep_tags=frozenset()):
        seen["ensured"].append((data["docker_image_tag"], set(keep_tags)))
        seen["protected_at_ensure"].append(getattr(pullers[0], "protected", None) if pullers else None)

    class FakePuller:
        def __init__(self, client, state_path=None):
            seen["pullers"] += 1
            pullers.append(self)

        async def sweep(self, entries, protected=frozenset(), deadline=None):
            seen["swept"].append([e["docker_image_tag"] for e in entries])
            seen["protected"].append(set(protected))
            seen["deadline"] = deadline

    async def stop(seconds):
        seen["slept"] = seconds
        seen["sleeps"].append(seconds)
        await real_sleep(0)
        if len(seen["sleeps"]) >= iterations:
            raise asyncio.CancelledError

    monkeypatch.setattr(cache_template_service, "_fetch_templates", fetch)
    monkeypatch.setattr(cache_template_service, "_ensure_template", ensure)
    if puller is not None:
        real_init = puller.__init__

        def init(self, client, state_path=None):
            pullers.append(self)
            real_init(self, client, state_path)

        puller.__init__ = init
    monkeypatch.setattr(cache_template_service, "PrePuller", puller or FakePuller)
    monkeypatch.setattr(cache_template_service, "_get_gpu_info", lambda: ("NVIDIA H100 80GB HBM3", "580.65.06", None))
    monkeypatch.setattr(cache_template_service.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cache_template_service.run_cache_template_prefetch(state_path=None))
    return seen


# --- on by default (DAH-3604): a node whose .env has no PRE_PULL_* line -----------------

# Prod's answer for an H100 on driver 580 (GET /executors/default-docker-image?include_pre_pull=true,
# 23 Sep 2026, backend PRE_PULL_TEMPLATES_TOP_N=2): the node's default image, then the top-2.
LIUM1_DEFAULT_TAG = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind-lium1"
LIUM1_CU128_TAG = "2.11.0-py3.12-cuda12.8-devel-ubuntu24.04-dind-lium1"
CUDA_REPO, CUDA_TAG = "nvidia/cuda", "13.0.3-devel-ubuntu22.04"
DIGEST_LIUM1_DEFAULT = "sha256:efcbc04991511200e4763069434aeb1daf7baf40a0376e3c3296aef929828d89"
DIGEST_LIUM1_CU128 = "sha256:b3aa9c485c5e91c051a3ef21c45a1b0f68dd1e101495cd8ade57d12eef861eb4"
DIGEST_CUDA = "sha256:3869b846a8cc495ce11c172d87cfc0da8874b910d14a9810bec6b6182e9ee9f8"
PROD_H100_ANSWER = [
    _entry(REPO, LIUM1_DEFAULT_TAG, DIGEST_LIUM1_DEFAULT, size=9_133_782_473, pre_pull=False),
    _entry(REPO, LIUM1_CU128_TAG, DIGEST_LIUM1_CU128, size=6_491_891_139),
    _entry(CUDA_REPO, CUDA_TAG, DIGEST_CUDA, size=3_961_262_240),
]
PRE_PULL_ENV = (
    "PRE_PULL_TEMPLATES_ENABLED",
    "PRE_PULL_MIN_FREE_GB",
    "PRE_PULL_TIMEOUT_SECONDS",
    "PRE_PULL_START_JITTER_SECONDS",
    "CACHE_TEMPLATE_REFRESH_SECONDS",
)


def _node_settings(monkeypatch, **env: str):
    """Settings the way a node builds them, with no PRE_PULL_* in its environment unless given."""
    from core.config import Settings

    for name in PRE_PULL_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_a_node_without_the_flag_in_its_env_pre_pulls_and_false_opts_it_out(monkeypatch):
    assert _node_settings(monkeypatch).PRE_PULL_TEMPLATES_ENABLED is True
    for off in ("false", "False", "0"):
        node = _node_settings(monkeypatch, PRE_PULL_TEMPLATES_ENABLED=off)
        assert node.PRE_PULL_TEMPLATES_ENABLED is False


def test_by_default_the_loop_asks_for_the_top_n_and_hands_only_those_to_the_puller(monkeypatch):
    monkeypatch.setattr(cache_template_service, "settings", _node_settings(monkeypatch))

    seen = _one_loop_iteration(monkeypatch, PROD_H100_ANSWER)

    assert seen["params"] == {
        "gpu_model": "NVIDIA H100 80GB HBM3",
        "driver_version": "580.65.06",
        "include_pre_pull": "true",
    }
    # the default image stays on the mandatory path (validator-checked), its old-tag cleanup
    # shielding the pre-pull tags; the puller gets the two top-N entries and nothing else
    assert seen["ensured"] == [(LIUM1_DEFAULT_TAG, {LIUM1_CU128_TAG, CUDA_TAG})]
    assert seen["swept"] == [[LIUM1_CU128_TAG, CUDA_TAG]]
    assert seen["protected"] == [{f"{REPO}:{LIUM1_DEFAULT_TAG}"}]


def test_by_default_a_sweep_that_raises_never_stops_the_default_image_refresh(monkeypatch, caplog):
    monkeypatch.setattr(cache_template_service, "settings", _node_settings(monkeypatch))
    sweeps: list[int] = []
    outcomes: list[str] = []
    real_record = cache_template_service.CachePrefetchState.record_loop_outcome

    class BrokenPuller:
        def __init__(self, client, state_path=None):
            pass

        async def sweep(self, entries, protected=frozenset(), deadline=None):
            sweeps.append(len(entries))
            raise RuntimeError("registry unreachable")

    def record(self, outcome, error=None):
        outcomes.append(outcome)
        real_record(self, outcome, error=error)

    monkeypatch.setattr(cache_template_service.CachePrefetchState, "record_loop_outcome", record)
    with caplog.at_level(logging.WARNING):
        seen = _one_loop_iteration(monkeypatch, PROD_H100_ANSWER, puller=BrokenPuller, iterations=2)

    assert [tag for tag, _ in seen["ensured"]] == [LIUM1_DEFAULT_TAG, LIUM1_DEFAULT_TAG]
    assert sweeps == [2, 2]  # the failed sweep is over, so the next refresh starts a new one
    assert outcomes == [cache_template_service.Outcome.SWEEP_OK] * 2
    failures = [r for r in caplog.records if "sweep failed: registry unreachable" in r.getMessage()]
    assert len(failures) == 2


# --- idle guard -------------------------------------------------------------------------


def test_no_pull_while_a_rental_container_exists(quiet_node, caplog):
    client = _client(containers=[_container("pod_1f2e3d", image=f"{REPO}:{DEFAULT_TAG}")])

    with caplog.at_level(logging.INFO):
        _sweep(PrePuller(client, state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert quiet_node == []
    assert any("node busy (rental container pod_1f2e3d)" in r.getMessage() for r in caplog.records)


def test_no_pull_while_the_validator_drives_docker_over_ssh(quiet_node, monkeypatch):
    # a rental's own `docker pull` / `docker run` arrive through this process on the host
    procs = [_proc(["sshd: root"]), _proc(["docker", "system", "dial-stdio"])]
    monkeypatch.setattr(pre_pull_service.psutil, "process_iter", lambda *_: procs)

    _sweep(PrePuller(_client(), state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert quiet_node == []


@pytest.mark.parametrize("status", ["exited", "dead"])
def test_a_stopped_rental_counts_until_its_container_is_removed(quiet_node, status):
    # a stopped rental's container is `exited` and can start again
    client = _client(containers=[_container("pod_stopped", status=status)])

    assert rental_activity(client) == "rental container pod_stopped"
    _sweep(PrePuller(client, state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])
    assert quiet_node == []


def _verifyx_ssh_cmdline() -> list[str]:
    # the validator's SSH command (verifyx_validation_service), as the host's process table splits it
    python, script = "/root/app/.venv/bin/python", "/root/app/src/verifyx_executor.py"
    return f"{python} {script} --seed 42 --cipher_text abc".split()


def _verifyx_local_cmdline() -> list[str]:
    from services import local_verify_service

    script = str(local_verify_service.VERIFYX_SCRIPT)
    return ["/usr/bin/python3", script, "--seed", "42", "--cipher_text", "abc"]


@pytest.mark.parametrize(
    "cmdline", [_verifyx_ssh_cmdline, _verifyx_local_cmdline], ids=["ssh", "local-verify"]
)
def test_no_pull_while_the_validator_measures_bandwidth(quiet_node, monkeypatch, caplog, cmdline):
    # fresh read of 941e5ace: its download sample feeds the 100 Mbps EMA gate; a pull sharing the link lowers it
    procs = [_proc(["sshd: root"]), _proc(cmdline())]
    monkeypatch.setattr(pre_pull_service.psutil, "process_iter", lambda *_: procs)

    assert rental_activity(_client()) == "validator bandwidth test"
    with caplog.at_level(logging.INFO):
        _sweep(PrePuller(_client(), state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])
    assert quiet_node == []
    assert any("node busy (validator bandwidth test)" in r.getMessage() for r in caplog.records)


def test_no_pull_while_the_validator_probes_container_download_speed(quiet_node, monkeypatch, caplog):
    # the validator's SSH shell command (container_download_speed), as the host's process table splits it
    from_validator = "net=bridge; echo \"lium_dl_net name=$net\"; /usr/bin/docker run --rm alpine -c 'echo lium_dl'"
    procs = [_proc(["sshd: root"]), _proc(["bash", "-c", from_validator])]
    monkeypatch.setattr(pre_pull_service.psutil, "process_iter", lambda *_: procs)

    assert rental_activity(_client()) == "validator download test"
    with caplog.at_level(logging.INFO):
        _sweep(PrePuller(_client(), state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])
    assert quiet_node == []
    assert any("node busy (validator download test)" in r.getMessage() for r in caplog.records)


# --- the pull itself: timeout, preemption, retag ----------------------------------------


def _pull_client(events: list[dict]) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    response = MagicMock()
    client.api._post.return_value = response
    client.api._stream_helper.return_value = iter(events)
    return client, response


def _stalling_client(events: list[dict], stall_seconds: float = 3.0) -> tuple[MagicMock, MagicMock]:
    """A pull whose stream sends `events`, then goes silent until its socket is shut down (or
    `stall_seconds` pass, when it raises what the socket read timeout raises). A close() of the
    response does not end it: on a real socket it does not wake a blocked read."""
    client = MagicMock()
    response = MagicMock()
    closed = threading.Event()
    sock = client.api._get_raw_response_socket.return_value = MagicMock(spec=["shutdown"])
    sock.shutdown.side_effect = lambda how: closed.set()
    client.api._post.return_value = response

    def stream(*_args, **_kwargs):
        yield from events
        if not closed.wait(stall_seconds):
            raise urllib3.exceptions.ReadTimeoutError(None, None, "Read timed out.")
        raise urllib3.exceptions.ProtocolError("connection closed")

    client.api._stream_helper.side_effect = stream
    return client, response


@pytest.fixture
def fast_checks(monkeypatch):
    monkeypatch.setattr(pre_pull_service, "BANDWIDTH_TEST_CHECK_SECONDS", 0.01)


def test_pull_is_cancelled_when_a_rental_starts_mid_pull(monkeypatch, fast_checks):
    activity = iter([None, "rental container pod_9"])
    monkeypatch.setattr(pre_pull_service, "rental_activity", lambda _: next(activity))
    monkeypatch.setattr(pre_pull_service, "ACTIVITY_CHECK_SECONDS", 0)
    client, response = _stalling_client([{"status": "Downloading"}] * 5)

    outcome, detail = _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, timeout_seconds=60)

    assert (outcome, detail) == ("preempted", "rental container pod_9")
    response.close.assert_called_once()  # closing the stream is what aborts the daemon-side pull
    client.images.get.assert_not_called()


def test_a_bandwidth_test_starting_mid_pull_stops_it_within_seconds(monkeypatch, fast_checks):
    # the full idle check (a docker call) runs every 10 s; the probe check alone every 2 s
    too_early = lambda _: pytest.fail("full check before 10 s")  # noqa: E731
    monkeypatch.setattr(pre_pull_service, "rental_activity", too_early)
    probe = iter([None, "validator bandwidth test"])
    monkeypatch.setattr(pre_pull_service, "host_activity", lambda: next(probe))
    client, response = _stalling_client([{"status": "Downloading"}] * 3)

    outcome, detail = _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, timeout_seconds=60)

    assert (outcome, detail) == ("preempted", "validator bandwidth test")
    response.close.assert_called_once()
    client.images.get.assert_not_called()


def test_a_bandwidth_test_stops_a_pull_whose_stream_has_gone_quiet(monkeypatch, fast_checks):
    # review (12 Oct): the probe check ran only when a stream event arrived, so a quiet stream
    # (a big layer extracting) kept pulling beside the bandwidth sample until the 60 s read
    # timeout. The check now runs on its own timer and closes the stream itself.
    monkeypatch.setattr(pre_pull_service, "rental_activity", lambda _: None)
    monkeypatch.setattr(pre_pull_service, "host_activity", lambda: "validator bandwidth test")
    client, response = _stalling_client([], stall_seconds=3.0)

    started = time.monotonic()
    outcome, detail = _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, timeout_seconds=60)

    assert (outcome, detail) == ("preempted", "validator bandwidth test")
    assert time.monotonic() - started < 1.0  # not the stall's (read timeout's) length
    response.close.assert_called_once()
    client.images.get.assert_not_called()


def test_pull_stops_at_the_per_image_timeout(monkeypatch, fast_checks):
    monkeypatch.setattr(pre_pull_service, "rental_activity", lambda _: None)
    monkeypatch.setattr(pre_pull_service, "host_activity", lambda: None)
    client, response = _stalling_client([{"status": "Downloading"}] * 3)

    outcome, detail = _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, timeout_seconds=0.05)

    assert (outcome, detail) == ("timeout", "exceeded 0s")
    response.close.assert_called_once()
    client.images.get.assert_not_called()


# --- disk guard -------------------------------------------------------------------------


def test_disk_guard_skips_the_pull_when_nothing_can_be_evicted(quiet_node, monkeypatch, caplog):
    monkeypatch.setattr(pre_pull_service.psutil, "disk_usage", lambda _: MagicMock(free=150 * GIB))
    client = _client()

    with caplog.at_level(logging.INFO):
        _sweep(PrePuller(client, state_path=None), [_entry(REPO, CU128_TAG, DIGEST_CU128)])

    assert quiet_node == []
    client.images.remove.assert_not_called()
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("pre_pull image="))
    assert "outcome=insufficient_disk" in line and "floor 200 GiB" in line


class _Disk:
    """A node whose free space moves with what is pulled and removed (on disk = 3 × compressed)."""

    def __init__(self, monkeypatch, free_gib: float):
        self.free = free_gib * GIB
        self.present: dict[str, int] = {}  # digest -> bytes on disk
        self.sizes: dict[str, int] = {}
        self.pulls: list[str] = []
        self.client = _client()
        self.client.images.get.side_effect = self._get
        self.client.images.remove.side_effect = self._remove
        free = lambda _: MagicMock(free=self.free)  # noqa: E731
        monkeypatch.setattr(pre_pull_service.psutil, "disk_usage", free)
        monkeypatch.setattr(pre_pull_service, "_pull_pinned", self._pull)

    def entry(self, repo, tag, digest, size):
        self.sizes[digest] = int(size * pre_pull_service.ON_DISK_MULTIPLIER)
        return _entry(repo, tag, digest, size=size)

    def _get(self, ref):
        if ref.rpartition("@")[2] in self.present:
            return MagicMock()
        raise docker.errors.ImageNotFound(ref)

    def _remove(self, ref):
        if "@" in ref and ref.rpartition("@")[2] in self.present:
            self.free += self.present.pop(ref.rpartition("@")[2])

    def _pull(self, client, repo, tag, digest, timeout_seconds):
        self.pulls.append(f"{repo}:{tag}")
        self.present[digest] = self.sizes[digest]
        self.free -= self.sizes[digest]
        return "pull_ok", None


def test_disk_guard_never_evicts_a_served_image_to_pull_another(quiet_node, monkeypatch, caplog):
    # fresh read of 941e5ace: with today's top-N (6.5 GB + 4.0 GB compressed) a node at 222 GiB free
    # fits either image above the 200 GiB floor but not both; evicting one served image to pull the
    # other re-pulled the pair alternately every sweep (8 sweeps -> 8 pulls). The node keeps the
    # first one and logs insufficient_disk for the second instead.
    disk = _Disk(monkeypatch, free_gib=222)
    served = [
        disk.entry(REPO, LIUM1_CU128_TAG, DIGEST_LIUM1_CU128, 6_491_891_139),
        disk.entry(CUDA_REPO, CUDA_TAG, DIGEST_CUDA, 3_961_262_240),
    ]
    puller = PrePuller(disk.client, state_path=None)

    with caplog.at_level(logging.INFO):
        for _ in range(8):
            _sweep(puller, served)

    assert disk.pulls == [f"{REPO}:{LIUM1_CU128_TAG}"]
    disk.client.images.remove.assert_not_called()
    assert set(puller.state.images) == {f"{REPO}:{LIUM1_CU128_TAG}"}
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("pre_pull image=")]
    assert len(lines) == 8 and all("outcome=insufficient_disk" in line for line in lines[1:])


def test_the_watcher_wakes_a_read_blocked_on_a_real_socket(monkeypatch, fast_checks):
    # close() from the watcher would leave this read (and the watcher) blocked until the read
    # timeout; the socket shutdown ends it at once
    monkeypatch.setattr(pre_pull_service, "rental_activity", lambda _: None)
    monkeypatch.setattr(pre_pull_service, "host_activity", lambda: "validator bandwidth test")
    near_end, daemon_end = socket.socketpair()
    near_end.settimeout(3)  # only so a regression fails instead of hanging
    reader = near_end.makefile("rb")
    client, response = _pull_client([])
    client.api._get_raw_response_socket.return_value = reader.raw  # the SocketIO, as docker-py
    client.api._stream_helper.side_effect = lambda *a, **k: iter(lambda: reader.read1(1024), b"")

    started = time.monotonic()
    outcome, detail = _pull_pinned(client, REPO, CU128_TAG, DIGEST_CU128, timeout_seconds=60)

    assert (outcome, detail) == ("preempted", "validator bandwidth test")
    assert time.monotonic() - started < 1.0
    response.close.assert_called_once()
    reader.close()
    near_end.close()
    daemon_end.close()
