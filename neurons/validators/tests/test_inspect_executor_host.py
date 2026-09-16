"""scripts/inspect_executor_host.py (DAH-3540): the read-only host inspection walks the validator's key path.

The fakes are the validator's own: ``_FakeSSHClient`` (test_failed_container_diagnostics) records every
command the script runs, ``_connect_returning`` (test_ssh_connect_timing) stands in for ``asyncssh.connect``
underneath ``connect_with_phase_timing`` and captures the connect kwargs.
"""

import asyncio
import importlib.util
import io
import pathlib
import re
import sys

import bittensor
import pytest

import services.ssh_connect_timing as sct
from datura.requests.miner_requests import AcceptSSHKeyRequest, ExecutorSSHInfo
from services.miner_service import MinerService
from test_failed_container_diagnostics import _FakeSSHClient, _SSHRunResult
from test_ssh_connect_timing import _connect_returning

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "inspect_executor_host.py"
_spec = importlib.util.spec_from_file_location("inspect_executor_host", _SCRIPT)
ieh = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ieh  # dataclasses resolve the module's annotations through sys.modules
_spec.loader.exec_module(ieh)

EXECUTOR_ID = "2e2cfbdf-75c2-427c-93b4-f3758d983802"
MINER_HOTKEY = "5DhzqmmkD99d8MFbEEFgtQYSUWyYHRLNJKpVjYQFn5d6hJsv"
HUB_DIGEST = "sha256:" + "f590fed3" * 8
OLD_DIGEST = "sha256:" + "8c07d3a9" * 8

READ_ONLY_PROGRAMS = {"cat", "echo", "head", "test", "hostname"}
READ_ONLY_DOCKER_VERBS = {"info", "images", "ps", "logs", "inspect"}
READ_ONLY_DOCKER_IMAGE_VERBS = {"inspect", "ls", "history"}


def _segments(command: str):
    """Every program invocation in a shell line: each `$(…)` on its own, then split on `||`, `&&`, `;`, `|`."""
    inner: list[str] = []

    def _lift(match):
        inner.append(match.group(1))
        return "SUBST"

    outer = command
    while re.search(r"\$\(([^()]*)\)", outer):  # innermost first, so nested substitutions each get a look
        outer = re.sub(r"\$\(([^()]*)\)", _lift, outer)
    for segment in inner + re.split(r"\|\||&&|;|\|", outer):
        words = segment.split()
        if words:
            yield words


def _executor(uuid=EXECUTOR_ID):
    return ExecutorSSHInfo(
        uuid=uuid, address="203.0.113.10", port=8001, ssh_username="root", ssh_port=2200,
        python_path="/usr/bin/python3", root_dir="/root",
    )


class _FakeConn(_FakeSSHClient):
    """The validator's fake SSH client plus the close pair ``connect_with_phase_timing`` calls on exit."""

    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class _FakeMinerRest:
    """Stands in for ``MinerService._make_rest_request``; answers the submit with the executors given."""

    def __init__(self, executors, submit_status=200):
        self.executors = executors
        self.submit_status = submit_status
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, *, method, url, json_data, headers, timeout, log_extra, operation_name):
        self.calls.append((url, json_data))
        if url.endswith("/ssh-pubkey-submit"):
            if self.submit_status != 200:
                return self.submit_status, {"message_type": "FailedRequest", "details": "no"}
            return 200, AcceptSSHKeyRequest(executors=self.executors).model_dump(mode="json")
        if url.endswith("/ssh-pubkey-remove"):
            return 200, {"message_type": "SSHKeyRemoved"}
        raise AssertionError(f"unexpected miner URL {url}")

    def urls(self):
        return [url.rsplit("/", 1)[-1] for url, _ in self.calls]


@pytest.fixture
def my_key():
    return bittensor.Keypair.create_from_uri("//Alice")


@pytest.fixture
def wired(monkeypatch):
    """A fake miner + fake sshd; returns (conn, captured connect kwargs, rest, inspect coroutine factory)."""

    def _wire(executors=None, submit_status=200, conn=None):
        conn = conn or _FakeConn()
        captured: dict = {}
        monkeypatch.setattr(sct.asyncssh, "connect", _connect_returning(conn, capture=captured))
        rest = _FakeMinerRest(executors if executors is not None else [_executor()], submit_status)
        monkeypatch.setattr(MinerService, "_make_rest_request", rest)
        return conn, captured, rest

    return _wire


async def _resolve(hotkey):
    assert hotkey == MINER_HOTKEY
    return "203.0.113.2", 8091


async def _inspect(my_key, resolve=_resolve, node_timeout=5.0):
    return await ieh.inspect_executor(
        target=ieh.Target(executor_id=EXECUTOR_ID, miner_hotkey=MINER_HOTKEY),
        miner_service=ieh.make_miner_service(),
        my_key=my_key,
        resolve_axon=resolve,
        node_timeout=node_timeout,
    )


def _is_read_only(words: list[str]) -> bool:
    words = [w for w in words if w not in ("2>/dev/null", "2>&1")]  # stderr routing writes no file
    if not words:
        return True
    if ">" in " ".join(words):  # any remaining redirect writes a file
        return False
    if words[0] in READ_ONLY_PROGRAMS:
        return True
    if words[0] == "docker" and len(words) > 1:
        if words[1] == "image":
            return len(words) > 2 and words[2] in READ_ONLY_DOCKER_IMAGE_VERBS
        return words[1] in READ_ONLY_DOCKER_VERBS
    return False


def test_every_command_is_read_only():
    """Regression: a later edit adds `docker compose pull`, `docker update` or a restart to COMMANDS."""
    for label, command in ieh.COMMANDS:
        for words in _segments(command):
            assert _is_read_only(words), f"{label}: {' '.join(words)!r} is not on the read-only allow-list"
    # the allow-list itself refuses what the ticket must never do
    for bad in ("docker compose pull", "docker update --restart=no c", "docker container create x", "docker rm x",
                "docker image prune", "cat a > /etc/docker/daemon.json", "docker ps > /tmp/x", "systemctl restart docker"):
        assert not all(_is_read_only(w) for w in _segments(bad)), bad


@pytest.mark.asyncio
async def test_connects_where_the_miner_said_and_takes_the_key_back(my_key, wired):
    """Regression: the script connects to an address it made up, or leaves the key installed."""
    conn, captured, rest = wired()

    report = await _inspect(my_key)

    assert report.error is None
    assert (captured["host"], captured["port"], captured["username"]) == ("203.0.113.10", 2200, "root")
    assert captured["known_hosts"] is None
    assert len(captured["client_keys"]) == 1
    assert conn.commands == [command for _, command in ieh.COMMANDS]
    assert rest.urls() == ["ssh-pubkey-submit", "ssh-pubkey-remove"]
    submit = rest.calls[0][1]
    assert submit["executor_id"] == EXECUTOR_ID
    assert submit["miner_hotkey"] == MINER_HOTKEY
    assert submit["is_rental_request"] is False
    assert report.key_removed is True
    assert conn.closed is True


@pytest.mark.asyncio
async def test_a_failing_read_keeps_the_other_reads_and_the_key_removal(my_key, wired):
    """Regression: one command raising ends the node with no output and the key still on the miner."""
    conn = _FakeConn()
    conn.errors_by_substring["docker logs"] = RuntimeError("channel closed")
    conn.results_by_substring["daemon.json"] = _SSHRunResult(stdout=ieh.NO_DAEMON_JSON + "\n")
    wired(conn=conn)

    report = await _inspect(my_key)

    assert report.error is None
    assert set(report.outputs) == {label for label, _ in ieh.COMMANDS}
    assert report.outputs["watchtower_log"].stderr == "RuntimeError: channel closed"
    assert report.outputs["daemon_json"].stdout == ieh.NO_DAEMON_JSON
    assert report.key_removed is True


def test_summary_says_unknown_not_yes_for_a_read_that_failed():
    """Regression: an errored `cat daemon.json` (empty stdout) read as `yes`, an errored dockerenv as `host`."""
    report = _report_with({})
    report.outputs["daemon_json"] = ieh.CommandOutput(None, "", "TimeoutError: ")
    report.outputs["dockerenv"] = ieh.CommandOutput(None, "", "TimeoutError: ")
    row = ieh.summarise(report, hub_runner_digest=None)
    assert (row["daemon_json"], row["shell_ran_in"]) == ("?", "?")


def test_main_refuses_a_bad_axon_and_zero_concurrency(capsys):
    with pytest.raises(SystemExit, match="HOTKEY=IP:PORT"):
        ieh.main(["--executor-ids", f"{EXECUTOR_ID} {MINER_HOTKEY}", "--axon", "nope", "--dry-run"])
    with pytest.raises(SystemExit, match="at least 1"):
        ieh.main(["--executor-ids", f"{EXECUTOR_ID} {MINER_HOTKEY}", "--concurrency", "0", "--dry-run"])


@pytest.mark.asyncio
async def test_an_executor_the_miner_did_not_list_is_not_connected(my_key, wired):
    """Regression: the miner answers with another executor and the script ssh-es into it anyway."""
    conn, captured, rest = wired(executors=[_executor(uuid="00000000-0000-0000-0000-000000000000")])

    report = await _inspect(my_key)

    assert "not this id" in report.error
    assert captured == {}
    assert conn.commands == []
    assert rest.urls() == ["ssh-pubkey-submit", "ssh-pubkey-remove"]


@pytest.mark.asyncio
async def test_a_refused_key_submit_sends_no_remove(my_key, wired):
    conn, captured, rest = wired(submit_status=403)

    report = await _inspect(my_key)

    assert report.error.startswith("miner refused the key submit: HTTP 403")
    assert rest.urls() == ["ssh-pubkey-submit"]
    assert report.key_removed is None
    assert captured == {}


@pytest.mark.asyncio
async def test_an_axon_lookup_failure_is_reported_without_a_key_submit(my_key, wired):
    conn, captured, rest = wired()

    async def boom(hotkey):
        raise ValueError(f"Miner with hotkey={hotkey!r} not present in this subnetwork")

    report = await _inspect(my_key, resolve=boom)

    assert report.error.startswith("miner axon lookup failed:")
    assert rest.calls == []


def _report_with(outputs: dict[str, str]) -> "ieh.NodeReport":
    report = ieh.NodeReport(target=ieh.Target(EXECUTOR_ID, MINER_HOTKEY), executor=_executor())
    report.outputs = {label: ieh.CommandOutput(0, text, "") for label, text in outputs.items()}
    return report


def test_summary_reads_mirror_daemon_json_digest_and_where_the_shell_ran():
    """Regression: the table says `none`/`yes`/`current` for a node that has a mirror, no file and the old runner."""
    images = (
        "REPOSITORY   TAG   DIGEST   IMAGE ID   CREATED   SIZE\n"
        f"{ieh.RUNNER_IMAGE}   latest   {OLD_DIGEST}   0123456789ab   4 weeks ago   200MB\n"
    )
    report = _report_with({
        "registry_config": '{"Mirrors":["https://mirror.example.internal"],"IndexConfigs":{}}',
        "daemon_json": ieh.NO_DAEMON_JSON,
        "pulled_runner_images": images,
        "running_runner_digest": f"{ieh.RUNNER_IMAGE}@{OLD_DIGEST}",
        "watchtower_log": "time=... msg=\"Found new image\"\ntime=... msg=\"Unable to update container\"\n",
        "dockerenv": "IN_CONTAINER\nabcdef012345",
    })

    row = ieh.summarise(report, hub_runner_digest=HUB_DIGEST)

    assert row["mirror"] == "https://mirror.example.internal"
    assert row["daemon_json"] == "no"
    assert row["running_digest"] == OLD_DIGEST[:19]
    assert row["pulled_digest"] == OLD_DIGEST[:19]
    assert row["vs_hub"] == "STALE"
    assert row["shell_ran_in"] == "executor container"
    assert row["watchtower_last_line"] == 'time=... msg="Unable to update container"'

    fresh = _report_with({
        "registry_config": '{"Mirrors":[]}',
        "daemon_json": '{"runtimes": {}}',
        "pulled_runner_images": images.replace(OLD_DIGEST, HUB_DIGEST),
        "running_runner_digest": f"{ieh.RUNNER_IMAGE}@{HUB_DIGEST}",
        "dockerenv": "NO_DOCKERENV\nhost-1",
    })
    row = ieh.summarise(fresh, hub_runner_digest=HUB_DIGEST)
    assert (row["mirror"], row["daemon_json"], row["vs_hub"], row["shell_ran_in"]) == ("none", "yes", "current", "host")
    assert row["watchtower_last_line"] == "?"


def test_vs_hub_follows_the_running_image_not_the_newest_pull():
    """Regression: Watchtower pulled the new runner and failed to restart; `docker images` lists the new one first."""
    images = (
        "REPOSITORY TAG DIGEST IMAGE_ID CREATED SIZE\n"
        f"{ieh.RUNNER_IMAGE} latest {HUB_DIGEST} aaaaaaaaaaaa 1 hour ago 200MB\n"
        f"{ieh.RUNNER_IMAGE} <none> {OLD_DIGEST} bbbbbbbbbbbb 4 weeks ago 200MB\n"
    )
    report = _report_with({"pulled_runner_images": images, "running_runner_digest": f"{ieh.RUNNER_IMAGE}@{OLD_DIGEST}"})
    row = ieh.summarise(report, hub_runner_digest=HUB_DIGEST)
    assert (row["pulled_digest"], row["running_digest"], row["vs_hub"]) == (HUB_DIGEST[:19], OLD_DIGEST[:19], "STALE")


def test_summary_marks_a_daemon_json_directory_and_a_missing_runner_container():
    report = _report_with({"daemon_json": ieh.DAEMON_JSON_IS_DIR})
    report.outputs["running_runner_digest"] = ieh.CommandOutput(1, "", "Error: No such object:")
    row = ieh.summarise(report, hub_runner_digest=HUB_DIGEST)
    assert row["daemon_json"] == "DIR"
    assert (row["running_digest"], row["vs_hub"]) == ("?", "?")


def test_summary_without_a_hub_digest_does_not_call_a_node_stale():
    report = _report_with({"running_runner_digest": f"{ieh.RUNNER_IMAGE}@{OLD_DIGEST}"})
    assert ieh.summarise(report, hub_runner_digest=None)["vs_hub"] == "?"


@pytest.mark.asyncio
async def test_a_hung_read_keeps_the_earlier_reads_and_names_the_budget(my_key, wired):
    """Regression: the node budget fires mid-read and every finished read is thrown away."""

    class _HangingConn(_FakeConn):
        async def run(self, command):
            if "docker logs" in command:
                await asyncio.sleep(3600)
            return await super().run(command)

    conn = _HangingConn()
    wired(conn=conn)

    report = await _inspect(my_key, node_timeout=1.0)

    labels = [label for label, _ in ieh.COMMANDS]
    done_before_logs = labels[: labels.index("watchtower_log")]
    assert report.error == f"node budget of 1 s ran out after {len(done_before_logs)} of {len(ieh.COMMANDS)} reads"
    assert list(report.outputs) == done_before_logs
    assert report.key_removed is True


@pytest.mark.asyncio
async def test_axons_are_resolved_once_per_miner_before_the_nodes_run(monkeypatch):
    """Regression: every node task ran its own metagraph lookup (four miners, 43 lookups)."""
    calls = []

    async def fake_lookup(hotkey):
        calls.append(hotkey)
        if hotkey == "hk-gone":
            raise ValueError("not present in this subnetwork")
        return f"203.0.113.{len(calls)}", 8091

    monkeypatch.setattr(ieh, "resolve_axon_from_metagraph", fake_lookup)

    resolve = await ieh.resolve_axons_once({"hk-a", "hk-gone", MINER_HOTKEY}, {MINER_HOTKEY: ("203.0.113.9", 1)})

    assert sorted(calls) == ["hk-a", "hk-gone"]
    assert await resolve("hk-a") == ("203.0.113.1", 8091)
    assert await resolve(MINER_HOTKEY) == ("203.0.113.9", 1)
    with pytest.raises(ValueError, match="not present"):
        await resolve("hk-gone")
    assert sorted(calls) == ["hk-a", "hk-gone"]


def test_node_block_shows_each_command_its_output_and_the_key_state():
    report = _report_with({"registry_config": '{"Mirrors":[]}'})
    report.key_removed = True
    block = ieh.render_node_block(report)
    assert block.startswith(f"### {EXECUTOR_ID} (miner {MINER_HOTKEY}) — 203.0.113.10:2200 as root")
    assert "$ docker info --format '{{json .RegistryConfig}}'" in block
    assert '{"Mirrors":[]}' in block
    assert "key removed from miner: True" in block


def test_parse_targets_takes_file_lines_inline_ids_and_comments():
    text = f"# stale nodes\n{EXECUTOR_ID} {MINER_HOTKEY}\n\nabc def  # trailing\n"
    assert ieh.parse_targets(text, None) == [
        ieh.Target(EXECUTOR_ID, MINER_HOTKEY),
        ieh.Target("abc", "def"),
    ]
    assert ieh.parse_targets("a,b", "hk") == [ieh.Target("a", "hk"), ieh.Target("b", "hk")]
    with pytest.raises(SystemExit, match="no miner hotkey"):
        ieh.parse_targets("a\n", None)
    with pytest.raises(SystemExit, match="no executor ids"):
        ieh.parse_targets("# only a comment\n", "hk")


def test_axon_override_parses_hotkey_ip_port():
    assert ieh.parse_axon_overrides([f"{MINER_HOTKEY}=203.0.113.2:8091"]) == {MINER_HOTKEY: ("203.0.113.2", 8091)}
    with pytest.raises(SystemExit, match="HOTKEY=IP:PORT"):
        ieh.parse_axon_overrides(["nope"])


def test_dry_run_prints_the_plan_and_opens_nothing(monkeypatch, capsys):
    """Regression: --dry-run still resolves the miner or submits a key."""

    def refuse(*a, **k):
        raise AssertionError("dry run reached the network")

    monkeypatch.setattr(sct.asyncssh, "connect", refuse)
    monkeypatch.setattr(MinerService, "_make_rest_request", refuse)
    monkeypatch.setattr(ieh, "resolve_axon_from_metagraph", refuse)
    monkeypatch.setattr(ieh, "fetch_hub_runner_digest", refuse)

    rc = ieh.main(["--executor-ids", f"{EXECUTOR_ID} {MINER_HOTKEY}", "--dry-run", "--concurrency", "2"])

    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("DRY RUN — 1 executor(s), concurrency 2, 60 s per node. Nothing opened.")
    assert f"- {EXECUTOR_ID}  miner {MINER_HOTKEY}" in out
    for _, command in ieh.COMMANDS:
        assert f"  $ {command}" in out


@pytest.mark.asyncio
async def test_run_prints_a_block_per_node_then_the_table(my_key, wired):
    conn = _FakeConn()
    conn.results_by_substring["docker info"] = _SSHRunResult(stdout='{"Mirrors":["https://m.example.internal"]}')
    wired(conn=conn)
    out = io.StringIO()

    reports = await ieh.run(
        [ieh.Target(EXECUTOR_ID, MINER_HOTKEY)],
        concurrency=4, node_timeout=5.0, resolve_axon=_resolve,
        miner_service=ieh.make_miner_service(), my_key=my_key, hub_runner_digest=HUB_DIGEST, out=out,
    )

    text = out.getvalue()
    assert len(reports) == 1 and reports[0].error is None
    assert text.index(f"### {EXECUTOR_ID}") < text.index("| node | mirror |")
    assert f"Docker Hub {ieh.RUNNER_IMAGE}:latest digest now: {HUB_DIGEST}" in text
    assert "| https://m.example.internal |" in text
