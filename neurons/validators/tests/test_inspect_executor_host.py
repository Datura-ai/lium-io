"""scripts/inspect_executor_host.py (DAH-3540): the read-only host inspection walks the validator's key path.

The fakes are the validator's own: ``_FakeSSHClient`` (test_failed_container_diagnostics) records every
command the script runs, ``_connect_returning`` (test_ssh_connect_timing) stands in for ``asyncssh.connect``
underneath ``connect_with_phase_timing`` and captures the connect kwargs.
"""

import asyncio
import importlib.util
import io
import json
import pathlib
import re
import subprocess
import sys

import asyncssh
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
EXECUTOR_HUB_DIGEST = "sha256:" + "1a2b3c4d" * 8
EXECUTOR_OLD_DIGEST = "sha256:" + "9e8f7a6b" * 8
HUB = ieh.HubDigests(executor=EXECUTOR_HUB_DIGEST, runner=HUB_DIGEST)

READ_ONLY_PROGRAMS = {"cat", "echo", "head", "test", "hostname", "printf"}
READ_ONLY_DOCKER_VERBS = {"info", "images", "ps", "logs", "inspect"}
READ_ONLY_DOCKER_IMAGE_VERBS = {"inspect", "ls", "history"}
_FD_DUP = re.compile(r"^\d*>&(\d+|-)$")  # 2>&1, 1>&3, 3>&- : fd plumbing, no file written


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
    """The validator's fake SSH client plus the close pair ``connect_with_phase_timing`` calls on exit.

    ``run`` answers like a node that ran the ``capped()`` wrapper on a small output: the configured
    result's exit status rides on the last stdout line and the pipeline's own status is 0. A test
    that wants the raw stream (a stdout ``head`` cut, so no marker) sets ``remote_marker = False``.
    """

    def __init__(self):
        super().__init__()
        self.closed = False
        self.remote_marker = True
        self.run_kwargs: list[dict] = []

    async def run(self, command, **kwargs):
        self.run_kwargs.append(kwargs)
        result = await super().run(command)
        if not self.remote_marker:
            return result
        marker = f"\n{ieh.RC_MARKER}{result.exit_status}\n"
        return _SSHRunResult(exit_status=0, stdout=(result.stdout or "") + marker, stderr=result.stderr)

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


_UNSET = object()  # `remove_body` not given: the remove answers SSHKeyRemoved (None is a real body, a JSON null)


class _FakeMinerRest:
    """Stands in for ``MinerService._make_rest_request``; answers the submit with the executors given."""

    def __init__(self, executors, submit_status=200, remove_status=200, submit_times_out=False, remove_body=_UNSET):
        self.executors = executors
        self.submit_status = submit_status
        self.remove_status = remove_status
        self.submit_times_out = submit_times_out  # `_make_rest_request` re-raises asyncio.TimeoutError
        self.remove_body = remove_body  # the 200 body of the remove; _UNSET = SSHKeyRemoved
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, *, method, url, json_data, headers, timeout, log_extra, operation_name):
        self.calls.append((url, json_data))
        if url.endswith("/ssh-pubkey-submit"):
            if self.submit_times_out:
                raise asyncio.TimeoutError()
            if self.submit_status != 200:
                return self.submit_status, {"message_type": "FailedRequest", "details": "no"}
            return 200, AcceptSSHKeyRequest(executors=self.executors).model_dump(mode="json")
        if url.endswith("/ssh-pubkey-remove"):
            if self.remove_status != 200:
                return self.remove_status, {"message_type": "FailedRequest", "details": "no"}
            return 200, {"message_type": "SSHKeyRemoved"} if self.remove_body is _UNSET else self.remove_body
        raise AssertionError(f"unexpected miner URL {url}")

    def urls(self):
        return [url.rsplit("/", 1)[-1] for url, _ in self.calls]


@pytest.fixture
def my_key():
    return bittensor.Keypair.create_from_uri("//Alice")


@pytest.fixture
def wired(monkeypatch):
    """A fake miner + fake sshd; returns (conn, captured connect kwargs, rest, inspect coroutine factory)."""

    def _wire(executors=None, submit_status=200, conn=None, remove_status=200, submit_times_out=False, remove_body=_UNSET):
        conn = conn or _FakeConn()
        captured: dict = {}
        monkeypatch.setattr(sct.asyncssh, "connect", _connect_returning(conn, capture=captured))
        rest = _FakeMinerRest(
            executors if executors is not None else [_executor()],
            submit_status,
            remove_status,
            submit_times_out,
            remove_body,
        )
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
    # stderr routing, fd-to-fd plumbing and the wrapper's group braces write no file
    words = [w for w in words if w != "2>/dev/null" and w not in ("{", "}") and not _FD_DUP.match(w)]
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
        for words in _segments(ieh.capped(command)):  # the wrapped form is what the node runs
            assert _is_read_only(words), f"{label}: {' '.join(words)!r} is not on the read-only allow-list"
    # the allow-list itself refuses what the ticket must never do, wrapped or not
    for bad in ("docker compose pull", "docker update --restart=no c", "docker container create x", "docker rm x",
                "docker image prune", "cat a > /etc/docker/daemon.json", "docker ps > /tmp/x", "systemctl restart docker"):
        assert not all(_is_read_only(w) for w in _segments(bad)), bad
        assert not all(_is_read_only(w) for w in _segments(ieh.capped(bad))), bad


@pytest.mark.parametrize(
    "command, want_stdout, want_stderr, want_rc",
    [
        # a small output: both streams whole, the exit status on the marker line
        ("printf out; printf err >&2; (exit 3)", b"out", b"err", 3),
        # 200 kB on stdout: the node sends exactly the cap and the marker never arrives
        ("head -c 200000 /dev/zero | tr '\\0' a; echo err >&2", b"a" * ieh.OUTPUT_CAP, b"err\n", None),
        # 200 kB on stderr: stderr is cut, stdout and its exit status are whole
        ("head -c 200000 /dev/zero | tr '\\0' b >&2; echo ok", b"ok\n", b"b" * ieh.OUTPUT_CAP, 0),
        # a `docker info` style warning on stderr stays out of the JSON on stdout
        ("echo '{\"Mirrors\":[]}'; echo 'WARNING: No swap limit support' >&2", b'{"Mirrors":[]}\n', b"WARNING: No swap limit support\n", 0),
    ],
)
def test_capped_wrapper_cuts_each_stream_on_the_node_and_keeps_the_exit_status(command, want_stdout, want_stderr, want_rc):
    """Regression: `ssh_client.run()` buffered a 30 MB daemon.json or log line whole (Rustam, #1385)."""
    done = subprocess.run(["sh", "-c", ieh.capped(command)], capture_output=True, timeout=30)
    stdout, rc = ieh.split_exit_status(done.stdout.decode())
    assert (stdout.encode(), done.stderr, rc) == (want_stdout, want_stderr, want_rc)
    assert len(done.stdout) <= ieh.OUTPUT_CAP and len(done.stderr) <= ieh.OUTPUT_CAP


@pytest.mark.asyncio
async def test_a_stream_at_the_cap_is_cut_and_marked_on_the_client_too(my_key, wired):
    """Regression: a node whose shell ignored the wrapper (no `head`, no marker) gets its whole stream printed, and read as `yes`."""
    conn = _FakeConn()
    conn.remote_marker = False  # raw streams: a stdout `head` cut has no marker; nothing capped the 3× stderr
    conn.results_by_substring["daemon.json"] = _SSHRunResult(stdout="x" * ieh.OUTPUT_CAP, stderr="e" * (3 * ieh.OUTPUT_CAP))
    conn.results_by_substring["docker ps --format"] = _SSHRunResult(stdout="executor-executor-1 daturaai/compute-subnet-executor@sha256:…")
    conn.results_by_substring["docker info"] = _SSHRunResult(stdout="null")  # the daemon behind the socket did not answer
    wired(conn=conn)

    report = await _inspect(my_key)

    daemon = report.outputs["daemon_json"]
    assert daemon.exit_status is None
    assert daemon.stdout.startswith("x" * 100) and daemon.stdout.endswith(ieh.TRUNCATED_NOTE)
    assert daemon.stderr == "e" * ieh.OUTPUT_CAP + "\n" + ieh.TRUNCATED_NOTE
    assert len(daemon.stdout) <= ieh.OUTPUT_CAP + len(ieh.TRUNCATED_NOTE) + 1
    running = report.outputs["running"]  # short stdout, no marker: the shell stopped early, the note says so
    assert running.exit_status is None
    assert running.stdout == "executor-executor-1 daturaai/compute-subnet-executor@sha256:…\n[no exit status: stdout cut at 65536 bytes or the shell stopped early]"
    row = ieh.summarise(report, HUB)
    assert (row["daemon_json"], row["mirror"]) == ("?", "unparsable")  # a cut read is not "yes"; a `null` registry config is not a crash
    assert all(kwargs == {"errors": "replace"} for kwargs in conn.run_kwargs)  # a split multibyte char is not a failed read


@pytest.mark.asyncio
async def test_connects_where_the_miner_said_and_takes_the_key_back(my_key, wired):
    """Regression: the script connects to an address it made up, or leaves the key installed."""
    conn, captured, rest = wired()

    report = await _inspect(my_key)

    assert report.error is None
    assert (captured["host"], captured["port"], captured["username"]) == ("203.0.113.10", 2200, "root")
    assert captured["known_hosts"] is None
    assert len(captured["client_keys"]) == 1
    assert conn.commands == [ieh.capped(command) for _, command in ieh.COMMANDS]  # never a bare, uncapped read
    assert all(output.exit_status == 0 for output in report.outputs.values())  # the marker's status, not head's
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
    report.outputs["registry_config"] = ieh.CommandOutput(0, "null", "")  # the daemon behind the socket did not answer
    row = ieh.summarise(report, ieh.HubDigests())
    assert (row["daemon_json"], row["shell_ran_in"], row["mirror"]) == ("?", "?", "unparsable")


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
async def test_a_refused_key_submit_still_takes_the_key_back_and_connects_nowhere(my_key, wired):
    """The remove is idempotent; a refused submit costs one extra request and never an SSH session."""
    conn, captured, rest = wired(submit_status=403)

    report = await _inspect(my_key)

    assert report.error.startswith("miner refused the key submit: HTTP 403")
    assert rest.urls() == ["ssh-pubkey-submit", "ssh-pubkey-remove"]
    assert report.key_removed is True
    assert captured == {}


@pytest.mark.asyncio
async def test_a_submit_that_times_out_is_still_followed_by_a_remove(my_key, wired):
    """Regression: the submit timed out after the miner had already pushed the key, and no remove was sent."""
    conn, captured, rest = wired(submit_times_out=True)

    report = await _inspect(my_key)

    assert report.error.startswith("TimeoutError")
    assert rest.urls() == ["ssh-pubkey-submit", "ssh-pubkey-remove"]
    assert report.key_removed is True
    assert captured == {}


@pytest.mark.asyncio
async def test_a_remove_the_miner_did_not_accept_is_a_node_error_and_exit_1(my_key, wired, monkeypatch):
    """Regression: `_remove_ssh_key_via_rest` returned False, `report.error` stayed empty and the run exited 0 with a key still installed."""
    monkeypatch.setattr(ieh, "KEY_REMOVE_RETRY_DELAY", 0.0)
    conn, captured, rest = wired(remove_status=500)

    report = await _inspect(my_key)

    assert report.key_removed is False
    assert report.error == "key remove not accepted by miner after 3 attempts: the key is still installed, remove it by hand (public key in the node block)"
    assert rest.urls() == ["ssh-pubkey-submit"] + ["ssh-pubkey-remove"] * ieh.KEY_REMOVE_ATTEMPTS
    assert ieh.exit_code([report]) == 1
    # an earlier error is kept, the remove note is appended after it
    conn, captured, rest = wired(submit_status=403, remove_status=500)
    refused = await _inspect(my_key)
    # a refused submit never installed the key for sure: the note says "may", not "is"
    assert refused.error == (
        "miner refused the key submit: HTTP 403 {'message_type': 'FailedRequest', 'details': 'no'}; "
        "key remove not accepted by miner after 3 attempts: the key may still be installed, remove it by hand (public key in the node block)"
    )
    assert "POSSIBLY LEFT ON THE EXECUTOR" in ieh.render_node_block(refused)
    clean = ieh.NodeReport(target=report.target, key_removed=True)
    assert ieh.exit_code([clean]) == 0
    never_submitted = ieh.NodeReport(target=report.target, error="miner axon lookup failed: x")  # key_removed None
    assert ieh.exit_code([never_submitted]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"message_type": "FailedRequest", "details": "deregister_pubkey: executor not found"},  # the route's except branch
        {},  # no message_type at all
        None,  # a JSON `null` body: `response.json()` gives None
    ],
)
async def test_a_200_whose_body_is_not_ssh_key_removed_is_not_a_removal(my_key, wired, body, monkeypatch):
    """Regression (taiberium, #1385): `/api/validator/ssh-pubkey-remove` answers HTTP 200 with a
    `FailedRequest` body when `deregister_pubkey` raised, and `_remove_ssh_key_via_rest` returned
    True on the status alone, so the node exited 0 with the key still installed."""
    monkeypatch.setattr(ieh, "KEY_REMOVE_RETRY_DELAY", 0.0)
    conn, captured, rest = wired(remove_body=body)

    report = await _inspect(my_key)

    assert report.key_removed is False
    assert report.error == "key remove not accepted by miner after 3 attempts: the key is still installed, remove it by hand (public key in the node block)"
    assert rest.urls() == ["ssh-pubkey-submit"] + ["ssh-pubkey-remove"] * ieh.KEY_REMOVE_ATTEMPTS
    assert ieh.exit_code([report]) == 1


@pytest.mark.asyncio
async def test_a_session_that_drops_after_login_is_a_failed_node_and_exit_1(my_key, wired):
    """Regression (taiberium, 17 Sep): sshd accepted the key and closed the channel; every read raised
    `ConnectionLost`, `report.error` stayed None and the run exited 0 with an empty row. Fails on the
    old code at the `error` assertion."""
    conn = _FakeConn()
    for _, command in ieh.COMMANDS:
        conn.errors_by_substring[command[:20]] = asyncssh.ConnectionLost("Connection lost")
    wired(conn=conn)

    report = await _inspect(my_key)

    assert len(report.outputs) == len(ieh.COMMANDS)
    assert all(out.exit_status is None and out.stderr.startswith("ConnectionLost") for out in report.outputs.values())
    assert report.error is not None and report.error.startswith(ieh.NO_READ_COMPLETED)
    assert "ConnectionLost" in report.error
    assert report.key_removed is True  # the key is still taken back
    assert ieh.exit_code([report]) == 1
    assert ieh.summarise(report, ieh.HubDigests())["error"].startswith(ieh.NO_READ_COMPLETED)
    # one completed read is enough to judge the node: no failure line for it
    conn = _FakeConn()
    for _, command in list(ieh.COMMANDS)[1:]:
        conn.errors_by_substring[command[:20]] = asyncssh.ConnectionLost("Connection lost")
    wired(conn=conn)
    partial = await _inspect(my_key)
    assert partial.error is None and ieh.exit_code([partial]) == 0
    assert [label for label, out in partial.outputs.items() if out.exit_status is not None] == ["registry_config"]


@pytest.mark.asyncio
async def test_the_key_remove_is_retried_with_the_same_public_key_and_printed_when_it_never_succeeds(
    my_key, wired, monkeypatch
):
    """Regression (taiberium, 17 Sep): one failed remove left the key on the executor and the note said
    "re-run this node" — a rerun mints a NEW key and cannot remove the first one. The remove is retried
    with the key this run submitted; when every attempt fails, the block prints that key for manual
    removal and the advice to rerun is gone."""
    monkeypatch.setattr(ieh, "KEY_REMOVE_RETRY_DELAY", 0.0)

    # two refusals, then the miner accepts: the retries carry the same public key, the node is clean
    class _FlakyRest(_FakeMinerRest):
        async def __call__(self, *, url, **kwargs):
            removes_so_far = sum(1 for u, _ in self.calls if u.endswith("/ssh-pubkey-remove"))
            self.remove_status = 500 if url.endswith("/ssh-pubkey-remove") and removes_so_far < 2 else 200
            return await super().__call__(url=url, **kwargs)

    wired()
    flaky = _FlakyRest([_executor()])
    monkeypatch.setattr(MinerService, "_make_rest_request", flaky)
    report = await _inspect(my_key)
    removes = [json_data for url, json_data in flaky.calls if url.endswith("/ssh-pubkey-remove")]
    submit = next(json_data for url, json_data in flaky.calls if url.endswith("/ssh-pubkey-submit"))
    assert len(removes) == 3
    assert {r["public_key"] for r in removes} == {submit["public_key"]}
    assert report.key_removed is True and report.error is None and report.public_key_left is None
    assert "PUBLIC KEY LEFT" not in ieh.render_node_block(report)

    # every attempt refused: the exact public key is printed, no advice to rerun
    conn, captured, rest = wired(remove_status=500)
    report = await _inspect(my_key)
    submit = next(json_data for url, json_data in rest.calls if url.endswith("/ssh-pubkey-submit"))
    assert rest.urls().count("ssh-pubkey-remove") == ieh.KEY_REMOVE_ATTEMPTS
    assert report.key_removed is False
    assert report.public_key_left == submit["public_key"].strip()
    block = ieh.render_node_block(report)
    assert "PUBLIC KEY LEFT ON THE EXECUTOR" in block and report.public_key_left in block
    assert "re-run" not in block and "rerun this node" not in block.lower()
    assert ieh.exit_code([report]) == 1


def test_runner_log_is_read_next_to_watchtowers():
    """Regression (taiberium, #1385): only Watchtower's log was read, and a `docker compose up --wait`
    that never came up healthy is in the runner's entrypoint output, not Watchtower's."""
    commands = dict(ieh.COMMANDS)
    assert commands["runner_log"].startswith(f"docker logs {ieh.RUNNER_CONTAINER} --tail 50")
    # the fallback finds the runner by service name when the compose project is not `executor`
    assert "docker ps -qf name=executor-runner" in commands["runner_log"]
    assert ieh.RUNNER_CONTAINER == "executor-executor-runner-1" and ieh.WATCHTOWER_CONTAINER == "executor-watchtower-1"
    labels = [label for label, _ in ieh.COMMANDS]
    assert labels.index("runner_log") == labels.index("watchtower_log") + 1


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


def _running(*containers: tuple[str, str, list[str] | None]) -> dict[str, str]:
    """The two reads for running containers: ``(name, image id, RepoDigests)`` per container, as docker prints them."""
    names = "\n".join(f"/{name} {image_id}" for name, image_id, _ in containers)
    digests = "\n".join(f"{image_id} {json.dumps(repo_digests)}" for _, image_id, repo_digests in containers)
    return {"running_containers": names, "running_image_digests": digests}


STANDARD_STACK = _running(
    ("executor-executor-1", "sha256:" + "e" * 64, [f"{ieh.EXECUTOR_IMAGE}@{EXECUTOR_OLD_DIGEST}"]),
    ("executor-monitor-1", "sha256:" + "e" * 64, [f"{ieh.EXECUTOR_IMAGE}@{EXECUTOR_OLD_DIGEST}"]),
    ("executor-executor-runner-1", "sha256:" + "a" * 64, [f"{ieh.RUNNER_IMAGE}@{OLD_DIGEST}"]),
    ("executor-watchtower-1", "sha256:" + "b" * 64, ["containrrr/watchtower@sha256:" + "c" * 64]),
    ("executor-autoheal-1", "sha256:" + "d" * 64, []),  # a locally built image has no RepoDigests
    ("executor-nginx-1", "sha256:" + "f" * 64, None),  # some engines print `null` instead of `[]`
)


def test_summary_reads_mirror_daemon_json_digests_and_where_the_shell_ran():
    """Regression: the table says `none`/`yes`/`current` for a node that has a mirror, no file and the old images."""
    images = (
        "REPOSITORY   TAG   DIGEST   IMAGE ID   CREATED   SIZE\n"
        f"{ieh.RUNNER_IMAGE}   latest   {OLD_DIGEST}   0123456789ab   4 weeks ago   200MB\n"
    )
    report = _report_with({
        "registry_config": '{"Mirrors":["https://mirror.example.internal"],"IndexConfigs":{}}',
        "daemon_json": ieh.NO_DAEMON_JSON,
        "pulled_runner_images": images,
        **STANDARD_STACK,
        "watchtower_log": "time=... msg=\"Found new image\"\ntime=... msg=\"Unable to update container\"\n",
        "dockerenv": "IN_CONTAINER\nabcdef012345",
    })

    row = ieh.summarise(report, HUB)

    assert row["mirror"] == "https://mirror.example.internal"
    assert row["daemon_json"] == "no"
    assert row["executor_running"] == EXECUTOR_OLD_DIGEST[:19] + " ×2"  # executor + monitor run the same image
    assert row["executor_vs_hub"] == "STALE"
    assert row["runner_running"] == OLD_DIGEST[:19]
    assert row["runner_pulled"] == OLD_DIGEST[:19]
    assert row["runner_vs_hub"] == "STALE"
    assert row["shell_ran_in"] == "executor container"
    assert row["watchtower_last_line"] == 'time=... msg="Unable to update container"'

    fresh = _report_with({
        "registry_config": '{"Mirrors":[]}',
        "daemon_json": '{"runtimes": {}}',
        "pulled_runner_images": images.replace(OLD_DIGEST, HUB_DIGEST),
        **{k: v.replace(EXECUTOR_OLD_DIGEST, EXECUTOR_HUB_DIGEST).replace(OLD_DIGEST, HUB_DIGEST) for k, v in STANDARD_STACK.items()},
        "dockerenv": "NO_DOCKERENV\nhost-1",
    })
    row = ieh.summarise(fresh, HUB)
    assert (row["mirror"], row["daemon_json"], row["shell_ran_in"]) == ("none", "yes", "host")
    assert (row["executor_vs_hub"], row["runner_vs_hub"]) == ("current", "current")
    assert row["watchtower_last_line"] == "?"


def test_executor_vs_hub_compares_the_executor_image_the_validator_scores_not_the_runner():
    """Regression: `vs_hub` read `current` on a node whose runner was current while its executor was the old image (Rustam, #1385)."""
    runner_current_executor_old = _running(
        ("executor-executor-1", "sha256:" + "e" * 64, [f"{ieh.EXECUTOR_IMAGE}@{EXECUTOR_OLD_DIGEST}"]),
        ("executor-executor-runner-1", "sha256:" + "a" * 64, [f"{ieh.RUNNER_IMAGE}@{HUB_DIGEST}"]),
    )
    row = ieh.summarise(_report_with(runner_current_executor_old), HUB)
    assert (row["executor_vs_hub"], row["runner_vs_hub"]) == ("STALE", "current")
    assert row["executor_running"] == EXECUTOR_OLD_DIGEST[:19]


@pytest.mark.asyncio
async def test_hub_digests_come_from_the_validators_own_executor_lookup(monkeypatch):
    """Regression: the run fetched the runner tag only and compared nothing against EXECUTOR_IMAGE_REF's digest."""
    asked = []

    async def fake_executor_digest():
        asked.append("executor")
        return EXECUTOR_HUB_DIGEST

    async def fake_registry_digest(session, ref):
        asked.append(ref)
        return HUB_DIGEST

    monkeypatch.setattr(ieh, "fetch_executor_image_digest", fake_executor_digest)
    monkeypatch.setattr(ieh, "fetch_registry_digest", fake_registry_digest)

    assert await ieh.fetch_hub_digests() == HUB
    assert asked == ["executor", f"{ieh.RUNNER_IMAGE}:latest"]


def test_runner_vs_hub_follows_the_running_image_not_the_newest_pull():
    """Regression: Watchtower pulled the new runner and failed to restart; `docker images` lists the new one first."""
    images = (
        "REPOSITORY TAG DIGEST IMAGE_ID CREATED SIZE\n"
        f"{ieh.RUNNER_IMAGE} latest {HUB_DIGEST} aaaaaaaaaaaa 1 hour ago 200MB\n"
        f"{ieh.RUNNER_IMAGE} <none> {OLD_DIGEST} bbbbbbbbbbbb 4 weeks ago 200MB\n"
    )
    report = _report_with({"pulled_runner_images": images, **STANDARD_STACK})
    row = ieh.summarise(report, HUB)
    assert (row["runner_pulled"], row["runner_running"], row["runner_vs_hub"]) == (HUB_DIGEST[:19], OLD_DIGEST[:19], "STALE")


def test_summary_reports_zero_and_several_runner_containers_instead_of_picking_one():
    """Regression: `docker ps -qf name=executor-runner` matched two containers and the table showed the last digest (Rustam, #1385)."""
    no_runner = _running(("executor-executor-1", "sha256:" + "e" * 64, [f"{ieh.EXECUTOR_IMAGE}@{EXECUTOR_HUB_DIGEST}"]))
    row = ieh.summarise(_report_with(no_runner), HUB)
    assert (row["runner_running"], row["runner_vs_hub"]) == ("none", "none running")
    assert (row["executor_running"], row["executor_vs_hub"]) == (EXECUTOR_HUB_DIGEST[:19], "current")

    two_runners = _running(
        ("executor-executor-runner-1", "sha256:" + "a" * 64, [f"{ieh.RUNNER_IMAGE}@{OLD_DIGEST}"]),
        ("old-executor-runner-1", "sha256:" + "f" * 64, [f"{ieh.RUNNER_IMAGE}@{HUB_DIGEST}"]),
    )
    row = ieh.summarise(_report_with(two_runners), HUB)
    assert row["runner_running"] == (
        f"2 images in 2 containers: executor-executor-runner-1={OLD_DIGEST[:19]}, old-executor-runner-1={HUB_DIGEST[:19]}"
    )
    assert row["runner_vs_hub"] == "MIXED"
    assert (row["executor_running"], row["executor_vs_hub"]) == ("none", "none running")

    # one image pulled by tag and by digest carries two RepoDigests of the repo: one container, both shown, either counts
    two_digests_one_image = _running(
        ("executor-executor-1", "sha256:" + "e" * 64, [f"{ieh.EXECUTOR_IMAGE}@{EXECUTOR_HUB_DIGEST}", f"{ieh.EXECUTOR_IMAGE}@{EXECUTOR_OLD_DIGEST}"]),
    )
    row = ieh.summarise(_report_with(two_digests_one_image), HUB)
    assert row["executor_running"] == f"{EXECUTOR_HUB_DIGEST[:19]}+{EXECUTOR_OLD_DIGEST[:19]}"
    assert row["executor_vs_hub"] == "current"


def test_summary_marks_a_daemon_json_directory_and_a_failed_container_read():
    report = _report_with({"daemon_json": ieh.DAEMON_JSON_IS_DIR, **STANDARD_STACK})
    report.outputs["running_image_digests"] = ieh.CommandOutput(1, "", "Error: No such object:")
    row = ieh.summarise(report, HUB)
    assert row["daemon_json"] == "DIR"
    assert (row["executor_running"], row["executor_vs_hub"], row["runner_running"], row["runner_vs_hub"]) == ("?", "?", "?", "?")

    # a container that started between the two `docker ps -q` has no digest line: "?", never "none"
    skewed = _report_with(STANDARD_STACK)
    skewed.outputs["running_containers"].stdout += "\n/executor-executor-runner-2 sha256:" + "9" * 64
    row = ieh.summarise(skewed, HUB)
    assert (row["runner_running"], row["runner_vs_hub"]) == ("?", "?")


def test_split_exit_status_refuses_a_marker_the_cap_cut_in_half():
    """Regression: `__inspect_rc=12` left of a cut `127` read as exit 12 with no truncation note."""
    assert ieh.split_exit_status("out\n__inspect_rc=127\n") == ("out", 127)
    assert ieh.split_exit_status("out\n__inspect_rc=12") == ("out\n__inspect_rc=12", None)
    assert ieh.split_exit_status("no marker") == ("no marker", None)


def test_summary_without_a_hub_digest_does_not_call_a_node_stale():
    row = ieh.summarise(_report_with(STANDARD_STACK), ieh.HubDigests())
    assert (row["executor_vs_hub"], row["runner_vs_hub"]) == ("?", "?")
    assert row["executor_running"] == EXECUTOR_OLD_DIGEST[:19] + " ×2"


@pytest.mark.asyncio
async def test_a_hung_read_keeps_the_earlier_reads_and_names_the_budget(my_key, wired):
    """Regression: the node budget fires mid-read and every finished read is thrown away."""

    class _HangingConn(_FakeConn):
        async def run(self, command, **kwargs):
            if "docker logs" in command:
                await asyncio.sleep(3600)
            return await super().run(command, **kwargs)

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
    # the miner's 200 means it accepted the request; the block must not claim the executor confirmed the removal
    assert "key remove request accepted by miner (not confirmed on executor): True" in block
    assert "key removed from miner" not in block


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
    monkeypatch.setattr(ieh, "fetch_hub_digests", refuse)

    rc = ieh.main(["--executor-ids", f"{EXECUTOR_ID} {MINER_HOTKEY}", "--dry-run", "--concurrency", "2"])

    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("DRY RUN — 1 executor(s), concurrency 2, 60 s per node. Nothing opened.")
    assert f"- {EXECUTOR_ID}  miner {MINER_HOTKEY}" in out
    for _, command in ieh.COMMANDS:
        assert f"  $ {command}" in out
    assert f"  $ {ieh.capped(ieh.COMMANDS[0][1])}" in out  # the plan shows the wrapped form that will run


@pytest.mark.asyncio
async def test_run_prints_a_block_per_node_then_the_table(my_key, wired):
    conn = _FakeConn()
    conn.results_by_substring["docker info"] = _SSHRunResult(stdout='{"Mirrors":["https://m.example.internal"]}')
    wired(conn=conn)
    out = io.StringIO()

    reports = await ieh.run(
        [ieh.Target(EXECUTOR_ID, MINER_HOTKEY)],
        concurrency=4, node_timeout=5.0, resolve_axon=_resolve,
        miner_service=ieh.make_miner_service(), my_key=my_key, hub=HUB, out=out,
    )

    text = out.getvalue()
    assert len(reports) == 1 and reports[0].error is None
    assert text.index(f"### {EXECUTOR_ID}") < text.index("| node | mirror |")
    assert f"fetch_executor_image_digest, what the validator scores against): {EXECUTOR_HUB_DIGEST}" in text
    assert f"Docker Hub {ieh.RUNNER_IMAGE}:latest digest now: {HUB_DIGEST}" in text
    assert "| executor_running | executor_vs_hub | runner_running | runner_pulled | runner_vs_hub |" in text
    assert "| https://m.example.internal |" in text
