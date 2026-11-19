"""DAH-6466: a Volume's s3fs plugin instance is made from the node's local copy, never pulled at rent time.

18 Nov 18:25Z, node 9881eb48: the rent path's per-volume `docker plugin install` failed on its Docker Hub round trip
(connection reset through the host's registry path), and the retried install failed the same way.
The node script runs here for real against a fake `docker` that keeps its plugins in a state file.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.s3fs_plugin_copy import S3FS_PLUGIN_MANIFEST, S3FS_PLUGIN_TAG, s3fs_plugin_command
from services.task.checks import s3fs_plugin_copy as check_module
from services.task.runner import SSHCommandResult

DOCKER_ROOT = "/dockerroot"

FAKE_DOCKER = r'''
import json, os, sys
state_path, root, host_root = os.environ["FAKE_STATE"], os.environ["FAKE_ROOT"], os.environ["FAKE_HOST_ROOT"]
state = json.load(open(state_path)) if os.path.exists(state_path) else {"plugins": {}, "calls": []}
args = sys.argv[1:]
state["calls"].append(args)
def done(code=0, out="", err=""):
    json.dump(state, open(state_path, "w"))
    sys.stdout.write(out); sys.stderr.write(err); sys.exit(code)
if args[:1] == ["info"]:
    done(out=root + "\n")
if args[:2] == ["plugin", "inspect"]:
    name = args[-1]
    if name not in state["plugins"]:
        done(1, err="Error: No such plugin: " + name)
    if "{{.Id}}" in args:
        done(out=state["plugins"][name]["id"] + "\n")
    if "{{json .Config}}" in args:
        done(out='{"Entrypoint":["/usr/bin/tini"]}\n')
    done(out="[{}]")
if args[:2] == ["plugin", "install"]:
    if os.environ.get("FAKE_INSTALL_ERROR"):
        done(1, err=os.environ["FAKE_INSTALL_ERROR"])
    name = args[args.index("--alias") + 1]
    plugin_id = "a" * 64
    plugin_dir = host_root + root + "/plugins/" + plugin_id
    os.makedirs(plugin_dir + "/rootfs/usr/bin", exist_ok=True)
    open(plugin_dir + "/rootfs/usr/bin/tini", "w").write("tini")
    json.dump({"Manifest": os.environ.get("FAKE_MANIFEST", "")}, open(plugin_dir + "/config.json", "w"))
    state["plugins"][name] = {"id": plugin_id}
    done()
if args[:2] == ["plugin", "create"]:
    name, directory = args[2], args[3]
    rootfs = directory + "/rootfs"
    state["plugins"][name] = {
        "id": "b" * 64,
        "marker": open(rootfs + "/.lium-instance").read(),
        "has_files": os.path.exists(rootfs + "/usr/bin/tini"),
        "config": json.load(open(directory + "/config.json")),
    }
    done()
if args[:2] == ["plugin", "rm"]:
    state["plugins"].pop(args[-1], None)
    done()
done(2, err="unexpected " + " ".join(args))
'''


@pytest.fixture
def node(tmp_path, monkeypatch):
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n{FAKE_DOCKER}")
    docker.chmod(0o755)
    host_root = tmp_path / "host"
    (host_root / DOCKER_ROOT.lstrip("/")).mkdir(parents=True)
    env = {
        **os.environ,
        "FAKE_STATE": str(tmp_path / "state.json"),
        "FAKE_ROOT": DOCKER_ROOT,
        "FAKE_HOST_ROOT": str(host_root),
        "FAKE_MANIFEST": S3FS_PLUGIN_MANIFEST,
    }

    def run(alias="", **extra_env):
        command = s3fs_plugin_command(alias, host_root=str(host_root), docker=str(docker))
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, env={**env, **extra_env})
        state = json.loads((tmp_path / "state.json").read_text())
        return result, state

    run.base = host_root / DOCKER_ROOT.lstrip("/") / "lium-s3fs-plugin"
    return run


def _installs(state):
    return [call for call in state["calls"] if call[:2] == ["plugin", "install"]]


def test_a_node_without_the_copy_installs_once_and_every_volume_after_is_made_locally(node):
    first, state = node("s3fs-celium-volume-one")
    second, state = node("s3fs-celium-volume-two")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    # one registry contact for the node, by tag (a digest reference must never reach `plugin install`)
    assert [call[2] for call in _installs(state)] == [S3FS_PLUGIN_TAG]
    plugins = state["plugins"]
    assert set(plugins) == {"s3fs-celium-volume-one", "s3fs-celium-volume-two"}  # the template is gone
    for alias in ("s3fs-celium-volume-one", "s3fs-celium-volume-two"):
        # each instance has the plugin's files plus its own marker, which keeps dockerd from refusing the second
        assert plugins[alias]["has_files"] and plugins[alias]["marker"] == alias + "\n"
        assert plugins[alias]["config"] == {"Entrypoint": ["/usr/bin/tini"]}
    copy = node.base / S3FS_PLUGIN_MANIFEST.split(":")[-1][:12]
    assert (copy / "ready").exists() and not (copy / "rootfs" / ".lium-instance").exists()
    assert not [path for path in node.base.iterdir() if path.name.startswith("instance-")]


def test_validation_stages_the_copy_and_the_rental_then_needs_no_registry(node):
    staged, _ = node()
    rented, state = node("s3fs-celium-volume-one", FAKE_INSTALL_ERROR="connection reset by peer")

    assert staged.returncode == 0 and rented.returncode == 0, rented.stderr
    assert len(_installs(state)) == 1
    assert "s3fs-celium-volume-one" in state["plugins"]


def test_a_registry_failure_on_a_node_without_the_copy_fails_with_dockers_reason(node):
    reset = 'Head "https://registry-1.docker.io/v2/mochoa/s3fs-volume-plugin/manifests/v2.0.9": connection reset by peer'

    result, state = node("s3fs-celium-volume-one", FAKE_INSTALL_ERROR=reset)

    assert result.returncode == 1
    assert "connection reset by peer" in result.stderr
    assert not any(call[:2] == ["plugin", "create"] for call in state["calls"])


def test_a_build_other_than_the_pinned_one_is_refused_and_removed(node):
    result, state = node("s3fs-celium-volume-one", FAKE_MANIFEST="sha256:" + "0" * 64)

    assert result.returncode == 1
    assert "does not match the pinned digest" in result.stderr
    assert state["plugins"] == {}
    assert not list(node.base.glob("*/ready"))


def test_an_instance_that_exists_is_left_alone(node):
    node("s3fs-celium-volume-one")

    result, state = node("s3fs-celium-volume-one")

    assert result.returncode == 0 and "already exists" in result.stdout
    assert sum(call[:2] == ["plugin", "create"] for call in state["calls"]) == 1


def _result(exit_code: int, stderr: str = "") -> SSHCommandResult:
    now = datetime.now(UTC)
    return SSHCommandResult(
        command="", command_id="c", exit_code=exit_code, stdout="", stderr=stderr, duration_ms=1,
        started_at=now, finished_at=now, success=exit_code == 0,
    )


class _Runner:
    def __init__(self, exit_code: int):
        self.exit_code, self.commands = exit_code, []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.commands.append(cmd)
        return _result(self.exit_code, "" if self.exit_code == 0 else "plugin install failed: reset")


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_code", "reason", "wait"), [(0, "S3FS_PLUGIN_COPY_STAGED", check_module.RECHECK_STAGED_SECONDS),
                                                          (1, "S3FS_PLUGIN_COPY_NOT_STAGED", check_module.RETRY_FAILED_SECONDS)])
async def test_the_check_stages_an_idle_node_then_waits_and_never_fails_it(context_factory, exit_code, reason, wait):
    runner = _Runner(exit_code)
    ctx = context_factory(runner=runner)
    check = check_module.S3fsPluginCopyCheck()
    clock = SimpleNamespace(now=1000.0)
    clock.monotonic = lambda: clock.now

    with patch.object(check_module, "time", clock), patch.object(check_module, "rental_of_this_node", lambda ctx: None):
        first = await check.run(ctx)
        clock.now += wait - 1
        await check.run(ctx)
        clock.now += 1
        await check.run(ctx)

    assert first.passed and reason in str(first.event)
    assert runner.commands == [s3fs_plugin_command()] * 2


@pytest.mark.asyncio
async def test_the_check_leaves_a_rented_node_alone(context_factory):
    runner = _Runner(0)
    rented = SimpleNamespace(pods=[object()])

    with patch.object(check_module, "rental_of_this_node", lambda ctx: rented):
        result = await check_module.S3fsPluginCopyCheck().run(context_factory(runner=runner))

    assert result.passed and runner.commands == []


def test_the_check_is_a_completion_step_so_it_never_runs_before_the_verdict():
    from services.task.pipeline import Pipeline
    from services.task.pipeline_factory import PipelineFactory

    pipeline = Pipeline(PipelineFactory.build_checks(), sink=None)
    assert [type(c) for c in pipeline.on_completion].count(check_module.S3fsPluginCopyCheck) == 1
    assert not any(isinstance(c, check_module.S3fsPluginCopyCheck) for c in pipeline.checks)


@pytest.mark.asyncio
async def test_the_check_skips_when_the_task_has_less_than_its_bound_left(context_factory):
    import time

    from core.config import settings
    from services.task.checks.task_time import TASK_TIMEOUT_MARGIN_SECONDS

    runner = _Runner(0)
    needed = check_module.S3FS_PLUGIN_COMMAND_TIMEOUT_SECONDS + check_module.BUDGET_MARGIN_SECONDS
    elapsed = settings.JOB_TIME_OUT - TASK_TIMEOUT_MARGIN_SECONDS - needed + 5
    ctx = context_factory(runner=runner, started_at_monotonic=time.monotonic() - elapsed)
    check = check_module.S3fsPluginCopyCheck()

    with patch.object(check_module, "rental_of_this_node", lambda ctx: None):
        result = await check.run(ctx)
        assert result.passed and "not enough task time left" in str(result.event) and runner.commands == []
        # not a run: a later run with time left stages at once
        assert (await check.run(context_factory(runner=runner))).passed

    assert runner.commands == [s3fs_plugin_command()]
