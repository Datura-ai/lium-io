"""DAH-6804: the boot-time layer reset reaches existing Sysbox hosts: the scrape reads its version, the check installs it."""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.task.checks.sysbox_layer_reset import (
    INSTALL_COMMAND,
    SysboxLayerResetCheck,
    setup_script,
)
from neurons.validators.src.services.task.messages import SysboxLayerResetMessages as Msg
from neurons.validators.tests.helpers import build_scrape_namespace

from core.config import settings
from tests.helpers import build_state
from tests.test_sysbox_required_check import _rented_data

SRC = Path(__file__).resolve().parents[1] / "src"
SETUP_SCRIPT = Path(__file__).resolve().parents[2] / "executor" / "nvidia_docker_sysbox_setup.sh"
UNIT = "etc/systemd/system/lium-sysbox-layer-reset.service"
HOTKEY = "5HotkeyOnTheAllowlist"
CURRENT = int(re.search(r"^LAYER_RESET_VERSION=(\d+)$", SETUP_SCRIPT.read_text(), re.M).group(1))


def _read(host_root: Path):
    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"get_sysbox_layer_reset_version", "SYSBOX_LAYER_RESET_UNIT", "SYSBOX_LAYER_RESET_VERSION_LINE"},
        {"os": os, "re": re},
    )
    return ns["get_sysbox_layer_reset_version"](str(host_root))


def _unit(tmp_path: Path, first_line: str) -> Path:
    (tmp_path / UNIT).parent.mkdir(parents=True)
    (tmp_path / UNIT).write_text(first_line + "\n[Unit]\nDescription=Lium: shift sysbox pods' layers back\n")
    return tmp_path


def test_the_unit_the_setup_script_writes_reads_as_its_version(tmp_path):
    # the line the setup script writes first, with its own version: scrape and script cannot drift apart
    line = re.search(r'^    cat > "\$unit" <<UNIT \|\| return 1\n(.*)$', SETUP_SCRIPT.read_text(), re.M).group(1)
    assert _read(_unit(tmp_path, line.replace("$LAYER_RESET_VERSION", str(CURRENT)))) == CURRENT


def test_no_unit_reads_as_not_installed(tmp_path):
    (tmp_path / "etc").mkdir()
    assert _read(tmp_path) == 0


@pytest.mark.parametrize("first_line", ["[Unit]", "# lium-layer-reset-version=", "# lium-layer-reset-version=2 x"])
def test_a_unit_without_a_version_line_reads_as_the_pre_marker_install(tmp_path, first_line):
    assert _read(_unit(tmp_path, first_line)) == 1


def test_no_host_view_raises_so_the_scrape_reports_unknown():
    with pytest.raises(RuntimeError, match="no host view"):
        _read(Path("/"))


def test_the_validator_ships_the_setup_scripts_version():
    assert setup_script() is not None
    assert setup_script()[1] == CURRENT >= 2


def _ssh(stdout: str):
    return SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stdout=stdout, stderr="")))


def _run(context_factory, specs, ssh, write=True, hotkey=HOTKEY, rented=False):
    state = build_state(specs=specs, rented_data=_rented_data() if rented else None)
    return SysboxLayerResetCheck(write=write).run(context_factory(state=state, ssh=ssh, miner_hotkey=hotkey))


@pytest.fixture
def mode(monkeypatch):
    def _set(value: str, hotkeys: str = HOTKEY):
        monkeypatch.setattr(settings, "SYSBOX_LAYER_RESET_INSTALL", value)
        monkeypatch.setattr(settings, "SYSBOX_LAYER_RESET_INSTALL_HOTKEYS", hotkeys)

    return _set


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [0, 1, CURRENT - 1])
@pytest.mark.parametrize("rented", [False, True])
async def test_an_outdated_host_gets_the_shipped_script_installed_and_keeps_its_score(context_factory, mode, installed, rented):
    mode("all")
    ssh = _ssh(f"layer_reset_installed {CURRENT}\n")

    result = await _run(context_factory, {"sysbox_layer_reset_version": installed}, ssh, rented=rented)

    assert (result.passed, result.event.reason_code, result.updates) == (True, Msg.INSTALLED.reason, {})
    assert ssh.run.await_args.args[0] == INSTALL_COMMAND
    assert ssh.run.await_args.kwargs["input"] == SETUP_SCRIPT.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("value,hotkeys,hotkey", [("shadow", HOTKEY, HOTKEY), ("off", HOTKEY, HOTKEY), ("allowlist", "5Other", HOTKEY)])
async def test_outside_the_rollout_it_reports_and_installs_nothing(context_factory, mode, value, hotkeys, hotkey):
    mode(value, hotkeys)
    ssh = _ssh("")

    result = await _run(context_factory, {"sysbox_layer_reset_version": 0}, ssh, hotkey=hotkey)

    assert (result.passed, result.event.reason_code, result.event.severity) == (True, Msg.OUTDATED.reason, "info")
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_allowlisted_hotkey_gets_it(context_factory, mode):
    mode("allowlist", f"5Other, {HOTKEY}")
    ssh = _ssh(f"layer_reset_installed {CURRENT}\n")

    result = await _run(context_factory, {"sysbox_layer_reset_version": 0}, ssh)

    assert result.event.reason_code == Msg.INSTALLED.reason


@pytest.mark.asyncio
async def test_dry_run_installs_nothing(context_factory, mode):
    mode("all")
    ssh = _ssh("")

    result = await _run(context_factory, {"sysbox_layer_reset_version": 0}, ssh, write=False)

    assert result.event.reason_code == Msg.OUTDATED.reason
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout",
    [
        "layer_reset_skipped no_python3\n",
        "layer_reset_failed install\n",
        "layer_reset_failed nsenter_failed\n",
        "layer_reset_installed 1\n",  # an older script answered: not this validator's version
        "",
    ],
)
async def test_a_host_the_install_does_not_reach_warns_and_keeps_its_score(context_factory, mode, stdout):
    mode("all")

    result = await _run(context_factory, {"sysbox_layer_reset_version": 0}, _ssh(stdout))

    assert (result.passed, result.event.reason_code, result.event.severity) == (True, Msg.INSTALL_FAILED.reason, "warning")
    assert "--install-layer-reset" in Msg.INSTALL_FAILED.remediation


@pytest.mark.asyncio
async def test_an_ssh_error_in_the_install_never_fails_the_check(context_factory, mode):
    mode("all")
    ssh = SimpleNamespace(run=AsyncMock(side_effect=OSError("connection lost")))

    result = await _run(context_factory, {"sysbox_layer_reset_version": 0}, ssh)

    assert (result.passed, result.event.reason_code) == (True, Msg.INSTALL_FAILED.reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "specs,reason",
    [
        ({"sysbox_layer_reset_version": CURRENT}, Msg.OK.reason),
        ({"sysbox_layer_reset_version": CURRENT + 1}, Msg.OK.reason),  # a newer setup script ran on the host
        ({}, Msg.UNKNOWN.reason),  # not a Sysbox host, or an old scrape
        ({"sysbox_layer_reset_scrape_error": "RuntimeError('no host view')"}, Msg.UNKNOWN.reason),
        ({"sysbox_layer_reset_version": True}, Msg.UNKNOWN.reason),
    ],
)
async def test_current_or_unknown_hosts_pass_quietly(context_factory, mode, specs, reason):
    mode("all")
    ssh = _ssh("")

    result = await _run(context_factory, specs, ssh)

    assert (result.passed, result.event.reason_code) == (True, reason)
    ssh.run.assert_not_awaited()


@pytest.mark.parametrize("pipeline", ["build_checks", "build_fast_path_checks", "build_dry_run_checks"])
def test_it_runs_right_after_the_needrestart_guard_before_any_fatal_check(pipeline):
    from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

    names = [type(c).__name__ for c in getattr(PipelineFactory, pipeline)()]

    assert names[names.index("NeedrestartGuardCheck") + 1] == "SysboxLayerResetCheck"
