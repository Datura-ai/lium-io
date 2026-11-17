"""DAH: hosts set up before 16 Nov lack the needrestart guard; the scrape reports it, the check warns."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from neurons.validators.src.services.task.checks.needrestart_guard import NeedrestartGuardCheck
from neurons.validators.src.services.task.messages import NeedrestartGuardMessages as Msg
from neurons.validators.tests.helpers import build_scrape_namespace

from tests.helpers import build_state
from tests.test_sysbox_required_check import _rented_data

SRC = Path(__file__).resolve().parents[1] / "src"
SETUP_SCRIPT = Path(__file__).resolve().parents[2] / "executor" / "nvidia_docker_sysbox_setup.sh"
GUARD = "/etc/needrestart/conf.d/lium-container-runtime.conf"


def _read(host_root: Path):
    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {"get_needrestart_guard", "NEEDRESTART_GUARD_FILE", "NEEDRESTART_GUARD_SERVICES"},
        {"os": os, "re": re},
    )
    return ns["get_needrestart_guard"](str(host_root))


def _host(tmp_path: Path, guard_text: str | None, needrestart: bool = True) -> Path:
    if needrestart:
        (tmp_path / "etc/needrestart/conf.d").mkdir(parents=True)
    if guard_text is not None:
        (tmp_path / GUARD.lstrip("/")).write_text(guard_text)
    return tmp_path


def _setup_script_guard() -> str:
    # the exact file the setup script writes, so the scrape and the script cannot drift apart
    script = SETUP_SCRIPT.read_text()
    return script.split("<<'NRCONF'\n", 1)[1].split("\nNRCONF\n", 1)[0]


def test_the_file_the_setup_script_writes_reads_as_present(tmp_path):
    assert _read(_host(tmp_path, _setup_script_guard())) == ("present", [])


def test_a_host_set_up_before_the_guard_reads_as_missing(tmp_path):
    status, unguarded = _read(_host(tmp_path, None))

    assert status == "missing"
    assert "sysbox" in unguarded and "nvidia-fabricmanager" in unguarded


def test_a_guard_without_one_service_reads_as_partial(tmp_path):
    text = "\n".join(line for line in _setup_script_guard().splitlines() if "containerd" not in line)

    assert _read(_host(tmp_path, text)) == ("partial", ["containerd"])


def test_a_host_without_needrestart_needs_no_guard(tmp_path):
    assert _read(_host(tmp_path, None, needrestart=False)) == ("no_needrestart", [])


def test_no_host_view_raises_so_the_scrape_reports_unknown():
    with pytest.raises(RuntimeError):
        _read(Path("/"))


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["missing", "partial"])
@pytest.mark.parametrize("rented", [False, True])
async def test_an_unguarded_host_warns_and_keeps_its_score(context_factory, guard, rented):
    specs = {"needrestart_guard": guard, "needrestart_guard_unguarded": ["sysbox"]}
    state = build_state(specs=specs, rented_data=_rented_data() if rented else None)

    result = await NeedrestartGuardCheck().run(context_factory(state=state))

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason
    assert result.event.severity == "warning"
    assert not result.updates
    assert GUARD in Msg.MISSING.remediation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "specs,reason",
    [
        ({"needrestart_guard": "present"}, Msg.OK.reason),
        ({"needrestart_guard": "no_needrestart"}, Msg.OK.reason),
        ({}, Msg.UNKNOWN.reason),  # old scrape or no host view: unknown, never a warning
    ],
)
async def test_guarded_or_unknown_hosts_pass_quietly(context_factory, specs, reason):
    result = await NeedrestartGuardCheck().run(context_factory(state=build_state(specs=specs)))

    assert result.passed is True
    assert result.event.reason_code == reason
