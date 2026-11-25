"""DAH-6530: every value the node or backend hands back stays ONE argument in the shell line the
validator sends. Each built command runs in a local sh with docker/rm stubbed to print their argv;
the planted `; touch`, `$(...)` and newline must neither run nor split the argument."""
import subprocess
from types import SimpleNamespace

import pytest

from neurons.validators.src.core.docker_utils import DockerCommand
from neurons.validators.src.services.container_cleanup import never_started_check_containers
from neurons.validators.src.services.port_utils import get_all_ports
from neurons.validators.src.services.task.checks.custom_build_orphan_sweep import (
    BUILD_DIND_PREFIX,
    BUILD_SCRATCH_PREFIX,
    CustomBuildOrphanSweepCheck,
)

_STUBS = 'docker(){ printf "<%s>\\n" "$@"; }; rm(){ printf "<%s>\\n" "$@"; }; '


def _payloads(marker):
    # relative marker: the shells run in tmp_path, and a "/" would make the scratch sweep refuse early
    return ["x; touch ran", "x$(touch ran)", "x`touch ran`", "x\ntouch ran"]


def _argv(command, tmp_path) -> str:
    command = command.replace("/usr/bin/docker", "docker").replace("/usr/bin/rm", "rm")
    done = subprocess.run(["sh", "-c", _STUBS + command], cwd=tmp_path, capture_output=True, text=True, timeout=20)
    return done.stdout


@pytest.mark.parametrize(
    "build",
    [
        lambda v: DockerCommand.run_alpine(v, "true", "none", 5),
        lambda v: DockerCommand.run_dind(v, 2222, "ssh-ed25519 AAAA", sysbox=False),
        lambda v: DockerCommand.run_dind("n", 2222, v, sysbox=False),
        lambda v: DockerCommand.remove_with_volumes(v),
        lambda v: DockerCommand.ps_filter(v),
        lambda v: DockerCommand.inspect_status(v),
        lambda v: DockerCommand.logs(v),
        lambda v: DockerCommand.inspect_exit_code(v),
    ],
)
def test_docker_command_builders_keep_node_values_inert(build, tmp_path):
    marker = tmp_path / "ran"
    for value in _payloads(marker):
        _argv(build(value), tmp_path)
        assert not marker.exists(), value


def test_run_dind_public_key_reaches_the_inner_shell_as_one_word(tmp_path):
    out = _argv(DockerCommand.run_dind("n", 2222, "k; touch x", sysbox=False), tmp_path)
    assert "echo 'k; touch x' >>" in out


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["_remove_dind_container", "_remove_image", "_remove_scratch"])
async def test_orphan_sweep_quotes_listed_names(method, tmp_path):
    prefix = BUILD_SCRATCH_PREFIX if method == "_remove_scratch" else BUILD_DIND_PREFIX
    sent = []

    async def run(command, **_):
        sent.append(command)
        return SimpleNamespace(exit_status=0, stdout="", stderr="")

    check = CustomBuildOrphanSweepCheck.__new__(CustomBuildOrphanSweepCheck)
    marker = tmp_path / "ran"
    for value in _payloads(marker):
        await getattr(check, method)(SimpleNamespace(run=run), prefix + value.replace("\n", " "))
    for command in sent:
        _argv(command, tmp_path)
    assert sent and not marker.exists()


@pytest.mark.asyncio
async def test_orphan_sweep_refuses_a_scratch_path_that_leaves_the_prefix():
    sent = []

    async def run(command, **_):
        sent.append(command)

    check = CustomBuildOrphanSweepCheck.__new__(CustomBuildOrphanSweepCheck)
    assert await check._remove_scratch(SimpleNamespace(run=run), BUILD_SCRATCH_PREFIX + "x/../../etc") is False
    assert sent == []


def test_container_cleanup_only_returns_full_container_ids():
    good = "a" * 64
    listing = f"{good} lium_capability_check_1_abcdef\n$(reboot) lium_capability_check_2_abcdef\n"
    assert never_started_check_containers(listing) == [good]


def test_port_mappings_from_the_node_must_be_integers():
    with pytest.raises(ValueError):
        get_all_ports(None, '[["2000; reboot", 2000]]', 22)
    assert get_all_ports(None, '[["2000", 3000]]', 22) == [(2000, 3000)]
