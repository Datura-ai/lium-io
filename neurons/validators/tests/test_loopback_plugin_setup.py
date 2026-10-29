import asyncio
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from services.loopback_plugin_setup import (
    _SETUP_SCRIPT,
    LOOPBACK_PLUGIN_BUILDS,
    loopback_plugin_setup_command,
)


_FAKE_DOCKER = r"""
import json
import sys
import time
from pathlib import Path

root = Path(__file__).parent
state_path = root / "state.json"
state = json.loads(state_path.read_text())
state["calls"].append(sys.argv[1:])
state_path.write_text(json.dumps(state))
containers = state.setdefault("containers", [])
if sys.argv[1] == "run":
    if state.get("run_hang"):
        # the daemon keeps the container once the client is killed by the script's timeout
        containers.append(sys.argv[sys.argv.index("--name") + 1])
        state_path.write_text(json.dumps(state))
        time.sleep(30)
    sys.exit(state.get("run_exit", 0))
if sys.argv[1] == "rm":
    if sys.argv[-1] not in containers:
        print("Error: No such container: " + sys.argv[-1], file=sys.stderr)
        sys.exit(1)
    if state.get("rm_exit"):
        print("Error: cannot remove container", file=sys.stderr)
        sys.exit(1)
    containers.remove(sys.argv[-1])
    state_path.write_text(json.dumps(state))
    sys.exit(0)
if sys.argv[1] == "volume":
    if sys.argv[2] == "rm" and containers:
        print("Error: volume is in use", file=sys.stderr)
        sys.exit(1)
    sys.exit(state.get("volume_" + sys.argv[2] + "_exit", 0))
action = sys.argv[2]
plugin_id = "a" * 64
generation = state["generation"]

def plugin():
    data_dir = "/srv" + str(root) + "/vloopback-" + generation
    if state.get("root_mismatch"):
        data_dir += "-wrong"
    env = ["DATA_DIR=" + data_dir, "STATE_DIR=/srv/run/docker-volume-loopback-" + generation]
    if state.get("duplicate_env"):
        env.append("DATA_DIR=/wrong")
    identity = "b" * 64 if state.get("enable_switch_id") and state.get("enabled") else plugin_id
    return {"Id": identity, "Enabled": state.get("enabled", False), "Settings": {"Env": env}}

if action == "install":
    time.sleep(state.get("install_delay", 0))
    state["installed"] = True
    state["enabled"] = False
    folder = root / "plugins" / plugin_id
    folder.mkdir(exist_ok=True)
    (folder / "config.json").write_text(json.dumps({
        "Manifest": state["manifest"], "plugin": {"Id": state.get("saved_id", plugin_id)}}))
elif action == "enable" and not state.get("enable_exit"):
    state["enabled"] = True
elif action == "disable" and not state.get("disable_exit"):
    state["enabled"] = False
state_path.write_text(json.dumps(state))
if action == "inspect":
    if state.get("inspect_error"):
        print("permission denied", file=sys.stderr)
        sys.exit(1)
    if not state.get("installed"):
        print("Error response from daemon: No such plugin: " + sys.argv[3], file=sys.stderr)
        sys.exit(1)
    print(json.dumps([plugin()]))
sys.exit(state.get(action + "_exit", 0))
"""


def _setup_arguments(
    tmp_path: Path, overrides: dict[str, str | int | bool], alias: str = "vloopback:v3"
) -> list[str]:
    # the real setup command's arguments, with the fake docker and tmp_path as the host
    tag, digest, generation = LOOPBACK_PLUGIN_BUILDS[alias]
    state = {"manifest": digest, "generation": generation, "calls": [], **overrides}
    (tmp_path / "state.json").write_text(json.dumps(state))
    plugin_dir = tmp_path / "plugins" / ("a" * 64)
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "config.json").write_text(
        json.dumps(
            {
                "Manifest": state["manifest"],
                "plugin": {"Id": state.get("saved_id", "a" * 64)},
            }
        )
    )
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n" + _FAKE_DOCKER)
    docker.chmod(0o700)
    arguments = shlex.split(loopback_plugin_setup_command(str(tmp_path), alias))
    arguments[0] = sys.executable
    arguments[3] = str(docker)
    arguments[-3] = ""
    arguments[-2] = str(tmp_path / "setup.lock")
    if arguments[-1]:
        arguments[-1] = str(tmp_path / "probe-ok")
        marker_content = {"current": digest, "old": "sha256:" + "0" * 64, "empty": ""}
        if overrides.get("marker_before"):
            (tmp_path / "probe-ok").write_text(marker_content[str(overrides["marker_before"])])
    return arguments


def _plugin_calls(tmp_path: Path) -> list[list[str]]:
    calls = json.loads((tmp_path / "state.json").read_text())["calls"]
    return [call for call in calls if call[0] == "plugin"]


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["vloopback:v2", "vloopback:v3"])
@pytest.mark.parametrize(
    "overrides, success, installs, enables",
    [
        ({}, True, 1, 1),
        ({"install_exit": 1}, True, 1, 1),
        ({"installed": True, "enabled": True}, True, 0, 0),
        ({"installed": True, "enabled": False}, True, 0, 1),
        ({"manifest": "sha256:" + "0" * 64}, False, 1, 0),
        ({"installed": True, "enabled": True, "manifest": "wrong"}, False, 0, 0),
        ({"root_mismatch": True}, False, 1, 0),
        ({"duplicate_env": True}, False, 1, 0),
        ({"saved_id": "wrong"}, False, 1, 0),
        ({"enable_exit": 1}, False, 1, 1),
        ({"enable_switch_id": True}, False, 1, 1),
        ({"inspect_error": True}, False, 0, 0),
    ],
)
async def test_setup_verifies_download_before_enable_and_preserves_existing_plugins(
    tmp_path: Path,
    alias: str,
    overrides: dict,
    success: bool,
    installs: int,
    enables: int,
) -> None:
    # Arrange: execute the real remote script against a stateful Docker command fixture.
    arguments = _setup_arguments(tmp_path, overrides, alias)

    # Act
    result = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Assert: no unverified binary runs and no existing plugin or volume is removed.
    calls = _plugin_calls(tmp_path)
    assert (result.returncode == 0) is success, result.stderr
    assert sum(call[1] == "install" for call in calls) == installs
    assert sum(call[1] == "enable" for call in calls) == enables
    assert not any(call[1] in ("rm", "disable", "set") for call in calls)
    all_calls = json.loads((tmp_path / "state.json").read_text())["calls"]
    if alias == "vloopback:v2":
        # v2 is set up exactly as before this generation existed: no probe volume, no container
        assert all(call[0] == "plugin" for call in all_calls)
    for call in calls:
        if call[1] == "install":
            assert "--disable" in call and "@sha256:" not in " ".join(call)
        elif call[1] == "enable":
            assert call[2] == "a" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides, success, mounts, disables",
    [
        ({}, True, 1, 0),
        ({"installed": True, "enabled": False}, True, 1, 0),
        ({"installed": True, "enabled": True}, True, 1, 0),
        ({"installed": True, "enabled": True, "marker_before": "current"}, True, 0, 0),
        ({"installed": True, "enabled": True, "marker_before": "old"}, True, 1, 0),
        ({"run_exit": 1}, False, 1, 1),
        ({"volume_create_exit": 1}, False, 0, 1),
        ({"run_exit": 1, "disable_exit": 1}, False, 1, 3),
    ],
    ids=[
        "fresh",
        "disabled",
        "enabled-unprobed",
        "enabled-probed",
        "enabled-probed-as-old-build",
        "mount-refused",
        "create-refused",
        "mount-refused-disable-refused",
    ],
)
async def test_setup_mounts_a_probe_volume_before_handing_out_v3_and_disables_it_when_the_host_refuses(
    tmp_path: Path,
    overrides: dict,
    success: bool,
    mounts: int,
    disables: int,
) -> None:
    # a host that refuses the direct-I/O loop (kernel, data-root filesystem) fails at Mount, which
    # install and enable never reach: the plugin must not stay enabled there, or every create and
    # the backend's health check would pick it (review, 27.10). A plugin found enabled is probed
    # too unless the marker of a passed probe is there (a probe the validator's deadline cut short
    # left the plugin enabled; reread, 27.10). The marker holds the pinned digest: one left by an
    # earlier build means this build was never probed here, and is rewritten once it mounts
    arguments = _setup_arguments(tmp_path, overrides)

    result = subprocess.run(arguments, capture_output=True, text=True, timeout=10)

    calls = json.loads((tmp_path / "state.json").read_text())["calls"]
    assert (result.returncode == 0) is success, result.stderr
    runs = [call for call in calls if call[0] == "run"]
    assert len(runs) == mounts
    assert all("lium-loopback-probe-v3:/mnt" in call and "--network" in call for call in runs)
    creates = [call for call in calls if call[:2] == ["volume", "create"]]
    removes = [call for call in calls if call[:2] == ["volume", "rm"]]
    assert len(creates) == (1 if mounts or "volume_create_exit" in overrides else 0)
    # leftovers of an earlier probe go before the create, the probe's own after it
    assert len(removes) == 2 * len(creates)
    assert all(
        call[3:5] == ["vloopback:v3", "-o"] and call[-1] == "lium-loopback-probe-v3"
        for call in creates
    )
    assert sum(call[:2] == ["plugin", "disable"] for call in calls) == disables
    assert all("size=1g" in call and "sparse=true" in call for call in creates)
    assert (tmp_path / "probe-ok").exists() is success
    if success:
        assert (tmp_path / "probe-ok").read_text() == LOOPBACK_PLUGIN_BUILDS["vloopback:v3"].digest
    if not success:
        assert "probe volume" in result.stderr
        if overrides.get("disable_exit"):
            assert "could not be disabled" in result.stderr
        else:
            assert json.loads((tmp_path / "state.json").read_text())["enabled"] is False


@pytest.mark.parametrize("disable_exit", [0, 1], ids=["disable-ok", "disable-refused"])
@pytest.mark.parametrize("run_exit", [0, 1], ids=["probe-ok", "probe-failed"])
@pytest.mark.parametrize("marker_before", ["empty", "old", "current"])
def test_setup_removes_a_marker_of_another_build_before_it_can_fail(
    tmp_path: Path, marker_before: str, run_exit: int, disable_exit: int
) -> None:
    # Arrange: the backend's health check tests the marker's existence only. Were a marker of an
    # earlier build left while this build's probe fails and the plugin cannot be disabled (volumes
    # keep it busy), the validator would rent on v2 while the backend checks v3 (reread, 29.10)
    overrides: dict[str, str | int | bool] = {
        "installed": True, "enabled": True, "marker_before": marker_before,
        "run_exit": run_exit, "disable_exit": disable_exit,
    }
    arguments = _setup_arguments(tmp_path, overrides)
    digest = LOOPBACK_PLUGIN_BUILDS["vloopback:v3"].digest

    # Act
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=10)

    # Assert
    probed = marker_before != "current"
    assert (result.returncode == 0) is (not probed or run_exit == 0), result.stderr
    marker = tmp_path / "probe-ok"
    if result.returncode == 0:
        assert marker.read_text() == digest
    else:
        assert not marker.exists()
        assert ("could not be disabled" in result.stderr) is bool(disable_exit)


def test_setup_reclaims_a_probe_container_an_earlier_setup_abandoned(tmp_path: Path) -> None:
    # Arrange: a setup killed before its cleanup left the probe container (`created`, never exits)
    # holding the probe volume. The next setup must reclaim both before it probes, and never write
    # the marker while they stay (reread, 29.10)
    arguments = _setup_arguments(tmp_path, {"containers": ["lium-loopback-probe-v3"]})

    # Act: two setups, as two rents would run them
    results = [subprocess.run(arguments, capture_output=True, text=True, timeout=10) for _ in range(2)]

    # Assert: the first reclaims and probes, the second finds the marker and skips
    state = json.loads((tmp_path / "state.json").read_text())
    assert [result.returncode for result in results] == [0, 0], [result.stderr for result in results]
    assert state["containers"] == []
    assert (tmp_path / "probe-ok").read_text() == LOOPBACK_PLUGIN_BUILDS["vloopback:v3"].digest
    calls = state["calls"]
    assert sum(call[0] == "run" for call in calls) == 1
    first_rm = next(i for i, call in enumerate(calls) if call[:2] == ["rm", "-f"])
    first_create = next(i for i, call in enumerate(calls) if call[:2] == ["volume", "create"])
    assert first_rm < first_create
    assert all(call[-1] == "lium-loopback-probe-v3" for call in calls if call[:2] == ["rm", "-f"])


def test_setup_fails_without_a_marker_when_an_abandoned_probe_container_cannot_be_removed(
    tmp_path: Path,
) -> None:
    # Arrange: the abandoned container refuses `docker rm -f`, so its volume stays in use
    arguments = _setup_arguments(tmp_path, {"containers": ["lium-loopback-probe-v3"], "rm_exit": 1})

    # Act
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=10)

    # Assert: a leftover is a setup error, not a warning, and no proof is written over it
    state = json.loads((tmp_path / "state.json").read_text())
    assert result.returncode == 1
    assert "an earlier probe left container lium-loopback-probe-v3, volume lium-loopback-probe-v3" in result.stderr
    assert not (tmp_path / "probe-ok").exists()
    assert not any(call[0] == "run" for call in state["calls"])
    assert state["enabled"] is False


def test_setup_removes_a_probe_container_its_timeout_cut_off_before_its_volume(tmp_path: Path) -> None:
    # Arrange: the probe's `sync` outlives the script's per-command timeout. Python kills the docker
    # client, the daemon keeps the container with the volume attached: unless the container goes
    # first, the volume stays in use and the plugin cannot be disabled, and every retry adds a
    # container (review, 29.10)
    arguments = _setup_arguments(tmp_path, {"run_hang": True})
    arguments[2] = _SETUP_SCRIPT.replace("timeout=120", "timeout=1")

    # Act: two attempts, as two rents would retry the setup
    results = [subprocess.run(arguments, capture_output=True, text=True, timeout=20) for _ in range(2)]

    # Assert
    state = json.loads((tmp_path / "state.json").read_text())
    for result in results:
        assert result.returncode == 1
        assert "probe volume failed: TimeoutExpired" in result.stderr
        assert "could not be disabled" not in result.stderr
    assert state["containers"] == []
    assert state["enabled"] is False
    assert not (tmp_path / "probe-ok").exists()
    calls = state["calls"]
    runs = [i for i, call in enumerate(calls) if call[0] == "run"]
    assert len(runs) == 2
    for run in runs:
        assert calls[run][calls[run].index("--name") + 1] == "lium-loopback-probe-v3"
        after = calls[run + 1 :]
        container_rm = next(i for i, call in enumerate(after) if call[:2] == ["rm", "-f"])
        volume_rm = next(i for i, call in enumerate(after) if call[:2] == ["volume", "rm"])
        assert container_rm < volume_rm


@pytest.mark.asyncio
async def test_concurrent_setup_waits_for_the_first_install(tmp_path: Path) -> None:
    # Arrange: two rents reach the same absent plugin while its installation is still running.
    arguments = _setup_arguments(tmp_path, {"install_delay": 0.2})

    # Act
    processes = await asyncio.gather(
        *(
            asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            for _ in range(2)
        )
    )
    outputs = await asyncio.wait_for(
        asyncio.gather(*(p.communicate() for p in processes)), timeout=10
    )

    # Assert: both rents see the verified plugin; the second cannot race another install or enable.
    assert all(p.returncode == 0 for p in processes), outputs
    calls = _plugin_calls(tmp_path)
    assert sum(call[1] == "install" for call in calls) == 1
    assert sum(call[1] == "enable" for call in calls) == 1


@pytest.mark.parametrize("alias, generation", [("vloopback:v2", "v2"), ("vloopback:v3", "v3")])
@pytest.mark.asyncio
async def test_setup_command_quotes_the_root_and_never_installs_a_digest(alias, generation) -> None:
    # Arrange
    root = "/data/docker space'$(touch /tmp/unwanted)"
    tag, digest, _ = LOOPBACK_PLUGIN_BUILDS[alias]

    # Act
    args = shlex.split(loopback_plugin_setup_command(root, alias))

    # Assert: the shell sees one literal path, with its generation's tag and digest passed separately.
    assert args[-4] == root
    assert args[-3] == "/proc/1/root"
    assert args[-2] == f"/run/lock/lium-loopback-{generation}.lock"
    # the marker path is read by the backend's health check too (lium-platform#2126) and stays fixed
    assert args[-1] == ("" if generation == "v2" else f"/run/lock/lium-loopback-{generation}.probe-ok")
    assert args[2] == _SETUP_SCRIPT
    assert args[4:8] == [alias, tag, digest, generation]
    assert args[8] == ("" if generation == "v2" else "docker.io/library/alpine:3.19")
    assert "sha256" not in tag
