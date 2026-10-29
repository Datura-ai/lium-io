import asyncio
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from services.loopback_plugin_setup import (
    LOOPBACK_PLUGIN_DIGEST,
    _SETUP_SCRIPT,
    loopback_plugin_setup_command,
)


_FAKE_DOCKER = r'''
import json
import sys
import time
from pathlib import Path

root = Path(__file__).parent
state_path = root / "state.json"
state = json.loads(state_path.read_text())
action = sys.argv[2]
state["calls"].append(sys.argv[1:])
plugin_id = "a" * 64

def plugin():
    data_dir = "/srv" + str(root) + "/vloopback-v2"
    if state.get("root_mismatch"):
        data_dir += "-wrong"
    env = ["DATA_DIR=" + data_dir, "STATE_DIR=/srv/run/docker-volume-loopback-v2"]
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
state_path.write_text(json.dumps(state))
if action == "inspect":
    if state.get("inspect_error"):
        print("permission denied", file=sys.stderr)
        sys.exit(1)
    if not state.get("installed"):
        print("Error response from daemon: No such plugin: vloopback:v2", file=sys.stderr)
        sys.exit(1)
    print(json.dumps([plugin()]))
sys.exit(state.get(action + "_exit", 0))
'''


def _setup_arguments(tmp_path: Path, overrides: dict) -> list[str]:
    state = {"manifest": LOOPBACK_PLUGIN_DIGEST, "calls": [], **overrides}
    (tmp_path / "state.json").write_text(json.dumps(state))
    plugin_dir = tmp_path / "plugins" / ("a" * 64)
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "config.json").write_text(json.dumps({
        "Manifest": state["manifest"], "plugin": {"Id": state.get("saved_id", "a" * 64)},
    }))
    docker = tmp_path / "docker"
    docker.write_text(f"#!{sys.executable}\n" + _FAKE_DOCKER)
    docker.chmod(0o700)
    return [
        sys.executable, "-c", _SETUP_SCRIPT, str(docker), "vloopback:v2",
        "daturaai/docker-volume-loopback:1.0.0-lium1", LOOPBACK_PLUGIN_DIGEST,
        str(tmp_path), "", str(tmp_path / "setup.lock"),
    ]


@pytest.mark.asyncio
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
    tmp_path: Path, overrides: dict, success: bool, installs: int, enables: int,
) -> None:
    # Arrange: execute the real remote script against a stateful Docker command fixture.
    arguments = _setup_arguments(tmp_path, overrides)

    # Act
    result = subprocess.run(
        arguments,
        capture_output=True, text=True, timeout=10,
    )

    # Assert: no unverified binary runs and no existing plugin or volume is removed.
    saved = json.loads((tmp_path / "state.json").read_text())
    calls = saved["calls"]
    assert (result.returncode == 0) is success, result.stderr
    assert sum(call[1] == "install" for call in calls) == installs
    assert sum(call[1] == "enable" for call in calls) == enables
    assert not any(call[1] in ("rm", "disable", "set") for call in calls)
    for call in calls:
        if call[1] == "install":
            assert "--disable" in call and "@sha256:" not in " ".join(call)
        elif call[1] == "enable":
            assert call[2] == "a" * 64


@pytest.mark.asyncio
async def test_concurrent_setup_waits_for_the_first_install(tmp_path: Path) -> None:
    # Arrange: two rents reach the same absent plugin while its installation is still running.
    arguments = _setup_arguments(tmp_path, {"install_delay": 0.2})

    # Act
    processes = await asyncio.gather(*(
        asyncio.create_subprocess_exec(
            *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        ) for _ in range(2)
    ))
    outputs = await asyncio.wait_for(asyncio.gather(*(p.communicate() for p in processes)), timeout=10)

    # Assert: both rents see the verified plugin; the second cannot race another install or enable.
    assert all(p.returncode == 0 for p in processes), outputs
    calls = json.loads((tmp_path / "state.json").read_text())["calls"]
    assert sum(call[1] == "install" for call in calls) == 1
    assert sum(call[1] == "enable" for call in calls) == 1


@pytest.mark.asyncio
async def test_setup_command_quotes_the_root_and_never_installs_a_digest() -> None:
    # Arrange
    root = "/data/docker space'$(touch /tmp/unwanted)"

    # Act
    args = shlex.split(loopback_plugin_setup_command(root, "vloopback:v2"))

    # Assert: the shell sees one literal path, with the expected digest passed separately.
    assert args[-3] == root
    assert args[-2] == "/proc/1/root"
    assert args[-1] == "/run/lock/lium-loopback-v2.lock"
    assert args[2] == _SETUP_SCRIPT
    assert args[-4] == LOOPBACK_PLUGIN_DIGEST
