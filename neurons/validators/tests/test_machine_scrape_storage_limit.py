import ast
import subprocess
from pathlib import Path
from types import SimpleNamespace

MACHINE_SCRAPE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "miner_jobs" / "machine_scrape.py"
)


def load_storage_limit_functions():
    names = {
        "_docker_info_supports_storage_limit",
        "_normalize_docker_info_key",
        "_parse_docker_storage_info",
        "check_storage_limit_ability",
    }
    tree = ast.parse(MACHINE_SCRAPE_PATH.read_text())
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "COMMANDS": {
            "GET_DOCKER_INFO": ["docker", "info"],
            "CHECK_STORAGE_LIMIT_ABILITY": [
                "docker",
                "run",
                "--rm",
                "--storage-opt",
                "size=1g",
                "--gpus",
                "all",
                "daturaai/compute-subnet-executor:latest",
                "nvidia-smi",
            ],
        },
        "re": __import__("re"),
        "subprocess": subprocess,
    }
    exec(compile(module, str(MACHINE_SCRAPE_PATH), "exec"), namespace)
    return SimpleNamespace(**namespace)


def test_docker_info_detects_overlay2_xfs_as_supported():
    module = load_storage_limit_functions()

    supported, message = module._docker_info_supports_storage_limit(
        """
 Storage Driver: overlay2
  Backing Filesystem: xfs
  Supports d_type: true
  Using metacopy: false
  Native Overlay Diff: true
  userxattr: false
 Logging Driver: json-file
"""
    )

    assert supported is True
    assert "overlay2" in message
    assert "xfs" in message


def test_docker_info_detects_containerd_overlayfs_snapshotter_as_unsupported():
    module = load_storage_limit_functions()

    supported, message = module._docker_info_supports_storage_limit(
        """
 Storage Driver: overlayfs
  driver-type: io.containerd.snapshotter.v1
 Logging Driver: json-file
"""
    )

    assert supported is False
    assert "overlayfs" in message


def test_check_storage_limit_rejects_false_positive_probe(monkeypatch):
    module = load_storage_limit_functions()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == module.COMMANDS["GET_DOCKER_INFO"]:
            return SimpleNamespace(
                returncode=0,
                stdout="""
 Storage Driver: overlayfs
  driver-type: io.containerd.snapshotter.v1
 Logging Driver: json-file
""",
                stderr="",
            )
        if command == module.COMMANDS["CHECK_STORAGE_LIMIT_ABILITY"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setitem(
        module.check_storage_limit_ability.__globals__,
        "subprocess",
        SimpleNamespace(
            PIPE=subprocess.PIPE,
            TimeoutExpired=subprocess.TimeoutExpired,
            run=fake_run,
        ),
    )

    supported, message = module.check_storage_limit_ability()

    assert supported is False
    assert "overlayfs" in message
    assert calls == [module.COMMANDS["GET_DOCKER_INFO"]]


def test_check_storage_limit_accepts_overlay2_xfs_after_successful_probe(monkeypatch):
    module = load_storage_limit_functions()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == module.COMMANDS["GET_DOCKER_INFO"]:
            return SimpleNamespace(
                returncode=0,
                stdout="""
 Storage Driver: overlay2
  Backing Filesystem: xfs
  Supports d_type: true
  Using metacopy: false
  Native Overlay Diff: true
  userxattr: false
 Logging Driver: json-file
""",
                stderr="",
            )
        if command == module.COMMANDS["CHECK_STORAGE_LIMIT_ABILITY"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setitem(
        module.check_storage_limit_ability.__globals__,
        "subprocess",
        SimpleNamespace(
            PIPE=subprocess.PIPE,
            TimeoutExpired=subprocess.TimeoutExpired,
            run=fake_run,
        ),
    )

    supported, message = module.check_storage_limit_ability()

    assert supported is True
    assert "Storage limit is supported" in message
    assert calls == [
        module.COMMANDS["GET_DOCKER_INFO"],
        module.COMMANDS["CHECK_STORAGE_LIMIT_ABILITY"],
    ]
