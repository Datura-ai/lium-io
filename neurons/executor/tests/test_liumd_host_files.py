"""The files the executor image gives `liumd run`."""

import hashlib
import json
import os
import platform
import re
import stat
import subprocess
from pathlib import Path

import pytest

import liumd_host_files as hf

EXECUTOR = Path(__file__).resolve().parents[1]
BINARY = EXECUTOR / "liumd" / "liumd"
DOCKERFILE = (EXECUTOR / "Dockerfile").read_text()
RUN_SH = (EXECUTOR / "run.sh").read_text()
WRAPPER = (EXECUTOR / "liumd" / "liumd.sh").read_text()

# liumd's own manifest names and paths (its children.json), less `gpu_sig`, which no verify step
# spawns and this image does not carry.
LIUMD_CHILDREN = {
    "verifyx": "/usr/lib/libverifyx.so",
    "inspector": "/usr/lib/libinspector.so",
    "matmul": "/usr/lib/libdmcompverify.so",
    "python": "/root/app/.venv/bin/python",
    "matmul_script": "/root/app/src/decrypt_challenge.py",
    "verifyx_script": "/root/app/src/verifyx_executor.py",
}


def _image_tree(root: Path) -> None:
    """The children as the Dockerfile lays them out, the interpreter behind the venv's symlink."""
    for name, path in hf.CHILDREN:
        target = root / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == "python":
            real = root / "usr/local/bin/python3.11"
            real.parent.mkdir(parents=True, exist_ok=True)
            real.write_bytes(b"\x7fELF interpreter")
            target.symlink_to(real)
        else:
            target.write_bytes(f"{name} bytes".encode())


def test_the_manifest_names_every_child_liumd_spawns_at_its_path():
    assert dict(hf.CHILDREN) == LIUMD_CHILDREN


def test_the_manifest_pins_each_child_and_follows_the_venv_symlink(tmp_path):
    _image_tree(tmp_path)

    manifest = hf.children_manifest(tmp_path)

    assert manifest["schema"] == "lium.children/1"
    by_name = {c["name"]: c for c in manifest["children"]}
    assert {n: c["path"] for n, c in by_name.items()} == LIUMD_CHILDREN
    for name, child in by_name.items():
        assert re.fullmatch(r"[0-9a-f]{64}", child["sha256"]), name
    assert by_name["matmul"]["sha256"] == hashlib.sha256(b"matmul bytes").hexdigest()
    assert by_name["python"]["sha256"] == hashlib.sha256(b"\x7fELF interpreter").hexdigest()


def test_a_missing_child_fails_the_build(tmp_path):
    _image_tree(tmp_path)
    (tmp_path / "usr/lib/libverifyx.so").unlink()

    with pytest.raises(FileNotFoundError, match="verifyx"):
        hf.children_manifest(tmp_path)


def test_children_writes_the_manifest_file(tmp_path, monkeypatch):
    _image_tree(tmp_path)
    real = hf.children_manifest
    monkeypatch.setattr(hf, "children_manifest", lambda: real(tmp_path))
    out = tmp_path / "etc/liumd/children.json"

    assert hf.main(["children", str(out)]) == 0

    assert json.loads(out.read_text()) == real(tmp_path)
    assert stat.S_IMODE(out.stat().st_mode) == 0o644


def test_the_host_files_have_the_shapes_liumd_reads(tmp_path):
    etc, nonces = tmp_path / "etc/liumd", tmp_path / "var/lib/liumd/nonces"

    hf.write_host_files(
        etc_dir=etc,
        nonce_dir=nonces,
        validator_hotkeys={"current": "5Current", "next": "5Next"},
        miner_hotkey=" 5Miner \n",
        port_range="20000-20010",
        port_mappings=None,
        ssh_port=2200,
    )

    assert (etc / "validator_hotkeys").read_text() == "5Current\n5Next\n"
    assert (etc / "miner_hotkey").read_text() == "5Miner\n"
    assert json.loads((etc / "ports.json").read_text()) == {
        "port_range": "20000-20010",
        "port_mappings": None,
        "ssh_port": 2200,
    }
    for name in ("validator_hotkeys", "miner_hotkey", "ports.json"):
        assert stat.S_IMODE((etc / name).stat().st_mode) == 0o644
    assert nonces.is_dir() and os.access(nonces, os.W_OK)
    assert stat.S_IMODE(nonces.stat().st_mode) == 0o700
    assert [p.name for p in etc.iterdir() if p.name.startswith(".")] == []


def test_an_unset_port_setting_is_null_not_an_empty_string():
    doc = json.loads(hf.ports_json_text("", "[[2000, 20000]]", 22))
    assert doc == {"port_range": None, "port_mappings": "[[2000, 20000]]", "ssh_port": 22}


def test_host_reads_the_executor_settings(tmp_path, monkeypatch):
    import core.config as config

    monkeypatch.setattr(hf, "ETC_DIR", tmp_path / "etc")
    monkeypatch.setattr(hf, "NONCE_DIR", tmp_path / "nonces")
    monkeypatch.setattr(config.settings, "RENTING_PORT_RANGE", "30000-30004")
    monkeypatch.setattr(config.settings, "RENTING_PORT_MAPPINGS", None)
    monkeypatch.setattr(config.settings, "SSH_PORT", 2222)

    assert hf.main(["host"]) == 0

    assert (tmp_path / "etc/validator_hotkeys").read_text().splitlines() == list(
        config.VALIDATOR_HOTKEYS_SS58.values()
    )
    assert (
        tmp_path / "etc/miner_hotkey"
    ).read_text() == f"{config.settings.MINER_HOTKEY_SS58_ADDRESS}\n"
    assert json.loads((tmp_path / "etc/ports.json").read_text()) == {
        "port_range": "30000-30004",
        "port_mappings": None,
        "ssh_port": 2222,
    }


def test_an_unknown_command_is_refused():
    assert hf.main(["everything"]) == 2


def test_the_dockerfile_pins_the_committed_binary():
    pinned = re.search(r'echo "([0-9a-f]{64})  liumd/liumd" \| sha256sum -c -', DOCKERFILE)
    assert pinned, "the Dockerfile must check liumd/liumd against a literal sha256"
    assert pinned.group(1) == hashlib.sha256(BINARY.read_bytes()).hexdigest()


def test_the_dockerfile_installs_liumd_and_hashes_the_children_after_the_provers():
    provers = DOCKERFILE.index("mv /root/app/libinspector.so /usr/lib/")
    for line in (
        "install -D -m 0755 liumd/liumd /usr/local/lib/liumd/liumd",
        "install -m 0755 liumd/liumd.sh /usr/local/bin/liumd",
        "install -d -m 0700 /var/lib/liumd/nonces",
        "/root/app/.venv/bin/python src/liumd_host_files.py children /etc/liumd/children.json",
    ):
        assert DOCKERFILE.index(line) > provers, line


def test_the_wrapper_hands_the_image_manifest_to_the_binary(tmp_path):
    # The image's wrapper with its binary path pointed at a stand-in that prints what it was given;
    # a wrapper naming another binary path would exec nothing and fail here.
    stand_in = tmp_path / "liumd"
    stand_in.write_text(
        '#!/bin/sh\necho "$LIUMD_CHILD_MANIFEST_FILE|${LIUMD_MINER_HOTKEY-unset}|$*"\n'
    )
    stand_in.chmod(0o755)
    wrapper = tmp_path / "wrapper"
    wrapper.write_text(WRAPPER.replace("/usr/local/lib/liumd/liumd", str(stand_in)))

    out = subprocess.run(
        ["/bin/sh", str(wrapper), "run"],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert out.stdout.strip() == "/etc/liumd/children.json|unset|run"


def test_run_sh_writes_the_host_files_before_sshd_and_never_stops_the_executor():
    line = "pdm run python src/liumd_host_files.py host || echo"
    assert line in RUN_SH
    assert RUN_SH.index(line) < RUN_SH.index("service ssh start")


@pytest.mark.skipif(platform.machine() != "x86_64", reason="the committed binary is x86_64 musl")
def test_the_committed_binary_is_the_keyless_dev_build():
    done = subprocess.run([str(BINARY), "version"], capture_output=True, timeout=30, check=True)
    version = json.loads(done.stdout)
    assert version["capability"] == "local_verify/1"
    assert version["key_id"] is None
    assert version["release"]["children_pinned"] == 0
