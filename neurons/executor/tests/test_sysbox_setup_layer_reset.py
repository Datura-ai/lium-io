"""nvidia_docker_sysbox_setup.sh: after an unclean stop, a boot oneshot shifts each sysbox pod's writable layer back."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

POD_BASE = 165536
SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nvidia_docker_sysbox_setup.sh"))


def _function(name: str) -> str:
    text = open(SCRIPT).read()
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    assert match, name
    return match.group(0)


def _install(root) -> None:
    script = (
        'host_path() { echo "${SYSBOX_SETUP_HOST_ROOT:-}$1"; }\nwarn() { echo "WARN $*"; }\n'
        + _function("docker_containerd_address")
        + _function("shift_sysbox_layers_back_at_boot")
        + "shift_sysbox_layers_back_at_boot\n"
    )
    return subprocess.run(
        ["bash", "-c", script], check=True, capture_output=True, text=True,
        env={**os.environ, "SYSBOX_SETUP_HOST_ROOT": str(root)},
    ).stdout


def _container(root, cid: str, runtime: str, mount: str) -> str:
    docker = root / "var/lib/docker"
    (docker / "containers" / cid).mkdir(parents=True)
    (docker / "containers" / cid / "hostconfig.json").write_text(json.dumps({"Runtime": runtime}))
    (docker / "image/overlay2/layerdb/mounts" / cid).mkdir(parents=True)
    (docker / "image/overlay2/layerdb/mounts" / cid / "mount-id").write_text(mount)
    upper = docker / "overlay2" / mount / "diff"
    (upper / "etc").mkdir(parents=True)
    (upper / "etc/sudoers").write_text("root ALL=(ALL) ALL\n")
    os.symlink("etc/sudoers", upper / "sudoers-link")
    if os.getuid() == 0:  # as root, give the layer a pod's uid base the way sysbox does
        for top, dirs, files in os.walk(upper):
            for path in [top, *(os.path.join(top, name) for name in dirs + files)]:
                os.lchown(path, POD_BASE, POD_BASE)
    return str(upper)


def _base() -> int:
    return POD_BASE if os.getuid() == 0 else os.getuid()


def _run(root, *args: str, path: str = "") -> subprocess.CompletedProcess:
    unit = (root / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()
    address = re.search(r"^Environment=LIUM_CONTAINERD_ADDRESS=(.*)$", unit, re.M).group(1)
    return subprocess.run(
        [sys.executable, "-I", str(root / "usr/local/sbin/lium-sysbox-layer-reset"), *args],
        env={
            **os.environ,
            "LIUM_LAYER_RESET_ROOT": str(root),
            "LIUM_CONTAINERD_ADDRESS": address,
            "PATH": path + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
    )


def test_the_boot_unit_runs_after_sysbox_and_before_docker(tmp_path):
    _install(tmp_path)
    unit = (tmp_path / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()
    assert "After=local-fs.target sysbox.service sysbox-mgr.service containerd.service" in unit
    assert "Before=docker.service\n" in unit
    assert "ExecStart=/usr/bin/python3 -I /usr/local/sbin/lium-sysbox-layer-reset" in unit


def test_a_shifted_sysbox_layer_is_found_and_a_runc_container_is_left_alone(tmp_path):
    # the test user's uid stands in for the pod's uid base (as root: POD_BASE)
    _install(tmp_path)
    _container(tmp_path, "a" * 64, "sysbox-runc", "m1")
    _container(tmp_path, "b" * 64, "runc", "m2")
    out = _run(tmp_path, "--dry-run")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"would shift 4 entries of {'a' * 12} back from {_base()}"]


def _containerd_store(tmp_path, cid: str) -> None:
    upper = Path(_container(tmp_path, cid, "sysbox-runc", "m1"))
    (tmp_path / "var/lib/docker/image/overlay2/layerdb/mounts" / cid / "mount-id").unlink()
    snap = tmp_path / "var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/7/fs"
    snap.parent.mkdir(parents=True)
    upper.rename(snap)


def _fake_ctr(tmp_path, cid: str, address: str) -> str:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "ctr").write_text(
        "#!/bin/sh\n"
        f'[ "$*" = "--address {tmp_path}{address} -n moby snapshots --snapshotter overlayfs mounts / {cid}" ] || exit 1\n'
        "echo mount -t overlay overlay / -o index=off,workdir=/x/7/work,"
        "upperdir=/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/7/fs,lowerdir=/x/6/fs\n"
    )
    (bindir / "ctr").chmod(0o755)
    return f"{bindir}:"


@pytest.mark.parametrize("jq_available", [True, False])
@pytest.mark.parametrize("address", ["/run/containerd/containerd.sock", "/run/lium-containerd/containerd.sock"])
def test_a_containerd_store_layer_is_found_through_its_snapshot(tmp_path, address, jq_available, monkeypatch):
    if not jq_available:  # the installer installs jq only later
        monkeypatch.setenv("BASH_FUNC_jq%%", "() { return 127; }")
    # Docker 29's containerd image store has no layerdb mount-id; ctr, asked at Docker's containerd address,
    # names the snapshot's upperdir
    (tmp_path / "etc/docker").mkdir(parents=True)
    (tmp_path / "etc/docker/daemon.json").write_text(json.dumps({"containerd": address}))
    (tmp_path / address.lstrip("/")).parent.mkdir(parents=True)
    (tmp_path / address.lstrip("/")).touch()  # the socket is up
    _install(tmp_path)
    cid = "c" * 64
    _containerd_store(tmp_path, cid)
    out = _run(tmp_path, "--dry-run", path=_fake_ctr(tmp_path, cid, address))
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"would shift 4 entries of {'c' * 12} back from {_base()}"]


def test_a_docker_managed_containerd_store_is_read_through_a_containerd_started_for_it(tmp_path):
    # no --containerd, no daemon.json "containerd", no system containerd socket: Docker starts its own containerd,
    # which is down before Docker, so the reset starts one on Docker's store and asks it
    _install(tmp_path)
    cid = "d" * 64
    _containerd_store(tmp_path, cid)
    path = _fake_ctr(tmp_path, cid, "/run/lium-sysbox-layer-reset/containerd.sock")
    (tmp_path / "bin/containerd").write_text(
        "#!/bin/sh\n"
        f'[ "$4" = "{tmp_path}/var/lib/docker/containerd/daemon" ] || exit 1\n'
        'mkdir -p "$(dirname "$8")" && touch "$8" && exec sleep 30\n'
    )
    (tmp_path / "bin/containerd").chmod(0o755)
    out = _run(tmp_path, "--dry-run", path=path)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"would shift 4 entries of {'d' * 12} back from {_base()}"]


@pytest.mark.parametrize("cut_after", [None, 0, 1, 3])
def test_a_reset_cut_off_part_way_is_finished_on_the_next_boot(tmp_path, cut_after):
    # the layer's top directory is shifted last: until it is, the next boot sees the layer as shifted and redoes it
    _install(tmp_path)
    upper = Path(_container(tmp_path, "a" * 64, "sysbox-runc", "m1"))
    renterfile = upper / "renterfile"
    renterfile.touch()
    modes = {str(p): os.lstat(p).st_mode for p in [upper, *upper.rglob("*")]}
    owners = dict.fromkeys(modes, POD_BASE)
    owners[str(renterfile)] = POD_BASE + 1000
    reset = {}
    program = (tmp_path / "usr/local/sbin/lium-sysbox-layer-reset").read_text().rsplit("\nmain()", 1)[0]
    exec(program, reset)
    reset.update(ROOT=str(tmp_path), DRY=False)
    reset["os"] = SimpleNamespace(
        path=os.path,
        walk=os.walk,
        kill=os.kill,
        lstat=lambda p: SimpleNamespace(st_uid=owners[p], st_gid=owners[p], st_mode=modes[p]),
        lchown=lambda p, u, g: owners.__setitem__(p, u),
        chmod=lambda *a: None,
        getxattr=lambda *a, **kw: None,
    )
    original = reset["shift_back"]
    calls = 0

    def interrupted(path, base):
        nonlocal calls
        if calls == cut_after:
            raise InterruptedError()
        calls += 1
        return original(path, base)

    reset["shift_back"] = interrupted
    if cut_after is None:
        reset["main"]()
    else:
        with pytest.raises(InterruptedError):
            reset["main"]()
    reset["shift_back"] = original
    reset["main"]()
    assert owners == {p: 1000 if p == str(renterfile) else 0 for p in modes}


def test_nothing_is_touched_while_docker_runs(tmp_path):
    _install(tmp_path)
    upper = _container(tmp_path, "a" * 64, "sysbox-runc", "m1")
    (tmp_path / "var/run").mkdir(parents=True)
    (tmp_path / "var/run/docker.pid").write_text(str(os.getpid()))
    out = _run(tmp_path)
    assert out.returncode != 0
    assert "Docker is running" in out.stderr
    assert os.lstat(upper).st_uid == _base()


def test_the_unit_is_installed_before_the_already_working_exit():
    # a host whose sysbox already works exits at "Already working?"; it runs pods, so it needs the unit most
    text = open(SCRIPT).read()
    assert text.count("\nshift_sysbox_layers_back_at_boot &&") == 1
    assert text.index("\nshift_sysbox_layers_back_at_boot &&") < text.index("# ── 2. Already working?")
