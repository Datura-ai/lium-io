"""nvidia_docker_sysbox_setup.sh: after an unclean stop, a boot oneshot shifts each sysbox pod's writable layer back."""

import json
import os
import re
import stat
import struct
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


def _run(root, *args: str, path: str = "", env: dict | None = None) -> subprocess.CompletedProcess:
    unit = (root / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()
    address = re.search(r"^Environment=LIUM_CONTAINERD_ADDRESS=(.*)$", unit, re.M).group(1)
    return subprocess.run(
        [sys.executable, "-I", str(root / "usr/local/sbin/lium-sysbox-layer-reset"), *args],
        env={
            **os.environ,
            "LIUM_LAYER_RESET_ROOT": str(root),
            "LIUM_CONTAINERD_ADDRESS": address,
            "PATH": path + os.environ["PATH"],
            **(env or {}),
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


def _acl(*entries) -> bytes:
    # version 2, then (tag, perm, id) per entry: USER_OBJ, a named USER, GROUP_OBJ, a named GROUP, MASK, OTHER
    return b"\x02\x00\x00\x00" + b"".join(struct.pack("<HHI", *e) for e in entries)


def _fake_fs(tmp_path):
    """The installed reset over an in-memory layer: owners, modes and xattrs per path; chown clears set-id bits and
    file capabilities the way the kernel does."""
    _install(tmp_path)
    upper = Path(_container(tmp_path, "a" * 64, "sysbox-runc", "m1"))
    for name in ("renterfile", "sudo"):
        (upper / name).touch()
    fs = SimpleNamespace(upper=upper, modes={}, owners={}, xattrs={})
    for p in [upper, *upper.rglob("*")]:
        fs.modes[str(p)] = os.lstat(p).st_mode
        fs.owners[str(p)] = POD_BASE
        fs.xattrs[str(p)] = {}
    fs.owners[str(upper / "renterfile")] = POD_BASE + 1000
    fs.modes[str(upper / "sudo")] = stat.S_IFREG | 0o4755
    fs.xattrs[str(upper / "sudo")]["security.capability"] = b"cap-bytes"

    def getxattr(p, name, **kw):
        if name not in fs.xattrs[p]:
            raise OSError(61, "No data available")
        return fs.xattrs[p][name]

    def lchown(p, u, g):
        fs.owners[p] = u
        fs.modes[p] &= ~(stat.S_ISUID | stat.S_ISGID)
        fs.xattrs[p].pop("security.capability", None)

    reset = {}
    program = (tmp_path / "usr/local/sbin/lium-sysbox-layer-reset").read_text().rsplit("\nmain()", 1)[0]
    exec(program, reset)
    reset.update(
        ROOT=str(tmp_path),
        DRY=False,
        JOURNAL=str(tmp_path / "var/lib/lium-sysbox-layer-reset/pending.jsonl"),
        STARTED=str(tmp_path / "run/lium-docker-started"),
    )
    reset["os"] = SimpleNamespace(
        path=os.path,
        walk=os.walk,
        kill=os.kill,
        environ={},
        makedirs=os.makedirs,
        fsync=os.fsync,
        remove=os.remove,
        lstat=lambda p: SimpleNamespace(st_uid=fs.owners[p], st_gid=fs.owners[p], st_mode=fs.modes[p]),
        lchown=lchown,
        chmod=lambda p, m: fs.modes.__setitem__(p, stat.S_IFMT(fs.modes[p]) | m),
        getxattr=getxattr,
        setxattr=lambda p, name, value, **kw: fs.xattrs[p].__setitem__(name, value),
    )
    return reset, fs


@pytest.mark.parametrize("cut_after", [None, 0, 1, 3])
def test_a_reset_cut_off_part_way_is_finished_on_the_next_boot(tmp_path, cut_after):
    # the layer's top directory is shifted last: until it is, the next boot sees the layer as shifted and redoes it
    reset, fs = _fake_fs(tmp_path)
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
    assert fs.owners == {p: 1000 if p.endswith("renterfile") else 0 for p in fs.modes}


@pytest.mark.parametrize("cut", ["after_journal", "after_chown", "after_chmod"])
def test_a_reset_cut_off_between_chown_and_restore_gets_the_setuid_bit_and_capabilities_back(tmp_path, cut):
    reset, fs = _fake_fs(tmp_path)
    sudo = str(fs.upper / "sudo")
    real = SimpleNamespace(**vars(reset["os"]))
    step = {"after_journal": "lchown", "after_chown": "chmod", "after_chmod": "setxattr"}[cut]

    def cut_off(*a, **kw):
        if a[0] == sudo:
            raise InterruptedError()
        return getattr(real, step)(*a, **kw)

    if step == "lchown":
        reset["os"].lchown = cut_off
    elif step == "chmod":
        reset["os"].chmod = cut_off
    else:
        reset["os"].setxattr = lambda p, n, v, **kw: cut_off(p) if n == "security.capability" else real.setxattr(p, n, v, **kw)
    with pytest.raises(InterruptedError):
        reset["main"]()
    reset["os"] = real
    reset["main"]()
    assert fs.owners[sudo] == 0
    assert stat.S_IMODE(fs.modes[sudo]) == 0o4755
    assert fs.xattrs[sudo]["security.capability"] == b"cap-bytes"
    assert not os.path.exists(reset["JOURNAL"])


def test_named_acl_entries_are_shifted_back_also_when_the_owner_already_matches(tmp_path):
    reset, fs = _fake_fs(tmp_path)
    renter, etc = str(fs.upper / "renterfile"), str(fs.upper / "etc")
    acl = _acl((0x01, 6, 0), (0x02, 6, POD_BASE + 1000), (0x04, 4, 0), (0x08, 4, POD_BASE + 1000), (0x10, 6, 0), (0x20, 4, 0))
    fs.xattrs[renter]["system.posix_acl_access"] = acl
    fs.xattrs[etc]["system.posix_acl_default"] = acl
    fs.owners[etc] = 0  # an owner already shifted back by a cut-off run
    reset["main"]()
    want = _acl((0x01, 6, 0), (0x02, 6, 1000), (0x04, 4, 0), (0x08, 4, 1000), (0x10, 6, 0), (0x20, 4, 0))
    assert fs.xattrs[renter]["system.posix_acl_access"] == want
    assert fs.xattrs[etc]["system.posix_acl_default"] == want


def test_a_failing_reset_keeps_docker_down_and_a_docker_restart_never_reruns_it(tmp_path):
    reset, fs = _fake_fs(tmp_path)

    def broken(*a):
        raise OSError(5, "Input/output error")

    reset["os"].lchown = broken
    with pytest.raises(OSError):  # the unit fails, and with it Docker's start (Requires=)
        reset["main"]()
    dropin = (tmp_path / "etc/systemd/system/docker.service.d/lium-sysbox-layer-reset.conf").read_text()
    assert "Requires=lium-sysbox-layer-reset.service" in dropin
    assert "After=lium-sysbox-layer-reset.service" in dropin
    assert "ExecStartPost=/bin/touch /run/lium-docker-started" in dropin
    assert "RemainAfterExit=yes" in (tmp_path / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()
    # once Docker has started this boot, a rerun (Docker restart under live-restore) touches nothing
    reset["os"].lchown = lambda *a: pytest.fail("touched a layer after Docker started")
    os.makedirs(os.path.dirname(reset["STARTED"]), exist_ok=True)
    open(reset["STARTED"], "w").close()
    reset["main"]()


def test_a_data_root_from_dockers_launch_settings_is_used(tmp_path):
    # --data-root=/srv/docker on dockerd's command line, nothing in daemon.json: install records docker info's root
    _install(tmp_path)
    cid = "e" * 64
    _container(tmp_path, cid, "sysbox-runc", "m1")
    os.makedirs(tmp_path / "srv")
    os.rename(tmp_path / "var/lib/docker", tmp_path / "srv/docker")
    out = _run(tmp_path, "--dry-run", env={"LIUM_DOCKER_DATA_ROOT": "/srv/docker"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"would shift 4 entries of {'e' * 12} back from {_base()}"]


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
