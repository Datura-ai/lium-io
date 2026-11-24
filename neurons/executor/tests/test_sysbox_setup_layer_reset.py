"""nvidia_docker_sysbox_setup.sh: after an unclean stop, a boot oneshot shifts each sysbox pod's writable layer back."""

import json
import os
import re
import shutil
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


def _layer_reset_version() -> int:
    return int(re.search(r"^LAYER_RESET_VERSION=(\d+)$", open(SCRIPT).read(), re.M).group(1))


def _install(root) -> None:
    script = (
        'host_path() { echo "${SYSBOX_SETUP_HOST_ROOT:-}$1"; }\nwarn() { echo "WARN $*"; }\n'
        + f"LAYER_RESET_VERSION={_layer_reset_version()}\n"
        + _function("docker_containerd_address")
        + _function("systemd_quote")
        + _function("systemd_env")
        + _function("shift_sysbox_layers_back_at_boot")
        + "shift_sysbox_layers_back_at_boot\n"
    )
    return subprocess.run(
        ["bash", "-c", script], check=True, capture_output=True, text=True,
        env={**os.environ, "SYSBOX_SETUP_HOST_ROOT": str(root)},
    ).stdout


def _container(root, cid: str, runtime: str, mount: str) -> str:
    _subids(root)
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


def _unit_env(unit: str, name: str) -> str:
    # the value systemd reads from a quoted Environment= line
    quoted = re.search(rf'^Environment="{name}=((?:[^"\\]|\\.)*)"$', unit, re.M).group(1)
    return re.sub(r"\\(.)", r"\1", quoted).replace("%%", "%")


def _subids(root, uid_base=None, gid_base=None) -> None:
    # sysbox's host range, the way its installer writes it
    (root / "etc").mkdir(parents=True, exist_ok=True)
    uid = uid_base if uid_base is not None else _base()
    gid = gid_base if gid_base is not None else (POD_BASE if os.getuid() == 0 else os.getgid())
    (root / "etc/subuid").write_text(f"ubuntu:100000:65536\nsysbox:{uid}:65536\n")
    (root / "etc/subgid").write_text(f"ubuntu:100000:65536\nsysbox:{gid}:65536\n")


def _run(root, *args: str, path: str = "", env: dict | None = None) -> subprocess.CompletedProcess:
    unit = (root / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()
    address = _unit_env(unit, "LIUM_CONTAINERD_ADDRESS")
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


def test_the_unit_carries_the_version_the_validator_compares(tmp_path):
    # DAH-6804: lium-io machine_scrape reads this line; 1 is reserved for units written before the marker
    _install(tmp_path)
    unit = (tmp_path / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()
    assert _layer_reset_version() >= 2
    assert unit.splitlines()[0] == f"# lium-layer-reset-version={_layer_reset_version()}"


def _install_on_a_live_host(root, enable_fails: bool) -> subprocess.CompletedProcess:
    # the live-host path (no SYSBOX_SETUP_HOST_ROOT), its files under `root`; systemctl and docker are stand-ins
    log = root / "systemctl.log"
    script = (
        f'host_path() {{ echo "{root}$1"; }}\nwarn() {{ echo "WARN $*"; }}\n'
        + f'systemctl() {{ echo "$*" >> "{log}"; [ "$1" = enable ] && {"return 1" if enable_fails else "return 0"}; [ "$1" = is-active ] && return 1; return 0; }}\n'
        + 'docker() { return 1; }\n'
        + f"LAYER_RESET_VERSION={_layer_reset_version()}\n"
        + _function("docker_containerd_address")
        + _function("systemd_quote")
        + _function("systemd_env")
        + _function("shift_sysbox_layers_back_at_boot")
        + "shift_sysbox_layers_back_at_boot\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "SYSBOX_SETUP_HOST_ROOT"}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)


def test_a_failed_enable_leaves_no_docker_dependency_and_no_new_version(tmp_path):
    # lio#2373 review: a failed install must not leave Docker requiring an unit that is not enabled, nor the new
    # version line, so the validator's next check installs again
    dropin = tmp_path / "etc/systemd/system/docker.service.d/lium-sysbox-layer-reset.conf"
    dropin.parent.mkdir(parents=True)
    dropin.write_text("[Unit]\nRequires=lium-sysbox-layer-reset.service\n")  # an older install's drop-in

    result = _install_on_a_live_host(tmp_path, enable_fails=True)

    assert result.returncode != 0
    assert not dropin.exists()
    unit = (tmp_path / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()
    assert not unit.startswith("# lium-layer-reset-version=")


def test_once_enable_works_the_next_install_adds_the_dependency_and_the_version(tmp_path):
    _install_on_a_live_host(tmp_path, enable_fails=True)

    result = _install_on_a_live_host(tmp_path, enable_fails=False)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "etc/systemd/system/docker.service.d/lium-sysbox-layer-reset.conf").read_text().startswith(
        "[Unit]\nRequires=lium-sysbox-layer-reset.service\n"
    )
    unit = (tmp_path / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()
    assert unit.splitlines()[0] == f"# lium-layer-reset-version={_layer_reset_version()}"
    assert unit.count("# lium-layer-reset-version=") == 1


@pytest.mark.skipif(os.getuid() == 0, reason="the not-root answer needs a non-root run")
def test_install_layer_reset_mode_answers_one_line_and_installs_nothing_without_root(tmp_path):
    result = subprocess.run(
        ["bash", SCRIPT, "--install-layer-reset"], capture_output=True, text=True, timeout=60,
        env={**os.environ, "SYSBOX_SETUP_HOST_ROOT": str(tmp_path)},
    )
    assert (result.returncode, result.stdout) == (0, "layer_reset_skipped not_root\n")
    assert not any(tmp_path.iterdir())


def test_install_layer_reset_mode_never_runs_the_reset_or_restarts_docker():
    # the validator runs it under rented pods: install only (unit + drop-in), as the setup script's own install does
    branch = re.search(r"^    --install-layer-reset\)\n(.*?)^        ;;$", open(SCRIPT).read(), re.S | re.M).group(1)
    assert "shift_sysbox_layers_back_at_boot" in branch
    assert not re.search(r"systemctl (re)?start|systemctl restart|/usr/local/sbin/lium-sysbox-layer-reset|--dry-run", branch)


def test_a_shifted_sysbox_layer_is_found_and_a_runc_container_is_left_alone(tmp_path):
    # the test user's uid stands in for the pod's uid base (as root: POD_BASE)
    _install(tmp_path)
    _container(tmp_path, "a" * 64, "sysbox-runc", "m1")
    _container(tmp_path, "b" * 64, "runc", "m2")
    out = _run(tmp_path, "--dry-run")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"would shift 4 entries of {'a' * 12} back from {_base()}"]


SNAPSHOT = "/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/7/fs"


def _containerd_store(tmp_path, cid: str, snapshot: str = SNAPSHOT) -> None:
    upper = Path(_container(tmp_path, cid, "sysbox-runc", "m1"))
    (tmp_path / "var/lib/docker/image/overlay2/layerdb/mounts" / cid / "mount-id").unlink()
    snap = tmp_path / snapshot.lstrip("/")
    snap.parent.mkdir(parents=True)
    upper.rename(snap)


def _fake_ctr(tmp_path, cid: str, address: str, snapshot: str = SNAPSHOT, failure: str = "") -> str:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "ctr").write_text(
        "#!/bin/sh\n"
        f'[ "$*" = "--address {tmp_path}{address} -n moby snapshots --snapshotter overlayfs mounts / {cid}" ] || exit 1\n'
        + (f'echo "{failure}" >&2; exit 1\n' if failure else "")
        + f'echo "mount -t overlay overlay / -o index=off,workdir=/x/7/work,upperdir={snapshot},lowerdir=/x/6/fs"\n'
    )
    (bindir / "ctr").chmod(0o755)
    return f"{bindir}:"


@pytest.mark.parametrize("flag", ["--config-file=/srv/docker.json", "--config-file /srv/docker.json"])
def test_the_containerd_address_comes_from_the_daemon_json_dockerd_is_given(tmp_path, flag):
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv/docker.json").write_text(json.dumps({"containerd": "/run/lium-containerd/containerd.sock"}))
    (tmp_path / "etc/docker").mkdir(parents=True)
    (tmp_path / "etc/docker/daemon.json").write_text("{}")
    script = (
        f'host_path() {{ echo "{tmp_path}$1"; }}\n'
        f'systemctl() {{ echo "{{ path=/usr/bin/dockerd ; argv[]=/usr/bin/dockerd -H fd:// {flag} ; }}"; }}\n'
        + _function("docker_containerd_address")
        + "docker_containerd_address\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "SYSBOX_SETUP_HOST_ROOT"}
    out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True, env=env).stdout
    assert out.strip() == "/run/lium-containerd/containerd.sock"


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


def _containerd_host(tmp_path, cid: str, snapshot: str = SNAPSHOT) -> None:
    address = "/run/containerd/containerd.sock"
    (tmp_path / "etc/docker").mkdir(parents=True)
    (tmp_path / "etc/docker/daemon.json").write_text(json.dumps({"containerd": address}))
    (tmp_path / address.lstrip("/")).parent.mkdir(parents=True)
    (tmp_path / address.lstrip("/")).touch()
    _install(tmp_path)
    _containerd_store(tmp_path, cid, snapshot)


def test_a_snapshot_path_with_a_space_is_read_whole(tmp_path):
    cid = "c" * 64
    snapshot = "/srv/docker data/containerd/snapshots/7/fs"
    _containerd_host(tmp_path, cid, snapshot)
    out = _run(tmp_path, "--dry-run", path=_fake_ctr(tmp_path, cid, "/run/containerd/containerd.sock", snapshot))
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"would shift 4 entries of {'c' * 12} back from {_base()}"]


def test_a_snapshot_containerd_says_is_missing_is_skipped(tmp_path):
    cid = "c" * 64
    _containerd_host(tmp_path, cid)
    path = _fake_ctr(tmp_path, cid, "/run/containerd/containerd.sock", failure=f'ctr: snapshot "{cid}": not found')
    out = _run(tmp_path, "--dry-run", path=path)
    assert out.returncode == 0, out.stderr
    assert "no writable layer found" in out.stderr


@pytest.mark.parametrize("failure", ["ctr: failed to dial: connection refused", None])
def test_a_snapshot_lookup_that_fails_fails_the_reset(tmp_path, failure):
    # nonzero exit for another reason, or ctr not runnable (the same branch a timeout takes): Docker must stay down
    cid = "c" * 64
    _containerd_host(tmp_path, cid)
    path = _fake_ctr(tmp_path, cid, "/run/containerd/containerd.sock", failure=failure or "")
    if failure is None:
        (tmp_path / "bin/ctr").write_text("not an executable format\n")
    out = _run(tmp_path, "--dry-run", path=path)
    assert out.returncode != 0
    assert "ctr could not read the snapshot" in out.stderr


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
    _subids(tmp_path, POD_BASE, POD_BASE)
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
        scandir=os.scandir,
        stat=os.stat,
        kill=os.kill,
        environ={},
        makedirs=os.makedirs,
        fsync=os.fsync,
        open=os.open,
        close=os.close,
        O_RDONLY=os.O_RDONLY,
        sync=lambda: None,
        remove=os.remove,
        lstat=lambda p: SimpleNamespace(st_uid=fs.owners[p], st_gid=fs.owners[p], st_mode=fs.modes[p]),
        lchown=lchown,
        chmod=lambda p, m: fs.modes.__setitem__(p, stat.S_IFMT(fs.modes[p]) | m),
        getxattr=getxattr,
        setxattr=lambda p, name, value, **kw: fs.xattrs[p].__setitem__(name, value),
    )
    return reset, fs


@pytest.mark.parametrize("batch", [512, 1])
@pytest.mark.parametrize("cut_after", [None, 0, 1, 3])
def test_a_reset_cut_off_part_way_is_finished_on_the_next_boot(tmp_path, cut_after, batch):
    # the layer's top directory is shifted last: until it is, the next boot sees the layer as shifted and redoes it
    reset, fs = _fake_fs(tmp_path)
    reset["BATCH"] = batch
    real = reset["os"].lchown
    calls = 0

    def interrupted(*a):
        nonlocal calls
        if calls == cut_after:
            raise InterruptedError()
        calls += 1
        return real(*a)

    reset["os"].lchown = interrupted
    if cut_after is None:
        reset["main"]()
    else:
        with pytest.raises(InterruptedError):
            reset["main"]()
    reset["os"].lchown = real
    reset["main"]()
    assert fs.owners == {p: 1000 if p.endswith("renterfile") else 0 for p in fs.modes}
    assert stat.S_IMODE(fs.modes[str(fs.upper / "sudo")]) == 0o4755


def test_the_journal_holds_one_batch_at_most(tmp_path):
    # a layer full of setuid files never grows the journal past one batch, and each batch costs one fsync
    reset, fs = _fake_fs(tmp_path)
    for i in range(7):
        f = fs.upper / f"suid{i}"
        f.touch()
        fs.modes[str(f)], fs.owners[str(f)], fs.xattrs[str(f)] = stat.S_IFREG | 0o4755, POD_BASE, {}
    reset["BATCH"] = 3
    written = []
    real = reset["journal"]
    reset["journal"] = lambda entries: (written.append(len(entries)), real(entries))
    reset["main"]()
    assert max(written) <= 3 and sum(written) == 8  # the 7 files and sudo
    assert all(stat.S_IMODE(fs.modes[str(fs.upper / f"suid{i}")]) == 0o4755 for i in range(7))
    assert not os.path.exists(reset["JOURNAL"])


def test_a_renter_path_the_host_cannot_reach_is_skipped_and_the_rest_is_shifted(tmp_path):
    # a path inside the pod below PATH_MAX can pass it on the host, after the upperdir prefix: ENAMETOOLONG
    reset, fs = _fake_fs(tmp_path)
    deep = str(fs.upper / "renterfile")
    real = reset["os"].lstat

    def lstat(p):
        if p == deep:
            raise OSError(36, "File name too long", p)
        return real(p)

    reset["os"].lstat = lstat
    reset["main"]()
    assert fs.owners[deep] == POD_BASE + 1000  # left as it is
    assert fs.owners[str(fs.upper / "sudo")] == 0
    assert fs.owners[str(fs.upper)] == 0  # the layer is done: the unit succeeds and Docker starts


def test_an_io_error_on_an_entry_fails_the_reset_and_leaves_the_layer_for_the_next_boot(tmp_path):
    reset, fs = _fake_fs(tmp_path)
    real = reset["os"].lstat

    def lstat(p):
        if p.endswith("renterfile"):
            raise OSError(5, "Input/output error", p)
        return real(p)

    reset["os"].lstat = lstat
    with pytest.raises(OSError):
        reset["main"]()
    assert fs.owners[str(fs.upper)] == POD_BASE  # not marked done: the unit fails, Docker stays down, next boot retries


def test_a_tree_deeper_than_the_recursion_limit_is_walked(tmp_path):
    # Python 3.10's os.walk recursed once per level: a renter's deep tree ended the reset in RecursionError
    reset, fs = _fake_fs(tmp_path)
    deep = fs.upper
    for _ in range(300):
        deep = deep / "a"
        deep.mkdir()
        fs.modes[str(deep)], fs.owners[str(deep)], fs.xattrs[str(deep)] = stat.S_IFDIR | 0o755, POD_BASE, {}
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        reset["main"]()
    finally:
        sys.setrecursionlimit(limit)
    assert fs.owners[str(deep)] == 0
    assert fs.owners[str(fs.upper)] == 0


@pytest.mark.parametrize(
    ("top", "renterfile", "top_after", "renterfile_after"),
    [
        (POD_BASE, POD_BASE + 1000, 0, 1000),  # shifted, / owned by root in the pod
        (POD_BASE + 1000, POD_BASE + 1000, 1000, 1000),  # shifted, the renter ran `chown 1000 /`
        (0, 1000, 0, 1000),  # unshifted, / owned by root
        (1000, 1000, 1000, 1000),  # unshifted (a clean stop), / owned by 1000 in the pod
    ],
)
def test_the_pod_slice_comes_from_sysboxs_range_not_from_the_tops_owner(tmp_path, top, renterfile, top_after, renterfile_after):
    reset, fs = _fake_fs(tmp_path)
    shifted = top >= POD_BASE
    for p in fs.owners:
        fs.owners[p] = POD_BASE if shifted else 0
    fs.owners[str(fs.upper)] = top
    fs.owners[str(fs.upper / "renterfile")] = renterfile
    reset["main"]()
    assert fs.owners[str(fs.upper)] == top_after
    assert fs.owners[str(fs.upper / "renterfile")] == renterfile_after
    assert fs.owners[str(fs.upper / "sudo")] == 0


@pytest.mark.parametrize("name", ["security.capability", "system.posix_acl_access"])
def test_an_xattr_that_cannot_be_read_stops_the_reset_before_any_chown(tmp_path, name):
    reset, fs = _fake_fs(tmp_path)
    sudo = str(fs.upper / "sudo")
    real = reset["os"].getxattr

    def getxattr(p, n, **kw):
        if p == sudo and n == name:
            raise OSError(5, "Input/output error", p)
        return real(p, n, **kw)

    reset["os"].getxattr = getxattr
    with pytest.raises(OSError):
        reset["main"]()
    assert fs.xattrs[sudo]["security.capability"] == b"cap-bytes"  # never dropped by a chown
    assert fs.owners[str(fs.upper)] == POD_BASE  # the next boot retries the layer


@pytest.mark.parametrize("value", ["/srv/docker data", 'a"b\\c%d', "/var/lib/docker"])
def test_the_unit_environment_carries_a_data_root_with_spaces_unchanged(value):
    line = subprocess.run(
        ["bash", "-c", _function("systemd_quote") + _function("systemd_env") + 'systemd_env LIUM_DOCKER_DATA_ROOT "$1"', "-", value],
        check=True, capture_output=True, text=True,
    ).stdout
    assert _unit_env(line, "LIUM_DOCKER_DATA_ROOT") == value


def test_a_replay_that_cannot_restore_keeps_the_journal_and_fails(tmp_path):
    reset, fs = _fake_fs(tmp_path)
    sudo = str(fs.upper / "sudo")
    reset["journal"]([{"p": sudo, "m": 0o4755, "c": None}])

    def eio(p, m):
        raise OSError(5, "Input/output error")

    reset["os"].chmod = eio
    with pytest.raises(OSError):
        reset["main"]()
    assert os.path.exists(reset["JOURNAL"])  # the unit fails, Docker stays down, the next boot retries


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


@pytest.mark.parametrize("root", ["/srv/docker", "/srv/docker[1]", "/srv/docker data"])
def test_a_data_root_from_dockers_launch_settings_is_used(tmp_path, root):
    # --data-root on dockerd's command line, nothing in daemon.json: install records docker info's root; glob
    # characters and spaces in it are taken literally
    _install(tmp_path)
    cid = "e" * 64
    _container(tmp_path, cid, "sysbox-runc", "m1")
    os.makedirs(tmp_path / "srv")
    os.rename(tmp_path / "var/lib/docker", tmp_path / root.lstrip("/"))
    out = _run(tmp_path, "--dry-run", env={"LIUM_DOCKER_DATA_ROOT": root})
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


@pytest.mark.parametrize(
    "unreadable",
    [
        "var/lib/docker/containers/{cid}/hostconfig.json",
        "var/lib/docker/image/overlay2/layerdb/mounts/{cid}/mount-id",
        "etc/subuid",
        "etc/subgid",
        "etc/docker/daemon.json",
    ],
)
def test_a_discovery_input_that_cannot_be_read_fails_the_reset(tmp_path, unreadable):
    # review 4601860269: a read error (EIO here stands as a directory in the file's place) used to skip the pod or the
    # whole reset and report success, so Docker started the pod with its layer still shifted
    _install(tmp_path)
    cid = "f" * 64
    _container(tmp_path, cid, "sysbox-runc", "m1")
    path = tmp_path / unreadable.format(cid=cid)
    if path.exists():
        path.unlink()
    path.mkdir(parents=True)

    out = _run(tmp_path, "--dry-run")

    assert out.returncode != 0
    assert "would shift" not in out.stdout


@pytest.mark.parametrize(
    ("listing", "layer", "fails", "shifted"),
    [
        ("readable", "readable", False, True),
        ("missing", "readable", False, False),
        ("eio", "readable", True, False),
        ("readable", "missing", False, False),
        ("readable", "eio", True, False),
    ],
)
def test_listing_the_containers_and_finding_the_layer_fail_only_on_a_read_error(tmp_path, listing, layer, fails, shifted):
    # review 4602552409: glob() and os.path.isdir() turned EIO into "no containers" or "no layer" and the reset succeeded;
    # a file in a directory's place stands for EIO (a read error other than "not there")
    _install(tmp_path)
    cid = "a" * 64
    _container(tmp_path, cid, "sysbox-runc", "m1")
    docker = tmp_path / "var/lib/docker"
    if listing != "readable":
        shutil.rmtree(docker / "containers")
        if listing == "eio":
            (docker / "containers").write_text("")
    if layer != "readable":
        shutil.rmtree(docker / "overlay2/m1")
        if layer == "eio":
            (docker / "overlay2/m1").write_text("")

    out = _run(tmp_path, "--dry-run")

    assert (out.returncode != 0) == fails, out.stderr
    assert ("would shift" in out.stdout) == shifted


def test_inputs_that_are_not_there_are_still_skipped(tmp_path):
    _install(tmp_path)
    _container(tmp_path, "a" * 64, "sysbox-runc", "m1")
    os.makedirs(tmp_path / "var/lib/docker/containers" / ("c" * 64))  # a container dir without its hostconfig yet

    out = _run(tmp_path, "--dry-run")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"would shift 4 entries of {'a' * 12} back from {_base()}"]

    (tmp_path / "etc/subuid").unlink()
    out = _run(tmp_path, "--dry-run")
    assert out.returncode == 0 and "nothing to reset" in out.stdout


@pytest.mark.parametrize(("data_root", "expected"), [(None, "/var/lib/docker"), ("/srv/docker data%1", "/srv/docker data%1")])
def test_the_reset_waits_for_the_disk_dockers_data_root_is_on(tmp_path, data_root, expected):
    # review 4601860262: a nofail data-disk mount is not part of local-fs.target; the reset scanned an empty
    # /var/lib/docker, succeeded, and Docker restarted the pods once the disk mounted
    if data_root is not None:
        (tmp_path / "etc/docker").mkdir(parents=True)
        (tmp_path / "etc/docker/daemon.json").write_text(json.dumps({"data-root": data_root}))
    _install(tmp_path)
    unit = (tmp_path / "etc/systemd/system/lium-sysbox-layer-reset.service").read_text()

    quoted = re.search(r'^RequiresMountsFor="((?:[^"\\]|\\.)*)"$', unit, re.M).group(1)
    assert re.sub(r"\\(.)", r"\1", quoted).replace("%%", "%") == expected
    assert unit.index("RequiresMountsFor=") < unit.index("[Service]")
