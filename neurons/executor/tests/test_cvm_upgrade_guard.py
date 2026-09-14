"""cvm_upgrade_guard.sh on a fake host layout (DAH-3188 prerequisite).

The regressions these tests guard:
- key-provider/run.sh ran `docker compose up --build -d`, and lium-cvm.sh run ran
  `docker-compose up -d`, both of which rebuild the SGX key provider when its image
  is missing. A rebuild changes MRENCLAVE and every CVM data disk on the host becomes
  unreadable at its next boot.
- Nothing looked at other checkouts, other VM directories or stopped CVMs before a
  rebuild, and nothing serialised a rebuild against `lium-cvm.sh new`/`run`.

Docker is a stub on PATH that records every call and keeps images/containers as
files, so the tests assert which docker commands the guard issued. They need
Linux `flock` (util-linux) and bash 4; CI's ubuntu runner has both.
"""

import fcntl
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

DSTACKTEE = Path(__file__).resolve().parents[1] / "dstacktee"
KP_IMAGE = "lium-key-provider:local"
OLD_ID = "sha256:" + "a" * 64
NEW_ID = "sha256:" + "b" * 64
STRAY_ID = "sha256:" + "c" * 64

FAKE_DOCKER = r"""#!/bin/bash
# docker stub: images/<name> holds an image id, containers/<name> holds the image a
# container runs, calls.log records every invocation. `compose up` without --no-build
# exits 99 so a test that triggers an implicit build fails loudly.
S="$FAKE_DOCKER_STATE"
echo "$*" >>"$S/calls.log"
echo "docker $*" >>"$S/seq.log"
resolve() { # name or id -> id
    if [ -f "$S/images/$1" ]; then cat "$S/images/$1"; return 0; fi
    for f in "$S"/images/*; do [ -f "$f" ] && [ "$(cat "$f")" = "$1" ] && { echo "$1"; return 0; }; done
    return 1
}
case "$1 $2" in
"info ")
    [ -f "$S/daemon_down" ] && { echo "Cannot connect to the Docker daemon" >&2; exit 1; }
    echo "Server Version: stub" ;;
"image inspect")
    shift 2; fmt=""; [ "$1" = "--format" ] && { fmt="$2"; shift 2; }
    id="$(resolve "$1")" || exit 1
    [ -n "$fmt" ] && echo "$id" || echo "[]"
    ;;
"inspect --format")
    [ -f "$S/containers/$4" ] || exit 1; cat "$S/containers/$4" ;;
"tag "*)
    id="$(resolve "$2")" || exit 1; echo "$id" >"$S/images/$3" ;;
"compose build")
    [ -f "$S/fail_build" ] && exit 1
    echo "sha256:$(printf 'e%.0s' $(seq 64))" >"$S/images/lium-aesmd:local"
    [ "$3" = "aesmd" ] || cat "$S/next_build_id" >"$S/images/lium-key-provider:local"
    ;;
"compose up")
    case " $* " in *" --no-build "*) ;; *) echo "IMPLICIT BUILD" >&2; exit 99 ;; esac
    id="$(cat "$S/images/lium-key-provider:local" 2>/dev/null)" || { echo "no image" >&2; exit 1; }
    echo "$id" >"$S/containers/dstack-key-provider"
    ;;
"compose logs") echo '{"mr_enclave":"deadbeef"}' ;;
*) echo "docker stub: unhandled $*" >&2; exit 2 ;;
esac
"""

FAKE_PYTHON3 = r"""#!/bin/bash
# python3 stub for lium-cvm.sh: `-c` (manifest reads) goes to the real interpreter,
# everything else (dstack.py) is only recorded.
if [ "$1" = "-c" ]; then exec "$REAL_PYTHON3" "$@"; fi
# Is the host lock held by our caller right now? (a probe from a separate process)
if flock -n "$LIUM_CVM_LOCK_FILE" true 2>/dev/null; then lock=free; else lock=held; fi
echo "$*" >>"$FAKE_DOCKER_STATE/python3.log"
echo "python3 $*" >>"$FAKE_DOCKER_STATE/seq.log"
echo "$lock $*" >>"$FAKE_DOCKER_STATE/lock-at-python3.log"
"""

FAKE_QEMU_IMG = r"""#!/bin/bash
# qemu-img create -f qcow2 <path> <size>
echo "$*" >>"$FAKE_DOCKER_STATE/qemu-img.log"
echo "qemu-img $*" >>"$FAKE_DOCKER_STATE/seq.log"
: >"$4"
"""


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class Host:
    """A fake TDX host: checkouts under a sweep root, state dir, lock, docker stub."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.sweep = tmp / "sweep"
        self.state = tmp / "state"
        self.lock = tmp / "lium-cvm.lock"
        self.docker_state = tmp / "docker"
        for d in (self.docker_state / "images", self.docker_state / "containers"):
            d.mkdir(parents=True)
        (self.docker_state / "next_build_id").write_text(NEW_ID + "\n")
        self.bin = tmp / "bin"
        self.bin.mkdir()
        _write_exe(self.bin / "docker", FAKE_DOCKER)
        _write_exe(self.bin / "python3", FAKE_PYTHON3)
        _write_exe(self.bin / "qemu-img", FAKE_QEMU_IMG)
        self.checkout = self.add_checkout("opt/lium-io")

    def add_checkout(self, rel: str) -> Path:
        """Copy the dstacktee tooling into <sweep>/<rel>/neurons/executor/dstacktee."""
        dest = self.sweep / rel / "neurons" / "executor" / "dstacktee"
        dest.mkdir(parents=True)
        for name in ("lium-cvm.sh", "cvm_upgrade_guard.sh"):
            shutil.copy2(DSTACKTEE / name, dest / name)
        shutil.copytree(DSTACKTEE / "key-provider", dest / "key-provider")
        (dest / "scripts").mkdir()
        (dest / "scripts" / "dstack.py").write_text("# stubbed by the python3 shim\n")
        return dest

    @staticmethod
    def add_cvm(checkout: Path, name: str, *, disk: bool = True, running: bool = False) -> Path:
        vm = checkout / "run" / "vms" / name
        vm.mkdir(parents=True)
        (vm / "vm-manifest.json").write_text('{"disk_size": 20, "image": "x"}')
        if disk:
            (vm / "hda.img").write_bytes(b"\0")
        if running:
            (vm / "runtime.json").write_text('{"cid": 7, "pid": 1}')
        return vm

    def set_image(self, name: str, image_id: str) -> None:
        (self.docker_state / "images" / name).write_text(image_id + "\n")

    def set_container(self, image_id: str) -> None:
        (self.docker_state / "containers" / "dstack-key-provider").write_text(image_id + "\n")

    def pin(self, image_id: str) -> None:
        self.state.mkdir(exist_ok=True)
        (self.state / "key-provider.image").write_text(f"{image_id} 2026-09-14T00:00:00Z\n")

    def pinned(self) -> str:
        return (self.state / "key-provider.image").read_text().split()[0]

    def register(self, vms_dir: Path) -> None:
        self.state.mkdir(exist_ok=True)
        with open(self.state / "vm-dirs", "a") as f:
            f.write(str(vms_dir) + "\n")

    def seq(self) -> list[str]:
        log = self.docker_state / "seq.log"
        return log.read_text().splitlines() if log.exists() else []

    def calls(self) -> list[str]:
        log = self.docker_state / "calls.log"
        return log.read_text().splitlines() if log.exists() else []

    def env(self, lock_wait: int = 5) -> dict:
        env = dict(os.environ)
        env.update(
            PATH=f"{self.bin}:{env['PATH']}",
            FAKE_DOCKER_STATE=str(self.docker_state),
            REAL_PYTHON3=sys.executable,
            LIUM_CVM_STATE_DIR=str(self.state),
            LIUM_CVM_LOCK_FILE=str(self.lock),
            LIUM_CVM_LOCK_WAIT=str(lock_wait),
            LIUM_CVM_SWEEP_ROOTS=str(self.sweep),
        )
        return env

    def guard(self, *args: str, checkout: Path | None = None, lock_wait: int = 5):
        script = (checkout or self.checkout) / "cvm_upgrade_guard.sh"
        return subprocess.run(
            ["bash", str(script), *args],
            env=self.env(lock_wait),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def lium_cvm(self, *args: str, lock_wait: int = 5):
        return subprocess.run(
            ["bash", str(self.checkout / "lium-cvm.sh"), *args],
            cwd=self.checkout,
            env=self.env(lock_wait),
            capture_output=True,
            text=True,
            timeout=60,
        )


@pytest.fixture
def host(tmp_path: Path) -> Host:
    if shutil.which("flock") is None:
        pytest.skip("needs util-linux flock (Linux)")
    return Host(tmp_path)


def _built(calls: list[str]) -> bool:
    return any(
        c.startswith("compose build") and not c.startswith("compose build aesmd") for c in calls
    )


# --- inventory and upgrade -----------------------------------------------------


def test_disk_in_another_checkout_refuses_upgrade(host: Host):
    other = host.add_checkout("home/operator/lium-io-old")
    vm = host.add_cvm(other, "old-executor")
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)

    r = host.guard("upgrade")

    assert r.returncode == 3, r.stdout + r.stderr
    assert str(vm / "hda.img") in r.stdout
    assert f"sudo rm -rf {vm}" in r.stdout
    assert not _built(host.calls()), host.calls()


def test_registered_vm_root_outside_sweep_is_found(host: Host, tmp_path: Path):
    elsewhere = tmp_path / "elsewhere" / "neurons" / "executor" / "dstacktee"
    vm = host.add_cvm(elsewhere, "custom")
    host.register(elsewhere / "run" / "vms")

    r = host.guard("upgrade")

    assert r.returncode == 3, r.stdout + r.stderr
    assert str(vm / "hda.img") in r.stdout
    assert not _built(host.calls())


def test_stopped_and_running_cvms_both_refuse(host: Host):
    stopped = host.add_cvm(host.checkout, "stopped-one")
    running = host.add_cvm(host.checkout, "running-one", running=True)

    r = host.guard("upgrade")

    assert r.returncode == 3, r.stdout + r.stderr
    lines = {
        ln.split()[0]: ln.split()[1]
        for ln in r.stdout.splitlines()
        if ln.startswith("  ") and "hda.img" in ln
    }
    assert lines == {"stopped": str(stopped / "hda.img"), "running": str(running / "hda.img")}
    assert f"sudo {host.checkout}/lium-cvm.sh stop running-one" in r.stdout
    assert "lium-cvm.sh stop stopped-one" not in r.stdout
    assert not _built(host.calls())


def test_created_but_never_run_cvm_does_not_block(host: Host):
    host.add_cvm(host.checkout, "fresh", disk=False)

    r = host.guard("inventory")

    assert r.returncode == 0, r.stdout + r.stderr
    assert "no data disk yet" in r.stdout


def test_swept_disk_without_manifest_is_an_orphan_and_refuses(host: Host):
    # Regression: the sweep skipped any hda.img with no vm-manifest.json beside
    # it, so a disk whose manifest was deleted did not block the upgrade.
    vm = host.sweep / "srv" / "old-cvms" / "lost-manifest"
    vm.mkdir(parents=True)
    (vm / "hda.img").write_bytes(b"\0")
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)

    r = host.guard("inventory")
    assert r.returncode == 3, r.stdout + r.stderr
    listed = [ln.split() for ln in r.stdout.splitlines() if ln.startswith("  ") and "hda.img" in ln]
    assert listed == [["orphan", str(vm / "hda.img")]]
    assert "orphan = hda.img with no vm-manifest.json" in r.stdout
    assert f"sudo rm -rf {vm}" in r.stdout

    r = host.guard("upgrade")
    assert r.returncode == 3, r.stdout + r.stderr
    assert not _built(host.calls()), host.calls()


def test_registered_disk_without_manifest_is_an_orphan(host: Host, tmp_path: Path):
    # The registry walk and the sweep share one predicate: hda.img present.
    vms_dir = tmp_path / "elsewhere" / "neurons" / "executor" / "dstacktee" / "run" / "vms"
    vm = vms_dir / "manifest-gone"
    vm.mkdir(parents=True)
    (vm / "hda.img").write_bytes(b"\0")
    host.register(vms_dir)

    r = host.guard("inventory")

    assert r.returncode == 3, r.stdout + r.stderr
    assert f"  orphan   {vm / 'hda.img'}" in r.stdout
    assert f"sudo rm -rf {vm}" in r.stdout


def test_running_disk_without_manifest_is_running_and_keeps_the_stop_line(host: Host):
    # runtime.json says QEMU runs; `lium-cvm.sh stop` reads it, not the
    # manifest, so the stop line stays.
    vm = host.checkout / "run" / "vms" / "live-no-manifest"
    vm.mkdir(parents=True)
    (vm / "hda.img").write_bytes(b"\0")
    (vm / "runtime.json").write_text('{"cid": 7, "pid": 1}')

    r = host.guard("inventory")

    assert r.returncode == 3, r.stdout + r.stderr
    assert f"  running  {vm / 'hda.img'}" in r.stdout
    assert "orphan" not in r.stdout
    assert f"sudo {host.checkout}/lium-cvm.sh stop live-no-manifest" in r.stdout
    assert f"sudo rm -rf {vm}" in r.stdout


def test_empty_host_upgrade_keeps_old_image_and_pins_new(host: Host):
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)

    r = host.guard("upgrade")

    assert r.returncode == 0, r.stdout + r.stderr
    calls = host.calls()
    assert any(c.startswith(f"tag {OLD_ID} lium-key-provider:pre-upgrade-") for c in calls), calls
    assert "compose build" in calls
    assert "compose up -d --no-build" in calls
    assert host.pinned() == NEW_ID
    assert (host.docker_state / "containers" / "dstack-key-provider").read_text().strip() == NEW_ID


def test_upgrade_rejects_an_unknown_argument(host: Host):
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)

    r = host.guard("upgrade", "--dry_run")

    assert r.returncode == 1, r.stdout + r.stderr
    assert "unknown argument '--dry_run'" in r.stderr
    assert host.calls() == [], host.calls()
    assert host.pinned() == OLD_ID


def test_upgrade_dry_run_changes_nothing(host: Host):
    # No pin on purpose: the adopt step (two `docker tag`s and the pin file) must not
    # run in a dry run either.
    host.set_container(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)

    r = host.guard("upgrade", "--dry-run")

    assert r.returncode == 0, r.stdout + r.stderr
    assert "dry run" in r.stdout
    assert not any(c.startswith(("tag ", "compose")) for c in host.calls()), host.calls()
    assert not (host.state / "key-provider.image").exists()


def test_swept_directory_with_a_space_is_quoted(host: Host):
    other = host.add_checkout("home/op/my checkouts/lium-io")
    vm = host.add_cvm(other, "exec-x")

    r = host.guard("inventory")

    assert r.returncode == 3, r.stdout + r.stderr
    quoted = str(vm).replace(" ", "\\ ")
    assert f"  sudo rm -rf {quoted}" in r.stdout.splitlines()


def test_docker_down_is_a_tool_error_not_a_lost_image(host: Host):
    host.add_cvm(host.checkout, "live")
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)
    (host.docker_state / "daemon_down").touch()

    r = host.guard("start")

    assert r.returncode == 1, r.stdout + r.stderr
    assert "docker is not reachable" in r.stderr
    assert "Recovery" not in r.stderr


def test_unreadable_vm_root_makes_inventory_incomplete(host: Host, tmp_path: Path):
    if os.geteuid() == 0:
        pytest.skip("root reads every directory")
    hidden = tmp_path / "hidden-vms"
    hidden.mkdir()
    host.register(hidden)
    hidden.chmod(0)
    try:
        r = host.guard("upgrade")
    finally:
        hidden.chmod(0o755)

    assert r.returncode == 4, r.stdout + r.stderr
    assert "INCOMPLETE" in r.stdout
    assert str(hidden) in r.stdout
    assert not _built(host.calls())


def test_missing_registered_vm_root_makes_inventory_incomplete(host: Host, tmp_path: Path):
    # A VM root on an unmounted disk looks like a removed checkout; neither may be
    # read as "no disks".
    gone = tmp_path / "unmounted" / "run" / "vms"
    host.register(gone)

    r = host.guard("upgrade")

    assert r.returncode == 4, r.stdout + r.stderr
    assert f"{gone} (registered VM root is missing" in r.stdout
    assert not _built(host.calls())


def test_lock_held_elsewhere_refuses_with_message(host: Host):
    host.lock.touch()
    with open(host.lock) as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        r = host.guard("upgrade", lock_wait=1)
        fcntl.flock(fh, fcntl.LOCK_UN)

    assert r.returncode == 5, r.stdout + r.stderr
    assert str(host.lock) in r.stderr
    assert "waited 1s" in r.stderr
    assert host.calls() == ["info"]  # the preflight only; no image or compose call


# --- start: the pinned image ------------------------------------------------------


def test_start_runs_pinned_image_without_build(host: Host):
    host.add_cvm(host.checkout, "live")
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)

    r = host.guard("start")

    assert r.returncode == 0, r.stdout + r.stderr
    calls = host.calls()
    assert "compose up -d --no-build" in calls
    assert not _built(calls), calls
    assert (host.docker_state / "containers" / "dstack-key-provider").read_text().strip() == OLD_ID


def test_start_retags_pinned_image_when_local_tag_moved(host: Host):
    host.add_cvm(host.checkout, "live")
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, STRAY_ID)  # someone ran `docker compose build` by hand
    host.set_image("lium-key-provider:pinned-" + OLD_ID[7:], OLD_ID)

    r = host.guard("start")

    assert r.returncode == 0, r.stdout + r.stderr
    assert f"tag {OLD_ID} {KP_IMAGE}" in host.calls()
    assert not _built(host.calls())
    assert (host.docker_state / "containers" / "dstack-key-provider").read_text().strip() == OLD_ID


def test_start_refuses_when_pinned_image_missing_and_disks_exist(host: Host):
    host.add_cvm(host.checkout, "live")
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, STRAY_ID)  # the pinned id exists nowhere

    r = host.guard("start")

    assert r.returncode == 6, r.stdout + r.stderr
    assert OLD_ID in r.stderr
    assert "Recovery" in r.stderr
    assert not _built(host.calls())
    assert not any(c.startswith("compose up") for c in host.calls())


def test_start_refuses_when_pinned_image_missing_on_empty_host_and_points_to_upgrade(host: Host):
    host.pin(OLD_ID)  # image pruned, no CVM left: start never builds over a pin

    r = host.guard("start")

    assert r.returncode == 6, r.stdout + r.stderr
    assert "No CVM disk exists" in r.stderr
    assert "upgrade' to rebuild" in r.stderr
    assert "docker load" not in r.stderr
    assert not _built(host.calls())


def test_start_adopts_running_containers_on_host_without_pin(host: Host):
    host.add_cvm(host.checkout, "live")
    host.set_container(OLD_ID)
    host.set_image("key-provider-gramine-sealing-key-provider", OLD_ID)
    aesmd_id = "sha256:" + "f" * 64
    (host.docker_state / "containers" / "dstack-aesmd").write_text(aesmd_id + "\n")
    host.set_image("key-provider-aesmd", aesmd_id)

    r = host.guard("start")

    assert r.returncode == 0, r.stdout + r.stderr
    assert host.pinned() == OLD_ID
    assert f"tag {OLD_ID} {KP_IMAGE}" in host.calls()
    assert f"tag {aesmd_id} lium-aesmd:local" in host.calls()
    assert not any(c.startswith("compose build") for c in host.calls()), host.calls()


def test_start_refuses_build_when_disks_exist_and_nothing_to_adopt(host: Host):
    host.add_cvm(host.checkout, "live")

    r = host.guard("start")

    assert r.returncode == 6, r.stdout + r.stderr
    assert not _built(host.calls())


def test_start_builds_once_on_empty_host(host: Host):
    r = host.guard("start")
    r2 = host.guard("start")

    assert r.returncode == 0 and r2.returncode == 0, r.stdout + r.stderr + r2.stdout + r2.stderr
    assert host.calls().count("compose build") == 1, host.calls()
    assert host.pinned() == NEW_ID


def test_run_sh_starts_through_the_guard(host: Host):
    host.add_cvm(host.checkout, "live")
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)

    r = subprocess.run(
        ["bash", str(host.checkout / "key-provider" / "run.sh")],
        cwd=host.checkout / "key-provider",
        env=host.env(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert r.returncode == 0, r.stdout + r.stderr
    assert "compose up -d --no-build" in host.calls()
    assert not _built(host.calls())
    assert "Services started!" in r.stdout


def test_lock_subcommand_holds_the_lock_while_the_command_runs(host: Host):
    probe = host.bin / "lockprobe"
    _write_exe(probe, f'#!/bin/bash\nflock -n "{host.lock}" true && echo free || echo held\n')

    r = host.guard("lock", "--", str(probe))

    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip() == "held"


# --- lium-cvm.sh routes through the guard ------------------------------------------


def test_lium_cvm_run_takes_lock_registers_root_and_creates_disk_first(host: Host):
    vm = host.add_cvm(host.checkout, "exec-1", disk=False)
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)

    r = host.lium_cvm("run", "exec-1")

    assert r.returncode == 0, r.stdout + r.stderr
    vms_dir = str(host.checkout / "run" / "vms")
    assert vms_dir in (host.state / "vm-dirs").read_text().splitlines()
    assert "compose up -d --no-build" in host.calls()
    assert not _built(host.calls())
    assert (
        host.docker_state / "qemu-img.log"
    ).read_text().strip() == f"create -f qcow2 {vm}/hda.img 20G"
    assert (vm / "hda.img").exists()
    # Order: provider up, then the disk, then dstack.py (which inherits no lock).
    seq = host.seq()
    up = seq.index("docker compose up -d --no-build")
    disk = next(i for i, s in enumerate(seq) if s.startswith("qemu-img create"))
    run = next(
        i for i, s in enumerate(seq) if s.startswith("python3 ") and f"dstack.py run {vm}" in s
    )
    assert up < disk < run, seq
    # dstack.py run starts QEMU, which must not inherit the host lock.
    probes = (host.docker_state / "lock-at-python3.log").read_text().splitlines()
    assert any(p.startswith("free ") and "dstack.py run" in p for p in probes), probes


def test_lium_cvm_run_propagates_the_guard_exit_code(host: Host):
    host.add_cvm(host.checkout, "exec-1")
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, STRAY_ID)  # pinned image gone, a disk exists

    r = host.lium_cvm("run", "exec-1")

    assert r.returncode == 6, r.stdout + r.stderr
    assert "cvm_upgrade_guard.sh exit 6" in r.stdout
    # The recovery text names the guard, not the sourcing lium-cvm.sh.
    assert f"'{host.checkout}/cvm_upgrade_guard.sh inventory'" in r.stderr
    assert "lium-cvm.sh upgrade" not in r.stdout + r.stderr
    assert not (host.docker_state / "python3.log").exists()


def test_symlinked_sweep_root_is_swept(host: Host, tmp_path: Path):
    real = tmp_path / "mnt" / "nvme0"
    real.mkdir(parents=True)
    other = host.add_checkout("data-real/lium-io")  # under host.sweep for the copy
    vm = host.add_cvm(other, "exec-nvme")
    shutil.move(str(host.sweep / "data-real"), str(real / "lium"))
    link = tmp_path / "data"
    link.symlink_to(real)
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)
    env = host.env()
    env["LIUM_CVM_SWEEP_ROOTS"] = str(link)

    r = subprocess.run(
        ["bash", str(host.checkout / "cvm_upgrade_guard.sh"), "upgrade"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert r.returncode == 3, r.stdout + r.stderr
    # Reported through the symlink, the path the operator knows.
    via_link = link / "lium" / vm.relative_to(host.sweep / "data-real")
    assert str(via_link / "hda.img") in r.stdout
    assert not _built(host.calls()), host.calls()


def test_submounts_lists_real_filesystems_under_the_root_only(host: Host, tmp_path: Path):
    mounts = tmp_path / "mounts"
    root = host.sweep / "opt"
    mounts.write_text(
        "\n".join(
            [
                "/dev/nvme1n1 " + str(root / "data") + " ext4 rw 0 0",
                "/dev/nvme2n1 " + str(root / "with\\040space") + " xfs rw 0 0",
                "proc " + str(root / "proc") + " proc rw 0 0",
                "tmpfs " + str(root / "shm") + " tmpfs rw 0 0",
                "/dev/nvme3n1 " + str(host.sweep / "home" / "data") + " ext4 rw 0 0",
                "/dev/nvme4n1 " + str(root) + "-other ext4 rw 0 0",
                "",
            ]
        )
    )
    env = host.env()
    env["LIUM_CVM_MOUNTS_FILE"] = str(mounts)

    r = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; cvm_guard_submounts "$2"',
            "_",
            str(host.checkout / "cvm_upgrade_guard.sh"),
            str(root),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        str(root / "data"),
        str(root / "with space"),
        str(root / "shm"),
    ]


def test_lium_cvm_run_refuses_while_upgrade_holds_lock(host: Host):
    host.add_cvm(host.checkout, "exec-1", disk=False)
    host.pin(OLD_ID)
    host.set_image(KP_IMAGE, OLD_ID)
    host.lock.touch()
    with open(host.lock) as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        r = host.lium_cvm("run", "exec-1", lock_wait=1)
        fcntl.flock(fh, fcntl.LOCK_UN)

    assert r.returncode == 5, r.stdout + r.stderr
    assert str(host.lock) in r.stderr
    assert not (host.docker_state / "python3.log").exists()


def test_lium_cvm_new_registers_vm_root_under_lock(host: Host):
    (host.checkout / ".env").write_text(
        "SSH_PUBLIC_PORT=2200\nSSH_PORT=22\nEXTERNAL_PORT=8000\nRENTING_PORT_RANGE=9001\n"
        "EXECUTOR_RUNNER_IMAGE_DIGEST=sha256:" + "d" * 64 + "\n"
    )
    (host.checkout / "run" / "images" / "dstack-nvidia-0.5.11").mkdir(parents=True)
    (host.checkout / "run" / "images" / "dstack-nvidia-0.5.11" / "metadata.json").write_text("{}")
    (host.checkout / "app").mkdir()
    (host.checkout / "app" / "docker-compose.yml").write_text("services: {}\n")
    (host.checkout / "app" / "init_script.sh").write_text("")
    (host.checkout / "app" / "pre_launch_script.sh").write_text("")
    # `curl -s ifconfig.me` after creation must not reach the network.
    _write_exe(host.bin / "curl", "#!/bin/bash\necho 203.0.113.10\n")

    r = host.lium_cvm("new", "exec-2")

    assert r.returncode == 0, r.stdout + r.stderr
    assert str(host.checkout / "run" / "vms") in (host.state / "vm-dirs").read_text().splitlines()
    assert "dstack.py new" in (host.docker_state / "python3.log").read_text()
    probes = (host.docker_state / "lock-at-python3.log").read_text().splitlines()
    assert any(p.startswith("held ") and "dstack.py new" in p for p in probes), probes
