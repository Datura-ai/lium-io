"""setup_disk_reserve.sh: `losetup -f` prints "/dev/loopN (lost)" when the free loop device has no
node in the container's /dev; the path handed to mknod/losetup must be "/dev/loopN" (GB10 tester
log: mknod: invalid minor device number '23 (lost)', executor ran without the reserve).

setup_backing_fs and ensure_loop_node are sourced from the script with the host tools stubbed as
bash functions that record their arguments.
"""

import os
import subprocess

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "setup_disk_reserve.sh"))

HARNESS = r"""
set -u
src=$(sed -n '/^log()/p; /^ensure_loop_node()/,/^}/p; /^setup_backing_fs()/,/^}/p' "$SCRIPT")
eval "$src"
SIZE_MB=1; RESERVE_DIR="$WORK"; IMG="$WORK/reserve.img"; MARKER="$WORK/.initialized"; MNT="$WORK/mnt"
CALLS="$WORK/calls"
fallocate() { :; }
losetup() {
    case "$1" in
        -j) return 0 ;;
        -f) echo "$LOSETUP_F_OUT" ;;
        *) echo "losetup $*" >>"$CALLS" ;;
    esac
}
mknod() { echo "mknod $*" >>"$CALLS"; }
blkid() { echo ext4; }
mountpoint() { return 0; }
setup_backing_fs
echo "rc=$?"
"""


def _run(tmp_path, losetup_f_out):
    env = dict(os.environ, SCRIPT=SCRIPT, WORK=str(tmp_path), LOSETUP_F_OUT=losetup_f_out)
    out = subprocess.run(["bash", "-c", HARNESS], env=env, capture_output=True, text=True, timeout=30)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return out.stdout, calls


def test_lost_suffix_is_stripped_before_mknod_and_attach(tmp_path):
    stdout, calls = _run(tmp_path, "/dev/loop23 (lost)")
    assert "rc=0" in stdout, stdout
    assert "mknod /dev/loop23 b 7 23" in calls
    assert f"losetup /dev/loop23 {tmp_path}/reserve.img" in calls


def test_plain_losetup_output_unchanged(tmp_path):
    stdout, calls = _run(tmp_path, "/dev/loop987654")
    assert "rc=0" in stdout, stdout
    assert "mknod /dev/loop987654 b 7 987654" in calls
    assert f"losetup /dev/loop987654 {tmp_path}/reserve.img" in calls


def test_garbage_output_still_refused(tmp_path):
    stdout, calls = _run(tmp_path, "losetup: cannot find an unused loop device")
    assert "rc=1" in stdout, stdout
    assert calls == []
