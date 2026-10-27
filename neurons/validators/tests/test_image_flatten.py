import asyncio
import json
import os
import subprocess
import time
from unittest.mock import AsyncMock, Mock

import pytest

from services import image_flatten
from services.image_flatten import (
    FLATTEN_DISK_MARGIN_BYTES,
    FLATTEN_LEASE_DIR,
    FLATTEN_LOCK_PATH,
    FLATTEN_USE_LOCK_PATH,
    FLATTEN_NO_DISK_EXIT,
    ImageFlattenNoDisk,
    _flatten_command,
    flatten_image_on_host,
    image_flatten_changes,
)

_IMAGE_ID = "sha256:" + "ab" * 32
_TAG = f"lium-flattened:{'ab' * 32}"
_CONFIG = {
    "Entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"],
    "Cmd": ["python3", "-m", "sglang.launch_server"],
    "Env": ["PATH=/usr/local/bin:/usr/bin", 'PS1=$ "x"\\', "EMPTY="],
    "WorkingDir": "/sgl-workspace",
    "User": "root",
    "ExposedPorts": {"30000/tcp": {}},
    "Volumes": {"/data": {}},
    "Labels": {"org.opencontainers.image.version": "24.04", "lium.volume_encryption.enable": "1", "doc": "a\nb"},
}


def test_changes_restore_the_run_config_and_escape_dockerfile_quoting():
    assert image_flatten_changes(_CONFIG) == [
        'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]',
        'CMD ["python3", "-m", "sglang.launch_server"]',
        'ENV PATH="/usr/local/bin:/usr/bin"',
        'ENV PS1="\\$ \\"x\\"\\\\"',
        'ENV EMPTY=""',
        'WORKDIR "/sgl-workspace"',
        'USER "root"',
        "EXPOSE 30000/tcp",
        'VOLUME ["/data"]',
        # a multi-line label cannot be one --change line: dropped, the others kept
        'LABEL "lium.volume_encryption.enable"="1"',
        'LABEL "org.opencontainers.image.version"="24.04"',
    ]


def test_changes_refuse_an_env_value_with_a_line_break():
    with pytest.raises(ValueError):
        image_flatten_changes({"Env": ["A=1\n2"]})


def _ssh(*, exists: bool, flatten_rc: int = 0, size: int = 24 * 1024**3, driver: str = "overlay2"):
    async def run(command, check=False, timeout=None):
        if command.startswith("/usr/bin/docker info --format '{{.Driver}}'"):
            return Mock(exit_status=0, stdout=f"{driver}\n")
        if command.startswith("/usr/bin/docker image inspect --format"):
            return Mock(exit_status=0, stdout=json.dumps({"Id": _IMAGE_ID, "Size": size, "Config": _CONFIG}))
        if "docker create" not in command and f"/usr/bin/docker image inspect {_TAG}" in command:
            return Mock(exit_status=0 if exists else 1, stdout="")
        return Mock(exit_status=flatten_rc, stdout="", stderr="boom")

    ssh = Mock()
    ssh.run = AsyncMock(side_effect=run)
    return ssh


@pytest.mark.asyncio
async def test_an_existing_copy_of_the_same_image_id_is_reused():
    ssh = _ssh(exists=True)
    assert await flatten_image_on_host(ssh, "lmsysorg/sglang:dev", lease="r1") == _TAG
    assert ssh.run.await_count == 3


@pytest.mark.asyncio
async def test_the_no_disk_exit_raises_the_renter_sentence_and_other_failures_raise():
    with pytest.raises(ImageFlattenNoDisk) as exc:
        await flatten_image_on_host(_ssh(exists=False, flatten_rc=FLATTEN_NO_DISK_EXIT), "lmsysorg/sglang:dev", lease="r1")
    assert "not enough free disk" in exc.value.renter_sentence
    with pytest.raises(RuntimeError, match="boom"):
        await flatten_image_on_host(_ssh(exists=False, flatten_rc=98), "lmsysorg/sglang:dev", lease="r1")


@pytest.mark.asyncio
async def test_the_copy_is_built_from_the_inspected_image_id_never_the_mutable_tag():
    ssh = _ssh(exists=False)
    assert await flatten_image_on_host(ssh, "model:dev", lease="r1", reserved_bytes=79 * 1024**3) == _TAG
    command = ssh.run.await_args_list[-1].args[0]
    create = command.split("docker create", 1)[1].split(")", 1)[0]
    assert create.endswith(f" {_IMAGE_ID}") and "model:dev" not in command
    assert "--change 'CMD [\"python3\"" in command and _TAG in command
    # the disk the pod's limits already granted stays free after the copy
    assert f"-ge {24 * 1024**3 + FLATTEN_DISK_MARGIN_BYTES + 79 * 1024**3} ]" in command


@pytest.mark.asyncio
async def test_no_flatten_off_the_classic_overlay2_store():
    # the containerd store keeps images outside the data root the disk check measures
    ssh = _ssh(exists=False, driver="overlayfs")
    with pytest.raises(RuntimeError, match="storage driver"):
        await flatten_image_on_host(ssh, "model:dev", lease="r1")
    assert ssh.run.await_count == 1


_HEALTH = {"Interval": 30 * 10**9, "Timeout": 5 * 10**9, "StartPeriod": 0, "Retries": 3}


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"StopSignal": "SIGQUIT"}, ["STOPSIGNAL SIGQUIT"]),
        ({"Healthcheck": {"Test": ["NONE"]}}, ["HEALTHCHECK NONE"]),
        (
            {"Healthcheck": {**_HEALTH, "Test": ["CMD-SHELL", "curl -f localhost:8000 || exit 1"]}},
            [
                "HEALTHCHECK --interval=30000000000ns --timeout=5000000000ns --retries=3 "
                'CMD ["/bin/sh", "-c", "curl -f localhost:8000 || exit 1"]'
            ],
        ),
        # the image's SHELL decides the shell-form verdict; the copy has no SHELL, so it is written out
        (
            {"Shell": ["/bin/bash", "-c"], "Healthcheck": {"Test": ["CMD-SHELL", "[[ -f /ok ]]\nexit 0"]}},
            ['HEALTHCHECK CMD ["/bin/bash", "-c", "[[ -f /ok ]]\\nexit 0"]'],
        ),
        (
            {"Healthcheck": {"Test": ["CMD", "/bin/check", "--fast"], "StartInterval": 10**9}},
            ['HEALTHCHECK --start-interval=1000000000ns CMD ["/bin/check", "--fast"]'],
        ),
    ],
)
def test_changes_restore_stop_signal_and_health_check(config, expected):
    assert image_flatten_changes(config) == expected


def test_changes_refuse_a_health_check_they_cannot_carry():
    with pytest.raises(ValueError):
        image_flatten_changes({"Healthcheck": {"Test": ["CMD-SHELL"]}})


def _flatten_script(
    tmp_path, *, lease="r1", export_rc=0, import_rc=0, rm_rc=0, avail=10**12, import_sleep=0, volumes=(), sized_mark=""
) -> list[str]:
    """The real shell line against a fake docker that logs its calls (one argv per line)."""
    log = tmp_path / "calls"
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {log}\n'
        'case "$1" in\n'
        "  info) echo /var/lib/docker ;;\n"
        f"  run) printf 'Filesystem 1-blocks Used Available Capacity Mounted\\n/dev/x 1 1 {avail} 1%% /hostfs\\n' ;;\n"
        f'  image) if [ "$2" = ls ]; then cat {tmp_path}/copies 2>/dev/null; else [ -e {tmp_path}/tagged ]; fi ;;\n'
        "  create) echo cid1 ;;\n"
        f"  export) echo tar; exit {export_rc} ;;\n"
        f"  import) cat >/dev/null; sleep {import_sleep}; echo sha256:new; exit {import_rc} ;;\n"
        f"  rm) exit {rm_rc} ;;\n"
        f"  tag) touch {tmp_path}/tagged ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    script = _flatten_command("sha256:id", "lium-flattened:x", ['CMD ["a"]'], list(volumes), 100, lease, sized_mark)
    for real, local in (
        ("/usr/bin/docker", fake),
        (FLATTEN_LOCK_PATH, tmp_path / "lock"),
        (FLATTEN_USE_LOCK_PATH, tmp_path / "use.lock"),
        (FLATTEN_LEASE_DIR, tmp_path / "leases"),
        (image_flatten.FLATTEN_SIZED_PATH, tmp_path / "sized"),
    ):
        script = script.replace(real, str(local))
    return ["sh", "-c", script]


def _calls(tmp_path) -> list[str]:
    return (tmp_path / "calls").read_text().splitlines()


@pytest.mark.parametrize(
    "export_rc, import_rc, rm_rc, avail, expected_rc, expected_calls",
    [
        (0, 0, 0, 100, 0, ["image", "image", "info", "run", "create", "export", "import", "rm", "tag"]),
        # a truncated export must never be tagged: the import of it is removed
        (1, 0, 0, 100, 98, ["image", "image", "info", "run", "create", "export", "import", "rm", "rmi"]),
        (0, 1, 0, 100, 98, ["image", "image", "info", "run", "create", "export", "import", "rm", "rmi"]),
        # the container or its volumes were not removed: no cached tag, so a later rental retries
        (0, 0, 1, 100, 98, ["image", "image", "info", "run", "create", "export", "import", "rm", "rmi"]),
        # too little disk: nothing is created
        (0, 0, 0, 99, FLATTEN_NO_DISK_EXIT, ["image", "image", "info", "run"]),
    ],
)
def test_flatten_script_tags_only_a_complete_copy_and_always_removes_the_container(
    tmp_path, export_rc, import_rc, rm_rc, avail, expected_rc, expected_calls
):
    argv = _flatten_script(tmp_path, export_rc=export_rc, import_rc=import_rc, rm_rc=rm_rc, avail=avail)
    assert subprocess.run(argv, capture_output=True, env={"PATH": os.environ["PATH"]}).returncode == expected_rc
    calls = _calls(tmp_path)
    assert sorted(c.split()[0] for c in calls) == sorted(expected_calls)  # export and import run concurrently
    if "rm" in expected_calls:
        assert "rm -fv cid1" in calls  # -v: the container's anonymous volumes go with it


def test_declared_volumes_get_empty_volumes_so_create_copies_nothing_into_them(tmp_path):
    argv = _flatten_script(tmp_path, volumes=["/models"])
    assert subprocess.run(argv, capture_output=True, env={"PATH": os.environ["PATH"]}).returncode == 0
    create = next(c for c in _calls(tmp_path) if c.startswith("create"))
    assert create == "create --entrypoint /lium-flatten-noop --mount type=volume,dst=/models,volume-nocopy sha256:id"


def test_each_new_flatten_first_removes_earlier_copies_no_container_uses(tmp_path):
    # renters renting one unique image after another: the earlier pods are gone, their copies go too
    (tmp_path / "copies").write_text("lium-flattened:old1\nlium-flattened:old2\n")
    argv = _flatten_script(tmp_path)
    assert subprocess.run(argv, capture_output=True, env={"PATH": os.environ["PATH"]}).returncode == 0
    calls = _calls(tmp_path)
    assert "image ls --format {{.Repository}}:{{.Tag}} lium-flattened" in calls
    # without -f: dockerd refuses a copy a pod still uses; before the disk check, so the space counts
    first = [c.split()[0] for c in calls]
    assert calls.index("rmi lium-flattened:old1") < calls.index("rmi lium-flattened:old2") < first.index("info")
    assert (tmp_path / "leases" / "x.r1").exists()  # the new copy is leased to the rental that waits for it


def test_cleanup_keeps_a_copy_a_rental_still_waits_to_create_from(tmp_path):
    # pending: flattened or reused by a rental that has not created its container yet
    (tmp_path / "copies").write_text("lium-flattened:pending\nlium-flattened:stale\nlium-flattened:unleased\n")
    leases = tmp_path / "leases"
    leases.mkdir()
    (leases / "pending.r0").touch()
    (leases / "stale.r9").touch()  # a crashed validator never released it
    old = time.time() - (image_flatten.FLATTEN_LEASE_MINUTES + 5) * 60
    os.utime(leases / "stale.r9", (old, old))
    argv = _flatten_script(tmp_path)
    assert subprocess.run(argv, capture_output=True, env={"PATH": os.environ["PATH"]}).returncode == 0
    rmi = [c for c in _calls(tmp_path) if c.startswith("rmi lium-flattened:")]
    assert rmi == ["rmi lium-flattened:stale", "rmi lium-flattened:unleased"]


def test_a_lease_that_cannot_be_written_stops_before_any_cleanup_or_copy(tmp_path):
    # e.g. the lease disk is full: without a lease nothing protects the copy, so none is made or removed
    (tmp_path / "copies").write_text("lium-flattened:unleased\n")
    (tmp_path / "leases").write_text("")  # a file where the dir should be: mkdir and touch fail
    argv = _flatten_script(tmp_path)
    result = subprocess.run(argv, capture_output=True, env={"PATH": os.environ["PATH"]})
    assert result.returncode == image_flatten.FLATTEN_LEASE_EXIT and b"cannot write the lease" in result.stderr
    assert not [c for c in _calls(tmp_path) if c.split()[0] in ("rmi", "create")]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
def test_an_unreadable_lease_dir_deletes_nothing(tmp_path):
    (tmp_path / "copies").write_text("lium-flattened:unleased\n")
    leases = tmp_path / "leases"
    leases.mkdir()
    leases.chmod(0o300)  # leases can be written, not listed: how many are held is unknown
    try:
        result = subprocess.run(_flatten_script(tmp_path), capture_output=True, env={"PATH": os.environ["PATH"]})
    finally:
        leases.chmod(0o700)
    assert result.returncode == image_flatten.FLATTEN_BUSY_EXIT
    assert not [c for c in _calls(tmp_path) if c.split()[0] in ("rmi", "create")]


def test_reusing_a_copy_leases_it_under_the_cleanup_lock(tmp_path):
    (tmp_path / "tagged").touch()
    argv = _flatten_script(tmp_path)
    assert subprocess.run(argv, capture_output=True, env={"PATH": os.environ["PATH"]}).returncode == 0
    assert (tmp_path / "leases" / "x.r1").exists() and "create" not in " ".join(_calls(tmp_path))
    reuse = image_flatten._reuse_command("lium-flattened:x", "r1")
    assert reuse.index("flock -w") < reuse.index("image inspect") < reuse.index("touch")


def _sh(command: str, tmp_path) -> int:
    command = command.replace(FLATTEN_LEASE_DIR, str(tmp_path / "leases"))
    return subprocess.run(["sh", "-c", command], env={"PATH": os.environ["PATH"]}).returncode


def test_a_released_lease_leaves_an_abandoned_copy_to_the_next_cleanup(tmp_path):
    # a renter flattens, then cancels: the create path releases the lease at once, not after the expiry
    leases = tmp_path / "leases"
    leases.mkdir()
    (leases / "abandoned.r0").touch()
    (leases / "other.r5").touch()  # another rental still waits on its copy
    assert _sh(image_flatten.flatten_lease_release_command("r0"), tmp_path) == 0
    assert sorted(p.name for p in leases.iterdir()) == ["other.r5"]
    (tmp_path / "copies").write_text("lium-flattened:abandoned\nlium-flattened:other\n")
    assert subprocess.run(_flatten_script(tmp_path), env={"PATH": os.environ["PATH"]}).returncode == 0
    assert [c for c in _calls(tmp_path) if c.startswith("rmi lium-flattened:")] == ["rmi lium-flattened:abandoned"]


def test_a_new_copy_waits_for_fewer_than_the_cap_of_held_leases(tmp_path):
    leases = tmp_path / "leases"
    leases.mkdir()
    for n in range(image_flatten.FLATTEN_MAX_LEASES):
        (leases / f"copy{n}.r{n}").touch()
    result = subprocess.run(_flatten_script(tmp_path), capture_output=True, env={"PATH": os.environ["PATH"]})
    assert result.returncode == image_flatten.FLATTEN_BUSY_EXIT and b"still wait" in result.stderr
    assert not [c for c in _calls(tmp_path) if c.split()[0] in ("rmi", "create")]
    # a reuse adds no copy: not capped
    (tmp_path / "tagged").touch()
    assert subprocess.run(_flatten_script(tmp_path), env={"PATH": os.environ["PATH"]}).returncode == 0


def test_two_concurrent_flattens_on_one_node_copy_once(tmp_path):
    argv = _flatten_script(tmp_path, import_sleep=1)
    env = {"PATH": os.environ["PATH"]}
    procs = [subprocess.Popen(argv, env=env), subprocess.Popen(argv, env=env)]
    assert [p.wait(timeout=30) for p in procs] == [0, 0]
    # the second waits for the node lock, then finds the copy the first tagged
    assert [c.split()[0] for c in _calls(tmp_path)].count("create") == 1


def test_timeout_bounds_both_sides_of_the_copy():
    script = _flatten_command("img", "t", [], [], 0, "r1")
    assert script.count(f"timeout {image_flatten.FLATTEN_TIMEOUT_SECONDS} ") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("copy", ["none", "running", "ended_after_the_early_probe"])
async def test_a_replacement_sizes_on_disk_measured_under_the_flatten_lock(tmp_path, monkeypatch, copy):
    # review: 120 GiB free; rental A was deleted while its 30 GiB copy is written. Rental B's early probe may
    # have measured 120 before the copy landed (also when it ended before B's wait): B sizes on the free disk
    # read under the lock the copy holds, so all it is granted stays backed.
    lock, used = tmp_path / "lock", tmp_path / "used"
    used.write_text("30" if copy == "ended_after_the_early_probe" else "0")
    gib = 1024**3
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  info) echo /var/lib/docker ;;\n"
        f"  run) printf 'Filesystem 1-blocks Used Available Capacity Mounted\\n/dev/x 1 1 %s 1%% /hostfs\\n' "
        f"$(( (120 - $(cat {used})) * {gib} )) ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(image_flatten, "FLATTEN_LOCK_PATH", str(lock))
    monkeypatch.setattr(image_flatten, "FLATTEN_SIZED_PATH", str(tmp_path / "sized"))
    env = {"PATH": os.environ["PATH"]}
    if copy == "running":
        lock.touch()
        running = subprocess.Popen(["flock", str(lock), "sh", "-c", f"sleep 1; echo 30 > {used}"], env=env)
        while subprocess.run(["flock", "-n", str(lock), "true"], env=env).returncode == 0:
            await asyncio.sleep(0.01)

    async def run(command, check=False, timeout=None):
        done = subprocess.run(
            ["sh", "-c", command.replace("/usr/bin/docker", str(fake))], env=env, capture_output=True, text=True
        )
        return Mock(exit_status=done.returncode, stdout=done.stdout)

    free = await image_flatten.wait_for_node_flatten(Mock(run=run))
    assert free + int(used.read_text()) * gib <= 120 * gib
    assert free == (120 if copy == "none" else 90) * gib
    assert (tmp_path / "sized").exists()
    if copy == "running":
        running.wait()


@pytest.mark.asyncio
async def test_a_node_without_flock_runs_no_flatten_and_gives_no_locked_measure():
    ssh = Mock(run=AsyncMock(return_value=Mock(exit_status=0, stdout="")))
    assert await image_flatten.wait_for_node_flatten(ssh) is None


@pytest.mark.asyncio
async def test_a_failed_wait_for_a_running_flatten_stops_the_create():
    ssh = Mock(run=AsyncMock(return_value=Mock(exit_status=1)))
    with pytest.raises(RuntimeError, match="image copy"):
        await image_flatten.wait_for_node_flatten(ssh)


def test_a_flatten_refuses_once_a_create_sized_the_disk_after_it_checked_for_other_creates(tmp_path, monkeypatch):
    # review: A saw no other create, then B waited (no flatten held the lock yet) and measured the full
    # free disk; A must not copy into what B was just granted
    monkeypatch.setattr(image_flatten, "FLATTEN_SIZED_PATH", str(tmp_path / "sized"))
    env = {"PATH": os.environ["PATH"]}
    mark = subprocess.run(["sh", "-c", image_flatten.flatten_sized_mark_command()], capture_output=True, text=True)
    _flatten_script(tmp_path)  # writes the fake docker
    wait = image_flatten.flatten_wait_command().replace(FLATTEN_LOCK_PATH, str(tmp_path / "lock"))
    assert _sh(wait.replace("/usr/bin/docker", str(tmp_path / "docker")), tmp_path) == 0
    result = subprocess.run(_flatten_script(tmp_path, sized_mark=mark.stdout.strip()), capture_output=True, env=env)
    assert result.returncode == FLATTEN_NO_DISK_EXIT and b"sized its disk" in result.stderr
    assert "create" not in [c.split()[0] for c in _calls(tmp_path)]
    # read after B's mark: B is visible to A's own check for other creates, so the copy goes ahead
    mark = subprocess.run(["sh", "-c", image_flatten.flatten_sized_mark_command()], capture_output=True, text=True)
    assert subprocess.run(_flatten_script(tmp_path, sized_mark=mark.stdout.strip()), env=env).returncode == 0
