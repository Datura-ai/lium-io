import asyncio
import json
import os
import shutil
import subprocess
import sys
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


def _ssh(
    *, exists: bool, flatten_rc: int = 0, size: int = 24 * 1024**3, driver: str = "overlay2", same_disk_rc=0,
    docker_version: str = "29.8.0",
):
    async def run(command, check=False, timeout=None):
        if command.startswith("/usr/bin/docker version"):
            return Mock(exit_status=0, stdout=f"{docker_version}\n")
        if "--pid=host" in command:
            return Mock(exit_status=0, stdout="/usr/bin/containerd \n")
        if command == image_flatten.containerd_plugins_command("/run/containerd/containerd.sock"):
            return Mock(exit_status=0, stdout=_plugins("/var/lib/containerd"))
        if command.startswith("/usr/bin/docker info --format '{{json .DriverStatus}}'"):
            return Mock(exit_status=same_disk_rc, stdout="version = 3\n")
        if command.startswith("root=$(/usr/bin/docker info --format '{{.DockerRootDir}}')"):
            return Mock(exit_status=0, stdout="")
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
async def test_no_flatten_on_a_store_off_the_measured_disk():
    # the containerd store keeps images under containerd's root, which may not be the data root the disk check measures
    ssh = _ssh(exists=False, driver="overlayfs", same_disk_rc=1)
    with pytest.raises(RuntimeError, match="storage driver"):
        await flatten_image_on_host(ssh, "model:dev", lease="r1")
    assert ssh.run.await_count == 4


@pytest.mark.asyncio
async def test_no_flatten_on_the_containerd_store_of_a_docker_that_undercounts_image_size():
    ssh = _ssh(exists=False, driver="overlayfs", docker_version="29.7.2")
    with pytest.raises(RuntimeError, match="storage driver 'overlayfs': docker version '29.7.2' does not count"):
        await flatten_image_on_host(ssh, "model:dev", lease="r1")
    assert ssh.run.await_count == 2


@pytest.mark.parametrize(
    "version, counts",
    [("29.8.0", True), ("29.10.1", True), ("30.0.0-rc.1", True), ("29.7.9", False), ("28.5.2", False), ("", False),
     ("dev", False)],
)
def test_only_a_docker_whose_containerd_size_counts_unpacked_snapshots(version, counts):
    assert image_flatten.docker_counts_unpacked_size(version) is counts


@pytest.mark.asyncio
async def test_no_flatten_on_a_containerd_snapshotter_other_than_overlayfs():
    # only the overlayfs snapshotter's directory is measured: stargz, native or zfs write elsewhere
    ssh = _ssh(exists=False, driver="stargz")
    with pytest.raises(RuntimeError, match="storage driver"):
        await flatten_image_on_host(ssh, "model:dev", lease="r1")
    assert ssh.run.await_count == 1


@pytest.mark.asyncio
async def test_containerd_store_flattens_when_its_root_shares_the_measured_disk():
    ssh = _ssh(exists=False, driver="overlayfs")
    assert await flatten_image_on_host(ssh, "model:dev", lease="r1") == _TAG


def _plugins(root) -> str:
    """`ctr plugins ls -d` of a daemon running on `root`."""
    return "".join(
        f"Type:          {kind}\nID:            {ident}\nPlatforms:     linux/amd64\nExports:\n"
        f"               root      {root}/{kind}.{ident}\n\n"
        for kind, ident in (("io.containerd.content.v1", "content"), ("io.containerd.snapshotter.v1", "overlayfs"))
    )


def _same_disk_rc(
    tmp_path, *, driver_status: str, config: str | None, daemons: str = "/usr/bin/containerd \n", link=None,
    running_root=None,
) -> int:
    """The real check line against a fake docker whose `run` maps each --mount dst to its src locally."""
    data_root = tmp_path / "docker"
    data_root.mkdir()
    (tmp_path / "etc" / "containerd").mkdir(parents=True)
    if config is not None:
        (tmp_path / "etc" / "containerd" / "config.toml").write_text(config)
    if link is not None:
        link(tmp_path / "etc")
    fake = tmp_path / "fake-docker"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys\n"
        "a = sys.argv[1:]\n"
        "if a[0] == 'version':\n"
        "    print('29.8.0'); sys.exit(0)\n"
        f"if '--pid=host' in a:\n    print({daemons!r}, end=''); sys.exit(0)\n"
        "if a[0] == 'info':\n"
        f"    print({driver_status!r} if 'DriverStatus' in a[2] else {str(data_root)!r}); sys.exit(0)\n"
        "mounts, i = {}, 2\n"
        "while a[i] == '--mount':\n"
        "    kv = dict(p.split('=', 1) for p in a[i + 1].split(',') if '=' in p); i += 2\n"
        f"    src = {str(tmp_path)!r} + kv['src'] if kv['src'] == '/etc' else kv['src']\n"
        "    if not __import__('os').path.isdir(src): sys.exit(125)\n"
        "    mounts[kv['dst']] = src\n"
        "script = a[i + 3]\n"
        "for dst, src in mounts.items(): script = script.replace(dst + '/', src + '/')\n"
        "sys.exit(subprocess.run(['sh', '-c', script]).returncode)\n"
    )
    fake.chmod(0o755)
    async def run(command, check=False, timeout=None):
        # only the host-side command answers: run in the executor container, ctr is missing and this check refuses
        if command == image_flatten.containerd_plugins_command(image_flatten.containerd_address(config)):
            live = running_root or image_flatten.containerd_root(config)
            return Mock(exit_status=0 if live != "unavailable" else 1, stdout=_plugins(live))
        done = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", str(fake))], capture_output=True, text=True)
        return Mock(exit_status=done.returncode, stdout=done.stdout)

    ssh = Mock()
    ssh.run = AsyncMock(side_effect=run)
    return 1 if asyncio.run(image_flatten.containerd_store_refusal(ssh)) else 0


_STORE = '[["driver-type","io.containerd.snapshotter.v1"]]'


def _ctrd(root, **elsewhere):
    """A containerd root with its snapshot and content directories; `elsewhere` links one to another filesystem."""
    for sub in image_flatten._CONTAINERD_DATA_DIRS:
        if any(sub.startswith(f"{linked}/") for linked in elsewhere):
            continue  # under a linked directory: the link decides
        (root / sub).parent.mkdir(parents=True, exist_ok=True)
        if sub in elsewhere:
            (root / sub).symlink_to(elsewhere[sub])
        else:
            (root / sub).mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize(
    "sub, target, expected",
    [
        (None, None, True),  # ordinary directories on the data root's disk
        ("io.containerd.snapshotter.v1.overlayfs", "/proc", False),  # snapshots on another filesystem
        ("io.containerd.content.v1.content", "/proc", False),  # the content store on another filesystem
        ("io.containerd.snapshotter.v1.overlayfs", "/nonexistent", False),  # a dangling link fails the mount
        ("io.containerd.snapshotter.v1.overlayfs/snapshots", "/proc", False),  # only the snapshots child elsewhere
        ("io.containerd.content.v1.content/ingest", "/proc", False),  # only the content ingest child elsewhere
        ("io.containerd.content.v1.content/blobs/sha256", "/proc", False),  # only the content blobs child elsewhere
    ],
)
def test_the_snapshot_and_content_dirs_must_share_the_data_root_filesystem(tmp_path, sub, target, expected):
    _ctrd(tmp_path / "ctrd", **({sub: target} if sub else {}))
    config = f"version = 3\nroot = '{tmp_path / 'ctrd'}'\n"
    assert (_same_disk_rc(tmp_path, driver_status=_STORE, config=config) == 0) is expected


def test_a_proxy_snapshotter_refuses_the_flatten():
    stargz = (
        "version = 2\n[proxy_plugins]\n  [proxy_plugins.stargz]\n    type = 'snapshot'\n"
        "    address = '/run/containerd-stargz-grpc/containerd-stargz-grpc.sock'\n"
    )
    assert image_flatten.containerd_root(stargz) is None


@pytest.mark.parametrize(
    "driver_status, root, expected_same",
    [
        (_STORE, "same", True),
        (_STORE, "other", False),  # containerd root on another filesystem than the measured data root
        (_STORE, "missing", False),
        ('[["Backing Filesystem","zfs"]]', "same", False),  # a classic driver other than overlay2
    ],
)
def test_containerd_store_check_requires_its_root_on_the_data_root_filesystem(
    tmp_path, driver_status, root, expected_same
):
    path = {"same": tmp_path / "ctrd", "other": "/proc", "missing": tmp_path / "nope"}[root]
    if root == "same":
        _ctrd(path)
    config = f'version = 2\nroot = "{path}"\n[plugins]\n  root = "/ignored"\n'
    assert (_same_disk_rc(tmp_path, driver_status=driver_status, config=config) == 0) is expected_same


@pytest.mark.parametrize(
    "config, expected",
    [
        ("version = 3\nroot = '{same}'\n", True),  # the reviewer's config: literal-string root, version 3
        ('version = 2\nroot = "{same}"\n', True),
        ('root="{same}" # comment\n', True),
        ("version = 3\nroot = '{other}'\n", False),  # points at another filesystem: never the default root
        ("version = 3\nroot = '{same}\n", False),  # unparsable: refuse, never assume
        ("imports = ['/etc/containerd/conf.d/*.toml']\nroot = '{same}'\n", False),  # imports may move it
        ("root = 'relative/ctrd'\n", False),
    ],
)
def test_containerd_root_comes_from_the_config_or_the_check_refuses(tmp_path, config, expected):
    _ctrd(tmp_path / "ctrd")
    if config is not None:
        config = config.format(same=tmp_path / "ctrd", other="/proc")
    rc = _same_disk_rc(tmp_path, driver_status=_STORE, config=config)
    assert (rc == 0) is expected


@pytest.mark.parametrize(
    "daemons, expected",
    [
        ("/usr/bin/containerd \n", True),
        ("/usr/bin/containerd --config /etc/containerd/config.toml \n", True),
        ("/usr/bin/containerd --root /mnt/b/containerd \n", False),  # the flag overrides the file's root
        ("/usr/bin/containerd --config /srv/containerd.toml \n", False),  # another file entirely
        ("/usr/bin/containerd \ncontainerd --config /var/run/docker/containerd/containerd.toml \n", False),
        ("", False),  # no containerd seen: the daemon serving Docker is unknown
    ],
)
def test_containerd_store_check_requires_one_daemon_reading_the_etc_config(tmp_path, daemons, expected):
    _ctrd(tmp_path / "ctrd")
    config = f"version = 3\nroot = '{tmp_path / 'ctrd'}'\n"
    assert (_same_disk_rc(tmp_path, driver_status=_STORE, config=config, daemons=daemons) == 0) is expected


def _link_config(etc):
    target = etc.parent / "srv-containerd.toml"
    (etc / "containerd" / "config.toml").rename(target)
    (etc / "containerd" / "config.toml").symlink_to(target)


def _link_dir(etc):
    (etc / "containerd").rename(etc.parent / "srv-containerd")
    (etc / "containerd").symlink_to(etc.parent / "srv-containerd")


@pytest.mark.parametrize("link", [_link_config, _link_dir])
def test_a_symlinked_containerd_config_is_refused_never_read_as_missing(tmp_path, link):
    _ctrd(tmp_path / "ctrd")
    config = f"version = 3\nroot = '{tmp_path / 'ctrd'}'\n"
    assert _same_disk_rc(tmp_path, driver_status=_STORE, config=config, link=link) == 1


@pytest.mark.parametrize("running, expected", [("same", True), ("other", False), ("unavailable", False)])
def test_a_config_rewritten_without_a_containerd_restart_is_refused(tmp_path, running, expected):
    # the daemon started on another disk, then the config was edited to a root on the measured disk
    _ctrd(tmp_path / "ctrd")
    config = f"version = 3\nroot = '{tmp_path / 'ctrd'}'\n"
    live = {"same": str(tmp_path / "ctrd"), "other": "/mnt/b/containerd", "unavailable": "unavailable"}[running]
    assert (_same_disk_rc(tmp_path, driver_status=_STORE, config=config, running_root=live) == 0) is expected


def test_the_running_root_check_runs_the_hosts_ctr_on_the_configured_socket(tmp_path):
    command = image_flatten.containerd_plugins_command("/run/containerd/containerd.sock")
    # the executor container has no ctr and no containerd socket: the check must enter the host's mount namespace
    assert command.startswith("nsenter -t 1 -m -- sh -c ")
    script = command.removeprefix("nsenter -t 1 -m -- ")
    log = tmp_path / "ctr.log"
    for path in image_flatten._CTR_PATHS:
        script = script.replace(path, str(tmp_path / path.lstrip("/")))
    bundled = tmp_path / "usr/bin/docker-containerd-ctr"
    bundled.parent.mkdir(parents=True)
    run = lambda: subprocess.run(["sh", "-c", script], env={"PATH": os.environ["PATH"]}).returncode  # noqa: E731
    assert run() == 127  # no ctr on the host: the caller refuses
    bundled.write_text(f'#!/bin/sh\necho "$*" > {log}\n')
    bundled.chmod(0o755)
    assert run() == 0 and log.read_text() == "--address /run/containerd/containerd.sock plugins ls -d\n"
    assert image_flatten.containerd_address(None) == "/run/containerd/containerd.sock"
    assert image_flatten.containerd_address("version = 2\n[grpc]\n  address = '/run/c/c.sock'\n") == "/run/c/c.sock"
    assert image_flatten.containerd_address("version = 3\n[grpc]\n  uid = 0\n") == "/run/containerd/containerd.sock"
    assert image_flatten.containerd_address("grpc = 'x'\n") is None


def test_a_snapshotter_root_path_elsewhere_refuses_the_flatten():
    v2 = "version = 2\n[plugins.'io.containerd.snapshotter.v1.overlayfs']\n  root_path = '/mnt/b/snap'\n"
    v3 = "version = 3\n[plugins]\n  [plugins.'io.containerd.snapshotter.v1.overlayfs']\n    root_path = '/mnt/b'\n"
    assert image_flatten.containerd_root(v2) is None
    assert image_flatten.containerd_root(v3) is None
    assert image_flatten.containerd_root("version = 2\n[plugins.'io.containerd.snapshotter.v1.overlayfs']\n  root_path = ''\n") == "/var/lib/containerd"


def test_the_default_root_is_used_only_without_a_config_file():
    assert image_flatten.containerd_root(None) == "/var/lib/containerd"
    assert image_flatten.containerd_root("version = 3\n[plugins]\n  root = '/x'\n") == "/var/lib/containerd"
    assert image_flatten.containerd_root("version = 3\nroot = '/mnt/ssd/containerd'\n") == "/mnt/ssd/containerd"
    assert image_flatten.containerd_root("root = [") is None


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
    tmp_path,
    *,
    lease="r1",
    export_rc=0,
    import_rc=0,
    rm_rc=0,
    avail=10**12,
    import_sleep=0,
    volumes=(),
    sized_mark="",
    need_bytes=100,
    export_bytes=3,
    export_factor=0,
    fixed_bytes=0,
    export_sleep=0,
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
        f"  export) sleep {export_sleep}; head -c {export_bytes} /dev/zero; exit {export_rc} ;;\n"
        f"  import) cat >/dev/null; sleep {import_sleep}; echo sha256:new; exit {import_rc} ;;\n"
        f"  rm) exit {rm_rc} ;;\n"
        f"  tag) touch {tmp_path}/tagged ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    script = _flatten_command(
        "sha256:id", "lium-flattened:x", ['CMD ["a"]'], list(volumes), need_bytes, lease, sized_mark,
        export_factor=export_factor, fixed_bytes=fixed_bytes,
    )
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


@pytest.mark.parametrize(
    "size, export_bytes, export_rc, expected_rc, expected_calls",
    [
        # dense, and snapshots-only (Size ~ its unpacked data): the export matches Size, the copy runs
        (40, 40, 0, 0, ["image", "image", "info", "run", "create", "export", "export", "import", "rm", "tag"]),
        (40, 38, 0, 0, ["image", "image", "info", "run", "create", "export", "export", "import", "rm", "tag"]),
        # sparse: Size counts allocated blocks, the export expands the holes; refused before any import
        (4, 60, 0, FLATTEN_NO_DISK_EXIT, ["image", "image", "info", "run", "create", "export", "rm"]),
        # the counting export failed: no copy
        (40, 40, 1, 98, ["image", "image", "info", "run", "create", "export", "rm"]),
    ],
)
def test_the_containerd_disk_check_budgets_the_bytes_the_export_really_writes(
    tmp_path, size, export_bytes, export_rc, expected_rc, expected_calls
):
    # 100 free, 10 kept: the import writes the export twice, so at most 45 bytes of export fit
    argv = _flatten_script(
        tmp_path, avail=100, need_bytes=2 * size + 10, export_bytes=export_bytes, export_rc=export_rc,
        export_factor=2, fixed_bytes=10,
    )
    assert subprocess.run(argv, capture_output=True, env={"PATH": os.environ["PATH"]}).returncode == expected_rc
    assert sorted(c.split()[0] for c in _calls(tmp_path)) == sorted(expected_calls)
    assert "rm -fv cid1" in _calls(tmp_path)


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
    assert script.count('timeout "$left" ') == 2


@pytest.mark.parametrize("export_factor", [0, 2])
def test_counting_and_copying_share_one_deadline(tmp_path, monkeypatch, export_factor):
    # scaled: each export pass takes 3/4 of the limit; two passes in a row must not hold the lock past the
    # limit the waiters (flatten lock wait, flatten_wait_command) allow for
    monkeypatch.setattr(image_flatten, "FLATTEN_TIMEOUT_SECONDS", 2)
    argv = _flatten_script(tmp_path, export_sleep=1.5, export_factor=export_factor)
    # a logging timeout: the limit each pass gets, not wall-clock time, so a loaded host cannot flake it
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "timeout").write_text(f'#!/bin/sh\necho "$1" >> {tmp_path}/limits\nexec {shutil.which("timeout")} "$@"\n')
    (shim / "timeout").chmod(0o755)
    rc = subprocess.run(argv, capture_output=True, env={"PATH": f"{shim}:{os.environ['PATH']}"}).returncode
    limits = [int(n) for n in (tmp_path / "limits").read_text().split()] if (tmp_path / "limits").exists() else []
    if export_factor:
        # the count used 1.5 of the 2 seconds: the copy gets at most what is left, or does not start
        assert limits[0] == 2 and all(n <= 1 for n in limits[1:]) and rc in (0, 98)
    else:
        assert limits == [2, 2] and rc == 0
    assert "rm -fv cid1" in _calls(tmp_path)


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


def test_no_flatten_lock_or_mark_lives_on_tmp():
    # review: /tmp is janitor-managed and can fill up
    for path in (FLATTEN_LOCK_PATH, FLATTEN_USE_LOCK_PATH, image_flatten.FLATTEN_SIZED_PATH):
        assert path.startswith("/root/app/")


@pytest.mark.asyncio
async def test_a_node_without_flock_runs_no_flatten_and_gives_no_locked_measure():
    ssh = Mock(run=AsyncMock(return_value=Mock(exit_status=0, stdout="")))
    assert await image_flatten.wait_for_node_flatten(ssh) is None


@pytest.mark.asyncio
async def test_a_failed_wait_for_a_running_flatten_stops_the_create():
    ssh = Mock(run=AsyncMock(return_value=Mock(exit_status=1)))
    with pytest.raises(RuntimeError, match="image copy") as raised:
        await image_flatten.wait_for_node_flatten(ssh)
    assert not isinstance(raised.value, image_flatten.FlattenMeasureFailed)


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", ["lock", "mark", "measure"])
async def test_only_a_failed_measure_under_the_held_lock_and_mark_may_fall_back(tmp_path, monkeypatch, broken):
    # review: a lock or mark that cannot be written must stop the create, not authorize fallback sizing
    fake = tmp_path / "docker"
    fake.write_text('#!/bin/sh\n[ "$1" = info ] && echo /var/lib/docker\nexit 1\n')
    fake.chmod(0o755)
    missing = str(tmp_path / "missing" / "f")  # parent dir absent: the file cannot be created
    monkeypatch.setattr(image_flatten, "FLATTEN_LOCK_PATH", missing if broken == "lock" else str(tmp_path / "lock"))
    monkeypatch.setattr(image_flatten, "FLATTEN_SIZED_PATH", missing if broken == "mark" else str(tmp_path / "sized"))
    env = {"PATH": os.environ["PATH"]}

    async def run(command, check=False, timeout=None):
        done = subprocess.run(["sh", "-c", command.replace("/usr/bin/docker", str(fake))], env=env, capture_output=True)
        return Mock(exit_status=done.returncode, stdout=done.stdout.decode())

    with pytest.raises(RuntimeError) as raised:
        await image_flatten.wait_for_node_flatten(Mock(run=run))
    assert isinstance(raised.value, image_flatten.FlattenMeasureFailed) == (broken == "measure")


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


_GIB24 = 24 * 1024**3
_MANY = image_flatten.FLATTEN_BUDGET_MIN_LAYERS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout, copy_exit, expected",
    [
        # uncached many-layer image: the sizing keeps room for the new copy
        (f"{_GIB24} {_MANY} sha256:abc\noverlay2\n", 1, _GIB24 + FLATTEN_DISK_MARGIN_BYTES),
        # containerd store, complete or snapshots-only image: the import writes a compressed blob and unpacks a snapshot
        (f"{_GIB24} {_MANY} sha256:abc\noverlayfs\n", 1, 2 * _GIB24 + FLATTEN_DISK_MARGIN_BYTES),
        # its copy is already on the node: the flatten reuses it, so no extra room is kept, on either store
        (f"{_GIB24} {_MANY} sha256:abc\noverlay2\n", 0, None),
        (f"{_GIB24} {_MANY} sha256:abc\noverlayfs\n", 0, None),
        # the copy cannot be looked up: keep the room
        (f"{_GIB24} {_MANY} sha256:abc\noverlay2\n", RuntimeError("ssh"), _GIB24 + FLATTEN_DISK_MARGIN_BYTES),
        (f"{_GIB24} {_MANY - 1} sha256:abc\noverlay2\n", 1, None),  # starts as is: no copy
        ("garbage\n", 1, None),
    ],
)
async def test_flatten_copy_bytes_budgets_only_a_copy_still_to_write(stdout, copy_exit, expected):
    async def run(command, **kwargs):
        if "--format" in command:
            return Mock(stdout=stdout)
        assert "lium-flattened:abc" in command
        if isinstance(copy_exit, Exception):
            raise copy_exit
        return Mock(exit_status=copy_exit)

    assert await image_flatten.flatten_copy_bytes(Mock(run=run), "sglang") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("driver, copies", [("overlay2", 1), ("overlayfs", 2)])
async def test_the_flattens_own_disk_check_budgets_the_containerd_import_twice(driver, copies):
    # a snapshots-only image (compressed blobs gone) still has Size ~ its unpacked data; the import writes both again
    ssh = _ssh(exists=False, driver=driver)
    await flatten_image_on_host(ssh, "model:dev", lease="r1", reserved_bytes=79 * 1024**3)
    command = ssh.run.await_args_list[-1].args[0]
    assert f"-ge {copies * 24 * 1024**3 + FLATTEN_DISK_MARGIN_BYTES + 79 * 1024**3} ]" in command
    # containerd also counts the real export bytes (a sparse file reports a small Size); overlay2's Size does not under-count
    assert ("head -c" in command) == (copies == 2)
