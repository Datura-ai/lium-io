import json
import os
import subprocess
from unittest.mock import AsyncMock, Mock

import pytest

from services import image_flatten
from services.image_flatten import (
    FLATTEN_DISK_MARGIN_BYTES,
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


def _ssh(*, exists: bool, avail: int, size: int = 24 * 1024**3):
    async def run(command, check=False, timeout=None):
        if command.startswith("/usr/bin/docker image inspect --format"):
            return Mock(exit_status=0, stdout=json.dumps({"Id": _IMAGE_ID, "Size": size, "Config": _CONFIG}))
        if command.startswith(f"/usr/bin/docker image inspect {_TAG}"):
            return Mock(exit_status=0 if exists else 1, stdout="")
        if "df -P -B1" in command:
            return Mock(exit_status=0, stdout=f"Filesystem 1-blocks Used Available Capacity Mounted\n/dev/x 1 1 {avail} 1% /hostfs\n")
        return Mock(exit_status=0, stdout="")

    ssh = Mock()
    ssh.run = AsyncMock(side_effect=run)
    return ssh


@pytest.mark.asyncio
async def test_an_existing_copy_of_the_same_image_id_is_reused():
    ssh = _ssh(exists=True, avail=0)
    assert await flatten_image_on_host(ssh, "lmsysorg/sglang:dev") == _TAG
    assert ssh.run.await_count == 2


@pytest.mark.asyncio
async def test_too_little_disk_refuses_before_copying():
    ssh = _ssh(exists=False, avail=24 * 1024**3 + FLATTEN_DISK_MARGIN_BYTES - 1)
    with pytest.raises(ImageFlattenNoDisk) as exc:
        await flatten_image_on_host(ssh, "lmsysorg/sglang:dev")
    assert "not enough free disk" in exc.value.renter_sentence
    assert not any("docker export" in c.args[0] for c in ssh.run.await_args_list)


@pytest.mark.asyncio
async def test_enough_disk_builds_the_copy_and_returns_its_tag():
    ssh = _ssh(exists=False, avail=24 * 1024**3 + FLATTEN_DISK_MARGIN_BYTES)
    assert await flatten_image_on_host(ssh, "lmsysorg/sglang:dev") == _TAG
    command = ssh.run.await_args_list[-1].args[0]
    assert "docker export" in command and "--change 'CMD [\"python3\"" in command and _TAG in command


def _run_flatten_script(tmp_path, *, export_rc: int, import_rc: int) -> tuple[int, str]:
    """Run the real shell line against a fake docker that logs its calls."""
    log = tmp_path / "calls"
    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$1" >> {log}\n'
        'case "$1" in\n'
        "  create) echo cid1 ;;\n"
        f"  export) echo tar; exit {export_rc} ;;\n"
        f"  import) cat >/dev/null; echo sha256:new; exit {import_rc} ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    script = _flatten_command("img", "lium-flattened:x", ['CMD ["a"]']).replace("/usr/bin/docker", str(fake))
    rc = subprocess.run(["sh", "-c", script], capture_output=True, env={"PATH": os.environ["PATH"]}).returncode
    return rc, log.read_text().split()


@pytest.mark.parametrize(
    "export_rc, import_rc, expected_rc, expected_calls",
    [
        (0, 0, 0, ["create", "export", "import", "rm", "tag"]),
        # a truncated export must never be tagged: the import of it is removed
        (1, 0, 98, ["create", "export", "import", "rm", "rmi"]),
        (0, 1, 98, ["create", "export", "import", "rm", "rmi"]),
    ],
)
def test_flatten_script_tags_only_a_complete_copy_and_always_removes_the_container(
    tmp_path, export_rc, import_rc, expected_rc, expected_calls
):
    rc, calls = _run_flatten_script(tmp_path, export_rc=export_rc, import_rc=import_rc)
    assert rc == expected_rc
    assert sorted(calls) == sorted(expected_calls)  # export and import run concurrently


def test_timeout_bounds_both_sides_of_the_copy():
    script = _flatten_command("img", "t", [])
    assert script.count(f"timeout {image_flatten.FLATTEN_TIMEOUT_SECONDS} ") == 2
