"""The mounted-volume listing skips a container `docker ps -a` lists but `docker inspect` cannot find."""

import os
import subprocess

import pytest

from services.prerun_host_probe import DOCKER_MOUNTED_VOLUME_NAMES_CMD

FAKE_DOCKER = """#!/bin/sh
if [ "$1" = ps ]; then printf '%s\\n' $CONTAINERS; exit 0; fi
shift 3; rc=0
for id in "$@"; do
  case "$id" in
    ghost*) echo "Error: No such object: $id" >&2; rc=1;;
    broken*) echo "Error response from daemon: permission denied" >&2; rc=1;;
    *) printf 'volume_%s\\n' "$id";;
  esac
done
exit $rc
"""


@pytest.mark.parametrize(
    ("containers", "rc", "mounted"),
    [
        ("a b", 0, "volume_a\nvolume_b\n"),
        ("a ghost1 b ghost2", 0, "volume_a\nvolume_b\n"),
        ("", 0, ""),
        ("a broken b", 1, None),
    ],
    ids=["all_found", "ghosts_skipped", "no_containers", "other_error_fails"],
)
def test_mounted_volume_listing_skips_only_missing_containers(tmp_path, containers, rc, mounted):
    docker = tmp_path / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    command = DOCKER_MOUNTED_VOLUME_NAMES_CMD.replace("/usr/bin/docker", str(docker))
    result = subprocess.run(
        ["sh", "-c", command],
        capture_output=True,
        text=True,
        env={**os.environ, "CONTAINERS": containers},
    )
    assert result.returncode == rc
    if mounted is not None:
        assert result.stdout == mounted
