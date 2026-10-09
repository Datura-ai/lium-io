"""lium-io#1435: an ``executor_cd_dev`` run from a branch other than main pushes to the ``-dev``
repositories, because the Docker Hub token it gets (environment ``dockerhub-push-dev``) covers
only those. The build scripts run here against a stub ``docker`` that records each build's tag
and keeps the compose file baked into the runner image.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

EXECUTOR_DIR = Path(__file__).resolve().parents[1]
DIGEST = "sha256:" + "ab" * 32
STUB_DOCKER = """#!/bin/bash
echo "$*" >> "$STUB_OUT/calls"
if [[ "$1" == build ]]; then
  context="${@: -1}"
  for arg in "$@"; do
    [[ "$prev" == --tag ]] && echo "$arg" >> "$STUB_OUT/tags"
    [[ "$arg" == targetFile=* ]] && cp "$context/${arg#targetFile=}" "$STUB_OUT/compose.yml"
    prev="$arg"
  done
fi
"""

pytestmark = pytest.mark.skipif(shutil.which("envsubst") is None, reason="docker_runner_build.sh needs envsubst")


def _build(tmp_path: Path, suffix: str | None) -> tuple[list[str], dict]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "docker"
    stub.write_text(STUB_DOCKER)
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VALIDATOR_", "IMAGE_REPO_SUFFIX"))}
    env.update(PATH=f"{bin_dir}:{env['PATH']}", STUB_OUT=str(tmp_path), TAG="dev", EXECUTOR_IMAGE_SHA256=DIGEST)
    if suffix is not None:
        env["IMAGE_REPO_SUFFIX"] = suffix
    for script in ("docker_build.sh", "docker_runner_build.sh"):
        subprocess.run(["bash", script], cwd=EXECUTOR_DIR, env=env, check=True, capture_output=True)
    tags = (tmp_path / "tags").read_text().split()
    services = yaml.safe_load((tmp_path / "compose.yml").read_text())["services"]
    return tags, services


def _executor_images(services: dict) -> set[str]:
    return {s["image"] for s in services.values() if s.get("image", "").startswith("daturaai/compute-subnet-executor@")
            or s.get("image", "").startswith("daturaai/compute-subnet-executor-dev@")}


def test_branch_build_names_only_dev_repositories(tmp_path):
    """Regression: a branch build tags or bakes in a release repository, so its push needs the
    release token (refused off main) or its runner pulls the release executor instead of its own."""
    tags, services = _build(tmp_path, "-dev")
    assert tags == ["daturaai/compute-subnet-executor-dev:dev", "daturaai/compute-subnet-executor-runner-dev:dev"]
    assert _executor_images(services) == {f"daturaai/compute-subnet-executor-dev@{DIGEST}"}


def test_release_build_names_are_unchanged_without_the_suffix(tmp_path):
    """Regression: the suffix leaks into the release path (``executor_cd_prod`` and a dev run from
    main set none), so the release images or the digest the runner pins move to a -dev name."""
    tags, services = _build(tmp_path, None)
    assert tags == ["daturaai/compute-subnet-executor:dev", "daturaai/compute-subnet-executor-runner:dev"]
    assert _executor_images(services) == {f"daturaai/compute-subnet-executor@{DIGEST}"}
