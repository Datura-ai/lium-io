"""Digest-aware template pulls in the legacy image-management loop (DAH-2461).

``manage_docker_images`` pulls the default template by tag unconditionally every
sweep; on hosts behind a stale provider registry mirror that silently re-poisons
the tag. With a backend digest the loop must skip when the local tag already
matches, and otherwise pull digest-pinned and re-point the tag.
"""

import subprocess
from unittest.mock import call, patch

import gpus_utility

DIGEST = "sha256:70bd5fa697877594b753a146e207ca4de66d9b875d606ae09e6ee7bac8f4f423"
REPO = "daturaai/pytorch"
TAG = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind"
TEMPLATE = f"{REPO}:{TAG}"


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_local_tag_matches_digest_true():
    # Arrange
    inspect_output = f'["{REPO}@{DIGEST}"]'
    with patch.object(
        gpus_utility.subprocess, "run", return_value=_completed(stdout=inspect_output)
    ) as run:
        # Act
        matches = gpus_utility._local_tag_matches_digest(TEMPLATE, DIGEST)

    # Assert
    assert matches is True
    run.assert_called_once_with(
        ['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', TEMPLATE],
        capture_output=True,
        text=True,
    )


def test_local_tag_matches_digest_false_on_other_digest():
    # Arrange
    inspect_output = '["daturaai/pytorch@sha256:2d19c94ce8a37c6fa364f8a6211d8b6dc1a44ece574c4c22ab5579925ce7a4c8"]'
    with patch.object(gpus_utility.subprocess, "run", return_value=_completed(stdout=inspect_output)):
        # Act
        matches = gpus_utility._local_tag_matches_digest(TEMPLATE, DIGEST)

    # Assert
    assert matches is False


def test_local_tag_matches_digest_false_when_inspect_fails():
    # Arrange: image not present locally
    with patch.object(gpus_utility.subprocess, "run", return_value=_completed(returncode=1)):
        # Act
        matches = gpus_utility._local_tag_matches_digest(TEMPLATE, DIGEST)

    # Assert
    assert matches is False


def test_pull_template_image_with_digest_pulls_pinned_and_retags():
    # Arrange
    with patch.object(
        gpus_utility.subprocess, "run", side_effect=[_completed(), _completed()]
    ) as run:
        # Act
        pulled = gpus_utility._pull_template_image(REPO, TAG, DIGEST)

    # Assert: content-addressed pull, then the tag is re-pointed at the pinned build
    assert pulled is True
    assert run.call_args_list == [
        call(['docker', 'pull', f'{REPO}@{DIGEST}'], capture_output=True, text=True),
        call(['docker', 'tag', f'{REPO}@{DIGEST}', TEMPLATE], capture_output=True, text=True),
    ]


def test_pull_template_image_with_digest_pull_failure_skips_retag():
    # Arrange
    with patch.object(
        gpus_utility.subprocess, "run", return_value=_completed(returncode=1, stderr="403")
    ) as run:
        # Act
        pulled = gpus_utility._pull_template_image(REPO, TAG, DIGEST)

    # Assert
    assert pulled is False
    assert run.call_count == 1


def test_pull_template_image_with_digest_tag_failure_reports_error():
    # Arrange
    with patch.object(
        gpus_utility.subprocess,
        "run",
        side_effect=[_completed(), _completed(returncode=1, stderr="no such image")],
    ):
        # Act
        pulled = gpus_utility._pull_template_image(REPO, TAG, DIGEST)

    # Assert
    assert pulled is False


def test_pull_template_image_without_digest_pulls_by_tag():
    # Arrange: legacy path for backends that send no digest
    with patch.object(gpus_utility.subprocess, "run", return_value=_completed()) as run:
        # Act
        pulled = gpus_utility._pull_template_image(REPO, TAG, None)

    # Assert
    assert pulled is True
    run.assert_called_once_with(['docker', 'pull', TEMPLATE], capture_output=True, text=True)
