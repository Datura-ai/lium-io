import pytest
from services.executor_image_policy import (
    ExpectedImage,
    ExpectedImageSnapshot,
    ImageVerdict,
    build_expected_image_snapshot,
)

EXECUTOR_DIGEST = f"sha256:{'a' * 64}"


def policy() -> ExpectedImageSnapshot:
    return ExpectedImageSnapshot(
        executor=ExpectedImage("daturaai/compute-subnet-executor:latest", EXECUTOR_DIGEST),
        executor_ref="daturaai/compute-subnet-executor:latest",
    )


def test_matching_digest_is_current():
    assert policy().report(EXECUTOR_DIGEST).status is ImageVerdict.CURRENT


def test_mismatch_is_outdated():
    assert policy().report(f"sha256:{'c' * 64}").status is ImageVerdict.OUTDATED


def test_missing_local_digest_is_outdated():
    report = policy().report(None)

    assert report.status is ImageVerdict.OUTDATED
    assert report.observed_digest is None


def test_missing_expected_digest_cannot_report():
    empty = ExpectedImageSnapshot(
        executor=None,
        executor_ref="daturaai/compute-subnet-executor:latest",
    )

    with pytest.raises(ValueError, match="expected executor digest"):
        empty.report(EXECUTOR_DIGEST)


def test_expected_image_rejects_non_sha256_digest():
    with pytest.raises(ValueError, match="Invalid image digest"):
        ExpectedImage("repo:tag", "sha256:short")


def test_build_expected_image_snapshot_from_digest():
    snapshot = build_expected_image_snapshot(EXECUTOR_DIGEST)

    assert snapshot.executor is not None
    assert snapshot.executor.digest == EXECUTOR_DIGEST


def test_build_snapshot_skips_missing_digest():
    snapshot = build_expected_image_snapshot(None)

    assert snapshot.executor is None
