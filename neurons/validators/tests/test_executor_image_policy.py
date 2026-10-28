from services.executor_image_policy import (
    ExpectedImage,
    ExpectedImageSnapshot,
    ImageVerdict,
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


