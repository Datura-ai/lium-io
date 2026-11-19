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




PREVIOUS_DIGEST = f"sha256:{'b' * 64}"


def test_the_replaced_release_is_current_only_while_the_rollout_window_names_it():
    # DAH-6560: one cycle after each executor release every node not yet pulled read OUTDATED
    in_window = ExpectedImageSnapshot(
        executor=ExpectedImage("daturaai/compute-subnet-executor:latest", EXECUTOR_DIGEST),
        executor_ref="daturaai/compute-subnet-executor:latest",
        rollout_previous_digest=PREVIOUS_DIGEST,
    )

    assert in_window.report(PREVIOUS_DIGEST).status is ImageVerdict.CURRENT
    assert in_window.report(EXECUTOR_DIGEST).status is ImageVerdict.CURRENT
    assert in_window.report(f"sha256:{'c' * 64}").status is ImageVerdict.OUTDATED
    assert in_window.report(None).status is ImageVerdict.OUTDATED
    assert policy().report(PREVIOUS_DIGEST).status is ImageVerdict.OUTDATED
    two_back = f"sha256:{'c' * 64}"
    after_two_pushes = ExpectedImageSnapshot(
        executor=ExpectedImage("daturaai/compute-subnet-executor:latest", EXECUTOR_DIGEST),
        executor_ref="daturaai/compute-subnet-executor:latest",
        rollout_previous_digest=PREVIOUS_DIGEST,
        rollout_earlier_digests=(two_back,),
    )
    assert after_two_pushes.report(two_back).status is ImageVerdict.CURRENT
    assert after_two_pushes.report(f"sha256:{'e' * 64}").status is ImageVerdict.OUTDATED
