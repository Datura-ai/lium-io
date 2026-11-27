from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from core.config import settings

_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ImageVerdict(StrEnum):
    CURRENT = "CURRENT"
    OUTDATED = "OUTDATED"
    UNOFFICIAL = "UNOFFICIAL"


@dataclass(frozen=True)
class ExpectedImage:
    ref: str
    digest: str

    def __post_init__(self) -> None:
        if not _SHA256_DIGEST.fullmatch(self.digest):
            raise ValueError(f"Invalid image digest: {self.digest}")


@dataclass(frozen=True)
class ExecutorImageReport:
    status: ImageVerdict
    observed_digest: str | None
    expected_ref: str
    expected_digest: str
    # DAH-6845: the rental state at the moment the digest was read; an OUTDATED verdict excludes only a node that was
    # free then, so a rental ending later in the same cycle never judges a reading taken while it ran
    rented_when_read: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "observed_digest": self.observed_digest,
            "expected_ref": self.expected_ref,
            "expected_digest": self.expected_digest,
            "rented_when_read": self.rented_when_read,
        }


@dataclass(frozen=True)
class ExpectedImageSnapshot:
    executor: ExpectedImage | None
    executor_ref: str
    # DAH-6560: the release just replaced, set only while the rollout window covers the cycle; a node
    # still on it has not been reached by Watchtower yet, so it reads CURRENT instead of OUTDATED
    rollout_previous_digest: str | None = None
    # releases replaced before that one whose own grace still runs (two pushes close together)
    rollout_earlier_digests: tuple[str, ...] = ()

    def report(self, observed_digest: str | None) -> ExecutorImageReport:
        if self.executor is None:
            raise ValueError("Cannot report without an expected executor digest")
        status = (
            ImageVerdict.CURRENT
            if observed_digest is not None
            and observed_digest in (self.executor.digest, self.rollout_previous_digest, *self.rollout_earlier_digests)
            else ImageVerdict.OUTDATED
        )
        return ExecutorImageReport(
            status=status,
            observed_digest=observed_digest,
            expected_ref=self.executor_ref,
            expected_digest=self.executor.digest,
        )


def normalize_sha256_digest(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    digest = value.strip().lower()
    return digest if _SHA256_DIGEST.fullmatch(digest) else None


def build_expected_image_snapshot(digest: str | None) -> ExpectedImageSnapshot:
    ref = settings.EXECUTOR_IMAGE_REF
    normalized = normalize_sha256_digest(digest)
    executor = ExpectedImage(ref=ref, digest=normalized) if normalized else None
    return ExpectedImageSnapshot(executor=executor, executor_ref=ref)


def outdated_image_remediation(expected_ref: str, status: str | None) -> str:
    # a rented node keeps its incentive only on an OUTDATED image; an UNOFFICIAL one earns nothing, rented or not
    consequence = (
        "While the node is free it fails validation, earns nothing and is off the rent list until the executor "
        "image matches the current release; a rented node keeps its rental and incentive."
        if status == "OUTDATED"
        else ""
    )
    return (
        f"This node is not running the current {expected_ref} image. "
        "On a standard stack, confirm executor-executor-runner-1 and executor-watchtower-1 "
        "are running so Watchtower can pull and redeploy the latest executor automatically. "
        "If auto-update stopped, follow "
        "https://docs.lium.io/providers/nodes/gpu-power-cap#the-standard-stack-which-stopped-updating. "
        f"{consequence}"
    ).rstrip()
