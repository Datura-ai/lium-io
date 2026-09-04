from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from core.config import settings

_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ImageVerdict(StrEnum):
    CURRENT = "CURRENT"
    OUTDATED = "OUTDATED"


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

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "observed_digest": self.observed_digest,
            "expected_ref": self.expected_ref,
            "expected_digest": self.expected_digest,
        }


@dataclass(frozen=True)
class ExpectedImageSnapshot:
    executor: ExpectedImage | None
    executor_ref: str

    def report(self, observed_digest: str | None) -> ExecutorImageReport:
        if self.executor is None:
            raise ValueError("Cannot report without an expected executor digest")
        status = (
            ImageVerdict.CURRENT
            if observed_digest == self.executor.digest
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


def outdated_image_remediation(expected_ref: str) -> str:
    return (
        f"This node is not running the current {expected_ref} image. "
        "On a standard stack, confirm executor-executor-runner-1 and executor-watchtower-1 "
        "are running so Watchtower can pull and redeploy the latest executor automatically. "
        "If auto-update stopped, follow "
        "https://docs.lium.io/providers/nodes/gpu-power-cap#the-standard-stack-which-stopped-updating. "
        "The node earns no incentive until the executor image matches the current release."
    )
