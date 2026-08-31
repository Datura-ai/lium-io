from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace

from services.executor_image_policy import (
    ImageVerdict,
    normalize_sha256_digest,
    outdated_image_remediation,
)

from ..messages import ExecutorImageMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

_EXECUTOR_NAME = re.compile(r"(?:^|-)executor-\d+$")


def _single_digest(
    containers: list[object],
    matches: Callable[[dict[str, object]], bool],
) -> str | None:
    matched = [
        container for container in containers if isinstance(container, dict) and matches(container)
    ]
    if len(matched) != 1:
        return None
    return normalize_sha256_digest(matched[0].get("digest"))


def observed_executor_digest(specs: dict[str, object]) -> str | None:
    docker = specs.get("docker")
    if not isinstance(docker, dict):
        return None
    containers = docker.get("containers")
    if not isinstance(containers, list):
        return None

    executor_container_id = docker.get("container_id")
    if isinstance(executor_container_id, str) and executor_container_id:
        digest = _single_digest(
            containers,
            lambda container: container.get("container_id") == executor_container_id,
        )
        if digest is not None:
            return digest

    return _single_digest(
        containers,
        lambda container: bool(
            isinstance(container.get("name"), str) and _EXECUTOR_NAME.search(container["name"])
        ),
    )


class ExecutorImageCheck:
    check_id = "executor.validate.image_version"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        snapshot = ctx.config.executor_image_snapshot
        if snapshot is None or snapshot.executor is None:
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        report = snapshot.report(observed_executor_digest(ctx.state.specs))
        updated_state = replace(ctx.state, executor_image_report=report)
        what = report.as_dict()

        if report.status is ImageVerdict.CURRENT:
            event = render_message(Msg.CURRENT, ctx=ctx, check_id=self.check_id, what=what)
            passed = True
        else:
            rented_data = ctx.state.rented_data
            rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
            is_rented = rented_executor is not None and len(rented_executor.pods) > 0
            passed = is_rented
            event = render_message(
                Msg.OUTDATED,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
                impact=(
                    "Rental can continue, but no incentive until the image is current"
                    if is_rented
                    else "Validation failed - executor unavailable for rent until the image is current"
                ),
                remediation=outdated_image_remediation(report.expected_ref),
                extra={"executor_image_status": report.status.value},
            )
        return CheckResult(
            passed=passed,
            event=event,
            updates={
                "state": updated_state,
                "default_extra": {
                    **ctx.default_extra,
                    "executor_image_status": report.status.value,
                },
            },
        )
