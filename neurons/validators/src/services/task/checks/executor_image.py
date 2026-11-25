from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import replace

from core.config import settings
from core.utils import _m, get_extra_info
from services.default_docker_image_digest_service import executor_build_published
from services.executor_image_policy import (
    ImageVerdict,
    normalize_sha256_digest,
    outdated_image_remediation,
)

from ..messages import ExecutorImageMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node

logger = logging.getLogger(__name__)

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
    replays_after_session_drop = True
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        if ctx.tdx_attestation_passed:
            event = render_message(
                Msg.SKIPPED,
                ctx=ctx,
                check_id=self.check_id,
                extra={"skip_reason": "cvm", "tdx_attestation_passed": True},
            )
            return CheckResult(passed=True, event=event)

        snapshot = ctx.config.executor_image_snapshot
        if snapshot is None or snapshot.executor is None:
            event = render_message(Msg.SKIPPED, ctx=ctx, check_id=self.check_id)
            return CheckResult(passed=True, event=event)

        report = snapshot.report(observed_executor_digest(ctx.state.specs))
        if report.status is ImageVerdict.OUTDATED:
            published = await executor_build_published(report.observed_digest)
            # a plain line, outside the event sink's repeat demotion, so a Loki count per cycle is exact
            logger.info(
                _m(
                    "EXECUTOR_BUILD",
                    extra=get_extra_info(
                        {
                            **ctx.default_extra,
                            "executor_build": {True: "official", False: "unofficial"}.get(published, "unknown"),
                            "observed_digest": report.observed_digest,
                            "executor_image_unofficial_enforced": True,
                        }
                    ),
                )
            )
            if published is False:
                report = replace(report, status=ImageVerdict.UNOFFICIAL)
        # DAH-6845: a free node fails and leaves the rent list until it updates; a rented one passes, and keeps its
        # score too when the image is only OUTDATED (score_calculator, DefaultIncentive), so its renter is untouched.
        # Judged on the rental state of this reading, carried on the report, not on the state at the end of the cycle
        rented_executor = rental_of_this_node(ctx)
        is_rented = rented_executor is not None and len(rented_executor.pods) > 0
        report = replace(report, rented_when_read=is_rented)
        updated_state = replace(ctx.state, executor_image_report=report)
        what = report.as_dict()

        if report.status is ImageVerdict.CURRENT:
            event = render_message(Msg.CURRENT, ctx=ctx, check_id=self.check_id, what=what)
            passed = True
        else:
            passed = is_rented
            impact = None  # the OUTDATED templates carry their own impact
            if report.status is ImageVerdict.UNOFFICIAL:
                template = Msg.UNOFFICIAL
                if is_rented:
                    impact = "Rental can continue, but no incentive until the image is official"
                else:
                    impact = "Validation failed - executor unavailable for rent until the image is current"
            else:
                template = Msg.RENTED_OUTDATED if is_rented else Msg.OUTDATED
            event = render_message(
                template,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
                impact=impact,
                remediation=outdated_image_remediation(report.expected_ref, report.status.value),
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
