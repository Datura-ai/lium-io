from __future__ import annotations

import asyncio
import re
import shlex
from collections.abc import Callable
from dataclasses import replace

from services.executor_connectivity.dind_probe import diagnose_docker_run_error
from services.executor_image_policy import (
    ImageVerdict,
    normalize_sha256_digest,
    outdated_image_remediation,
)

from ..messages import ExecutorImageMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context
from .rental_owner import rental_of_this_node

_EXECUTOR_NAME = re.compile(r"(?:^|-)executor-\d+$")
# Same GPU probe the executor updater runs before an update (neurons/executor/EXECUTOR_UPDATE.md),
# with its label so the updater removes a container this one leaves behind.
_UPDATE_HOLD_PROBE_TIMEOUT_SECONDS = 60


async def updater_holds_on_nvidia_mismatch(ctx: Context) -> bool:
    """Whether a new GPU container fails on this host with the NVIDIA driver/library mismatch, the
    condition on which the executor updater holds the node on its old image. False on any doubt."""
    docker = (ctx.state.specs or {}).get("docker")
    container_id = docker.get("container_id") if isinstance(docker, dict) else None
    if not isinstance(container_id, str) or not container_id:
        return False
    image = f"$(/usr/bin/docker inspect --format '{{{{.Image}}}}' {shlex.quote(container_id)})"
    cmd = (
        "/usr/bin/docker run --rm --gpus all --label io.lium.watchtower.probe=nvidia-runtime "
        f'--entrypoint true "{image}"'
    )
    try:
        result = await asyncio.wait_for(
            ctx.ssh.run(cmd, check=False), timeout=_UPDATE_HOLD_PROBE_TIMEOUT_SECONDS
        )
    except Exception:  # noqa: BLE001
        return False
    if getattr(result, "exit_status", 0) == 0:
        return False
    cause = diagnose_docker_run_error(getattr(result, "stderr", None))
    return cause is not None and cause.code == "NVIDIA_RUNTIME_MISMATCH"


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
        updated_state = replace(ctx.state, executor_image_report=report)
        what = report.as_dict()

        if report.status is ImageVerdict.CURRENT:
            event = render_message(Msg.CURRENT, ctx=ctx, check_id=self.check_id, what=what)
            passed = True
        else:
            rented_executor = rental_of_this_node(ctx)
            is_rented = rented_executor is not None and len(rented_executor.pods) > 0
            passed = is_rented
            if is_rented and await updater_holds_on_nvidia_mismatch(ctx):
                report = replace(report, update_held=True)
                updated_state = replace(updated_state, executor_image_report=report)
                what = report.as_dict()
            event = render_message(
                Msg.OUTDATED,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
                impact=(
                    "Rental continues and incentive is kept: the updater holds this host on the old "
                    "image until the NVIDIA driver/library mismatch is fixed (reboot once the rental ends)"
                    if report.update_held
                    else "Rental can continue, but no incentive until the image is current"
                    if is_rented
                    else "Validation failed - executor unavailable for rent until the image is current"
                ),
                remediation=(
                    "Once the rental ends, reboot the host or reload the NVIDIA driver, then run "
                    "`docker compose up -d`; the updater then installs the current image."
                    if report.update_held
                    else outdated_image_remediation(report.expected_ref)
                ),
                extra={
                    "executor_image_status": report.status.value,
                    "executor_image_check_enforced": True,
                },
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
