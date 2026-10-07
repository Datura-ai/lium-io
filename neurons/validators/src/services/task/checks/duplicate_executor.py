from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from core.config import settings
from core.utils import _m, get_extra_info, get_logger
from services.redis_service import DUPLICATED_MACHINE_SET

from ..messages import DuplicateExecutorMessages as Msg, render_message
from ..models import JobResult
from ..pipeline import CheckResult, Context

logger = get_logger(__name__)

CYCLE_CHECK_ID = "executor.cycle.duplicate_across_miners"
# Loki key for the cycle-level verdicts below, enforced or observed.
ACROSS_MINERS_OUTCOME = "DUPLICATE_ACROSS_MINERS"

MATCH_SSH_ENDPOINT = "ssh_endpoint"
MATCH_EXECUTOR_UUID = "executor_uuid"
MATCH_IP_PORT = "ip_port"
MATCH_GPU_UUID = "gpu_uuid"


class DuplicateExecutorCheck:
    """Ensure a miner is not registering the same executor UUID multiple times.

    The original pipeline cleared verification when Redis flagged duplicates. Keeping the
    guard avoids wasted scoring cycles and enforces one-to-one miner/executor mappings.

    The set is fed by the backend and keyed per miner hotkey, so it cannot see one machine that
    two hotkeys report under one executor UUID (the backend keeps a single row for it). That case
    is settled once per cycle, after every pipeline has ended, by `keep_one_miner_per_executor`.
    """

    check_id = "executor.validate.duplicate"
    fatal = True

    async def run(self, ctx: Context) -> CheckResult:
        redis_service = ctx.services.redis
        is_duplicate = await redis_service.is_elem_exists_in_set(
            DUPLICATED_MACHINE_SET,
            f"{ctx.miner_hotkey}:{ctx.executor.uuid}",
        )

        if is_duplicate:
            what: dict[str, str] = {
                "executor_uuid": ctx.executor.uuid,
                "miner_hotkey": ctx.miner_hotkey,
            }
            if settings.DUPLICATE_EXECUTOR_DRY_RUN:
                event = render_message(
                    Msg.DUPLICATE_OBSERVED,
                    ctx=ctx,
                    check_id=self.check_id,
                    what=what,
                )
                return CheckResult(passed=True, event=event)

            event = render_message(
                Msg.DUPLICATE,
                ctx=ctx,
                check_id=self.check_id,
                what=what,
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"clear_verified_job_info": True},
            )

        event = render_message(
            Msg.UNIQUE,
            ctx=ctx,
            check_id=self.check_id,
            what={"executor_uuid": ctx.executor.uuid},
        )
        return CheckResult(passed=True, event=event)


@dataclass(frozen=True)
class AcrossMinersDuplicate:
    """One scored copy of a machine that two or more miner hotkeys reported this cycle."""

    executor_uuid: str
    hotkey: str
    other_hotkeys: tuple[str, ...]
    matched_on: str
    # set when this copy loses its score: the hotkey that keeps it
    kept_by: str | None
    enforced: bool


def _ssh_endpoint(result: JobResult) -> str:
    info = result.executor_info
    return f"{info.address.strip().lower()}:{info.ssh_port}"


def _gpu_uuids(result: JobResult) -> list[str]:
    details = ((result.spec or {}).get("gpu") or {}).get("details") or []
    return sorted({str(d["uuid"]) for d in details if isinstance(d, dict) and d.get("uuid")})


def _reported_keys(result: JobResult) -> list[tuple[str, str]]:
    info = result.executor_info
    keys = [
        (MATCH_EXECUTOR_UUID, str(info.uuid).lower()),
        (MATCH_IP_PORT, f"{info.address.strip().lower()}:{info.port}"),
    ]
    keys.extend((MATCH_GPU_UUID, uuid) for uuid in _gpu_uuids(result))
    return keys


def _keeper(copies: list[tuple[str, JobResult]], rental_hotkeys: dict[str, str]) -> str:
    """The hotkey the backend's rental names, then a hotkey with a rented copy, then the hotkey
    whose SS58 address sorts first. Copies under one executor UUID all read as rented, so the
    rental's own hotkey decides between them."""
    hotkeys = {hotkey for hotkey, _ in copies}
    holders = {
        rental_hotkeys.get(str(result.executor_info.uuid).lower()) for _, result in copies if result.is_rented
    } & hotkeys
    rented = {hotkey for hotkey, result in copies if result.is_rented}
    return min(holders or rented or hotkeys)


def _log(duplicate: AcrossMinersDuplicate, result: JobResult, default_extra: dict | None) -> None:
    if duplicate.kept_by is not None:
        message = (
            "Executor scored under two miner hotkeys this cycle; one hotkey keeps the score"
            if duplicate.enforced
            else "Executor scored under two miner hotkeys this cycle (observe mode, both keep the score)"
        )
    else:
        message = "Executor reported under two miner hotkeys this cycle on a node-reported key (logged only)"
    logger.warning(
        _m(
            message,
            extra=get_extra_info(
                {
                    **(default_extra or {}),
                    "outcome": ACROSS_MINERS_OUTCOME,
                    "executor_uuid": duplicate.executor_uuid,
                    "miner_hotkey": duplicate.hotkey,
                    "other_hotkeys": list(duplicate.other_hotkeys),
                    "kept_by_hotkey": duplicate.kept_by,
                    "matched_on": duplicate.matched_on,
                    "enforced": duplicate.enforced,
                    "job_batch_id": result.job_batch_id,
                    "score": result.score,
                    "job_score": result.job_score,
                }
            ),
        )
    )


def keep_one_miner_per_executor(
    job_results: dict[str, list[JobResult]],
    default_extra: dict | None = None,
    rental_hotkeys: dict[str, str] | None = None,
) -> list[AcrossMinersDuplicate]:
    """One machine earns under one miner hotkey per cycle.

    Scored copies from different hotkeys that the validator reached on the same SSH endpoint
    (address and ssh_port) are one machine: each request installs its own fresh key, so a
    passing login there means that machine admitted that hotkey. One hotkey keeps the score
    (`_keeper`; `rental_hotkeys` maps a rented executor UUID to the hotkey its rental names); the copies of the others are logged with every hotkey involved and, with
    DUPLICATE_EXECUTOR_DRY_RUN off, score 0 with EXECUTOR_DUPLICATE_ACROSS_MINERS.

    Matches on what the node itself reports (executor UUID, listed ip:port, GPU UUIDs) are
    logged only, never zeroed: any miner can report another node's values. One hotkey's own
    repeats are left to MinerService, unscored copies have nothing to pay twice, and a result
    the validator did not log in for (a forced pass, `ssh_port` 0) proves no endpoint.
    """
    enforce = not settings.DUPLICATE_EXECUTOR_DRY_RUN
    scored = [
        (hotkey, result)
        for hotkey in sorted(job_results)
        for result in job_results[hotkey]
        if result.is_successful
    ]

    by_endpoint: dict[str, list[tuple[str, JobResult]]] = defaultdict(list)
    for hotkey, result in scored:
        if result.executor_info.ssh_port > 0:
            by_endpoint[_ssh_endpoint(result)].append((hotkey, result))

    duplicates: list[AcrossMinersDuplicate] = []
    zeroed: list[tuple[JobResult, str, bool, dict]] = []
    for endpoint, copies in by_endpoint.items():
        hotkeys = {hotkey for hotkey, _ in copies}
        if len(hotkeys) < 2:
            continue
        kept_by = _keeper(copies, rental_hotkeys or {})
        kept_uuids = {str(r.executor_info.uuid).lower() for h, r in copies if h == kept_by}
        for hotkey, result in copies:
            if hotkey == kept_by:
                continue
            duplicate = AcrossMinersDuplicate(
                executor_uuid=str(result.executor_info.uuid),
                hotkey=hotkey,
                other_hotkeys=tuple(sorted(hotkeys - {hotkey})),
                matched_on=MATCH_SSH_ENDPOINT,
                kept_by=kept_by,
                enforced=enforce,
            )
            duplicates.append(duplicate)
            _log(duplicate, result, default_extra)
            what = {
                "executor_uuid": duplicate.executor_uuid,
                "ssh_endpoint": endpoint,
                "miner_hotkey": hotkey,
                "kept_by_hotkey": kept_by,
            }
            zeroed.append((result, kept_by, str(result.executor_info.uuid).lower() in kept_uuids, what))

    by_reported: dict[tuple[str, str], list[tuple[str, JobResult]]] = defaultdict(list)
    for hotkey, result in scored:
        for key in _reported_keys(result):
            by_reported[key].append((hotkey, result))
    reported_once: set[tuple[str, int]] = set()
    for (kind, _), copies in sorted(by_reported.items()):
        hotkeys = {hotkey for hotkey, _ in copies}
        if len(hotkeys) < 2 or len({_ssh_endpoint(result) for _, result in copies}) == 1:
            continue
        for hotkey, result in copies:
            if (hotkey, id(result)) in reported_once:
                continue
            reported_once.add((hotkey, id(result)))
            duplicate = AcrossMinersDuplicate(
                executor_uuid=str(result.executor_info.uuid),
                hotkey=hotkey,
                other_hotkeys=tuple(sorted(hotkeys - {hotkey})),
                matched_on=kind,
                kept_by=None,
                enforced=False,
            )
            duplicates.append(duplicate)
            _log(duplicate, result, default_extra)

    if enforce:
        for result, kept_by, shares_row, what in zeroed:
            result.score = 0
            result.job_score = 0
            result.failure_reason_code = Msg.ACROSS_MINERS.reason
            result.validation_event = render_message(
                Msg.ACROSS_MINERS, ctx=None, check_id=CYCLE_CHECK_ID, what=what
            )
            result.duplicate_kept_by = kept_by
            result.duplicate_shares_kept_row = shares_row
    return duplicates
