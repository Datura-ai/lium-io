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
# Loki key for the cycle-level verdict below, enforced or observed.
ACROSS_MINERS_OUTCOME = "DUPLICATE_ACROSS_MINERS"

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
    """One copy of a machine that another miner hotkey keeps the score for this cycle."""

    executor_uuid: str
    ip_port: str
    kept_hotkey: str
    dropped_hotkey: str
    matched_on: tuple[str, ...]
    enforced: bool


def _gpu_uuids(result: JobResult) -> list[str]:
    details = ((result.spec or {}).get("gpu") or {}).get("details") or []
    return sorted({str(d["uuid"]) for d in details if isinstance(d, dict) and d.get("uuid")})


def _identity_keys(result: JobResult, match_gpu_uuid: bool) -> list[tuple[str, str]]:
    info = result.executor_info
    keys = [
        (MATCH_EXECUTOR_UUID, str(info.uuid).lower()),
        (MATCH_IP_PORT, f"{info.address}:{info.port}"),
    ]
    if match_gpu_uuid:
        keys.extend((MATCH_GPU_UUID, uuid) for uuid in _gpu_uuids(result))
    return keys


def keep_one_miner_per_executor(
    job_results: dict[str, list[JobResult]],
    default_extra: dict | None = None,
) -> list[AcrossMinersDuplicate]:
    """One machine earns under one miner hotkey per cycle.

    Two scored results are one machine when they share an executor UUID, an executor ip:port, or
    (with DUPLICATE_EXECUTOR_MATCH_GPU_UUID) a GPU UUID; the match is transitive. When the copies
    of one machine come from more than one hotkey, the hotkey that sorts first (plain string
    order of the SS58 address) keeps the score: the rule needs no data beyond the cycle and picks
    the same hotkey every cycle. Every copy of every other hotkey is logged with both hotkeys.

    DUPLICATE_EXECUTOR_DRY_RUN on: logged only, no result changes. Off: those copies score 0
    this cycle with EXECUTOR_DUPLICATE_ACROSS_MINERS and `duplicate_kept_by` set. Copies within
    one hotkey are not touched here (MinerService keeps one entry per executor UUID).
    Unscored results take no part: they earn nothing to pay twice.
    """
    enforce = not settings.DUPLICATE_EXECUTOR_DRY_RUN
    match_gpu_uuid = settings.DUPLICATE_EXECUTOR_MATCH_GPU_UUID

    entries: list[tuple[str, JobResult, list[tuple[str, str]]]] = []
    for hotkey in sorted(job_results):
        for result in job_results[hotkey]:
            if result.is_successful:
                entries.append((hotkey, result, _identity_keys(result, match_gpu_uuid)))

    parent = list(range(len(entries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_with_key: dict[tuple[str, str], int] = {}
    for index, (_, _, keys) in enumerate(entries):
        for key in keys:
            other = first_with_key.setdefault(key, index)
            if other != index:
                parent[find(index)] = find(other)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(entries)):
        groups[find(index)].append(index)

    duplicates: list[AcrossMinersDuplicate] = []
    for members in groups.values():
        hotkeys = {entries[i][0] for i in members}
        if len(hotkeys) < 2:
            continue
        kept_hotkey = min(hotkeys)
        kept_keys = {key for i in members if entries[i][0] == kept_hotkey for key in entries[i][2]}
        kept_uuids = sorted(
            str(entries[i][1].executor_info.uuid) for i in members if entries[i][0] == kept_hotkey
        )
        group_keys: dict[tuple[str, str], int] = defaultdict(int)
        for i in members:
            for key in entries[i][2]:
                group_keys[key] += 1
        for i in members:
            hotkey, result, keys = entries[i]
            if hotkey == kept_hotkey:
                continue
            shared = {key for key in keys if key in kept_keys} or {key for key in keys if group_keys[key] > 1}
            matched_on = tuple(sorted({kind for kind, _ in shared}))
            info = result.executor_info
            duplicate = AcrossMinersDuplicate(
                executor_uuid=str(info.uuid),
                ip_port=f"{info.address}:{info.port}",
                kept_hotkey=kept_hotkey,
                dropped_hotkey=hotkey,
                matched_on=matched_on,
                enforced=enforce,
            )
            duplicates.append(duplicate)
            what = {
                "executor_uuid": duplicate.executor_uuid,
                "ip_port": duplicate.ip_port,
                "miner_hotkey": hotkey,
                "kept_by_hotkey": kept_hotkey,
                "kept_executor_uuids": kept_uuids,
                "matched_on": list(matched_on),
            }
            logger.warning(
                _m(
                    "Executor scored under two miner hotkeys this cycle; one hotkey keeps the score"
                    if enforce
                    else "Executor scored under two miner hotkeys this cycle (observe mode, both keep the score)",
                    extra=get_extra_info(
                        {
                            **(default_extra or {}),
                            **what,
                            "outcome": ACROSS_MINERS_OUTCOME,
                            "enforced": enforce,
                            "job_batch_id": result.job_batch_id,
                            "score": result.score,
                            "job_score": result.job_score,
                        }
                    ),
                )
            )
            if not enforce:
                continue
            result.score = 0
            result.job_score = 0
            result.failure_reason_code = Msg.ACROSS_MINERS.reason
            result.validation_event = render_message(
                Msg.ACROSS_MINERS, ctx=None, check_id=CYCLE_CHECK_ID, what=what
            )
            result.duplicate_kept_by = kept_hotkey
    return duplicates
