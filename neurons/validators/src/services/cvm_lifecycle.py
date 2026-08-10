"""The validation-CVM lifecycle, driven by this validator (DAH-2629 + DAH-2630).

Two closely-related jobs share this module because they share everything else — the host
registry, the signed state read, and the budgets:

**DAH-2629 — bring the validation CVM up.** The locked design says the validator requests
validation CVMs from the daemon; cvmd grants ``validation`` to the validator key alone.
So when a cvmd host has no CVM — first sight after registration, after a renter teardown,
after a crash — this validator launches one, pinning the triple from the newest
validation entry of the platform's signed catalog, and verifies the launch report against
that same triple. A launch failure is a scoring signal against the node (it simply stays
unverifiable), not an operator page.

**DAH-2630 — account for the switch window (FR-I3).** During the RENTED↔UNRENTED
transition the node is neither rentable nor probeable *by design*, so a node inside its
budgeted switch window is exempt from the zero-score its unreachability would otherwise
earn. The grace is synthesized the same way the special-manual-rental forced pass is —
a JobResult with ``spec=None`` so the backend's executor upsert is untouched — and a
switch that exceeds its budget is a flag, never a silently longer window.

**The registry.** Both jobs need to know, for a node that is currently unreachable, that
it is a cvmd host at all and where its daemon listens. That cannot come from the miner's
per-cycle answer (the executor inside a torn-down CVM is exactly what is missing), so it
is remembered: every attested validation cycle records the host, and every relayed
provision/teardown refreshes it. Entries age out so a decommissioned node stops being
probed. A brand-new host therefore enters the registry with its first attested cycle —
the very first CVM on a virgin host still comes from day-zero provisioning.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from core.config import settings
from core.utils import _m, get_extra_info
from services.cvmd_client import CvmdClient
from services.cvmd_relay import CvmdRelayError
from services.redis_service import RedisService

logger = logging.getLogger(__name__)

CVMD_HOSTS_KEY = "cvmd_hosts"
CVM_LAUNCH_ATTEMPT_PREFIX = "cvm_launch_attempt"

# Node states as cvmd's /v1/state reports them (cvmd/state/machine.py — string values are
# the wire contract; RECONCILING doubles as idle).
STATE_RECONCILING = "RECONCILING"
STATE_LAUNCHING = "LAUNCHING"
STATE_VALIDATION_RUNNING = "VALIDATION_RUNNING"
STATE_RENTER_RUNNING = "RENTER_RUNNING"
STATE_TEARDOWN = "TEARDOWN"
STATE_SWITCHING = "SWITCHING"
STATE_FAILED = "FAILED"

# Log event names — Grafana/Loki query keys, so they are constants rather than prose.
CVM_SWITCH_GRACE_EVENT = "CVM_SWITCH_GRACE"
CVM_SWITCH_GRACE_OBSERVED_EVENT = "CVM_SWITCH_GRACE_OBSERVED"
CVM_SWITCH_BUDGET_EXCEEDED_EVENT = "CVM_SWITCH_BUDGET_EXCEEDED"
CVM_LAUNCH_STARTED_EVENT = "CVM_VALIDATION_LAUNCH_STARTED"
CVM_LAUNCH_VERIFIED_EVENT = "CVM_VALIDATION_LAUNCH_VERIFIED"
CVM_LAUNCH_FAILED_EVENT = "CVM_VALIDATION_LAUNCH_FAILED"


@dataclass(frozen=True)
class CvmdHost:
    """One registered cvmd host: enough to reach its daemon and to score its grace."""

    executor_uuid: str
    address: str
    miner_hotkey: str
    gpu_model: str | None = None
    gpu_count: int = 0
    updated_at: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "executor_uuid": self.executor_uuid,
                "address": self.address,
                "miner_hotkey": self.miner_hotkey,
                "gpu_model": self.gpu_model,
                "gpu_count": self.gpu_count,
                "updated_at": self.updated_at,
            }
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "CvmdHost | None":
        try:
            data = json.loads(raw)
            return cls(
                executor_uuid=str(data["executor_uuid"]),
                address=str(data["address"]),
                miner_hotkey=str(data["miner_hotkey"]),
                gpu_model=data.get("gpu_model"),
                gpu_count=int(data.get("gpu_count") or 0),
                updated_at=float(data.get("updated_at") or 0.0),
            )
        except (ValueError, KeyError, TypeError):
            return None

    def cvmd_url(self) -> str:
        if settings.CVMD_URL_OVERRIDE:
            return settings.CVMD_URL_OVERRIDE
        return f"https://{self.address}:{settings.CVMD_PORT}"


@dataclass(frozen=True)
class SwitchAssessment:
    """What one state read says about a possibly-switching node."""

    reachable: bool
    state: str | None = None
    has_cvm: bool = False
    switching: bool = False
    elapsed_seconds: float | None = None

    @property
    def idle_without_cvm(self) -> bool:
        return self.state == STATE_RECONCILING and not self.has_cvm


def _parse_started_at(last_switch: dict | None) -> datetime | None:
    if not isinstance(last_switch, dict):
        return None
    raw = last_switch.get("started_at")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class CvmLifecycleService:
    """Registry, state assessment, and validation-CVM launches for cvmd hosts."""

    def __init__(self, redis_service: RedisService, whitelist_source, keypair) -> None:
        self.redis_service = redis_service
        self.whitelist_source = whitelist_source
        self.client = CvmdClient(keypair)
        # Hosts with a launch in flight in THIS process, so one slow launch is not joined
        # by a second from the next cycle. The Redis cooldown covers everything else.
        self._launching: set[str] = set()

    # ------------------------------------------------------------------ registry

    async def record_host(
        self,
        *,
        executor_uuid: str,
        address: str,
        miner_hotkey: str,
        gpu_model: str | None = None,
        gpu_count: int = 0,
    ) -> None:
        """Remember (or refresh) one cvmd host. Merge-updates so a relay that knows only
        the address does not erase the gpu shape an attested cycle recorded."""
        existing = None
        try:
            raw = await self.redis_service.hget(CVMD_HOSTS_KEY, executor_uuid)
            if raw:
                existing = CvmdHost.from_json(raw)
        except Exception:  # noqa: BLE001 - registry reads must never break a cycle
            existing = None

        host = CvmdHost(
            executor_uuid=executor_uuid,
            address=address,
            miner_hotkey=miner_hotkey,
            gpu_model=gpu_model or (existing.gpu_model if existing else None),
            gpu_count=gpu_count or (existing.gpu_count if existing else 0),
            updated_at=time.time(),
        )
        try:
            await self.redis_service.hset(CVMD_HOSTS_KEY, executor_uuid, host.to_json())
        except Exception as exc:  # noqa: BLE001
            logger.warning(_m(
                "Could not record a cvmd host",
                extra=get_extra_info({"executor_uuid": executor_uuid, "error": str(exc)}),
            ))

    async def hosts_for_miner(self, miner_hotkey: str) -> list[CvmdHost]:
        """This miner's registered cvmd hosts, with aged-out entries pruned as they are seen."""
        try:
            raw_map = await self.redis_service.hgetall(CVMD_HOSTS_KEY) or {}
        except Exception:  # noqa: BLE001
            return []

        hosts: list[CvmdHost] = []
        now = time.time()
        for field, raw in raw_map.items():
            host = CvmdHost.from_json(raw)
            if host is None:
                continue
            if now - host.updated_at > settings.CVMD_HOST_REGISTRY_TTL_SECONDS:
                field_name = field.decode() if isinstance(field, bytes) else str(field)
                try:
                    await self.redis_service.hdel(CVMD_HOSTS_KEY, field_name)
                except Exception:  # noqa: BLE001
                    pass
                continue
            if host.miner_hotkey == miner_hotkey:
                hosts.append(host)
        return hosts

    # ------------------------------------------------------------------ assessment

    async def assess(self, host: CvmdHost) -> SwitchAssessment:
        """One signed state read, folded into what the two callers need to decide."""
        try:
            result = await self.client.state(host.cvmd_url())
        except CvmdRelayError as exc:
            logger.info(_m(
                "cvmd unreachable during lifecycle assessment",
                extra=get_extra_info({"executor_uuid": host.executor_uuid, "error": str(exc)}),
            ))
            return SwitchAssessment(reachable=False)
        if not result.ok:
            return SwitchAssessment(reachable=False)

        body = result.body
        state = body.get("state")
        has_cvm = body.get("cvm") is not None
        switching = state in (STATE_TEARDOWN, STATE_SWITCHING)
        elapsed = None
        if switching:
            started_at = _parse_started_at(body.get("last_switch"))
            if started_at is not None:
                elapsed = max(0.0, (datetime.now(UTC) - started_at).total_seconds())
        return SwitchAssessment(
            reachable=True,
            state=state,
            has_cvm=has_cvm,
            switching=switching,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------ DAH-2629: launch

    async def ensure_validation_cvm(self, host: CvmdHost, *, assessment: SwitchAssessment | None = None) -> bool:
        """Launch the validation CVM on an idle, empty host. True only on a verified launch.

        Refusals are all fail-closed and named: no catalog in force, no validation entry,
        a host that is not actually idle, a cooldown still running. The launch itself is
        slow (it holds until the guest boots) — callers run this as its own task.
        """
        if not settings.ENABLE_CVM_LIFECYCLE:
            return False
        if host.executor_uuid in self._launching:
            return False

        extra = get_extra_info({
            "executor_uuid": host.executor_uuid,
            "miner_hotkey": host.miner_hotkey,
            "cvmd_url": host.cvmd_url(),
        })

        cooldown_key = f"{CVM_LAUNCH_ATTEMPT_PREFIX}:{host.executor_uuid}"
        try:
            if await self.redis_service.get(cooldown_key):
                return False
        except Exception:  # noqa: BLE001 - no Redis, no cooldown; the in-process set still guards
            pass

        catalog = self.whitelist_source.current()
        entry = catalog.newest_validation_entry() if catalog else None
        if entry is None:
            # Never guess a stack. A validator that launched from anything but the signed
            # catalog would be creating exactly the unattributable CVM the design forbids.
            logger.warning(_m(
                f"{CVM_LAUNCH_FAILED_EVENT} — no signed catalog entry to launch from",
                extra=extra,
            ))
            return False

        if assessment is None:
            assessment = await self.assess(host)
        if not assessment.reachable or not assessment.idle_without_cvm:
            return False

        self._launching.add(host.executor_uuid)
        try:
            try:
                await self.redis_service.set_with_expiration(
                    cooldown_key, str(time.time()), settings.CVM_LAUNCH_COOLDOWN_SECONDS
                )
            except Exception:  # noqa: BLE001
                pass

            logger.info(_m(
                CVM_LAUNCH_STARTED_EVENT,
                extra={**extra, "artifact_id": entry.id, "qemu": entry.qemu,
                       "compose_hash": entry.compose_hash},
            ))
            try:
                result = await self.client.launch_validation(
                    host.cvmd_url(),
                    qemu=entry.qemu,
                    os_image_hash=entry.os_image_hash,
                    compose_hash=entry.compose_hash,
                )
            except CvmdRelayError as exc:
                logger.warning(_m(
                    f"{CVM_LAUNCH_FAILED_EVENT} — host not reached",
                    extra={**extra, "error": str(exc)},
                ))
                return False

            if not result.ok:
                logger.warning(_m(
                    f"{CVM_LAUNCH_FAILED_EVENT} — cvmd refused",
                    extra={**extra, "status": result.status, "detail": result.reason()},
                ))
                return False

            measurements = (result.body or {}).get("measurements") or {}
            mismatched = [
                field
                for field, expected in (
                    ("qemu", entry.qemu),
                    ("os_image_hash", entry.os_image_hash),
                    ("compose_hash", entry.compose_hash),
                )
                if measurements.get(field) != expected
            ]
            if mismatched:
                # A 201 measuring as something else is worse than a refusal: the host is
                # running a stack the catalog does not describe. The normal verification
                # cycle will fail it; this log says why before it does.
                logger.error(_m(
                    f"{CVM_LAUNCH_FAILED_EVENT} — launch report disagrees with the pinned triple",
                    extra={**extra, "mismatched": mismatched},
                ))
                return False

            logger.info(_m(
                CVM_LAUNCH_VERIFIED_EVENT,
                extra={**extra, "instance_id": (result.body or {}).get("instance_id")},
            ))
            return True
        finally:
            self._launching.discard(host.executor_uuid)

    def schedule_ensure(self, host: CvmdHost, *, assessment: SwitchAssessment | None = None) -> None:
        """Fire the launch as its own task — a boot must never extend a validation cycle."""
        asyncio.create_task(self.ensure_validation_cvm(host, assessment=assessment))
