from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import TypedDict

from protocol.vc_protocol.compute_requests import RentedExecutor, RentedExecutorsResponse
from services.const import DEFAULT_JOB_OWNER_LIUM, PORT_VERIFY_FIRST_MISS_TTL_SECONDS
from services.executor_connectivity.models import PortVerificationResult
from services.redis_service import VerifiedPortsMemory

from core.config import settings
from core.utils import _m, get_extra_info

from ..host_disk import (
    DISK_PRESSURE_MAX_DEFERRALS,
    SNAPSHOT_SECONDS,
    SYNCFS_POLL_SECONDS,
    DiskPressureReading,
    disk_pressure_command,
    parse_disk_pressure,
)
from ..messages import PortConnectivityMessages as Msg
from ..messages import render_message
from ..pipeline import CheckResult, Context

logger = logging.getLogger(__name__)

# machine_scrape.py check_sysbox_gpu_compatibility's message when its 30 s `docker run --runtime=sysbox-runc` timed
# out: a busy host, not a verdict on Sysbox
SYSBOX_TEST_TIMED_OUT = "Test command timed out."

VERIFIED_PORTS_MEMORY_TTL_SECONDS = 24 * 3600
DISK_PRESSURE_DEFERRALS_TTL_SECONDS = 24 * 3600
# the two snapshots, the syncfs poll cap and a margin for the sysfs walk
DISK_PRESSURE_READ_TIMEOUT_SECONDS = SNAPSHOT_SECONDS + SYNCFS_POLL_SECONDS + 15
# (miner hotkey, executor uuid) -> monotonic time its latest completed check went on record in full - the ports
# and the sysbox memory both; only such a memory is republished, so a check Redis refused or cut short, or a
# restart with an older verdict still in Redis, means a real check first (the module outlives the per-cycle
# check instance). An entry as old as the port memory's TTL is gone with it.
_COMPLETED_CHECK_SAVED_AT: dict[tuple[str, str], float] = {}


def _has_fresh_completed_check(key: tuple[str, str]) -> bool:
    saved_at = _COMPLETED_CHECK_SAVED_AT.get(key)
    return saved_at is not None and time.monotonic() - saved_at < VERIFIED_PORTS_MEMORY_TTL_SECONDS


def _record_completed_check_save_time(key: tuple[str, str]) -> None:
    # departed executors leave here on the next completion anywhere
    now = time.monotonic()
    for expired in [k for k, at in _COMPLETED_CHECK_SAVED_AT.items() if now - at >= VERIFIED_PORTS_MEMORY_TTL_SECONDS]:
        del _COMPLETED_CHECK_SAVED_AT[expired]
    _COMPLETED_CHECK_SAVED_AT[key] = now


class _DiskPressureSpec(TypedDict, total=False):
    disk_pressure: dict[str, str | float | int | bool | None]


class _PressureGateFields(_DiskPressureSpec, total=False):
    disk_pressure_eligible: bool
    disk_pressure_self_rent: bool
    default_job_owner: str | None
    has_verified_ports_memory: bool
    disk_pressure_would_defer: bool
    disk_pressure_deferrals_in_a_row: int


@dataclass(frozen=True)
class _PressureGate:
    """DAH-5244: the host disk reading before the port check and what the gate makes of it."""

    reading: DiskPressureReading | None = None
    # a rental not contracted as a self-rent, or a Lium-owned filler: the writer is not the node's operator
    deferral_eligible: bool = False
    self_rent: bool = False
    default_job_owner: str | None = None
    memory: VerifiedPortsMemory | None = None
    # the validator's own DinD probe found sysbox within its memory's lifetime (never the host's scrape)
    sysbox_verified: bool = False
    # the enforced streak this cycle included once reserved in Redis, else the count found there
    deferrals_in_a_row: int = 0
    would_defer: bool = False
    deferred: bool = False

    def event_fields(self) -> _PressureGateFields:
        if self.reading is None:
            return {}
        return {
            "disk_pressure": self.reading.as_spec(),
            "disk_pressure_eligible": self.deferral_eligible,
            "disk_pressure_self_rent": self.self_rent,
            "default_job_owner": self.default_job_owner,
            "has_verified_ports_memory": self.memory is not None,
            "disk_pressure_would_defer": self.would_defer,
            "disk_pressure_deferrals_in_a_row": self.deferrals_in_a_row,
        }


@dataclass(frozen=True)
class _SysboxMemoryUpdate:
    # every write this cycle owed the sysbox memory is in Redis (a read in the fallback included)
    is_saved: bool
    # ticket-0494: a rented node's probe measured nothing and the memory says sysbox
    use_last_verified: bool = False


class PortConnectivityCheck:
    """Verify Docker port mappings by running the batch verifier exactly like before.

    Connectivity failures used to abort the task immediately because miners could not be
    rented. This check preserves that contract and updates sysbox state for later scoring.
    """

    check_id = "executor.validate.port_connectivity"
    fatal = False

    async def run(self, ctx: Context) -> CheckResult:
        extra = {**ctx.default_extra}

        if not ctx.config.job_batch_id:
            event = render_message(
                Msg.CONFIG_MISSING,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=False, event=event)

        # Extract rented ports and pod names from context
        rented_data = ctx.state.rented_data
        rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
        rented_ports = rented_executor.get_rented_ports() if rented_executor else []
        rented_pod_names = [p.container_name for p in rented_executor.pods] if rented_executor else []
        # DAH-2527: an idle filler holds ports without creating a pod, so an executor running only
        # fillers has no rented_executor entry at all — hence the separate lookup.
        filler_ports = rented_data.get_filler_ports(ctx.executor.uuid) if rented_data else []

        gate = await self._disk_pressure_gate(ctx, rented_data, rented_executor, filler_ports)
        if gate.deferred:
            return await self._deferred(ctx, gate, extra, busy_ports=set(rented_ports) | set(filler_ports))

        # 26 Oct: an unrented node's scrape sysbox test timed out under load, so the probe ran without sysbox-runc,
        # failed, and the node scored 0 with Sysbox working. The timeout is provider-reported, so it is neither a
        # pass nor a miss: the validator's own probe measures Sysbox with sysbox-runc instead.
        scrape_error = (ctx.state.specs or {}).get("sysbox_runtime_scrape_error")
        probe_sysbox = ctx.state.sysbox_runtime or (rented_executor is None and scrape_error == SYSBOX_TEST_TIMED_OUT)

        connectivity_service = ctx.services.connectivity
        result = await connectivity_service.verify_ports(
            ctx.ssh,
            ctx.miner_hotkey,
            ctx.executor,
            probe_sysbox,
            rented_ports=rented_ports,
            rented_pod_names=rented_pod_names,
            filler_ports=filler_ports,
            log_ctx={
                "pipeline_id": ctx.pipeline_id,
                "job_batch_id": ctx.config.job_batch_id,
                "miner_hotkey": ctx.miner_hotkey,
                "executor_uuid": ctx.executor.uuid,
                "executor_ip": ctx.executor.address,
            },
        )
        if probe_sysbox and not ctx.state.sysbox_runtime:
            # F-16: the request alone is no verdict (a no-ports or errored run echoes it back); only the probe's
            # own in-container measurement may say Sysbox works
            result = replace(result, sysbox_runtime=result.sysbox_measured is True)
        # this probe's verdict is on record only once every write below succeeded; a cancellation thrown into
        # one of them leaves the memory in Redis older than the probe and the readiness gone
        readiness_key = (ctx.miner_hotkey, ctx.executor.uuid)
        _COMPLETED_CHECK_SAVED_AT.pop(readiness_key, None)
        ports_saved = await self._save_completed_ports_and_deferral_counter(ctx, result, gate)
        verified_port_count = len(result.successful_ports)
        probed_port_count = len(result.selected_ports)
        declared_port_count = result.declared_port_count
        extra_info: dict[str, object] = {
            "sysbox_runtime": result.sysbox_runtime,
            "verified_port_count": verified_port_count,
            "probed_port_count": probed_port_count,
            "declared_port_count": declared_port_count,
            "probe_tier": result.probe_tier,
            "dind_ok": result.dind_ok,
        }
        if result.dind_error:
            extra_info["dind_error"] = result.dind_error.text
        # event-only: kept out of default_extra so later checks' log lines stay small
        event_extra: dict[str, object] = {
            "port_ranges": [r.as_dict() for r in result.port_ranges],
            **gate.event_fields(),
        }
        updated_state = replace(
            ctx.state,
            specs={
                # DAH-3698: the grace marker is the validator's alone; one the provider's specs carry never survives
                **{key: value for key, value in ctx.state.specs.items() if key != "port_verify_first_miss"},
                "sysbox_runtime": result.sysbox_runtime,
                "verified_ports": [p.external for p in result.successful_ports],
                **_disk_pressure_spec(gate),
            },
            sysbox_runtime=result.sysbox_runtime,
            verified_port_count=verified_port_count,
            probed_port_count=probed_port_count,
            declared_port_count=declared_port_count,
            verified_port_pairs=[(p.internal, p.external) for p in result.successful_ports],
            dind_probe_error=result.dind_error,
        )

        if await self._should_keep_last_known_sysbox(ctx, result, extra_info):
            extra_info["sysbox_downgrade_tolerated"] = True
            updated_state = replace(
                updated_state,
                specs={
                    **updated_state.specs,
                    "sysbox_runtime": ctx.state.sysbox_runtime,
                },
                sysbox_runtime=ctx.state.sysbox_runtime,
            )

        sysbox_memory = await self._update_last_verified_sysbox(
            ctx, result, rented=rented_executor is not None, sysbox_now=updated_state.sysbox_runtime
        )
        if sysbox_memory.use_last_verified:
            extra_info["sysbox_from_last_verified"] = True
            updated_state = replace(
                updated_state,
                specs={**updated_state.specs, "sysbox_runtime": True},
                sysbox_runtime=True,
            )
        if ports_saved and sysbox_memory.is_saved:
            _record_completed_check_save_time(readiness_key)

        # status "ok" means at least one port answered; a partial loss is a miss too, or one transient probe
        # failure would cut a split remainder's idle pay at once
        if await self._is_first_port_verify_miss(ctx, ok=result.status == "ok" and not result.failed_ports):
            extra_info["port_verify_first_miss"] = True
            updated_state = replace(
                updated_state, specs={**updated_state.specs, "port_verify_first_miss": True}
            )

        total = len(result.successful_ports) + len(result.failed_ports)
        pct = (len(result.successful_ports) / total * 100) if total > 0 else 0
        dind_status = "ok" if result.dind_ok else "failed"
        batch_count = len(result.successful_ports) - (1 if result.dind_ok else 0)
        batch_status = "ok" if batch_count > 0 else "failed"
        ok_sample = sorted([p.internal for p in result.successful_ports])[:5]
        fail_sample = sorted([p.internal for p in result.failed_ports])[:5]
        elapsed = result.elapsed_sec or 0.0

        msg = (
            f"verification complete total_time={elapsed:.2f}s {pct:.0f}% available, "
            f"dind={dind_status} batch={batch_status} ok={len(result.successful_ports)}{ok_sample}"
            f" port_range={ctx.executor.port_range}, port_mappings={ctx.executor.port_mappings}"
        )
        if result.failed_ports:
            msg += f" fail={len(result.failed_ports)}{fail_sample}"

        if result.status != "ok":
            # Fetch fresh rental data from backend to ensure we have latest state
            backend_client = ctx.services.backend
            fresh_rented_data = await backend_client.get_all_rented_executors()

            rental_info = {}
            if fresh_rented_data:
                # Update the state with fresh rental data
                updated_state = replace(
                    updated_state,
                    rented_data=fresh_rented_data,
                )
                # Add rental context to help debug
                rented_executor = fresh_rented_data.executors.get(ctx.executor.uuid) if fresh_rented_data else None
                if rented_executor:
                    rental_info = {
                        "has_rental": True,
                        "rental_pod_count": len(rented_executor.pods),
                        "rental_port_count": len(rented_executor.get_rented_ports()),
                        "rental_pods": [p.container_name for p in rented_executor.pods],
                    }

            # Provide detailed error messages based on status
            if result.status == "no_ports":
                details = "No ports available for docker container - all ports may be in use or misconfigured"
            elif result.status == "no_working_ports":
                details = f"No working ports found - verified {len(result.failed_ports)} ports, all failed connectivity test"
            elif result.status == "externally_unreachable":
                details = (
                    "Ports answer the validator but not a probe from outside the validator's network: a firewall "
                    "on the host admits only some source addresses, so renters cannot reach their pods. Allow "
                    "inbound TCP from any address on the port range"
                )
            elif result.status == "skipped_rental_active":
                # This shouldn't happen anymore but provide clear message if it does
                details = "Port verification was incorrectly skipped due to rental detection - this is a bug"
            elif result.status == "error":
                details = f"Port verification error: {result.error}" if result.error else "Port verification encountered an unexpected error"
            else:
                details = f"Port verification failed with status: {result.status}"

            event = render_message(
                Msg.VERIFY_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "details": details,
                    "message": msg,
                    "port_range": ctx.executor.port_range,
                    "port_mappings": ctx.executor.port_mappings,
                    "verification_status": result.status,
                    "total_ports_tested": len(result.successful_ports) + len(result.failed_ports),
                    "successful_ports": len(result.successful_ports),
                    "failed_ports": len(result.failed_ports),
                    **event_extra,
                    **rental_info,
                },
                extra={**extra_info, **event_extra},
            )
            return CheckResult(
                passed=False,
                event=event,
                updates={"default_extra": {**extra, **extra_info}, "state": updated_state},
            )

        event = render_message(
            Msg.VERIFY_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={"message": msg},
            extra={**extra_info, **event_extra},
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={
                "default_extra": {**extra, **extra_info},
                "state": updated_state,
            },
        )

    @staticmethod
    def _port_setup_fingerprint(ctx: Context) -> str:
        # every input of the port selector: a memory from another setup is not republished
        return (
            f"{ctx.executor.address}|{ctx.executor.port_range}|{ctx.executor.port_mappings}|{ctx.executor.ssh_port}"
        )

    async def _disk_pressure_gate(
        self,
        ctx: Context,
        rented_data: RentedExecutorsResponse | None,
        rented_executor: RentedExecutor | None,
        filler_ports: list[int],
    ) -> _PressureGate:
        # read the host disk right before the check's containers; defer only with pressure, an eligible
        # node, a memory to republish, deferrals left and the deferral on record in Redis - any Redis
        # trouble, a refused write included, runs the check as today
        if not settings.DISK_PRESSURE_CHECK_ENABLED:
            return _PressureGate()
        reading = await self._read_disk_pressure(ctx)
        self_rent = rented_executor is not None and rented_executor.self_rent
        default_job_owner = rented_data.get_default_job_owner(ctx.executor.uuid) if rented_data else None
        # the backend says whose filler it is: the provider's own default job is the provider's writer; a
        # self-rent vetoes the node whatever else runs on it
        deferral_eligible = not self_rent and (
            rented_executor is not None or (bool(filler_ports) and default_job_owner == DEFAULT_JOB_OWNER_LIUM)
        )
        enforce = settings.DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED
        gate = await self._load_pressure_gate_memory(
            ctx,
            _PressureGate(
                reading=reading,
                deferral_eligible=deferral_eligible,
                self_rent=self_rent,
                default_job_owner=default_job_owner,
            ),
            shadow=not enforce,
        )
        would_defer = (
            reading.pressure
            and deferral_eligible
            and gate.memory is not None
            and gate.deferrals_in_a_row < DISK_PRESSURE_MAX_DEFERRALS
        )
        if would_defer and not _has_fresh_completed_check((ctx.miner_hotkey, ctx.executor.uuid)):
            # the memory in Redis may be older than the last completed check
            logger.info(
                _m(
                    "deferral_blocked: no completed check saved by this process",
                    extra=get_extra_info({**ctx.default_extra, "disk_pressure": reading.as_spec()}),
                )
            )
            would_defer = False
        gate = replace(gate, would_defer=would_defer)
        if enforce and would_defer:
            gate = await self._reserve_pressure_deferral(ctx, gate)
        # its own line per reading: the port event repeats and LoggerSink drops a repeat to DEBUG
        logger.info(
            _m(
                "DISK_PRESSURE_READING",
                extra=get_extra_info(
                    {**ctx.default_extra, **gate.event_fields(), "disk_pressure_deferred": gate.deferred}
                ),
            )
        )
        return gate

    async def _load_pressure_gate_memory(self, ctx: Context, gate: _PressureGate, *, shadow: bool) -> _PressureGate:
        memory = None
        sysbox_verified = False
        deferrals = 0
        if gate.deferral_eligible:
            try:
                redis = ctx.services.redis
                memory = await redis.last_verified_ports(
                    ctx.miner_hotkey, ctx.executor.uuid, self._port_setup_fingerprint(ctx)
                )
                sysbox_verified = await redis.is_sysbox_verified(ctx.miner_hotkey, ctx.executor.uuid)
                deferrals = await redis.disk_pressure_deferrals(ctx.miner_hotkey, ctx.executor.uuid, shadow=shadow)
            except Exception as e:  # noqa: BLE001 - a safety net; never fail the port check over it
                logger.warning(
                    _m(
                        "verified ports memory unavailable",
                        extra=get_extra_info({**ctx.default_extra, "error": str(e)}),
                    )
                )
                memory = None
        return replace(gate, memory=memory, sysbox_verified=sysbox_verified, deferrals_in_a_row=deferrals)

    async def _reserve_pressure_deferral(self, ctx: Context, gate: _PressureGate) -> _PressureGate:
        deferred = True
        deferrals = gate.deferrals_in_a_row
        # the bound lives in this counter: a deferral Redis did not record (reachable, refusing writes)
        # would never count, so the check runs instead
        try:
            deferrals = await ctx.services.redis.record_disk_pressure_deferral(
                ctx.miner_hotkey,
                ctx.executor.uuid,
                shadow=False,
                ttl_seconds=DISK_PRESSURE_DEFERRALS_TTL_SECONDS,
            )
        except Exception as e:  # noqa: BLE001 - not on record, so not deferred
            logger.warning(
                _m("disk pressure deferral not recorded", extra=get_extra_info({**ctx.default_extra, "error": str(e)}))
            )
            deferred = False
        else:
            if deferrals > DISK_PRESSURE_MAX_DEFERRALS:
                # the slot INCR handed out is past the bound: another cycle took the last one between the read
                # in _load_pressure_gate_memory and this write, so this cycle runs the check
                logger.info(
                    _m(
                        "deferral_blocked: streak bound reached",
                        extra=get_extra_info({**ctx.default_extra, "deferrals": deferrals}),
                    )
                )
                deferred = False
        return replace(gate, deferred=deferred, deferrals_in_a_row=deferrals)

    @staticmethod
    async def _read_disk_pressure(ctx: Context) -> DiskPressureReading:
        try:
            command_result = await ctx.runner.run(
                disk_pressure_command(), timeout=DISK_PRESSURE_READ_TIMEOUT_SECONDS, retryable=False
            )
            return parse_disk_pressure(command_result.stdout or "")
        except Exception as e:  # noqa: BLE001 - a reading that fails is "not read", never pressure
            logger.warning(_m("disk pressure not read", extra=get_extra_info({**ctx.default_extra, "error": str(e)})))
            return DiskPressureReading(error="read_failed")

    async def _save_completed_ports_and_deferral_counter(
        self, ctx: Context, result: PortVerificationResult, gate: _PressureGate
    ) -> bool:
        # a completed check (any status, an empty result included) is what the next deferral republishes,
        # with the reading off too; the streak of deferrals ends here and nowhere earlier, so a timed-out
        # check cannot reopen it. True once every write is in Redis.
        try:
            redis = ctx.services.redis
            await redis.record_verified_ports(
                ctx.miner_hotkey,
                ctx.executor.uuid,
                self._port_setup_fingerprint(ctx),
                [(p.internal, p.external) for p in result.successful_ports],
                VERIFIED_PORTS_MEMORY_TTL_SECONDS,
            )
            if settings.DISK_PRESSURE_GATE_ENFORCEMENT_ENABLED:
                await redis.clear_disk_pressure_deferrals(ctx.miner_hotkey, ctx.executor.uuid, shadow=False)
            elif gate.would_defer:
                # shadow: the simulated streak follows the same rule, so the forced 4th cycle shows in Loki
                await redis.record_disk_pressure_deferral(
                    ctx.miner_hotkey,
                    ctx.executor.uuid,
                    shadow=True,
                    ttl_seconds=DISK_PRESSURE_DEFERRALS_TTL_SECONDS,
                )
            else:
                await redis.clear_disk_pressure_deferrals(ctx.miner_hotkey, ctx.executor.uuid, shadow=True)
            return True
        except Exception as e:  # noqa: BLE001 - a safety net; never fail the port check over it
            logger.warning(
                _m("verified ports not recorded", extra=get_extra_info({**ctx.default_extra, "error": str(e)}))
            )
            return False

    async def _deferred(
        self, ctx: Context, gate: _PressureGate, extra: dict[str, object], *, busy_ports: set[int]
    ) -> CheckResult:
        """No containers on a Docker root whose sync already stalls: the last completed check's ports are
        republished (PortCountCheck counts them as today) and sysbox is what the validator's own probe verified
        (the host's scrape never counts); the sysbox and DinD-miss memories are left as they are."""
        assert gate.memory is not None and gate.reading is not None
        sysbox = gate.sysbox_verified
        deferrals = gate.deferrals_in_a_row
        # a port a rental or filler took since the memory was written would collide with the rental probe's ssh
        pairs = [(internal, external) for internal, external in gate.memory.pairs if external not in busy_ports]
        extra_info: dict[str, object] = {
            "sysbox_runtime": sysbox,
            "verified_port_count": len(pairs),
            "probed_port_count": len(pairs),
            "disk_pressure_deferred": True,
            "disk_pressure_deferrals_in_a_row": deferrals,
            "verified_ports_now_busy": len(gate.memory.pairs) - len(pairs),
        }
        updated_state = replace(
            ctx.state,
            specs={
                # a deferral is no miss: the grace marker never survives it, the provider's one included
                **{key: value for key, value in ctx.state.specs.items() if key != "port_verify_first_miss"},
                "sysbox_runtime": sysbox,
                "verified_ports": [external for _, external in pairs],
                **_disk_pressure_spec(gate),
            },
            sysbox_runtime=sysbox,
            verified_port_count=len(pairs),
            probed_port_count=len(pairs),
            verified_port_pairs=list(pairs),
            port_check_deferred=True,
        )
        event = render_message(
            Msg.VERIFY_DEFERRED,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "message": (
                    f"sync -f on the Docker root took {gate.reading.syncfs_seconds} s"
                    f"{' (timed out)' if gate.reading.syncfs_timed_out else ''}; "
                    f"{len(pairs)} ports from the last completed check republished, "
                    f"deferral {deferrals} of {DISK_PRESSURE_MAX_DEFERRALS}"
                ),
                "verified_ports_measured_at": gate.memory.measured_at,
                **gate.event_fields(),
            },
            extra=extra_info,
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={"default_extra": {**extra, **extra_info}, "state": updated_state},
        )

    @staticmethod
    async def _is_first_port_verify_miss(ctx: Context, *, ok: bool) -> bool:
        """DAH-3698: the first failed verification inside the window is not a verdict on the
        node's ports; the split-remainder port gate skips that cycle. A repeat inside the window
        counts, even with passes in between: a pass never clears the record, or alternating a
        pass and a failure would earn grace every time. Fails closed (no grace) on a Redis error."""
        try:
            if ok:
                return False
            return await ctx.services.redis.record_port_verify_miss(
                ctx.miner_hotkey, ctx.executor.uuid, PORT_VERIFY_FIRST_MISS_TTL_SECONDS
            )
        except Exception as e:  # noqa: BLE001 — never fail the port check over the grace
            logger.warning(
                _m("port verify miss record unavailable", extra=get_extra_info({**ctx.default_extra, "error": str(e)}))
            )
            return False

    @staticmethod
    async def _should_keep_last_known_sysbox(
        ctx: Context, result: PortVerificationResult, extra_info: dict[str, object]
    ) -> bool:
        # DAH-2272 (tolerate): a customer rental force-removes port-check / DinD
        # probe containers the instant a ContainerCreateRequest lands (see
        # DockerService.wait_for_port_check_containers). That race can flip
        # sysbox_runtime to False for this cycle even though the executor is
        # fine. Don't record a rental-induced sysbox downgrade — keep the last
        # known value and let the next verification cycle re-measure. Mirrors
        # the rented-executor sysbox fallback in ExecutorConnectivityService.
        sysbox_downgraded = ctx.state.sysbox_runtime and not result.sysbox_runtime
        if sysbox_downgraded and await ctx.services.redis.renting_in_progress(ctx.miner_hotkey, ctx.executor.uuid):
            return True
        if not settings.DIND_PROBE_FIRST_MISS_GRACE:
            return False
        # DAH-3597: a probe that never reached its container (dind_ok False: docker run
        # refused, sshd not up in 30 s, SSH reset) measured nothing about sysbox, so the first
        # such miss inside the TTL window keeps the last known value and the next cycle
        # re-measures. A second miss inside the window is recorded as before. A probe that
        # reached its container and still says no sysbox is a verdict, never tolerated.
        if sysbox_downgraded and not result.dind_ok:
            first_miss = await ctx.services.redis.record_dind_probe_miss(
                ctx.miner_hotkey,
                ctx.executor.uuid,
                settings.DIND_PROBE_FIRST_MISS_GRACE_TTL_SECONDS,
            )
            if first_miss:
                extra_info["sysbox_downgrade_tolerated_reason"] = "first_dind_probe_miss"
                return True
            extra_info["dind_probe_miss_repeated"] = True
        elif result.dind_ok:
            await ctx.services.redis.clear_dind_probe_miss(ctx.miner_hotkey, ctx.executor.uuid)
        return False

    @staticmethod
    async def _update_last_verified_sysbox(
        ctx: Context, result: PortVerificationResult, *, rented: bool, sysbox_now: bool
    ) -> _SysboxMemoryUpdate:
        """ticket-0494: keep a rented executor's verified sysbox when this cycle measured nothing.

        On a busy rented host the scrape's own sysbox test can time out (sysbox_runtime False), so
        the DinD probe runs without sysbox-runc and then is refused by the host or never reaches its
        container. The fallbacks above key off ctx.state.sysbox_runtime, which is that same scrape,
        so neither applies and a rented node scored 0 for the cycle. Every probe that reached its
        container with sysbox-runc is remembered here (pass refreshes, fail forgets); a rented
        executor whose probe never reached its container reads that memory. A probe that reached
        its container and found no sysbox stays a verdict. Fails open to the current behaviour on a
        Redis error, which also leaves the memory unsaved (DAH-5244: a deferral would republish it).
        """
        redis = ctx.services.redis
        try:
            if result.sysbox_measured is True:
                await redis.record_sysbox_verified(
                    ctx.miner_hotkey, ctx.executor.uuid, settings.SYSBOX_VERIFIED_TTL_SECONDS
                )
            elif result.sysbox_measured is False:
                await redis.clear_sysbox_verified(ctx.miner_hotkey, ctx.executor.uuid)
            if not rented:
                if result.sysbox_measured is None and not sysbox_now:
                    # DAH-5244: a deferred cycle republishes this memory, so a cycle an unrented node ends
                    # without sysbox forgets it - the next deferral may not pass SysboxRequired on it
                    await redis.clear_sysbox_verified(ctx.miner_hotkey, ctx.executor.uuid)
                return _SysboxMemoryUpdate(is_saved=True)
            if sysbox_now or result.dind_ok:
                return _SysboxMemoryUpdate(is_saved=True)
            use_last_verified = await redis.is_sysbox_verified(ctx.miner_hotkey, ctx.executor.uuid)
            return _SysboxMemoryUpdate(is_saved=True, use_last_verified=use_last_verified)
        except Exception as e:  # noqa: BLE001 — a safety net; never fail the port check over it
            logger.warning(
                _m(
                    "last verified sysbox unavailable",
                    extra=get_extra_info({**ctx.default_extra, "error": str(e)}),
                )
            )
            return _SysboxMemoryUpdate(is_saved=False)


def _disk_pressure_spec(gate: _PressureGate) -> _DiskPressureSpec:
    # the backend keeps one cycle's reading (lium-platform MachineSpecs.disk_pressure); none read, no key
    return {"disk_pressure": gate.reading.as_spec()} if gate.reading is not None else {}
