from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import shlex
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

from services import executor_payload
from services.file_encrypt_service import ORIGINAL_KEYS
from services.gpu_spec_table import classify_gpu_model, list_sxm_on_pcie_adapter_as_pcie
from services.nvidia_devices import read_kernel_gpu_view

from core.config import settings
from core.utils import _m, get_extra_info

from ..messages import MachineSpecMessages as Msg
from ..messages import MessageTemplate, render_message
from ..models import ValidationEvent
from ..pipeline import CheckResult, Context
from ..runner import SSHCommandResult
from .container_download_speed import SPECS_KEY as IN_POD_DOWNLOAD_KEY
from .container_download_speed import SPECS_MEASURED_AT_KEY as IN_POD_DOWNLOAD_AT_KEY
from .gpu_loss_hold import GpuLossHoldCheck
from .gpu_identity import enforced, evaluate_gpu_identity, event_fields, log_gpu_identity
from .rented_gpu_drop import RentedGpuDropCheck, missing_gpus_under_reset
from .task_time import task_seconds_left
from .upload_files import UPLOAD_TIMEOUT, UploadFailed, upload_validation_files_to_fresh_remote_dir

logger = logging.getLogger(__name__)

# How long a failed stdin attempt may have taken and still be worth retrying with the binary.
# Above a full scrape (~15 s on a real box, plus up to ~90 s for the vloopback mount test and its
# cleanup, so a payload that will not decrypt is only knowable around then), below the point where
# the retry stops fitting: 120 s here + 300 s upload + 300 s scrape = 720 s against the 780 s
# per-executor timeout. The upload gets one attempt for exactly that reason.
STDIN_FAILURE_RETRY_WINDOW_MS = 120_000
FALLBACK_UPLOAD_ATTEMPTS = 1

# Specs mode reads liumd first, then falls back to the Python scrape when liumd cannot produce a
# usable document. The read is capped so the fallback still fits inside the machine-scrape timeout.
LIUMD_SPECS_READ_TIMEOUT_SECONDS = 180
LIUMD_SPECS_FALLBACK_RESERVE_SECONDS = 90
LIUMD_SPECS_MIN_READ_SECONDS = 5.0
# reasons a container's own read cannot settle: the card list, or a card source it could not see
_KERNEL_VIEW_REASONS = frozenset(
    {"nvidia_module_missing", "proc_driver_missing", "proc_source_missing", "nvml_source_missing"}
)

# The scrape says only two things on stdout: a Fernet token when it worked, and a JSON object
# with an "error" key when it ran but had nothing to report. Every Fernet token is base64url of
# a 0x80 version byte, so this prefix separates ours from whatever the miner's image printed.
FERNET_TOKEN_PREFIX = "gAAAAA"

# How much stdout the miner puts in front of the scrape's own line is his choice, not a bounded
# amount, so only the last lines are searched and only so many candidates are decrypted — both
# run synchronously on the event loop shared with every other executor in the cycle.
STDOUT_SEARCH_TAIL_BYTES = 256 * 1024
MAX_PAYLOAD_DECRYPT_ATTEMPTS = 20


def _lines_from_the_end(stdout: str) -> Iterator[str]:
    # whole lines, newest first, until the window is spent — a token that is itself longer than
    # the window would lose its prefix to a byte-wise cut and be dropped by every reader below
    end = len(stdout)
    scanned = 0
    while end > 0 and scanned < STDOUT_SEARCH_TAIL_BYTES:
        start = stdout.rfind("\n", 0, end)
        yield stdout[start + 1 : end]
        scanned += end - start
        end = start


def _update_keys(data: Any, key_mapping: dict[str, str]) -> Any:
    if isinstance(data, dict):
        updated: dict[str, Any] = {}
        for key, value in data.items():
            original_key = key_mapping.get(key, key)
            updated[original_key] = _update_keys(value, key_mapping)
        return updated
    if isinstance(data, list):
        return [_update_keys(item, key_mapping) for item in data]
    return data


def _deobfuscate(spec: dict[str, Any], obfuscation_keys: dict[str, str] | None) -> dict[str, Any]:
    if not obfuscation_keys:
        return spec
    reverse = {v: k for k, v in obfuscation_keys.items()}
    first_pass = _update_keys(spec, reverse)
    return _update_keys(first_pass, ORIGINAL_KEYS)


def _normalize_liumd_document(raw: dict[str, Any]) -> dict[str, Any]:
    return _update_keys(_deobfuscate(raw, None), ORIGINAL_KEYS)


def normalize_liumd_specs(raw: dict[str, Any]) -> dict[str, Any]:
    """The liumd machine-scrape document in the validator's normalized specs shape."""
    specs = _normalize_liumd_document(raw)
    # the vloopback mount test is reported in the scrape event only; the stored specs do not carry it
    specs.pop("vloopback_check", None)
    return specs


def _normalize_gpu_details(gpu_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for detail in gpu_details:
        name = detail.get("name")
        if isinstance(name, str) and name:
            detail = {**detail, "name": classify_gpu_model(name, detail.get("capacity"))}
        elif name is not None and not isinstance(name, str):
            detail = {**detail, "name": None}
        normalized.append(detail)
    return normalized


def _identity_cards(block: Any) -> list[dict[str, Any]]:
    cards = block.get("cards") if isinstance(block, dict) else None
    return [card for card in cards if isinstance(card, dict)] if isinstance(cards, list) else []


async def _identity_kernel_uuids(ctx: Context, cards: list[dict[str, Any]]) -> list[str] | None:
    """The container's own kernel card list, else the host's through PID 1's root."""
    if ctx.ssh is None:
        return None
    wanted = {
        str(card.get("uuid")).strip().lower()
        for card in cards
        if isinstance(card, dict) and card.get("uuid")
    }
    container: list[str] | None = None
    for host_view in (False, True):
        try:
            view = await read_kernel_gpu_view(ctx.ssh, host_view=host_view)
        except Exception:
            continue
        uuids = [uuid for uuid in view.uuids or [] if str(uuid).strip()]
        if not uuids:
            continue
        if not host_view:
            container = uuids
        if not wanted or wanted <= {str(uuid).strip().lower() for uuid in uuids}:
            return uuids
    return container


def _decrypt_payload(ctx: Context, stdout: str) -> str:
    # decrypt the scrape's Fernet token out of whatever else the executor printed
    #
    # The token is the scrape's own print, but the image decides what surrounds it: a .pth or
    # sitecustomize prints before it, an atexit handler after it. Searching newest-first beats
    # betting on a fixed position.
    if not ctx.encrypt_key:
        raise ValueError("Missing encrypt_key in context")

    last_exc: Exception | None = None
    attempts = 0
    for line in _lines_from_the_end(stdout):
        candidate = line.strip()
        if not candidate.startswith(FERNET_TOKEN_PREFIX):
            continue
        try:
            return ctx.services.ssh.decrypt_payload(ctx.encrypt_key, candidate)
        except Exception as exc:
            last_exc = exc
        attempts += 1
        if attempts == MAX_PAYLOAD_DECRYPT_ATTEMPTS:
            break

    raise last_exc or ValueError("No scrape payload on stdout")


def _scrape_error_report(stdout: str) -> dict[str, Any] | None:
    # the scrape prints {"error": ...} and exits non-zero when it ran but found nothing to report;
    # a source the interpreter could not run prints nothing of ours at all. Searched newest-first
    # like the token, since an atexit handler in the image may print after it.
    for line in _lines_from_the_end(stdout):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            report = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(report, dict) and "error" in report:
            return report
    return None


def _scrape_reported_its_own_failure(stdout: str) -> bool:
    return _scrape_error_report(stdout) is not None


def _is_no_gpu_details_error(error: Any, obfuscation_keys: dict[str, str] | None) -> bool:
    # the key substitution that renames `gpu_details` in the shipped scrape (file_encrypt_service)
    # is a plain text replace, so it renames it inside this string literal too
    obfuscated_gpu_details_key = (obfuscation_keys or {}).get("gpu_details", "gpu_details")
    return isinstance(error, str) and error in {"no_gpu_details", f"no_{obfuscated_gpu_details_key}"}


@dataclass(frozen=True)
class ScrapeFailure:
    """The reason code a failed scrape run gets, plus the cause fields its event carries."""

    template: MessageTemplate
    error_type: str | None = None
    scrape_error: str | None = None
    gpu_scrape_error: str | None = None

    def cause_event_fields(self) -> dict[str, str]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "template" and getattr(self, field.name) is not None
        }


def _classify_scrape_failure(
    scrape_run: SSHCommandResult, obfuscation_keys: dict[str, str] | None
) -> ScrapeFailure:
    # which side failed, from what came back: the runner sets error_type exactly when no exit status
    # came back from the host (timed out, raised, or the channel closed without one). Those stay
    # undetermined; only an exit status the host sent puts the failure on the host.
    if scrape_run.error_type == "timeout":
        return ScrapeFailure(Msg.SCRAPE_TIMEOUT, error_type=scrape_run.error_type)
    if scrape_run.error_type is not None:
        return ScrapeFailure(Msg.SCRAPE_TRANSPORT_FAILED, error_type=scrape_run.error_type)

    report = _scrape_error_report(scrape_run.stdout)
    if report is None or not _is_no_gpu_details_error(report.get("error"), obfuscation_keys):
        return ScrapeFailure(Msg.SCRAPE_FAILED_ON_HOST)

    report_data = report.get("data")
    report_data = _deobfuscate(report_data, obfuscation_keys) if isinstance(report_data, dict) else {}
    gpu_scrape_error = report_data.get("gpu_scrape_error")
    if gpu_scrape_error:
        return ScrapeFailure(
            Msg.SCRAPE_FAILED_DRIVER,
            scrape_error="no_gpu_details",
            gpu_scrape_error=str(gpu_scrape_error)[:200],
        )
    return ScrapeFailure(Msg.SCRAPE_FAILED_NO_GPU, scrape_error="no_gpu_details")


def _liumd_no_gpu_failure(specs: dict[str, Any]) -> ScrapeFailure:
    gpu_scrape_error = specs.get("gpu_scrape_error")
    if gpu_scrape_error:
        return ScrapeFailure(
            Msg.SCRAPE_FAILED_DRIVER,
            scrape_error="no_gpu_details",
            gpu_scrape_error=str(gpu_scrape_error)[:200],
        )
    return ScrapeFailure(Msg.SCRAPE_FAILED_NO_GPU, scrape_error="no_gpu_details")


# The scrape's own NVML reading on the host says the node cannot serve a GPU: NVML answered with zero cards
# (SCRAPE_FAILED_NO_GPU), or raised one of these nvml.h return codes (SCRAPE_FAILED_DRIVER): driver not
# loaded, GPU lost, GPU requires reset, GPU not found. Anything else stays a plain halt: the library copied
# to a temp file that a full or read-only disk refuses (LIBRARY_NOT_FOUND), a driver/library mismatch that
# leaves pods started before the upgrade working, a permission or memory error, UNKNOWN (999, which names
# no fault), and SCRAPE_FAILED_ON_HOST (a missing interpreter or a traceback says nothing of the GPU).
HOST_GPU_RUNTIME_DEAD_NVML_CODES = frozenset({9, 15, 16, 28})
_NVML_ERROR_CODE_RX = re.compile(r"^NVMLError\w*\((\d+)\)")
HOST_GPU_RUNTIME_FAULT_IMPACT = (
    "Validation halted — GPU runtime dead on the host of a rented node: verified job cleared, "
    "executor marked inactive until a clean scrape"
)


def _gpu_runtime_is_dead(failure: ScrapeFailure) -> bool:
    if failure.template.reason == Msg.SCRAPE_FAILED_NO_GPU.reason:
        return True
    if failure.template.reason != Msg.SCRAPE_FAILED_DRIVER.reason:
        return False
    match = _NVML_ERROR_CODE_RX.match(failure.gpu_scrape_error or "")
    return match is not None and int(match.group(1)) in HOST_GPU_RUNTIME_DEAD_NVML_CODES


def _host_gpu_fault_reset(ctx: Context, failure: ScrapeFailure, check_id: str) -> dict[str, Any]:
    """The reset a rented node's host-confirmed GPU runtime fault carries, or nothing.

    It clears the verified job as POD_NOT_RUNNING and GPU_MISSING do, so the backend marks the executor inactive
    and billing stops. It is handled like GPU_MISSING: with the matching backend change the backend proposes
    EXECUTOR_INACTIVE_MID_RENTAL for it, and the node is listed again only after a full GPU re-verification.
    A node without a customer pod keeps the plain halt.
    """
    if not settings.RENTED_HOST_GPU_FAULT_RESET_ENABLED:
        return {}
    if not _gpu_runtime_is_dead(failure):
        return {}
    rented_data = ctx.state.rented_data
    rented_executor = rented_data.executors.get(ctx.executor.uuid) if rented_data else None
    if not rented_executor or not rented_executor.pods:
        return {}
    # The executor UUID is the miner's word and the backend resets by UUID alone, so a rental another miner
    # owns must never be cleared on this miner's report.
    if rented_executor.miner_hotkey != ctx.miner_hotkey:
        return {}
    pod_ids = [pod.pod_id for pod in rented_executor.pods]
    evidence: dict[str, Any] = {
        "reason_code": failure.template.reason,
        "check_id": check_id,
        "pod_id": pod_ids[0],
        "rented_pod_ids": pod_ids,
        "scrape_error": failure.scrape_error,
    }
    if failure.gpu_scrape_error is not None:
        evidence["gpu_scrape_error"] = failure.gpu_scrape_error
    return {
        "clear_verified_job_info": True,
        "clear_verified_job_evidence": evidence,
    }


def _no_gpu_report_specs(stdout: str, obfuscation_keys: dict[str, str] | None) -> dict[str, Any]:
    """The specs a `no_gpu_details` report carries, in the key shape a successful scrape has."""
    report = _scrape_error_report(stdout) or {}
    data = report.get("data")
    if not isinstance(data, dict):
        return {"gpu": {}}
    specs = _update_keys(_deobfuscate(data, obfuscation_keys), ORIGINAL_KEYS)
    return specs if isinstance(specs.get("gpu"), dict) else {**specs, "gpu": {}}


async def _report_rented_gpu_loss(ctx: Context, specs: dict[str, Any]) -> list[ValidationEvent]:
    # A scrape that lists no GPU (card 0 lost, the only card lost, the driver down) fails here, and
    # this check is fatal, so RentedGpuDropCheck never runs on that cycle: hand it the empty scrape.
    gpu = specs["gpu"]
    count = gpu.get("count")
    state = replace(
        ctx.state,
        specs=specs,
        gpu_count=count if isinstance(count, int) else 0,
        gpu_details=[],
    )
    try:
        lost_ctx = ctx.model_copy(update={"state": state})
        await RentedGpuDropCheck().run(lost_ctx)
        # DAH-4457: the hold starts here too, so the next clean scrape does not list the node again at once.
        # This failed scrape ends the run before GpuLossHoldCheck, so its GPU_LOST entry rides on the scrape's
        # result: without it a node whose driver died stayed listed for as long as the scrape kept failing.
        hold_result = await GpuLossHoldCheck().run(lost_ctx)
        return [hold_result.event] if hold_result.event.is_availability_error else []
    except Exception:
        logger.warning(
            _m("RENTED_GPU_DROP_ON_SCRAPE_FAILURE_ERROR", extra=get_extra_info(ctx.default_extra)),
            exc_info=True,
        )
        return []


@dataclass(frozen=True)
class StdinAttempt:
    """Why the stdin delivery failed, carried into the event of the binary retry.

    The retry discards the stdin attempt's own event, and a run that exited 0 with an unreadable
    payload says nothing in ``exit_code`` or ``stderr`` — so the reason travels with it.
    """

    reason: str
    command_id: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    stderr_tail: str | None = None
    exception: str | None = None
    stdout_head: str | None = None

    @classmethod
    def from_failed_result(cls, result: CheckResult) -> StdinAttempt:
        what_we_saw = result.event.what_we_saw
        carried = {field.name for field in fields(cls)} - {"reason"}
        return cls(
            reason=result.event.reason_code,
            **{name: what_we_saw.get(name) for name in carried},
        )

    def as_event_field(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        }


def _interconnect_summary(specs: dict[str, Any]) -> dict[str, Any] | None:
    # what the scrape said about GPU-to-GPU wiring, minus the matrix (it is in specs for the backend)
    interconnect = specs.get("interconnect")
    if not isinstance(interconnect, dict):
        return {"scrape_error": specs.get("interconnect_scrape_error")} if specs.get("interconnect_scrape_error") else None
    summary = {key: value for key, value in interconnect.items() if key != "matrix"}
    if specs.get("interconnect_scrape_error"):
        summary["scrape_error"] = specs["interconnect_scrape_error"]
    return summary


VLOOPBACK_TEXT_FIELDS = ("verdict", "reason_code", "detail", "runtime")
# the host sends these fields; a modified scrape can send any type or length, so the validator keeps
# only strings, cut to the scrape's own detail cap
VLOOPBACK_TEXT_CAP = 300


def _host_text(value: Any) -> str | None:
    return value[:VLOOPBACK_TEXT_CAP] if isinstance(value, str) else None


def _vloopback_fields(vloopback: dict[str, Any]) -> dict[str, Any]:
    return {field: _host_text(vloopback.get(field)) for field in VLOOPBACK_TEXT_FIELDS}


def _storage_limit_summary(specs: dict[str, Any], vloopback: Any) -> dict[str, Any]:
    # the storage-limit verdict and the vloopback mount test's own verdict, in the validator's own log
    supported = bool(specs.get("storage_limit_supported", False))
    summary: dict[str, Any] = {"supported": supported}
    scrape_error = specs.get("storage_limit_scrape_error")
    if not supported and scrape_error:
        summary["detail"] = str(scrape_error)[:VLOOPBACK_TEXT_CAP]
    if isinstance(vloopback, dict):
        summary["vloopback"] = _vloopback_fields(vloopback)
    return summary


def _binary_command(remote_dir: str, script_filename: str) -> str:
    script_path = shlex.quote(f"{remote_dir.rstrip('/')}/{script_filename.lstrip('/')}")
    return f"chmod +x {script_path} && {script_path}"


async def scrape_gpu_details(ctx: Context, timeout: int) -> list[dict[str, Any]]:
    """A fresh scrape's gpu_details, normalised as MachineSpecScrapeCheck stores them; [] when no payload decrypted.

    DAH-6469: GpuThermalThrottleCheck reads the cards under its own load with it. It writes nothing to ctx.state, so
    the checks after it keep the cycle's own scrape."""
    if ctx.config.machine_scrape_source:
        command, stdin_text = f"{shlex.quote(executor_payload.EXECUTOR_PYTHON)} -I -", ctx.config.machine_scrape_source
    elif ctx.state.remote_dir and ctx.config.machine_scrape_filename:
        command, stdin_text = _binary_command(ctx.state.remote_dir, ctx.config.machine_scrape_filename), None
    else:
        return []
    scrape_run = await ctx.runner.run(command, timeout=timeout, retryable=False, stdin_text=stdin_text)
    if not scrape_run.success:
        return []
    try:
        specs = _deobfuscate(json.loads(_decrypt_payload(ctx, scrape_run.stdout)), ctx.config.obfuscation_keys)
    except Exception:  # noqa: BLE001 - no token this validator can read: no reading
        return []
    gpu_info = specs.get("gpu") if isinstance(specs, dict) else None
    details = gpu_info.get("details") if isinstance(gpu_info, dict) else None
    if not isinstance(details, list):
        return []
    return list_sxm_on_pcie_adapter_as_pcie(
        specs, _normalize_gpu_details([detail for detail in details if isinstance(detail, dict)])
    )


class MachineSpecScrapeCheck:
    """Run the obfuscated scrape script and unpack the executor's hardware profile.

    This is the backbone for nearly every other check: GPU inventory, UUIDs, process
    lists, and sysbox hints are all extracted here exactly as the legacy flow did.
    Skipping or weakening it would starve later checks of their source data.
    """

    check_id = "gpu.scrape.machine_spec"
    # a replay would advance this cycle's counters a second time
    replays_after_session_drop = False
    fatal = True

    DEFAULT_TIMEOUT = 300

    async def run(self, ctx: Context) -> CheckResult:
        liumd_specs = settings.VALIDATOR_LIUMD_SPECS and not settings.DRY_RUN
        timeout = ctx.config.machine_scrape_timeout or self.DEFAULT_TIMEOUT

        if liumd_specs:
            result, fallback_reason = await self._scrape_with_liumd(ctx, timeout)
            if result.passed or fallback_reason is None:
                return result
            if settings.VALIDATOR_LIUMD_EVIDENCE_MODE == "enforce":
                # Enforce is fail-closed: without the scrape's identity block the node cannot be
                # verified, and the Python scrape would bypass the gate.
                result.event.what_we_saw["liumd_fallback_refused"] = fallback_reason
                return result
            detail = str(result.event.what_we_saw.get("detail") or "")[:300]
            logger.warning(
                _m(
                    "liumd machine specs unavailable; using the Python scrape",
                    extra=get_extra_info(
                        {**ctx.default_extra, "liumd_reason": fallback_reason, "detail": detail}
                    ),
                )
            )
            if ctx.config.machine_scrape_source:
                fallback = await self._scrape_from_source(ctx, timeout, start_shadow=False)
            else:
                fallback = await self._scrape_uploaded_binary(ctx, timeout, start_shadow=False)
            fallback.event.what_we_saw["liumd_fallback_reason"] = fallback_reason
            if detail:
                fallback.event.what_we_saw["liumd_fallback_detail"] = detail
            return fallback

        if ctx.config.machine_scrape_source:
            return await self._scrape_from_source(ctx, timeout, start_shadow=True)
        return await self._scrape_uploaded_binary(ctx, timeout, start_shadow=True)

    async def _scrape_with_liumd(
        self, ctx: Context, timeout: int
    ) -> tuple[CheckResult, str | None]:
        """Use the liumd machine-scrape document, or report why the Python fallback must run."""
        from services.local_verify_client import LocalVerifyAnswer

        from ..liumd_shadow import start_facts_after_scrape

        task = start_facts_after_scrape(ctx)
        if task is None:
            result = self._fail_liumd(ctx, None, "no early liumd read")
            return result, self._liumd_reason(result)
        budget = max(
            LIUMD_SPECS_MIN_READ_SECONDS,
            min(
                float(timeout) - LIUMD_SPECS_FALLBACK_RESERVE_SECONDS,
                LIUMD_SPECS_READ_TIMEOUT_SECONDS,
            ),
        )
        try:
            answer = await asyncio.wait_for(task, timeout=budget)
        except TimeoutError:
            result = self._fail_liumd(ctx, None, "liumd read timed out", reason="timeout")
            return result, "timeout"
        except Exception as exc:  # a shadow bug must not fail the scrape
            result = self._fail_liumd(
                ctx, None, f"{type(exc).__name__}: {exc}"[:300], reason="error"
            )
            return result, "error"
        if not isinstance(answer, LocalVerifyAnswer):
            result = self._fail_liumd(ctx, answer)
            return result, self._liumd_reason(result)
        if not answer.specs:
            result = self._fail_liumd(ctx, answer, "empty liumd specs")
            return result, self._liumd_reason(result)

        specs = _normalize_liumd_document(answer.specs)
        gpu_info = specs.get("gpu")
        raw_gpu_details = gpu_info.get("details", []) if isinstance(gpu_info, dict) else None
        if not isinstance(gpu_info, dict) or not isinstance(raw_gpu_details, list) or not all(
            isinstance(detail, dict) for detail in raw_gpu_details
        ):
            result = self._fail_liumd(ctx, answer, "malformed liumd specs")
            return result, self._liumd_reason(result)
        if not raw_gpu_details:
            return await self._fail_without_gpus(ctx, answer, specs), None
        identity = evaluate_gpu_identity(
            specs.get("gpu_identity"),
            len(raw_gpu_details),
            claimed_model=raw_gpu_details[0].get("name"),
        )
        if _KERNEL_VIEW_REASONS & set(identity["reasons"]):
            block = specs.get("gpu_identity")
            kernel_uuids = await _identity_kernel_uuids(ctx, _identity_cards(block))
            if kernel_uuids:
                identity = evaluate_gpu_identity(
                    block,
                    len(raw_gpu_details),
                    claimed_model=raw_gpu_details[0].get("name"),
                    kernel_uuids=kernel_uuids,
                )
        mode = settings.VALIDATOR_LIUMD_EVIDENCE_MODE
        log_gpu_identity(ctx, identity, mode)
        if enforced(mode, identity):
            return self._fail_gpu_identity(ctx, identity, answer), None
        return self._result_from_specs(ctx, specs, delivery="liumd", task=task, snapshot=False), None

    @staticmethod
    def _liumd_reason(result: CheckResult) -> str:
        return str(result.event.what_we_saw.get("liumd_reason") or "unavailable")

    async def _fail_without_gpus(
        self, ctx: Context, answer: Any, specs: dict[str, Any]
    ) -> CheckResult:
        failure = _liumd_no_gpu_failure(specs)
        what: dict[str, Any] = {"delivery": "liumd", "round_trip_ms": answer.round_trip_ms}
        # DAH-4658: every card is missing; held only when each one is a GPU the renter resets
        if failure.scrape_error == "no_gpu_details" and await missing_gpus_under_reset(ctx, []):
            event = render_message(
                Msg.SCRAPE_GPU_RESET_IN_PROGRESS,
                ctx=ctx,
                check_id=self.check_id,
                what={**what, "held_reason_code": failure.template.reason, **failure.cause_event_fields()},
            )
            return CheckResult(passed=False, event=event)
        gpu_lost_events = await _report_rented_gpu_loss(ctx, specs)
        reset = _host_gpu_fault_reset(ctx, failure, self.check_id)
        event = render_message(
            failure.template,
            ctx=ctx,
            check_id=self.check_id,
            impact=HOST_GPU_RUNTIME_FAULT_IMPACT if reset else None,
            what={
                **what,
                **failure.cause_event_fields(),
                **({"host_gpu_fault_reset": True} if reset else {}),
            },
        )
        return CheckResult(passed=False, event=event, updates=reset, extra_events=gpu_lost_events)

    def _fail_gpu_identity(
        self, ctx: Context, judged: dict[str, Any], answer: Any
    ) -> CheckResult:
        event = render_message(
            Msg.GPU_IDENTITY_MISMATCH,
            ctx=ctx,
            check_id=self.check_id,
            what={
                "delivery": "liumd",
                "round_trip_ms": answer.round_trip_ms,
                **event_fields(judged),
            },
        )
        return CheckResult(passed=False, event=event)

    def _result_from_specs(
        self,
        ctx: Context,
        specs: dict[str, Any],
        *,
        delivery: Literal["stdin", "upload", "liumd"],
        task: Any = None,
        what: dict[str, Any] | None = None,
        snapshot: bool = True,
    ) -> CheckResult:
        specs.pop("port_verify_first_miss", None)
        specs.pop("split_disk", None)
        # the in-pod download figure is the validator's own reading (ContainerDownloadSpeedCheck), never the node's
        specs.pop(IN_POD_DOWNLOAD_KEY, None)
        specs.pop(IN_POD_DOWNLOAD_AT_KEY, None)
        gpu_identity = specs.pop("gpu_identity", None)
        vloopback = specs.pop("vloopback_check", None)
        gpu_info = specs.get("gpu", {}) or {}
        gpu_count = gpu_info.get("count", 0) or 0
        raw_gpu_details = gpu_info.get("details", []) or []
        for detail in raw_gpu_details:
            if isinstance(detail, dict):
                detail.pop("power_host_limit", None)
        gpu_details = list_sxm_on_pcie_adapter_as_pcie(specs, _normalize_gpu_details(raw_gpu_details))
        gpu_model = gpu_details[0].get("name") if gpu_count > 0 and gpu_details else None
        gpu_model_count = f"{gpu_model}:{gpu_count}" if gpu_model is not None else None
        gpu_uuids = ",".join(detail.get("uuid", "") for detail in gpu_details if detail.get("uuid"))
        sysbox_runtime = specs.get("sysbox_runtime", False)
        hardware_supports = specs.get("storage_limit_supported", False)
        gpu_splitting_config = ctx.state.rented_data.gpu_splitting_config if ctx.state.rented_data else {}
        gpu_splitting_min_count = gpu_splitting_config.get(ctx.executor.uuid)
        supports_gpu_splitting = hardware_supports and gpu_splitting_min_count is not None
        # only the Python scrape has a second read to compare against
        scrape_specs = copy.deepcopy(specs) if snapshot else None
        extra_info = {
            "sysbox_runtime": sysbox_runtime,
            "supports_gpu_splitting": supports_gpu_splitting,
            "gpu_splitting_min_count": gpu_splitting_min_count,
        }
        event = render_message(
            Msg.SCRAPE_OK,
            ctx=ctx,
            check_id=self.check_id,
            what={
                **(what or {"delivery": delivery}),
                "gpu_count": gpu_count,
                "gpu_model": gpu_model,
                "network": specs.get("network"),
                "interconnect": _interconnect_summary(specs),
                "storage_limit": _storage_limit_summary(specs, vloopback),
            },
            extra=extra_info,
        )
        updated_state = replace(
            ctx.state,
            specs=specs,
            scrape_specs=scrape_specs,
            gpu_model=gpu_model,
            gpu_count=gpu_count,
            gpu_details=gpu_details,
            gpu_processes=specs.get("gpu_processes", []) or [],
            sysbox_runtime=sysbox_runtime,
            supports_gpu_splitting=supports_gpu_splitting,
            gpu_splitting_min_count=gpu_splitting_min_count,
            gpu_model_count=gpu_model_count,
            gpu_uuids=gpu_uuids,
            gpu_identity=gpu_identity if isinstance(gpu_identity, dict) else None,
            liumd_shadow_facts_task=task,
            liumd_shadow_facts_attempted=True,
        )
        return CheckResult(
            passed=True,
            event=event,
            updates={"state": updated_state, "default_extra": {**ctx.default_extra, **extra_info}},
        )

    @staticmethod
    def _fail_liumd(
        ctx: Context,
        answer: Any,
        detail: str | None = None,
        *,
        reason: str | None = None,
    ) -> CheckResult:
        from services.liumd_exec_client import LiumdRefusal
        from services.local_verify_client import LocalVerifyUnavailable

        if reason is None:
            if isinstance(answer, LocalVerifyUnavailable):
                reason = answer.reason or "unavailable"
                detail = detail or answer.detail
            elif isinstance(answer, LiumdRefusal):
                # liumd answered and refused: its own error code and detail say why, not "malformed"
                reason = answer.error or "refused"
                detail = detail or answer.detail or f"liumd exit {answer.exit_status}"
            elif answer is None:
                reason = "no_early_read"
            else:
                reason = "malformed"
        detail = (detail or reason)[:300]
        template = Msg.SCRAPE_TIMEOUT if reason == "timeout" else Msg.SCRAPE_FAILED_ON_HOST
        event = render_message(
            template,
            ctx=ctx,
            check_id=MachineSpecScrapeCheck.check_id,
            what={
                "delivery": "liumd",
                "liumd_reason": reason,
                "scrape_error": reason,
                "detail": detail,
            },
        )
        return CheckResult(passed=False, event=event)

    async def _scrape_from_source(
        self, ctx: Context, timeout: int, *, start_shadow: bool = True
    ) -> CheckResult:
        # pipe the obfuscated source into the executor's own interpreter; binary on failure
        #
        # `-I` implies -E -s -P, which drops PYTHONPATH, the user site-dir and the cwd entry
        # from sys.path, so a file left next to the SSH login shell cannot shadow psutil or
        # json. It does not disable site processing: a .pth or sitecustomize inside the image
        # still runs and prints, which is what `_decrypt_payload` searches through.
        command = f"{shlex.quote(executor_payload.EXECUTOR_PYTHON)} -I -"
        scrape_run = await ctx.runner.run(
            command,
            timeout=timeout,
            retryable=False,
            stdin_text=ctx.config.machine_scrape_source,
        )
        result = await self._check_result_from_scrape_run(
            ctx, scrape_run, delivery="stdin", start_shadow=start_shadow
        )
        if result.passed:
            return result

        # The source did not run here: wrong interpreter, no psutil, or a cryptography too old
        # to produce a token this validator can read. The binary carries its own interpreter and
        # every dependency, so retry with it — but past STDIN_FAILURE_RETRY_WINDOW_MS the scrape
        # hung rather than failed to start, and the binary runs the very same code.
        if (
            scrape_run.duration_ms > STDIN_FAILURE_RETRY_WINDOW_MS
            or not ctx.config.machine_scrape_filename
        ):
            return result

        # The scrape reporting its own failure (no GPU details, for one) means the interpreter
        # ran it, so the binary would print the same thing for the price of a 13 MB upload.
        if scrape_run.exit_code != 0 and _scrape_reported_its_own_failure(scrape_run.stdout):
            return result

        if not await self._binary_ready_in_time(ctx, timeout):
            return result

        return await self._retry_scrape_with_uploaded_binary(
            ctx, timeout, fallback_from=StdinAttempt.from_failed_result(result), start_shadow=start_shadow
        )

    async def _binary_ready_in_time(self, ctx: Context, timeout: int) -> bool:
        # DAH-5899: the cycle builds its binary only now, once for every node that needs it; the build counts
        # against the task's time, and a retry whose scrape would no longer fit keeps the stdin verdict
        # (upload up to UPLOAD_TIMEOUT, then the scrape's own timeout)
        if ctx.config.machine_scrape_binary_build is None:
            return True
        seconds_the_retry_needs = timeout + UPLOAD_TIMEOUT
        wait_seconds = task_seconds_left(ctx) - seconds_the_retry_needs
        if wait_seconds <= 0:
            return False
        try:
            await ctx.config.machine_scrape_binary_build.ensure(wait_seconds)
        except Exception as exc:
            logger.error(
                _m("[scrape] binary not ready; keeping the stdin result", extra=get_extra_info(
                    {**ctx.default_extra, "error": repr(exc)[:400]}
                ))
            )
            return False
        return task_seconds_left(ctx) >= seconds_the_retry_needs

    async def _retry_scrape_with_uploaded_binary(
        self,
        ctx: Context,
        timeout: int,
        *,
        fallback_from: StdinAttempt,
        start_shadow: bool = True,
    ) -> CheckResult:
        # upload the frozen scrape and run it, after the stdin delivery failed to start
        try:
            remote_dir = await upload_validation_files_to_fresh_remote_dir(ctx, attempts=FALLBACK_UPLOAD_ATTEMPTS)
        except UploadFailed as exc:
            event = render_message(
                Msg.SCRAPE_TRANSPORT_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "delivery": "stdin",
                    "fallback_from": fallback_from.as_event_field(),
                    "fallback_upload_error": str(exc),
                },
            )
            return CheckResult(passed=False, event=event)

        scrape_run = await ctx.runner.run(
            _binary_command(remote_dir, ctx.config.machine_scrape_filename),
            timeout=timeout,
            retryable=False,
        )
        return await self._check_result_from_scrape_run(
            ctx, scrape_run, delivery="upload", fallback_from=fallback_from, start_shadow=start_shadow
        )

    async def _scrape_uploaded_binary(
        self, ctx: Context, timeout: int, *, start_shadow: bool = True
    ) -> CheckResult:
        # run the frozen scrape that UploadFilesCheck put on the executor
        remote_dir = ctx.state.remote_dir
        if not remote_dir:
            event = render_message(
                Msg.REMOTE_DIR_MISSING,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=False, event=event)

        script_filename = ctx.config.machine_scrape_filename
        if not script_filename:
            event = render_message(
                Msg.CONFIG_MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what={"machine_scrape_filename": script_filename},
            )
            return CheckResult(passed=False, event=event)

        scrape_run = await ctx.runner.run(
            _binary_command(remote_dir, script_filename), timeout=timeout, retryable=False
        )
        return await self._check_result_from_scrape_run(
            ctx, scrape_run, delivery="upload", start_shadow=start_shadow
        )

    async def _check_result_from_scrape_run(
        self,
        ctx: Context,
        scrape_run: SSHCommandResult,
        *,
        delivery: Literal["stdin", "upload"],
        fallback_from: StdinAttempt | None = None,
        start_shadow: bool = True,
    ) -> CheckResult:
        what: dict[str, Any] = {"delivery": delivery}
        if fallback_from:
            what["fallback_from"] = fallback_from.as_event_field()

        if not scrape_run.success or not scrape_run.stdout.strip():
            failure = _classify_scrape_failure(scrape_run, ctx.config.obfuscation_keys)
            # DAH-4658: every card is missing; held only when each one is a GPU the renter resets
            if failure.scrape_error == "no_gpu_details" and await missing_gpus_under_reset(ctx, []):
                event = render_message(
                    Msg.SCRAPE_GPU_RESET_IN_PROGRESS,
                    ctx=ctx,
                    check_id=self.check_id,
                    what={**what, "held_reason_code": failure.template.reason, **failure.cause_event_fields()},
                )
                return CheckResult(passed=False, event=event)
            gpu_lost_events: list[ValidationEvent] = []
            if failure.scrape_error == "no_gpu_details":
                gpu_lost_events = await _report_rented_gpu_loss(
                    ctx, _no_gpu_report_specs(scrape_run.stdout, ctx.config.obfuscation_keys)
                )
            reset = _host_gpu_fault_reset(ctx, failure, self.check_id)
            event = render_message(
                failure.template,
                ctx=ctx,
                check_id=self.check_id,
                impact=HOST_GPU_RUNTIME_FAULT_IMPACT if reset else None,
                what={
                    **what,
                    "command_id": scrape_run.command_id,
                    "exit_code": scrape_run.exit_code,
                    "duration_ms": scrape_run.duration_ms,
                    "stderr_tail": scrape_run.stderr[-400:],
                    **failure.cause_event_fields(),
                    **({"host_gpu_fault_reset": True} if reset else {}),
                },
            )
            return CheckResult(passed=False, event=event, updates=reset, extra_events=gpu_lost_events)

        try:
            decrypted = _decrypt_payload(ctx, scrape_run.stdout)
            raw = json.loads(decrypted)
            obfuscation_keys = ctx.config.obfuscation_keys
            specs = _deobfuscate(raw, obfuscation_keys)
            from ..liumd_shadow import start_facts_after_scrape  # liumd_shadow imports checks

            return self._result_from_specs(
                ctx,
                specs,
                delivery=delivery,
                task=start_facts_after_scrape(ctx) if start_shadow else None,
                what=what,
            )

        except Exception as exc:
            event = render_message(
                Msg.SCRAPE_PARSE_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={**what, "exception": str(exc)[:300], "stdout_head": scrape_run.stdout[:200]},
            )
            return CheckResult(passed=False, event=event)
