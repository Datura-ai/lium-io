from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .models import build_msg

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import Context


@dataclass(frozen=True)
class MessageTemplate:
    event: str
    reason: str
    severity: str
    category: str
    impact: str
    remediation: str | None = None
    help_uri: str | None = None


def render_message(
    template: MessageTemplate,
    *,
    ctx: Context,
    check_id: str,
    what: dict[str, Any] | None = None,
    severity: str | None = None,
    category: str | None = None,
    impact: str | None = None,
    remediation: str | None = None,
    help_uri: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Render a `MessageTemplate` into a structured message via `build_msg`."""
    return build_msg(
        event=template.event,
        reason=template.reason,
        severity=severity or template.severity,
        category=category or template.category,
        impact=impact or template.impact,
        remediation=(remediation if remediation is not None else template.remediation) or "",
        what=what or {},
        check_id=check_id,
        pipeline_id=ctx.pipeline_id,
        ctx={**ctx.default_extra, **(extra or {})},
        help_uri=help_uri or template.help_uri,
    )


class StartGpuMonitorMessages:
    CONFIG_MISSING = MessageTemplate(
        event="GPU monitor configuration missing",
        reason="MONITOR_CONFIG_MISSING",
        severity="error",
        category="prep",
        impact="Validation halted",
        remediation="Validator bug: missing monitor configuration in context.",
    )
    ALREADY_RUNNING = MessageTemplate(
        event="GPU monitor already running",
        reason="MONITOR_RUNNING",
        severity="info",
        category="prep",
        impact="Proceed",
    )
    STARTED = MessageTemplate(
        event="GPU monitor started",
        reason="MONITOR_STARTED",
        severity="info",
        category="prep",
        impact="Proceed with monitoring enabled",
    )
    START_FAILED = MessageTemplate(
        event="Failed to start GPU monitor",
        reason="MONITOR_START_FAILED",
        severity="warning",
        category="prep",
        impact="Validation continues without real-time GPU monitoring",
        remediation="Check Python installation and script permissions on executor.",
    )


class UploadFilesMessages:
    CONFIG_MISSING = MessageTemplate(
        event="Upload configuration missing",
        reason="UPLOAD_CONFIG_MISSING",
        severity="error",
        category="prep",
        impact="Validation halted",
        remediation="Validator bug: missing upload metadata in context",
    )
    UPLOAD_OK = MessageTemplate(
        event="Validation files uploaded",
        reason="UPLOAD_OK",
        severity="info",
        category="prep",
        impact="Proceed to validation",
    )
    UPLOAD_FAILED = MessageTemplate(
        event="Failed to upload validation files",
        reason="UPLOAD_FAILED",
        severity="error",
        category="prep",
        impact="Validation halted",
        remediation="Check network connectivity, disk space on executor, and SSH permissions",
    )


class MachineSpecMessages:
    REMOTE_DIR_MISSING = MessageTemplate(
        event="Remote directory not set",
        reason="MISSING_REMOTE_DIR",
        severity="error",
        category="prep",
        impact="Cannot locate scrape script",
        remediation="Internal error - UploadFilesCheck must run before MachineSpecScrapeCheck",
    )
    CONFIG_MISSING = MessageTemplate(
        event="Machine scrape configuration missing",
        reason="SCRAPE_CONFIG_MISSING",
        severity="error",
        category="prep",
        impact="Validation halted",
        remediation="Validator bug: missing scrape configuration in context",
    )
    DECRYPT_MISSING = MessageTemplate(
        event="SSH service unavailable",
        reason="SCRAPE_DECRYPT_MISSING",
        severity="error",
        category="prep",
        impact="Validation halted",
        remediation="Validator bug: missing SSH service in context",
    )
    SCRAPE_FAILED = MessageTemplate(
        event="Machine specs scrape failed",
        reason="SCRAPE_FAILED",
        severity="error",
        category="env",
        impact="Validation halted — GPU unverified",
        remediation=(
            "Ensure the scrape script exists and is executable:"
            "\n  chmod +x <script>\nCheck stderr and environment on the executor."
        ),
    )
    SCRAPE_OK = MessageTemplate(
        event="Machine specs scraped",
        reason="SCRAPE_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    SCRAPE_PARSE_FAILED = MessageTemplate(
        event="Machine specs parse/decrypt failed",
        reason="SCRAPE_PARSE_FAILED",
        severity="error",
        category="env",
        impact="Validation halted — GPU unverified",
        remediation="Confirm encryption key, payload, and repo versions on both validator and executor.",
    )


class GpuCountMessages:
    POLICY_MISSING = MessageTemplate(
        event="GPU count policy missing",
        reason="GPU_COUNT_POLICY_MISSING",
        severity="error",
        category="policy",
        impact="Validation halted",
        remediation="Validator bug: max GPU count not configured in context",
    )
    COUNT_EXCEEDS = MessageTemplate(
        event="GPU count exceeds policy",
        reason="GPU_COUNT_EXCEEDS_MAX",
        severity="error",
        category="policy",
        impact="Score set to 0",
        remediation="Reduce visible GPU count to policy limit (e.g., via CUDA_VISIBLE_DEVICES).",
    )
    COUNT_OK = MessageTemplate(
        event="GPU count within limits",
        reason="GPU_COUNT_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )


class GpuModelMessages:
    POLICY_MISSING = MessageTemplate(
        event="GPU model policy missing",
        reason="GPU_MODEL_POLICY_MISSING",
        severity="error",
        category="policy",
        impact="Validation halted",
        remediation="Validator bug: GPU model rates not configured in context",
    )
    MODEL_UNSUPPORTED = MessageTemplate(
        event="GPU model not supported",
        reason="GPU_MODEL_UNSUPPORTED",
        severity="warning",
        category="policy",
        impact="Job skipped; score set to 0",
    )
    COUNT_ZERO = MessageTemplate(
        event="No GPUs detected",
        reason="GPU_COUNT_ZERO",
        severity="warning",
        category="env",
        impact="Job skipped; score set to 0",
        remediation="Ensure GPUs are properly installed and visible to the system",
    )
    DETAILS_MISMATCH = MessageTemplate(
        event="GPU count mismatch",
        reason="GPU_DETAILS_MISMATCH",
        severity="warning",
        category="env",
        impact="Job skipped; score set to 0",
        remediation="GPU count and details length don't match. Check GPU detection.",
    )
    MODEL_OK = MessageTemplate(
        event="GPU model validated",
        reason="GPU_MODEL_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )


class GpuVramMessages:
    VRAM_REJECTED = MessageTemplate(
        event="GPU model/VRAM precheck rejected",
        reason="GPU_VRAM_MISMATCH",
        severity="warning",
        category="policy",
        impact="Job skipped; score set to 0",
        remediation="Reported GPU model and VRAM capacity are inconsistent. "
        "Ensure the executor advertises unmodified, matching GPU specs.",
    )
    VRAM_OK = MessageTemplate(
        event="GPU model/VRAM precheck passed",
        reason="GPU_VRAM_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )


class GpuPowerLimitMessages:
    LIMIT_BELOW_DEFAULT = MessageTemplate(
        event="GPU power limit below default threshold",
        reason="GPU_POWER_LIMIT_BELOW_DEFAULT",
        severity="warning",
        category="policy",
        impact="Job skipped; score set to 0",
        remediation=(
            "Restore the GPU power limit to at least 90% of the default limit "
            "reported by NVML."
        ),
    )
    DATA_INCOMPLETE = MessageTemplate(
        event="GPU power limit baseline incomplete",
        reason="GPU_POWER_LIMIT_BASELINE_INCOMPLETE",
        severity="info",
        category="env",
        impact="Proceed without power-limit penalty",
        remediation="Ensure NVML exposes the current and default power limits for every GPU.",
    )
    LIMIT_OK = MessageTemplate(
        event="GPU power limit check passed",
        reason="GPU_POWER_LIMIT_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )


class NvmlDigestMessages:
    DIGEST_MISMATCH = MessageTemplate(
        event="NVML library digest mismatch",
        reason="NVML_DIGEST_MISMATCH",
        severity="error",
        category="env",
        impact="Score set to 0; previous verification cleared",
        remediation="Reinstall the NVIDIA driver matching this version and ensure libnvidia-ml is not tampered.",
    )
    DRIVER_UNKNOWN = MessageTemplate(
        event="Unknown NVIDIA driver version",
        reason="NVML_DRIVER_UNKNOWN",
        severity="error",
        category="env",
        impact="Score set to 0; previous verification cleared",
        remediation="Update to a supported NVIDIA driver version. Your current driver version is not recognized.",
    )
    DIGEST_OK = MessageTemplate(
        event="NVML library digest verified",
        reason="NVML_DIGEST_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )


class SpecChangeMessages:
    SPEC_CHANGED = MessageTemplate(
        event="GPU inventory changed",
        reason="SPEC_CHANGED",
        severity="warning",
        category="env",
        impact="Verification reset; score set to 0",
        remediation="Keep GPU configuration stable between checks.",
    )
    SPEC_UNCHANGED = MessageTemplate(
        event="GPU inventory stable",
        reason="SPEC_UNCHANGED",
        severity="info",
        category="env",
        impact="Proceed",
    )


class GpuFingerprintMessages:
    UUID_CHANGED = MessageTemplate(
        event="GPU fingerprints changed",
        reason="GPU_UUID_CHANGED",
        severity="warning",
        category="env",
        impact="Verification reset; score set to 0",
        remediation="Ensure the same physical GPUs remain attached and stable.",
    )
    UUID_OK = MessageTemplate(
        event="GPU fingerprints stable",
        reason="GPU_UUID_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )


class BannedGpuMessages:
    UUID_EMPTY = MessageTemplate(
        event="No GPU fingerprints captured",
        reason="GPU_UUID_EMPTY",
        severity="info",
        category="env",
        impact="Proceed",
        remediation="Ensure the scrape script emits GPU UUIDs via nvidia-smi.",
    )
    GPU_BANNED = MessageTemplate(
        event="GPU model temporarily ineligible",
        reason="GPU_BANNED",
        severity="warning",
        category="policy",
        impact="Score set to 0; verification cleared",
        remediation="Swap to eligible GPUs or wait for policy updates before retrying validation.",
    )
    GPU_ALLOWED = MessageTemplate(
        event="GPU fingerprints allowed",
        reason="GPU_BANNED_CHECK_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )


class SysboxRequiredMessages:
    SYSBOX_MISSING = MessageTemplate(
        event="Sysbox required for unrented executor",
        reason="SYSBOX_REQUIRED_MISSING",
        severity="warning",
        category="policy",
        impact="Score set to 0; verification cleared; executor kept off the network",
        remediation="Install the sysbox runtime; unrented machines without sysbox are not allowed on the network.",
    )
    SYSBOX_OK = MessageTemplate(
        event="Sysbox requirement satisfied",
        reason="SYSBOX_REQUIRED_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    DISABLED = MessageTemplate(
        event="Sysbox requirement disabled",
        reason="SYSBOX_REQUIRED_DISABLED",
        severity="info",
        category="policy",
        impact="Proceed",
    )


class DuplicateExecutorMessages:
    DUPLICATE = MessageTemplate(
        event="Duplicate executor registration",
        reason="EXECUTOR_DUPLICATE",
        severity="warning",
        category="policy",
        impact="Score set to 0; verification cleared",
        remediation="Ensure every executor has a unique UUID and deregister duplicates before retrying.",
    )
    UNIQUE = MessageTemplate(
        event="Executor registration unique",
        reason="EXECUTOR_NOT_DUPLICATE",
        severity="info",
        category="policy",
        impact="Proceed",
    )


class CollateralMessages:
    VERIFIED = MessageTemplate(
        event="Collateral verified",
        reason="COLLATERAL_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    MISSING = MessageTemplate(
        event="No collateral deposited",
        reason="COLLATERAL_MISSING",
        severity="warning",
        category="policy",
        impact="Score may be reduced or set to 0 based on policy",
        remediation="Deposit collateral for this executor.",
    )


class StaleContainerCleanupMessages:
    CLEANED = MessageTemplate(
        event="Stale container cleanup complete",
        reason="STALE_CLEANUP_DONE",
        severity="info",
        category="prep",
        impact="Proceed",
    )


class CustomBuildOrphanSweepMessages:
    """DAH-2211 — Phase 3.4(ii) — periodic GC of `lium-build-*` image+scratch
    orphans left behind by validator crashes / aborted releases. The happy
    path (inline cleanup at pod release) means this sweep usually finds
    nothing."""

    SKIPPED = MessageTemplate(
        event="Custom build orphan sweep skipped (cadence)",
        reason="CUSTOM_BUILD_SWEEP_SKIPPED",
        severity="info",
        category="prep",
        impact="Proceed",
    )
    SWEPT = MessageTemplate(
        event="Custom build orphan sweep complete",
        reason="CUSTOM_BUILD_SWEEP_DONE",
        severity="info",
        category="prep",
        impact="Proceed",
    )


class InspectorMessages:
    SKIPPED = MessageTemplate(
        event="Inspector validation skipped",
        reason="INSPECTOR_SKIPPED_NOT_RENTED",
        severity="info",
        category="runtime",
        impact="No active customer rental on this executor",
    )
    DISABLED = MessageTemplate(
        event="Inspector validation disabled",
        reason="INSPECTOR_SKIPPED_DISABLED",
        severity="info",
        category="runtime",
        impact="Inspector checks are disabled",
    )
    CLEAN = MessageTemplate(
        event="Inspector validation clean",
        reason="INSPECTOR_CLEAN",
        severity="info",
        category="runtime",
        impact="No malicious findings reported",
    )
    COLLECTOR_RECENTLY_STARTED = MessageTemplate(
        event="Inspector collector recently started",
        reason="INSPECTOR_COLLECTOR_RECENTLY_STARTED",
        severity="warning",
        category="runtime",
        impact="Collection window may be incomplete; possible evasion",
        remediation="Review collector uptime before trusting a clean result.",
    )
    COLLECTOR_NOT_RUNNING = MessageTemplate(
        event="Inspector collector not running",
        reason="INSPECTOR_COLLECTOR_NOT_RUNNING",
        severity="warning",
        category="runtime",
        impact="No collector telemetry; possible evasion",
        remediation="Verify the collector process is running and reachable.",
    )
    CANARY_FAILED = MessageTemplate(
        event="Inspector canary check failed",
        reason="INSPECTOR_CANARY_FAILED",
        severity="warning",
        category="runtime",
        impact="Canary check failed; result not trustworthy",
        remediation="Investigate collector integrity and libinspector canary state.",
    )
    MALICIOUS_FINDINGS = MessageTemplate(
        event="Inspector malicious findings detected",
        reason="INSPECTOR_MALICIOUS_FINDINGS",
        severity="warning",
        category="runtime",
        impact="Findings logged; score unchanged",
        remediation="Review the libinspector report and tenant workload before taking action.",
    )
    VALIDATION_ERROR = MessageTemplate(
        event="Inspector validation error",
        reason="INSPECTOR_VALIDATION_ERROR",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable; score unchanged",
        remediation="Check libinspector installation and executor SSH process logs.",
    )
    FAILED_LIB_MISMATCH = MessageTemplate(
        event="Inspector libinspector.so mismatch",
        reason="INSPECTOR_FAILED_LIB_MISMATCH",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable; score unchanged",
        remediation=(
            "The executor's libinspector.so SHA256 does not match the validator's. "
            "Verify both sides ship the same libinspector build and restart the executor "
            "container (docker compose restart) if it does not."
        ),
    )
    FAILED_SSH_TRANSPORT = MessageTemplate(
        event="Inspector SSH transport failed",
        reason="INSPECTOR_FAILED_SSH_TRANSPORT",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable; score unchanged",
        remediation=(
            "Validator could not reach the executor over SSH. Verify the executor is running "
            "and reachable on its SSH port, and that the validator's SSH key is present in the "
            "executor's authorized_keys."
        ),
    )
    FAILED_EXECUTOR_CRASH = MessageTemplate(
        event="Inspector executor crashed",
        reason="INSPECTOR_FAILED_EXECUTOR_CRASH",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable; score unchanged",
        remediation=(
            "The inspector_executor.py process exited before completing the interactive session. "
            "Check executor container logs (docker logs <executor-container>) for tracebacks."
        ),
    )
    FAILED_CIPHER_REJECTED = MessageTemplate(
        event="Inspector cipher rejected",
        reason="INSPECTOR_FAILED_CIPHER_REJECTED",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable; score unchanged",
        remediation=(
            "The executor returned a response that the validator's libinspector.so rejected. "
            "Verify the executor's libinspector.so SHA256 matches the validator's, and restart "
            "the executor container (docker compose restart) if it does not."
        ),
    )
    FAILED_TIMEOUT = MessageTemplate(
        event="Inspector interactive command timed out",
        reason="INSPECTOR_FAILED_TIMEOUT",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable; score unchanged",
        remediation=(
            "The inspector_executor.py REPL did not respond within the configured timeout. "
            "Check for host load, stuck collector startup, or OOM on the executor."
        ),
    )
    FAILED_INTERACTIVE = MessageTemplate(
        event="Inspector executor command failed",
        reason="INSPECTOR_FAILED_INTERACTIVE",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable; score unchanged",
        remediation=(
            "The executor REPL returned an error for a handshake, collect, or collector command. "
            "Read executor_stderr in the validation event and inspect libinspector on the executor."
        ),
    )


class TenantEnforcementMessages:
    NOT_RENTED = MessageTemplate(
        event="Executor not rented",
        reason="EXECUTOR_NOT_RENTED",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    POD_NOT_RUNNING = MessageTemplate(
        event="Pod not running",
        reason="POD_NOT_RUNNING",
        severity="error",
        category="runtime",
        impact="Score set to 0; verification cleared",
        remediation="Start container and ensure it stays healthy.",
    )
    STALE_POD_NOT_RUNNING = MessageTemplate(
        event="Stale rented pod not running signal skipped",
        reason="STALE_POD_NOT_RUNNING",
        severity="info",
        category="runtime",
        impact="No reset sent because backend no longer considers the pod rental active",
        remediation="No action needed.",
    )
    EXECUTOR_TRANSPORT_UNREACHABLE = MessageTemplate(
        event="Executor unreachable over SSH",
        reason="EXECUTOR_TRANSPORT_UNREACHABLE",
        severity="warning",
        category="transport",
        impact="No verdict for this cycle - pod state unknown",
        remediation=(
            "Transient SSH transport failure (network blip, conntrack flush). "
            "Persistent unreachability is handled by the stale-executor sweep."
        ),
    )
    GPU_OUTSIDE_TENANT = MessageTemplate(
        event="Tenant container does not own GPU",
        reason="GPU_USAGE_OUTSIDE_TENANT",
        severity="warning",
        category="runtime",
        impact="Validation failed; score set to 0",
        remediation="Terminate host-level GPU processes, ensure nvidia-smi shows no extra workloads.",
    )
    ALREADY_RENTED = MessageTemplate(
        event="Executor already rented",
        reason="RENTED",
        severity="info",
        category="policy",
        impact="Proceed",
    )


class GpuUsageMessages:
    USAGE_OK = MessageTemplate(
        event="GPU usage within limits",
        reason="GPU_USAGE_OK",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    USAGE_HIGH = MessageTemplate(
        event="GPU busy outside validator",
        reason="GPU_USAGE_HIGH",
        severity="warning",
        category="runtime",
        impact="Validation skipped; score set to 0",
        remediation="Stop all GPU processes and re-run your node. If using Docker, ensure no host processes are running.",
    )
    ORPHANED_CONTAINER = MessageTemplate(
        event="Orphaned rental container detected",
        reason="ORPHANED_RENTAL_CONTAINER",
        severity="error",
        category="runtime",
        impact="Validation skipped; score set to 0",
        remediation="Rental ended but container still running. Remove it: docker stop {orphaned_container}",
    )


class PortConnectivityMessages:
    SKIPPED_RENTED = MessageTemplate(
        event="Port connectivity skipped for rented executor",
        reason="PORT_CONNECTIVITY_SKIPPED",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    RENTING_IN_PROGRESS = MessageTemplate(
        event="Renting already in progress",
        reason="RENTING_IN_PROGRESS",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    CONFIG_MISSING = MessageTemplate(
        event="Port connectivity configuration missing",
        reason="PORT_CONNECTIVITY_CONFIG_MISSING",
        severity="error",
        category="runtime",
        impact="Validation halted",
        remediation="Validator bug: missing port verification config in context",
    )
    VERIFY_FAILED = MessageTemplate(
        event="Port verification failed",
        reason="PORT_VERIFY_FAILED",
        severity="error",
        category="runtime",
        impact="Score set to 0",
        remediation="Check Docker access and port mappings, then retry validation.",
    )
    VERIFY_OK = MessageTemplate(
        event="Port verification completed",
        reason="PORT_VERIFY_OK",
        severity="info",
        category="runtime",
        impact="Proceed",
    )


class PortCountMessages:
    PORT_COUNT_RECORDED = MessageTemplate(
        event="Port availability inspected",
        reason="PORT_COUNT_RECORDED",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    INSUFFICIENT_PORTS = MessageTemplate(
        event="Insufficient available ports",
        reason="INSUFFICIENT_PORTS",
        severity="warning",
        category="runtime",
        impact="Score set to 0",
        remediation="Increase port range or check port mappings configuration.",
    )


VERIFYX_DEBUG_DOC_URL = (
    "https://github.com/Datura-ai/lium-io/blob/main/docs/lium-io/verifyx-debug.md"
)


def _verify_failed(label: str, reason_suffix: str, remediation: str) -> MessageTemplate:
    """Build a `VerifyX validation failed` template. event/severity/category/impact/help_uri
    are identical across all failure classes; only label, reason suffix, and remediation vary."""
    event = "VerifyX validation failed" + (f" ({label})" if label else "")
    reason = "VERIFYX_FAILED" + (f"_{reason_suffix}" if reason_suffix else "")
    return MessageTemplate(
        event=event,
        reason=reason,
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=remediation,
        help_uri=VERIFYX_DEBUG_DOC_URL,
    )


class VerifyXMessages:
    DISABLED = MessageTemplate(
        event="VerifyX validation skipped",
        reason="VERIFYX_DISABLED",
        severity="info",
        category="env",
        impact="Proceed",
    )
    FILLER_SKIPPED = MessageTemplate(
        event="VerifyX validation skipped",
        reason="VERIFYX_SKIPPED_ACTIVE_FILLER",
        severity="info",
        category="policy",
        impact="Active filler runtime preserved; reused last-known VerifyX EMA when available",
    )
    NO_SPECS = MessageTemplate(
        event="VerifyX validation skipped (no specs)",
        reason="VERIFYX_NO_SPECS",
        severity="error",
        category="env",
        impact="Validation halted",
        remediation="Run the machine scrape before executing VerifyX.",
    )
    VERIFY_SUCCESS = MessageTemplate(
        event="VerifyX validation passed",
        reason="VERIFYX_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    VERIFY_FAILED = _verify_failed(
        label="",
        reason_suffix="",
        remediation="Run VerifyX locally to debug network, disk, and RAM probes. See the debug doc for detailed steps.",
    )
    VERIFY_FAILED_SSH_TRANSPORT = _verify_failed(
        label="SSH transport",
        reason_suffix="SSH_TRANSPORT",
        remediation=(
            "Validator could not reach the executor over SSH. Verify the executor is running "
            "and reachable on its SSH port, and that the validator's SSH key is present in the "
            "executor's authorized_keys."
        ),
    )
    VERIFY_FAILED_EXECUTOR_CRASH = _verify_failed(
        label="executor crashed",
        reason_suffix="EXECUTOR_CRASH",
        remediation=(
            "The verifyx_executor.py process exited with a non-zero status. Read the executor "
            "container logs (docker logs <executor-container>) for the traceback, then reproduce "
            "by running verifyx_executor.py directly with the seed and cipher_text captured in "
            "the validator log line."
        ),
    )
    VERIFY_FAILED_EMPTY_RESPONSE = _verify_failed(
        label="empty response",
        reason_suffix="EMPTY_RESPONSE",
        remediation=(
            "The executor returned empty or truncated stdout. Check for OOM-killer events "
            "(dmesg) and disk-full conditions (df -h) on the executor host."
        ),
    )
    VERIFY_FAILED_CIPHER_REJECTED = _verify_failed(
        label="cipher rejected",
        reason_suffix="CIPHER_REJECTED",
        remediation=(
            "The executor returned a response that the validator's libverifyx.so rejected. "
            "Verify the executor's libverifyx.so SHA256 matches the validator's, and restart "
            "the executor container (docker compose restart) if it does not."
        ),
    )
    VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW = _verify_failed(
        label="network speed too slow",
        reason_suffix="NETWORK_SPEED_TOO_SLOW",
        remediation=(
            "EMA download speed is below the minimum threshold. "
            "Improve the executor's network connection or wait for the EMA to recover "
            "once speeds improve."
        ),
    )


class TdxHostMessages:
    TDX_SUPPORTED = MessageTemplate(
        event="Intel TDX CPU capability detected",
        reason="TDX_HOST_SUPPORTED",
        severity="info",
        category="env",
        impact="CPU silicon supports Intel TDX workloads",
    )
    TDX_NOT_SUPPORTED = MessageTemplate(
        event="Intel TDX CPU capability not detected",
        reason="TDX_HOST_NOT_SUPPORTED",
        severity="info",
        category="env",
        impact="CPU does not support TDX; tdx_host_supported=False recorded in spec",
    )


class CapabilityMessages:
    NO_SPECS = MessageTemplate(
        event="GPU capability verification skipped (no specs)",
        reason="GPU_VERIFY_NO_SPECS",
        severity="error",
        category="env",
        impact="Validation halted",
        remediation="Run machine scrape before capability validation.",
    )
    VERIFY_OK = MessageTemplate(
        event="GPU capability validated",
        reason="GPU_VERIFY_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    FILLER_SKIPPED = MessageTemplate(
        event="GPU capability validation skipped for active filler",
        reason="GPU_VERIFY_SKIPPED_ACTIVE_FILLER",
        severity="info",
        category="policy",
        impact="Active filler runtime preserved; capability probe would compete for VRAM.",
    )
    VERIFY_FAILED = MessageTemplate(
        event="GPU capability verification failed",
        reason="GPU_VERIFY_FAILED",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation="Run Docker GPU diagnostics (nvidia-smi) and ensure containers can access GPUs.",
    )


class ScoreMessages:
    SCORE_COMPUTED = MessageTemplate(
        event="Scores computed",
        reason="SCORE_COMPUTED",
        severity="info",
        category="policy",
        impact="Proceed",
    )


class RentalVerificationMessages:
    SKIPPED = MessageTemplate(
        event="Rental verification skipped",
        reason="RENTAL_CHECK_DISABLED",
        severity="info",
        category="policy",
        impact="Proceed without backend rental verification",
    )
    VERIFIED = MessageTemplate(
        event="Rental verification successful",
        reason="RENTAL_VERIFIED",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    FAILED = MessageTemplate(
        event="Rental verification failed",
        reason="RENTAL_NOT_VERIFIED",
        severity="error",
        category="policy",
        impact="Validation halted",
        remediation="Check executor health with backend or review rental status",
    )
    GPU_RUNTIME_NVML_MISMATCH = MessageTemplate(
        event="GPU runtime health check failed",
        reason="GPU_RUNTIME_NVML_MISMATCH",
        severity="error",
        category="runtime",
        impact="New rentals are disabled because Docker cannot initialize NVIDIA NVML on this host.",
        remediation=(
            "Reconcile the NVIDIA driver, libnvidia-ml.so.1, NVIDIA Container "
            "Toolkit, and Docker runtime. Then reboot the host with `sudo reboot`."
        ),
    )
    API_ERROR = MessageTemplate(
        event="Rental verification API error",
        reason="RENTAL_CHECK_API_ERROR",
        severity="warning",
        category="policy",
        impact="Proceed with existing rental status",
        remediation="Check backend API connectivity and logs",
    )


class FinalizeMessages:
    COMPLETED = MessageTemplate(
        event="Validation task completed",
        reason="VALIDATION_COMPLETED",
        severity="info",
        category="runtime",
        impact="Proceed",
    )


class CachedTemplateMessages:
    """DAH-2265 Plan 2 — advisory cached-template verification.

    Confirms whether the executor has the recommended default image pre-pulled so
    DOCKER_PULL is a no-op for default-template rentals. Purely observational: every
    template is severity="info" and the check never fails the pipeline or changes score.
    """

    CACHED = MessageTemplate(
        event="Recommended default image is cached on executor",
        reason="RECOMMENDED_IMAGE_CACHED",
        severity="info",
        category="runtime",
        impact="None — default-template DOCKER_PULL will be a no-op",
    )
    NOT_CACHED = MessageTemplate(
        event="Recommended default image not cached on executor",
        reason="RECOMMENDED_IMAGE_NOT_CACHED",
        severity="info",
        category="runtime",
        impact=(
            "None (advisory) — default-template rentals will pay a docker pull "
            "until the executor pre-pull catches up"
        ),
        remediation=(
            "Executor cache pre-pull keeps the recommended image warm; "
            "verify cache_template_service is running on the host."
        ),
    )
    SKIPPED = MessageTemplate(
        event="Cached-template verification skipped",
        reason="RECOMMENDED_IMAGE_CHECK_SKIPPED",
        severity="info",
        category="runtime",
        impact="None — could not determine the recommended image this cycle",
    )
    DIGEST_MATCH = MessageTemplate(
        event="Recommended image digest matches the published manifest",
        reason="RECOMMENDED_IMAGE_DIGEST_MATCH",
        severity="info",
        category="runtime",
        impact="None — node holds current content under this tag",
    )
    DIGEST_MISMATCH = MessageTemplate(
        event="Recommended image digest differs from the published manifest",
        reason="RECOMMENDED_IMAGE_DIGEST_MISMATCH",
        severity="info",
        category="runtime",
        impact="None (advisory) — node serves STALE content under an unchanged tag",
        remediation=(
            "Provider should re-pull repo:tag to fetch the latest content; "
            "the executor cache pre-pull may be broken or disabled."
        ),
    )
    DIGEST_SKIPPED = MessageTemplate(
        event="Recommended image digest check skipped",
        reason="RECOMMENDED_IMAGE_DIGEST_SKIPPED",
        severity="info",
        category="runtime",
        impact="None — digest unknown this cycle (no backend digest / unreadable RepoDigests)",
    )
