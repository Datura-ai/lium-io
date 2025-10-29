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
    ctx: "Context",
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


class VerifyXMessages:
    DISABLED = MessageTemplate(
        event="VerifyX validation skipped",
        reason="VERIFYX_DISABLED",
        severity="info",
        category="env",
        impact="Proceed",
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
    VERIFY_FAILED = MessageTemplate(
        event="VerifyX validation failed",
        reason="VERIFYX_FAILED",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation="Run VerifyX locally to debug network, disk, and RAM probes.",
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


class FinalizeMessages:
    COMPLETED = MessageTemplate(
        event="Validation task completed",
        reason="VALIDATION_COMPLETED",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
