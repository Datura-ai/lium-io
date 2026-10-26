from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .models import AVAILABILITY_CATEGORY, build_msg

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import Context


# Where every failing verdict sends the provider when its template names no page of its
# own: the docs table that lists each reason code with its cause and fix. The miner portal
# renders `help_uri` as the "Learn more" link on the node's error panel.
REASON_CODE_DOCS_URL = "https://docs.lium.io/providers/troubleshooting#6-validator-reason-codes"
_HELP_URI_SEVERITIES = frozenset({"error", "warning"})


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
    ctx: Context | None,
    check_id: str,
    what: dict[str, Any] | None = None,
    severity: str | None = None,
    category: str | None = None,
    impact: str | None = None,
    remediation: str | None = None,
    help_uri: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Render a `MessageTemplate` into a structured message via `build_msg`.

    `ctx` is None for a verdict the cycle reaches after every pipeline has ended."""
    resolved_severity = severity or template.severity
    resolved_help_uri = help_uri or template.help_uri
    if resolved_help_uri is None and resolved_severity in _HELP_URI_SEVERITIES:
        resolved_help_uri = REASON_CODE_DOCS_URL
    return build_msg(
        event=template.event,
        reason=template.reason,
        severity=resolved_severity,
        category=category or template.category,
        impact=impact or template.impact,
        remediation=(remediation if remediation is not None else template.remediation) or "",
        what=what or {},
        check_id=check_id,
        pipeline_id=ctx.pipeline_id if ctx is not None else None,
        ctx={**(ctx.default_extra if ctx is not None else {}), **(extra or {})},
        help_uri=resolved_help_uri,
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
    UPLOAD_SKIPPED = MessageTemplate(
        event="Validation files not uploaded",
        reason="UPLOAD_SKIPPED",
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
    # What a validator before the split reported for every scrape failure below. Kept because the
    # backend still reads it from those validators.
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
    # Host side: the scrape ran on the executor and the host sent back an exit status (or a signal)
    # with a failure.
    SCRAPE_FAILED_NO_GPU = MessageTemplate(
        event="Machine specs scrape found zero GPUs",
        reason="SCRAPE_FAILED_NO_GPU",
        severity="error",
        category="env",
        impact="Validation halted — GPU unverified",
        remediation=(
            "NVML answered on the host and listed zero GPUs. Run `nvidia-smi` on the host and make sure"
            " the executor container starts with GPU access (NVIDIA container runtime, `--gpus all`)."
        ),
    )
    SCRAPE_FAILED_DRIVER = MessageTemplate(
        event="Machine specs scrape hit a GPU driver error",
        reason="SCRAPE_FAILED_DRIVER",
        severity="error",
        category="env",
        impact="Validation halted — GPU unverified",
        remediation=(
            "NVML raised on the host (driver unloaded, a GPU fallen off the bus, or a driver/library"
            " version mismatch). Run `nvidia-smi` on the host, check `dmesg` for Xid errors, then"
            " reboot or reinstall the driver."
        ),
    )
    # DAH-4658: the renter's own reset takes the cards out of NVML; the cycle withholds the verdict
    SCRAPE_GPU_RESET_IN_PROGRESS = MessageTemplate(
        event="Machine specs scrape found no GPU: the renter is resetting the pod's GPUs",
        reason="SCRAPE_GPU_RESET_IN_PROGRESS",
        severity="info",
        category="env",
        impact="Validation halted — no verdict this cycle, the previous result stands",
    )
    SCRAPE_FAILED_ON_HOST = MessageTemplate(
        event="Machine specs scrape failed on the executor",
        reason="SCRAPE_FAILED_ON_HOST",
        severity="error",
        category="env",
        impact="Validation halted — GPU unverified",
        remediation=(
            "The scrape exited with an error on the executor, or ended without output. Check the"
            " stderr tail in this event, keep the executor image up to date, and make sure the host"
            " has free disk and memory."
        ),
    )
    # Undetermined: no exit status came back from the host, and a host fault and a validator fault
    # both end this way.
    SCRAPE_TIMEOUT = MessageTemplate(
        event="Machine specs scrape timed out",
        reason="SCRAPE_TIMEOUT",
        severity="error",
        category="env",
        impact="Validation halted — GPU unverified",
        remediation=(
            "The validator stopped waiting for the scrape at its timeout. Either side can cause this:"
            " a GPU query that hangs on the host, a host link that drops packets, or a busy validator."
            " If it repeats, check that `nvidia-smi` answers within seconds on the host."
        ),
    )
    SCRAPE_TRANSPORT_FAILED = MessageTemplate(
        event="Machine specs scrape returned no exit status",
        reason="SCRAPE_TRANSPORT_FAILED",
        severity="error",
        category="env",
        impact="Validation halted — GPU unverified",
        remediation=(
            "The SSH session ended before the host sent the scrape's exit status, or the scrape could"
            " not be delivered. Either side can cause this: the host's network, sshd or container going"
            " down, a full disk on the host during the upload, or the validator's own network. The next"
            " cycle retries; if it fails on every cycle, check that the host's SSH port answers from"
            " outside and that the host has free disk."
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


SCRAPE_HOST_SIDE_FAILURE_REASONS = frozenset(
    {
        MachineSpecMessages.SCRAPE_FAILED_NO_GPU.reason,
        MachineSpecMessages.SCRAPE_FAILED_DRIVER.reason,
        MachineSpecMessages.SCRAPE_FAILED_ON_HOST.reason,
    }
)
# No split code is validator-side: each one without an exit status can come from the host's link,
# sshd, disk or a hung GPU query as much as from the validator.
SCRAPE_UNDETERMINED_FAILURE_REASONS = frozenset(
    {
        MachineSpecMessages.SCRAPE_TIMEOUT.reason,
        MachineSpecMessages.SCRAPE_TRANSPORT_FAILED.reason,
    }
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
    MODEL_MIXED = MessageTemplate(
        event="Mixed GPU models on one node",
        reason="GPU_MODEL_MIXED",
        severity="warning",
        category="policy",
        impact="Job skipped; score set to 0",
        remediation="Every GPU on a node must be the same model.",
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
        impact="Verification reset; score set to 0 until the scrape lists every GPU again",
        remediation=(
            "The GPUs the scrape enumerated do not match the count the node reports. Check `nvidia-smi -L` and `dmesg` on the host; "
            "the node is verified again when every card is listed."
        ),
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


class DiskHealthMessages:
    NOT_WRITABLE = MessageTemplate(
        event="Executor disk refuses writes",
        reason="DISK_NOT_WRITABLE",
        severity="warning",
        category="env",
        impact="Proceed; score not changed (observation only until proven on live executors)",
        remediation="The filesystem holding the docker root is mounted read-only or refused a write "
        "probe (read_only, io_error, no_space or quota in write_probe_error). For no_space or quota: "
        "free space on the docker root (docker system prune, a larger disk or quota). Otherwise: check "
        "dmesg for I/O errors, run a filesystem check, replace the disk if it is failing, then remount "
        "read-write and restart the executor.",
    )
    UNKNOWN = MessageTemplate(
        event="Executor disk health not reported",
        reason="DISK_HEALTH_UNKNOWN",
        severity="info",
        category="env",
        impact="Proceed",
    )
    OK = MessageTemplate(
        event="Executor disk health ok",
        reason="DISK_HEALTH_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )


class GpuThermalThrottleMessages:
    THROTTLED = MessageTemplate(
        event="GPU thermal throttling under load",
        reason="GPU_THERMAL_THROTTLE",
        severity="warning",
        category="env",
        impact="Proceed; score not changed (observation only)",
        remediation="A busy GPU is held far below its clock by a thermal or hardware slowdown (see gpus[]: "
        "temperature_c, graphics_clock_mhz, max_graphics_clock_mhz, reasons). Check `nvidia-smi -q -d "
        "PERFORMANCE,TEMPERATURE`, the fans, airflow and the datacenter cooling; the renter is getting a "
        "fraction of the card's speed.",
    )
    COLLAPSED = MessageTemplate(
        event="GPU clock collapsed by thermal throttling",
        reason="GPU_THERMAL_COLLAPSE",
        severity="error",
        category="env",
        impact="Job skipped; score set to 0 until a busy GPU holds at least half its max clock",
        remediation="A busy GPU runs below half its max clock with a thermal or hardware slowdown set (see "
        "gpus[]: temperature_c, graphics_clock_mhz, max_graphics_clock_mhz, reasons). Fix the cooling (fans, "
        "airflow, datacenter temperature; `nvidia-smi -q -d PERFORMANCE,TEMPERATURE`); the node passes again "
        "on the next validation once the clock recovers.",
    )
    OK = MessageTemplate(
        event="GPU clocks not thermally throttled",
        reason="GPU_THERMAL_THROTTLE_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )


class StorageLimitMessages:
    MISSING = MessageTemplate(
        event="Docker cannot limit a container's disk",
        reason="STORAGE_LIMIT_MISSING",
        severity="warning",
        category="env",
        impact="Proceed; score not changed while the requirement is not enforced (see enforced). A renter who "
        "fills their disk fills the host's Docker storage and takes the host and every pod on it down",
        remediation="Set up Docker storage that enforces a per-container limit (overlay2 on xfs mounted with "
        "pquota): https://docs.lium.io/providers/nodes/docker-storage . Check with `sudo docker run --rm "
        "--storage-opt size=1g --gpus all daturaai/compute-subnet-executor:latest nvidia-smi`.",
    )
    REQUIRED = MessageTemplate(
        event="Node rejected: Docker cannot limit a container's disk",
        reason="STORAGE_LIMIT_REQUIRED",
        severity="error",
        category="env",
        impact="Job skipped; score set to 0 until the host enforces a per-container storage limit",
        remediation="Set up Docker storage that enforces a per-container limit (overlay2 on xfs mounted with "
        "pquota): https://docs.lium.io/providers/nodes/docker-storage . The node passes again on the next "
        "validation once `sudo docker run --rm --storage-opt size=1g --gpus all "
        "daturaai/compute-subnet-executor:latest nvidia-smi` works.",
    )
    OK = MessageTemplate(
        event="Docker limits a container's disk",
        reason="STORAGE_LIMIT_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )


class RentedGpuDropMessages:
    DROP = MessageTemplate(
        event="Rented node lost a GPU",
        reason="RENTED_GPU_DROP",
        severity="error",
        category="env",
        impact="Proceed; score not changed here. Reported to the backend unless held for a second cycle or a dry "
        "run (see pods[].posted, held and delivery); provider, support and renter are told once per incident",
        remediation="A card fell out of NVML on a rented node (see faults, missing_uuids, nvml_error_code). Check "
        "`nvidia-smi` and `dmesg` for Xid 79 / 'fallen off the bus', then reset or reboot the host; the renter's "
        "workload on the missing card is already broken.",
    )
    RECOVERED = MessageTemplate(
        event="Rented node shows every GPU again",
        reason="RENTED_GPU_RECOVERED",
        severity="info",
        category="env",
        impact="Proceed; the end of the incident is reported to the backend (see pods[].delivery; one with no "
        "answer is sent again next cycle)",
    )
    OK = MessageTemplate(
        event="Rented node shows every rented GPU",
        reason="RENTED_GPU_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    NOT_RENTED = MessageTemplate(
        event="Rented GPU drop check skipped: no running rental",
        reason="RENTED_GPU_DROP_NOT_RENTED",
        severity="info",
        category="env",
        impact="Proceed",
    )
    GPU_RESET_IN_PROGRESS = MessageTemplate(
        event="Rented GPU drop check skipped: the renter is resetting the pod's GPUs",
        reason="RENTED_GPU_DROP_GPU_RESET_IN_PROGRESS",
        severity="info",
        category="env",
        impact="Proceed",
    )


class GpuLossHoldMessages:
    LOST = MessageTemplate(
        event="A GPU dropped out on this node",
        reason="GPU_LOST",
        severity="error",
        # an availability error: the backend hides the node and refuses new rentals, a rental on it runs on
        category=AVAILABILITY_CATEGORY,
        impact="The node is hidden and takes no new rentals until its GPUs read clean for the hold window "
        "(see hold_until); an idle node scores 0 meanwhile, a rental already on it is not touched",
        remediation="A card fell out of NVML (see faults, missing_uuids, nvml_error_code). Run `nvidia-smi` and "
        "`dmesg | grep -i xid` on the host; 'Unknown Error' or Xid 79 ('fallen off the bus') means the card must "
        "be reseated, reset or replaced. The node is listed again once every card has read clean for the hold "
        "window. If you replaced the cards on purpose, remove the node in the portal and add it again: "
        "a node keeps the cards it was first verified with.",
    )
    PENDING = MessageTemplate(
        event="A GPU may have dropped out; judged again next cycle",
        reason="GPU_LOST_PENDING",
        severity="warning",
        category="env",
        impact="Proceed; the node is hidden if the next cycle shows it again",
    )
    OK = MessageTemplate(
        event="Every GPU is listed",
        reason="GPU_LOST_NONE",
        severity="info",
        category="env",
        impact="Proceed",
    )
    GPU_RESET_IN_PROGRESS = MessageTemplate(
        event="GPU loss hold skipped: the renter is resetting the pod's GPUs",
        reason="GPU_LOST_GPU_RESET_IN_PROGRESS",
        severity="info",
        category="env",
        impact="Proceed",
    )
    # DAH-4658: the reason of the zero-GPU scrape hold, so the cycle carries the last score the same way
    MISSING_GPUS_UNDER_RESET = MessageTemplate(
        event="GPUs missing from the scrape: the renter is resetting them",
        reason=MachineSpecMessages.SCRAPE_GPU_RESET_IN_PROGRESS.reason,
        severity="info",
        category="env",
        impact="Validation halted — no verdict this cycle, the previous result stands",
    )


class PostRentalGpuRecheckMessages:
    """DAH-4482 — the recheck a node owes after the rented GPU health read flagged it."""

    NO_MARKER = MessageTemplate(
        event="Post-rental GPU recheck not needed",
        reason="POST_RENTAL_GPU_RECHECK_NOT_NEEDED",
        severity="info",
        category="env",
        impact="Proceed",
    )
    FILLER_SKIPPED = MessageTemplate(
        event="Post-rental GPU recheck waits: active filler",
        reason="POST_RENTAL_GPU_RECHECK_SKIPPED_ACTIVE_FILLER",
        severity="info",
        category="policy",
        impact="Proceed; the recheck runs on a cycle without a filler, or after a few filler cycles anyway.",
    )
    INCONCLUSIVE = MessageTemplate(
        event="Post-rental GPU recheck gave no verdict",
        reason="POST_RENTAL_GPU_RECHECK_INCONCLUSIVE",
        severity="info",
        category="env",
        impact="Proceed; the recheck runs again next cycle.",
    )
    PASSED = MessageTemplate(
        event="Post-rental GPU recheck passed",
        reason="POST_RENTAL_GPU_RECHECK_PASSED",
        severity="info",
        category="env",
        impact="Proceed",
    )
    FAILED = MessageTemplate(
        event="Post-rental GPU recheck failed",
        reason="POST_RENTAL_GPU_RECHECK_FAILED",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=(
            "A GPU showed a fault during the last rental, and after it at least one card failed the "
            "all-cards matmul or the kernel-fault probe. Check `nvidia-smi -q` and the host `dmesg` for NVRM "
            "Xid lines; reset the card (`nvidia-smi -r`) or reboot the host, replace a card that still fails, "
            "and wait one cycle."
        ),
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
    DATA_INCOMPLETE_REJECTED = MessageTemplate(
        event="GPU power limit could not be read",
        reason="GPU_POWER_LIMIT_UNREADABLE",
        severity="warning",
        category="policy",
        impact="Job skipped; score set to 0",
        remediation="Ensure NVML exposes the current and default power limits for every GPU.",
    )
    LIMIT_OK = MessageTemplate(
        event="GPU power limit check passed",
        reason="GPU_POWER_LIMIT_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    SKIPPED_ACTIVE_LIUM_FILLER = MessageTemplate(
        event="GPU power limit check skipped for active Lium default job",
        reason="GPU_POWER_LIMIT_SKIPPED_LIUM_FILLER",
        severity="info",
        category="policy",
        impact="Proceed; Lium lowered the limit for its own idle filler, node stays rentable",
    )
    RESTORED_STALE_CAP = MessageTemplate(
        event="Below-floor GPU power limits are Lium's own filler caps",
        reason="GPU_POWER_LIMIT_RESTORED_STALE_CAP",
        severity="warning",
        category="policy",
        impact="Proceed without penalty; stale caps past the grace period were restored to the pre-cap limit",
    )
    RESCUE_STATE_UNAVAILABLE = MessageTemplate(
        event="GPU power limit records unreadable, cannot rule out Lium's own cap",
        reason="GPU_POWER_LIMIT_STATE_UNAVAILABLE",
        severity="warning",
        category="env",
        impact="Proceed without penalty this cycle; the check re-runs once Redis answers again",
        remediation="Check the validator's Redis health.",
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
        impact="Score set to 0 for this cycle",
        remediation="Update to a supported NVIDIA driver version. Your current driver version is not recognized.",
    )
    DRIVER_MISSING = MessageTemplate(
        event="NVIDIA driver version not reported",
        reason="NVML_DRIVER_MISSING",
        severity="error",
        category="env",
        impact="Score set to 0 for this cycle",
        remediation="Make sure `nvidia-smi` reports the driver version on the host.",
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
    # Warn mode (GPU_ANCHOR_HARD_ENABLED off): the legacy verdict. Its what_we_saw carries `anchor_hard_would_be`
    # (GPU_MISSING or ANCHOR_BROKEN) so the hard rule can be read from a week of rows before it is switched on.
    UUID_CHANGED = MessageTemplate(
        event="GPU fingerprints changed",
        reason="GPU_UUID_CHANGED",
        severity="warning",
        category="env",
        impact="Verification reset; score set to 0",
        remediation="Ensure the same physical GPUs remain attached and stable. If you replaced the cards on "
        "purpose, remove the node in the portal and add it again: a node keeps the cards it was first "
        "verified with.",
    )
    # Hard mode: the scrape lists a strict subset of the anchored set. Transient by definition: the node
    # scores again on the first cycle that shows the full set.
    GPU_MISSING = MessageTemplate(
        event="GPU missing from the listed set",
        reason="GPU_MISSING",
        severity="warning",
        category="env",
        impact="Verification reset; score set to 0 until every listed GPU is visible again",
        remediation=(
            "One or more GPUs this node listed are not visible. Check `nvidia-smi -L` and `dmesg` on the host. "
            "The node scores again when the full set is back."
        ),
    )
    # Hard mode: the scrape shows a GPU that is not in the anchored set (swap, added card, or a set
    # replaced by a different one). Permanent for this executor id; re-registering the node is the only way out.
    ANCHOR_BROKEN = MessageTemplate(
        event="GPU set differs from the listed set",
        reason="GPU_UUID_CHANGED",
        severity="error",
        category="env",
        impact="Score 0 on every cycle for this node id; verification does not restart",
        remediation=(
            "A listed node keeps one fixed set of GPUs. To list a different set, re-register the node. "
            "Uptime starts over; penalties stay."
        ),
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


class BannedProviderMessages:
    PROVIDER_BANNED = MessageTemplate(
        event="Provider banned for network abuse",
        reason="PROVIDER_BANNED",
        severity="warning",
        category="policy",
        impact="Score set to 0; verification and active rentals retained",
        remediation="Contact Lium support to appeal the provider ban.",
    )
    PROVIDER_ALLOWED = MessageTemplate(
        event="Provider allowed",
        reason="PROVIDER_BANNED_CHECK_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    # DAH-2662: a mount that is not procfs sits on /proc/driver/nvidia/gpus, so the kernel's GPU
    # list cannot be read (the 2026-08-19 kit). Emitted only when KERNEL_GPU_BAN_ENFORCEMENT_ENABLED.
    KERNEL_GPU_VIEW_OVERLAID = MessageTemplate(
        event="A foreign mount covers the kernel's GPU list",
        reason="KERNEL_GPU_VIEW_OVERLAID",
        severity="warning",
        category="policy",
        impact="Score set to 0 while the mount is in place",
        remediation="Remove the mount over /proc/driver/nvidia/gpus on the host.",
    )


class HardwareIdentityMessages:
    # DAH-4547: emitted as a warning in shadow; fails the node under HARDWARE_FINGERPRINT_ENFORCEMENT_ENABLED.
    SPOOFED = MessageTemplate(
        event="Host hardware identity does not match the hardware",
        reason="HARDWARE_IDENTITY_SPOOFED",
        severity="warning",
        category="policy",
        impact="Score set to 0 while the finding stands",
        remediation="Remove any mount or script over /sys/class/dmi/id on the host and in the executor "
        "container, and make sure NVML reports the same GPUs as /proc/driver/nvidia/gpus.",
    )
    OK = MessageTemplate(
        event="Host hardware identity ok",
        reason="HARDWARE_IDENTITY_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )


class ExecutorImageMessages:
    CURRENT = MessageTemplate(
        event="Executor image is current",
        reason="EXECUTOR_IMAGE_CURRENT",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    SKIPPED = MessageTemplate(
        event="Executor image check skipped",
        reason="EXECUTOR_IMAGE_SKIPPED",
        severity="info",
        category="policy",
        impact="No image penalty this cycle",
    )
    OUTDATED = MessageTemplate(
        event="Executor image is outdated",
        reason="EXECUTOR_IMAGE_OUTDATED",
        severity="error",
        category="policy",
        impact="Score set to 0 until the image is current",
    )
    # Same reason code as OUTDATED so log queries keep matching; severity says it is not scored.
    OUTDATED_WARNING = MessageTemplate(
        event="Executor image is outdated",
        reason="EXECUTOR_IMAGE_OUTDATED",
        severity="warning",
        category="policy",
        impact="Warning only until EXECUTOR_IMAGE_CHECK_ENFORCE is on: score and incentive unchanged",
    )


class SysboxRequiredMessages:
    SYSBOX_MISSING = MessageTemplate(
        event="Sysbox required for unrented executor",
        reason="SYSBOX_REQUIRED_MISSING",
        severity="warning",
        category="policy",
        impact="Score set to 0 for this cycle; verification is kept until repeated failures deactivate the executor",
        remediation="Install the sysbox runtime; unrented machines without sysbox are not allowed on the network.",
    )
    # DAH-3634: the probe's `docker run` was refused by the NVIDIA container hook, so no sysbox
    # verdict was measured and the node cannot start any GPU container. Same impact as
    # SYSBOX_MISSING (the check still fails, score 0); the reason code and the advice are truthful.
    # The call site prefixes the remediation with the probe's cause (dind_probe.DOCKER_RUN_CAUSES).
    # Same code as the executor updater's hold for this host condition (DAH-3481).
    NVIDIA_RUNTIME_MISMATCH = MessageTemplate(
        event="NVIDIA driver/library version mismatch on the host",
        reason="NVIDIA_RUNTIME_MISMATCH",
        severity="warning",
        category="runtime",
        impact="Score set to 0 for this cycle; verification is kept until repeated failures deactivate the executor",
        remediation=(
            "Reboot the host after the NVIDIA driver update, or reinstall the NVIDIA container toolkit. "
            "Sysbox was not measured; fix this first and the sysbox check runs again on the next cycle."
        ),
    )
    NVIDIA_CONTAINER_HOOK_FAILED = MessageTemplate(
        event="NVIDIA container hook refused the GPU container",
        reason="NVIDIA_CONTAINER_HOOK_FAILED",
        severity="warning",
        category="runtime",
        impact="Score set to 0 for this cycle; verification is kept until repeated failures deactivate the executor",
        remediation=(
            "Make `nvidia-smi` work on the host (reset or reboot), then reinstall the NVIDIA container "
            "toolkit; the sysbox check runs again on the next cycle."
        ),
    )
    SYSBOX_OK = MessageTemplate(
        event="Sysbox requirement satisfied",
        reason="SYSBOX_REQUIRED_OK",
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
    DUPLICATE_OBSERVED = MessageTemplate(
        event="Duplicate executor registration (observe mode)",
        reason="EXECUTOR_DUPLICATE_OBSERVED",
        severity="warning",
        category="policy",
        impact="Detected; no score impact (observe mode)",
        remediation="Ensure every executor has a unique UUID and deregister duplicates before retrying.",
    )
    UNIQUE = MessageTemplate(
        event="Executor registration unique",
        reason="EXECUTOR_NOT_DUPLICATE",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    ACROSS_MINERS = MessageTemplate(
        event="Executor also scored under another miner this cycle",
        reason="EXECUTOR_DUPLICATE_ACROSS_MINERS",
        severity="warning",
        category="policy",
        impact="Score set to 0 for this cycle; the other miner keeps it",
        remediation="One machine earns under one miner per cycle. Register it under one miner only.",
    )
    GPU_ALIAS = MessageTemplate(
        event="GPU also reported by another executor of this miner this cycle",
        reason="EXECUTOR_GPU_ALIAS",
        severity="warning",
        category="policy",
        impact="Score set to 0 for this cycle; the other executor keeps it",
        remediation="Each physical GPU earns under one executor. Register every machine once.",
    )

class CollateralStatusMessages:
    DEPOSITED = MessageTemplate(
        event="Collateral verified",
        reason="COLLATERAL_OK",
        severity="info",
        category="policy",
        impact="None; collateral has no score effect",
    )
    NOT_DEPOSITED = MessageTemplate(
        event="No collateral deposited",
        reason="COLLATERAL_MISSING",
        severity="info",
        category="policy",
        impact="None; collateral has no score effect",
    )
    READ_FAILED = MessageTemplate(
        event="Collateral read failed",
        reason="COLLATERAL_READ_FAILED",
        severity="info",
        category="policy",
        impact="None; the last known collateral status is reported, or none deposited when there is none",
    )


class StaleContainerCleanupMessages:
    CLEANED = MessageTemplate(
        event="Stale container cleanup complete",
        reason="STALE_CLEANUP_DONE",
        severity="info",
        category="prep",
        impact="Proceed",
    )
    RENTED_LIST_UNKNOWN = MessageTemplate(
        event="Stale container removal skipped: rented list empty or unknown",
        reason="STALE_CLEANUP_RENTED_LIST_UNKNOWN",
        severity="warning",
        category="prep",
        impact="No rental container removed this cycle; live pods and fillers are left alone",
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
        impact="Provider-origin findings recorded; score unchanged",
        remediation="Review the verdict's evidence; findings are recorded only, no score or marketplace effect.",
    )
    PLATFORM_ORIGIN_ONLY = MessageTemplate(
        event="Inspector findings were the platform's own execs",
        reason="INSPECTOR_PLATFORM_ORIGIN_ONLY",
        severity="info",
        category="runtime",
        impact="Only validator/executor execs seen (sshd ancestry lost on this host); nothing provider-side",
    )
    VALIDATION_ERROR = MessageTemplate(
        event="Inspector validation error",
        reason="INSPECTOR_VALIDATION_ERROR",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable; score unchanged",
        remediation="Check libinspector installation and executor SSH process logs.",
    )
    UNREADABLE = MessageTemplate(
        event="Inspector executor response unreadable",
        reason="INSPECTOR_UNREADABLE",
        severity="warning",
        category="runtime",
        impact="Inspector result unavailable on this executor; score unchanged; the node is counted as unreadable, not as clean",
        remediation=(
            "The inspector_executor.py --interactive stdout line for this command was not one JSON "
            "object (payload_head / json_error_pos say where it broke; payload_terminated=false "
            "means the line was cut at EOF or at the size cap). Check the executor container logs "
            "for what else wrote to its stdout, and the response size against "
            "INSPECTOR_RESPONSE_MAX_BYTES."
        ),
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
    REBOOT_FAILED_HOST_FAULT = MessageTemplate(
        event="Host cannot start a GPU container after a failed pod reboot",
        reason="REBOOT_FAILED_HOST_FAULT",
        severity="error",
        category="runtime",
        impact="Score set to 0; verification cleared; executor inactive until the host passes again",
        remediation=(
            "Run `docker run --rm --gpus all <image> nvidia-smi -L` on the host and fix the Docker daemon or"
            " the NVIDIA container runtime; check `dmesg` for Xid errors and reboot if a GPU is lost."
        ),
    )
    POD_RESTARTING = MessageTemplate(
        event="Pod restarting under its own restart policy",
        reason="POD_RESTARTING",
        severity="warning",
        category="runtime",
        impact="No verdict for this cycle - verification kept",
        remediation=(
            "The pod is restarting under its restart policy; the host cannot tell whether the renter's process or "
            "the host caused the exits."
        ),
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
        event="Provider is using the GPU",
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
    GPU_RESET_IN_PROGRESS = MessageTemplate(
        event="GPU usage check skipped: the renter is resetting the pod's GPUs",
        reason="GPU_USAGE_GPU_RESET_IN_PROGRESS",
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
    USAGE_HIGH_BESIDE_A_RENTAL_REMEDIATION = (
        "Stop the GPU processes outside Lium's containers and re-run your node: {outside_processes}. "
        "Do not stop the pod container of the rental on this node ({pod_containers}): Lium stops or "
        "starts it itself."
    )
    ORPHANED_CONTAINER = MessageTemplate(
        event="Orphaned rental container detected",
        reason="ORPHANED_RENTAL_CONTAINER",
        severity="error",
        category="runtime",
        impact="Validation skipped; score set to 0",
        remediation=(
            "{orphaned_container} is named like a Lium pod container{pod}, and no live rental on this "
            "node uses it: {rental_status}. Lium stops a pod's container when its rental ends and the "
            "validator removes leftovers, so this one should not be holding the GPU. Before you touch "
            "it, confirm it is not a live rental: the node's page in the Provider Portal must show no "
            "active rental. Only then remove it: docker rm -f {orphaned_container}. If the portal "
            "still shows a rental on the node, leave the container running and contact Lium support."
        ),
    )
    ORPHANED_CONTAINER_OF_A_LIVE_RENTAL_REMEDIATION = (
        "{orphaned_container} is named like a Lium pod container{pod}, and no live rental on this "
        "node uses it: {rental_status}. Do not stop or remove it: contact Lium support with the "
        "container name."
    )
    FOREIGN_PROCESS = MessageTemplate(
        event="Foreign GPU process on idle executor",
        reason="GPU_FOREIGN_PROCESS",
        severity="error",
        category="runtime",
        impact="Validation skipped; score set to 0",
        remediation=(
            "A GPU process outside Lium pod/filler containers is holding the card "
            "(foreign container or bare host process). Stop all non-Lium GPU workloads "
            "and re-run your node."
        ),
    )
    VRAM_HELD = MessageTemplate(
        event="VRAM held on idle executor with no visible owner",
        reason="GPU_VRAM_HELD",
        severity="error",
        category="runtime",
        impact="Validation skipped; score set to 0",
        remediation=(
            "GPU memory is occupied but no owning process is visible to the validator. "
            "Free the VRAM (stop hidden or host-side GPU workloads) and re-run your node."
        ),
    )
    GPU_WEDGE_CURED = MessageTemplate(
        event="Ghost GPU cured in place",
        reason="GPU_WEDGE_CURED",
        severity="warning",
        category="runtime",
        impact="Latched busy counter cleared by a CUDA context cycle; validation continues",
    )
    GPU_WEDGED = MessageTemplate(
        event="Wedged GPU detected",
        reason="GPU_WEDGED",
        severity="error",
        category="runtime",
        impact="Validation skipped; score set to 0; the CUDA context cure did not clear the card",
        remediation=(
            "GPU pinned at full utilization with no memory and no process — a latched Blackwell "
            "busy counter left by a hard-killed GPU process (ghost GPU, DAH-2427). If the "
            "automatic cure did not clear it, run a CUDA context open/close on the card "
            "(CUDA_VISIBLE_DEVICES=<uuid> python3 + cuInit/cuCtxCreate/cuCtxDestroy); "
            "nvidia-smi -r is unreliable across driver builds."
        ),
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


NO_OUTBOUND_INTERNET_REMEDIATION = (
    "containers on this host cannot reach the internet; check the docker bridge / FORWARD chain and DNS"
)


class OutboundInternetMessages:
    # the rental probe's egress step, under NO_OUTBOUND_INTERNET_ENFORCEMENT_ENABLED
    NO_OUTBOUND_INTERNET = MessageTemplate(
        event="Containers cannot reach the internet",
        reason="NO_OUTBOUND_INTERNET",
        severity="warning",
        category="runtime",
        impact="Score set to 0",
        remediation=NO_OUTBOUND_INTERNET_REMEDIATION,
    )


REGISTRY_PULL_REMEDIATION = (
    "this node's Docker daemon could not pull a small Docker Hub image, so a renter's template that is not "
    "cached fails to start; check the registry mirrors in /etc/docker/daemon.json (`docker info`, Registry "
    "Mirrors) resolve and answer from the host, or remove the broken mirror and restart docker, then "
    "`docker pull hello-world` on the host must finish"
)


class RegistryPullMessages:
    REGISTRY_PULL_FAILED = MessageTemplate(
        event="Docker Hub image pull fails through this node's registry path",
        reason="REGISTRY_PULL_FAILED",
        severity="warning",
        category="runtime",
        impact="Score set to 0",
        remediation=REGISTRY_PULL_REMEDIATION,
    )
    # one failed pull: a finding only once the next pull fails too
    REGISTRY_PULL_FAILED_ONCE = MessageTemplate(
        event="Docker Hub image pull failed once; the next pull decides",
        reason="REGISTRY_PULL_FAILED_ONCE",
        severity="info",
        category="runtime",
        impact="Proceed",
        remediation=REGISTRY_PULL_REMEDIATION,
    )
    REGISTRY_PULL_OK = MessageTemplate(
        event="Docker Hub image pull verified",
        reason="REGISTRY_PULL_OK",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    # the pull failed, but the validator's own GET of Docker Hub failed too: an outage, not this node
    REGISTRY_PULL_NO_VERDICT_HUB_DOWN = MessageTemplate(
        event="Docker Hub image pull failed while Docker Hub was unreachable from the validator; no verdict",
        reason="REGISTRY_PULL_NO_VERDICT_HUB_DOWN",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    # the pull ran but says nothing about the path: Docker Hub's 429, an auth or unclassified error, or
    # the probe itself did not run
    REGISTRY_PULL_UNMEASURED = MessageTemplate(
        event="Docker Hub image pull reached no verdict",
        reason="REGISTRY_PULL_UNMEASURED",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    SKIPPED = MessageTemplate(
        event="Registry pull check skipped",
        reason="REGISTRY_PULL_SKIPPED",
        severity="info",
        category="runtime",
        impact="Proceed",
    )


CONTAINER_DOWNLOAD_SLOW_REMEDIATION = (
    "a container on this node downloads from PyPI or Hugging Face far slower than the host does, so a renter's "
    "`pip install` or model download crawls; check the docker bridge MTU against the uplink (`ip link`; set "
    '"mtu" in /etc/docker/daemon.json to the uplink\'s and restart docker), NAT/firewall rules and any traffic '
    "shaping on docker traffic, then `docker run --rm alpine:3.19 wget -O /dev/null "
    "https://files.pythonhosted.org/packages/9b/b4/e3c7e6fab0f77fff6194afa173d1f2342073d91b1d3b4b30b17c3fb4407a/"
    "numpy-2.1.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` on the host must finish in seconds"
)


class ContainerDownloadSpeedMessages:
    SLOW = MessageTemplate(
        event="Downloads from inside a container on this node are too slow",
        reason="CONTAINER_DOWNLOAD_SLOW",
        severity="warning",
        category="runtime",
        impact="Score set to 0",
        remediation=CONTAINER_DOWNLOAD_SLOW_REMEDIATION,
    )
    # the same finding when a failing reading in the streak was not on the rentals network: logged, score unchanged
    SLOW_OBSERVED = MessageTemplate(
        event="Downloads from inside a container on this node are too slow (not enforced)",
        reason="CONTAINER_DOWNLOAD_SLOW_OBSERVED",
        severity="warning",
        category="runtime",
        impact="None: a failing reading was not taken on the rentals network",
        remediation=CONTAINER_DOWNLOAD_SLOW_REMEDIATION,
    )
    SLOW_ONCE = MessageTemplate(
        event="A container download was slow once; the next reading decides",
        reason="CONTAINER_DOWNLOAD_SLOW_ONCE",
        severity="info",
        category="runtime",
        impact="Proceed",
        remediation=CONTAINER_DOWNLOAD_SLOW_REMEDIATION,
    )
    OK = MessageTemplate(
        event="Container download speed verified",
        reason="CONTAINER_DOWNLOAD_SPEED_OK",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    # an endpoint the validator cannot reach either, an HTTP error, or a container that did not run
    UNMEASURED = MessageTemplate(
        event="Container download speed reached no verdict",
        reason="CONTAINER_DOWNLOAD_SPEED_UNMEASURED",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    SKIPPED = MessageTemplate(
        event="Container download speed check skipped",
        reason="CONTAINER_DOWNLOAD_SPEED_SKIPPED",
        severity="info",
        category="runtime",
        impact="Proceed",
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
    RENTED_SKIPPED = MessageTemplate(
        event="GPU capability verification not scored: a Lium workload took the cards during this cycle",
        reason="GPU_VERIFY_SKIPPED_RENTED",
        severity="info",
        category="policy",
        impact=(
            "Score unchanged; the probe could not get the GPU because a Lium pod or default job "
            "started after this cycle's rental snapshot. The next cycle verifies again."
        ),
        remediation="No action needed.",
    )
    VERIFY_FAILED = MessageTemplate(
        event="GPU capability verification failed",
        reason="GPU_VERIFY_FAILED",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation="Run Docker GPU diagnostics (nvidia-smi) and ensure containers can access GPUs.",
    )
    # One reason code per CUDA failure the probe can hit (DAH-3362, ticket-0318), as
    # GPU_VERIFY_TIMEOUT and the VERIFYX_FAILED_* family already do; the full steps live in the
    # reason-code table the default help_uri points at. Wording is provider-facing; no backticks,
    # the portal renders it as plain text.
    VERIFY_FAILED_CUDA_NOT_READY = MessageTemplate(
        event="GPU capability verification failed: CUDA cannot start",
        reason="GPU_VERIFY_CUDA_NOT_READY",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=(
            "CUDA cannot start on the GPU although nvidia-smi works (CUDA error 802, 'system not "
            "yet initialized'); on HGX H100/H200 boards the NVLink fabric is not ready. Start "
            "Fabric Manager, have the hosting provider activate the fabric partition, or on a "
            "board without NVSwitch reinstall the driver; the steps per case are on the linked "
            "page."
        ),
    )
    VERIFY_FAILED_NO_CUDA_DEVICE = MessageTemplate(
        event="GPU capability verification failed: no CUDA device in the container",
        reason="GPU_VERIFY_NO_CUDA_DEVICE",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=(
            "The probe found no usable CUDA device inside the executor container: the NVIDIA "
            "Container Toolkit is not exposing the GPUs to it, or the host driver is older than "
            "the probe's CUDA runtime. docker exec executor-executor-1 nvidia-smi must list every "
            "GPU — the toolkit, nvidia_uvm and driver steps are on the linked page."
        ),
    )
    VERIFY_FAILED_VRAM_UNAVAILABLE = MessageTemplate(
        event="GPU capability verification could not allocate VRAM",
        reason="GPU_VERIFY_VRAM_UNAVAILABLE",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=(
            "The matrix-multiplication probe could not allocate GPU memory: another process held "
            "VRAM while it ran. Check nvidia-smi for processes or leftover containers on the GPU; "
            "the next verification cycle retries."
        ),
    )
    VERIFY_TIMEOUT = MessageTemplate(
        event="GPU capability verification timed out",
        reason="GPU_VERIFY_TIMEOUT",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=(
            "The matrix-multiplication probe did not finish in time. Check nvidia-smi for "
            "stuck processes or leftover containers holding VRAM on the executor."
        ),
    )


class GpuMemoryHealthMessages:
    FAILED = MessageTemplate(
        event="GPU memory reported broken by NVML",
        reason="GPU_MEMORY_HEALTH_FAILED",
        severity="error",
        # an availability error: the backend hides the node and refuses new rentals, a rental on it runs on
        category=AVAILABILITY_CATEGORY,
        impact="Node hidden and not rentable until the card is reset or replaced (a running rental "
        "is kept)",
        remediation="NVML reports uncorrected ECC errors, a pending or failed row remap, or a "
        "required recovery action on the card(s) in flags; a rented card in this state faults the "
        "renter's jobs. Check `nvidia-smi -q -d ECC,ROW_REMAPPER`: a pending remap or a "
        "DRAIN_AND_RESET / GPU_RESET action needs the card drained and reset (`nvidia-smi -r` with "
        "no process on it, or a host reboot); a failed remap, or uncorrected errors that return "
        "after a reset, mean the card must be replaced. The node passes again on the first "
        "validation after NVML reads the card clean.",
    )
    FLAGGED = MessageTemplate(
        event="GPU memory reported broken by NVML",
        reason="GPU_MEMORY_HEALTH_FLAGGED",
        severity="warning",
        category="gpu",
        impact="Proceed; score not changed (GPU_MEMORY_HEALTH_ENFORCEMENT_ENABLED is off)",
        remediation="NVML reports uncorrected ECC errors, a pending or failed row remap, or a "
        "required recovery action on the card(s) in flags. Check `nvidia-smi -q -d "
        "ECC,ROW_REMAPPER`: a pending remap or a DRAIN_AND_RESET / GPU_RESET action needs the card "
        "drained and reset (`nvidia-smi -r` with no process on it, or a host reboot); a failed "
        "remap, or uncorrected errors that return after a reset, mean the card should be replaced.",
    )
    UNKNOWN = MessageTemplate(
        event="GPU memory health not read",
        reason="GPU_MEMORY_HEALTH_UNKNOWN",
        severity="info",
        category="gpu",
        impact="Proceed",
    )
    RESET_ADVISED = MessageTemplate(
        event="GPU memory needs a reset",
        reason="GPU_MEMORY_HEALTH_RESET_ADVISED",
        severity="warning",
        category="gpu",
        impact="Proceed; score not changed",
        remediation="NVML reports a pending row remap or a recovery action from correctable errors "
        "on the card(s) in warnings. The card still works, but the remap takes effect only after "
        "a reset: drain the card and reset it (`nvidia-smi -r` with no process on it, or a host "
        "reboot) when it is free.",
    )
    OK = MessageTemplate(
        event="GPU memory health ok",
        reason="GPU_MEMORY_HEALTH_OK",
        severity="info",
        category="gpu",
        impact="Proceed",
    )


class GpuFaultProbeMessages:
    """DAH-3035 — the kernel-fault probe that follows the matmul. Shadow (without enforcement) emits PROBE_FAILED as a warning and keeps passed=True; only
    GPU_FAULT_PROBE_ENFORCEMENT_ENABLED renders it error/score-zeroing."""

    FILLER_SKIPPED = MessageTemplate(
        event="GPU kernel-fault probe skipped for active filler",
        reason="GPU_FAULT_PROBE_SKIPPED_ACTIVE_FILLER",
        severity="info",
        category="policy",
        impact="Active filler runtime preserved; the probe would compete for VRAM.",
    )
    FIRST_PASS_SKIPPED = MessageTemplate(
        event="GPU kernel-fault probe skipped on the first pass",
        reason="GPU_FAULT_PROBE_SKIPPED_FIRST_PASS",
        severity="info",
        category="policy",
        impact="Proceed; in shadow the verdict changes nothing, so the probe waits for the next cycle.",
    )
    PROBE_OK = MessageTemplate(
        event="GPU kernel-fault probe passed",
        reason="GPU_FAULT_PROBE_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    PROBE_FAILED = MessageTemplate(
        event="GPU kernel-fault probe failed",
        reason="GPU_FAULT_PROBE_FAILED",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=(
            "A GPU returned a CUDA error, wrong data or hung under indexed memory access, or NVML reported "
            "new uncorrected ECC errors / remapped rows. Check `nvidia-smi -q -d ECC,ROW_REMAPPER` and the host "
            "`dmesg` for NVRM Xid lines; reseat or replace the card, or reset it (`nvidia-smi -r`) and wait one cycle."
        ),
    )
    UNKNOWN = MessageTemplate(
        event="GPU kernel-fault probe could not run",
        reason="GPU_FAULT_PROBE_UNKNOWN",
        severity="info",
        category="env",
        impact="Proceed without penalty",
    )


class PinnedMemoryProbeMessages:
    """F-1599 — the pinned-host-memory H2D/D2H round trip on every GPU. On by default: PROBE_FAILED fails
    the fatal check and keeps the node unrentable."""

    DISABLED = MessageTemplate(
        event="Pinned-memory copy probe disabled",
        reason="PINNED_MEMORY_PROBE_DISABLED",
        severity="info",
        category="env",
        impact="Proceed",
    )
    PROBE_OK = MessageTemplate(
        event="Pinned-memory copy probe passed",
        reason="PINNED_MEMORY_PROBE_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    PROBE_FAILED = MessageTemplate(
        event="Pinned-memory copy probe failed",
        reason="PINNED_MEMORY_PROBE_FAILED",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=(
            "A GPU could not copy a page-locked (pinned) host buffer to the device and back: wrong data, a CUDA "
            "error or a copy that never finished. PyTorch pin_memory() / non_blocking copies fail the same way. "
            "On a TDX guest (Intel TDX confidential VM) the GPUs must be in confidential-computing mode "
            "(`nvidia-smi conf-compute -f` shows CC status: ON) or the VM must run without TDX. A multi-GPU "
            "H100/H200 node runs in Protected PCIe mode instead, where CC status stays OFF: check "
            "`nvidia-smi conf-compute --get-multigpu-mode`. Also check the IOMMU / swiotlb settings and the "
            "host `dmesg` for NVRM Xid lines."
        ),
    )
    UNKNOWN = MessageTemplate(
        event="Pinned-memory copy probe could not run",
        reason="PINNED_MEMORY_PROBE_UNKNOWN",
        severity="info",
        category="env",
        impact="Proceed without penalty",
    )
    FILLER_SKIPPED = MessageTemplate(
        event="Pinned-memory copy probe skipped for active filler",
        reason="PINNED_MEMORY_PROBE_SKIPPED_ACTIVE_FILLER",
        severity="info",
        category="policy",
        impact="Active filler runtime preserved; the probe would compete with it for the GPUs.",
    )
    WORKLOAD_SKIPPED = MessageTemplate(
        event="Pinned-memory copy probe not scored: a Lium workload took the cards during this cycle",
        reason="PINNED_MEMORY_PROBE_SKIPPED_RENTED",
        severity="info",
        category="policy",
        impact=(
            "Score unchanged; a copy hung beside a Lium pod or default job that started after this "
            "cycle's rental snapshot. The next cycle probes again."
        ),
    )


class NvlinkFaultMessages:
    """A sticky NVLink fault: NVLink multicast (NVLS) cannot bind on an all-NVLink board. PROBE_FAILED is the
    enforced form; with NVLINK_FAULT_ENFORCEMENT_ENABLED off it renders as a warning that changes no score."""

    PROBE_OK = MessageTemplate(
        event="NVLink multicast probe passed",
        reason="NVLINK_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    SKIPPED = MessageTemplate(
        event="NVLink multicast probe not run: not an all-NVLink board",
        reason="NVLINK_SKIPPED",
        severity="info",
        category="env",
        impact="Proceed",
    )
    FILLER_SKIPPED = MessageTemplate(
        event="NVLink multicast probe skipped for active filler",
        reason="NVLINK_SKIPPED_ACTIVE_FILLER",
        severity="info",
        category="policy",
        impact="Active filler runtime preserved; the probe would compete with it on every GPU.",
    )
    PROBE_FAILED = MessageTemplate(
        event="NVLink is faulted: multicast memory cannot be bound across the GPUs",
        reason="NVLINK_FAULT",
        severity="error",
        category="env",
        impact="Score set to 0 until NVLink works again",
        remediation=(
            "The GPUs report an uncorrectable NVLink error state: NCCL NVLink SHARP (NVLS) and other "
            "multi-GPU jobs fail. Check the host `dmesg` for NVRM Xid lines and `nvidia-smi nvlink -e`. "
            "Stop the workloads, reset the GPUs (`nvidia-smi -r`) and restart nvidia-fabricmanager, or "
            "reboot. If the errors come back, service the board."
        ),
    )
    UNKNOWN = MessageTemplate(
        event="NVLink multicast probe could not run",
        reason="NVLINK_UNKNOWN",
        severity="info",
        category="env",
        impact="Proceed without penalty",
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
    GPU_RUNTIME_DEVICE_FAULT = MessageTemplate(
        event="GPU runtime health check failed",
        reason="GPU_RUNTIME_DEVICE_FAULT",
        severity="error",
        category="runtime",
        impact="New rentals are disabled because the GPU is not addressable on this host.",
        remediation=(
            "The GPU has entered a fatal state and needs a hard reset. Reboot the host with "
            "`sudo reboot`, then check `nvidia-smi` and `dmesg` for Xid errors."
        ),
    )
    CPU_QUOTA_EXCEEDS_HOST = MessageTemplate(
        event="Advertised CPU cores exceed the host's physical cores",
        reason="CPU_QUOTA_EXCEEDS_HOST",
        severity="error",
        category="runtime",
        impact="Score set to 0: the rental-verification container demanded the advertised cores and the host's Docker daemon rejected the count.",
        remediation=(
            "Advertise the real CPU(s) count. The host's Docker daemon refused `--cpus=<advertised>` "
            "because the machine has fewer physical cores than reported."
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
    FILLER_VERIFIED = MessageTemplate(
        event="Lium default job runtime verified",
        reason="FILLER_RUNTIME_VERIFIED",
        severity="info",
        category="policy",
        impact="Unrented incentive continues: the Lium default job container is running",
    )
    FILLER_KILLED = MessageTemplate(
        event="Lium default job container was removed on the host",
        reason="FILLER_CONTAINER_MISSING",
        severity="error",
        category="policy",
        impact="Unrented incentive withheld: the Lium default job container disappeared while its run is active",
        remediation=(
            "Do not stop or remove Lium default-job (filler_*) containers. "
            "Incentive resumes once a default job runs undisturbed. "
            "Contact Lium support if this container was not removed by you."
        ),
    )
    FILLER_KILLED_AT_CREATE = MessageTemplate(
        event="Lium default job container is removed on the host right after it is created",
        reason="FILLER_KILLED_AT_CREATE",
        severity="error",
        category="policy",
        impact=(
            "Unrented incentive withheld: Lium default jobs assigned to this machine were "
            "destroyed seconds after they started, repeatedly over the last 24 hours"
        ),
        remediation=(
            "Stop the host-side process that removes containers it does not recognise, or "
            "allow-list Lium default-job (filler_*) containers in it. "
            "Incentive resumes once a default job runs undisturbed. "
            "Contact Lium support if nothing on this host removes containers."
        ),
    )
    FILLER_STOPPED_BY_HOST = MessageTemplate(
        event="Lium default job container was stopped on the host",
        reason="FILLER_STOPPED_BY_HOST",
        severity="error",
        category="policy",
        impact=(
            "Unrented incentive withheld: something on this machine stopped the Lium default job "
            "while its run is active"
        ),
        remediation=(
            "Do not stop or remove Lium default-job (filler_*) containers. "
            "Incentive resumes once a default job runs undisturbed. "
            "Contact Lium support if this container was not stopped by you."
        ),
    )
    FILLER_CONTAINER_EXITED = MessageTemplate(
        event="Lium default job container exited on its own",
        reason="FILLER_CONTAINER_EXITED",
        severity="warning",
        category="runtime",
        impact="No penalty: the container is still on the host, so nobody removed it",
        remediation=(
            "The Lium default job stopped by itself on this machine. Check the container's exit "
            "code and logs; incentive is unaffected."
        ),
    )
    FILLER_STATE_UNKNOWN = MessageTemplate(
        event="Lium default job state could not be verified",
        reason="FILLER_STATE_UNKNOWN",
        severity="warning",
        category="policy",
        impact="Proceed without penalty",
    )
    FILLER_TRANSPORT_UNREACHABLE = MessageTemplate(
        event="SSH transport lost while verifying Lium default job",
        reason="FILLER_TRANSPORT_UNREACHABLE",
        severity="warning",
        category="runtime",
        impact="Filler state unknown for this cycle",
        remediation="Ensure the executor is reachable over SSH and stays online",
    )


class RentalProbeMessages:
    """DAH-3436: the synthetic rental probe. A failed step zeroes the score and clears the verified job
    (RENTAL_PROBE_FAILED) until a probe passes; an idle node with the image pulled is probed again next cycle."""

    SKIPPED = MessageTemplate(
        event="Rental probe skipped",
        reason="RENTAL_PROBE_SKIPPED",
        severity="info",
        category="policy",
        impact="Proceed; the probe runs on an idle node once per interval",
    )
    PROBE_OK = MessageTemplate(
        event="Rental probe passed: a renter container started, sshd listened and the GPUs were visible",
        reason="RENTAL_PROBE_OK",
        severity="info",
        category="runtime",
        impact="Proceed",
    )
    PROBE_FAILED = MessageTemplate(
        event="Rental probe failed: a renter could not use this node",
        reason="RENTAL_PROBE_FAILED",
        severity="error",
        category="runtime",
        impact="Score set to 0 and the verified job cleared until a probe passes; the node is probed again next cycle while it stays idle",
    )
    INCONCLUSIVE = MessageTemplate(
        event="Rental probe could not reach a verdict",
        reason="RENTAL_PROBE_INCONCLUSIVE",
        severity="warning",
        category="runtime",
        impact="Proceed without penalty; the probe runs again next cycle",
    )


class CpuTruthMessages:
    """DAH-2671 item 2a — corroborate advertised CPU(s) against sources the lscpu wrapper does not
    author. CPU_MISMATCH is an error that zeroes the score."""

    CPU_OK = MessageTemplate(
        event="CPU core count corroborated",
        reason="CPU_TRUTH_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    CPU_MISMATCH = MessageTemplate(
        event="Advertised CPU cores exceed the kernel-present CPU population",
        reason="CPU_TRUTH_MISMATCH",
        severity="error",
        category="env",
        impact="Score set to 0",
        remediation=(
            "Advertise the real CPU(s) count. The kernel-present CPU population "
            "(/sys/devices/system/cpu/present) is far below the reported core count."
        ),
    )
    CPU_UNKNOWN = MessageTemplate(
        event="CPU core count could not be corroborated",
        reason="CPU_TRUTH_UNKNOWN",
        severity="info",
        category="env",
        impact="Proceed without penalty",
    )


class GpuKernelCountMessages:
    """Reported GPU list against the kernel's list (/proc/driver/nvidia/gpus). COUNT_MISMATCH
    fails the validation; UUID_MISMATCH is logged only."""

    COUNT_OK = MessageTemplate(
        event="Reported GPUs match the kernel's GPU list",
        reason="GPU_KERNEL_COUNT_OK",
        severity="info",
        category="env",
        impact="Proceed",
    )
    COUNT_MISMATCH = MessageTemplate(
        event="Reported GPU count exceeds the kernel's GPU list",
        reason="GPU_KERNEL_COUNT_MISMATCH",
        severity="error",
        category="env",
        impact="Validation failed; score set to 0",
        remediation=(
            "Report only the GPUs this host has. The NVIDIA kernel driver "
            "(/proc/driver/nvidia/gpus) lists fewer cards than the node reports."
        ),
    )
    UUID_MISMATCH = MessageTemplate(
        event="Reported GPU UUIDs are missing from the kernel's GPU list",
        reason="GPU_KERNEL_UUID_MISMATCH",
        severity="warning",
        category="env",
        impact="Shadow observation only: the validation result was NOT changed",
    )
    NOT_EVALUATED = MessageTemplate(
        event="Kernel GPU list not available; reported GPU count not evaluated",
        reason="GPU_KERNEL_COUNT_NOT_EVALUATED",
        severity="info",
        category="env",
        impact="Proceed without penalty",
    )
    SKIPPED = MessageTemplate(
        event="Kernel GPU count check disabled",
        reason="GPU_KERNEL_COUNT_DISABLED",
        severity="info",
        category="env",
        impact="Proceed",
    )


class ProviderSideLoadMessages:
    """DAH-2734 — the CPU/disk twin of the foreign-GPU gate: a provider who runs another
    subnet's miner on a listed machine sells capacity he already uses himself. Shadow
    (without enforcement) renders LOAD_ABOVE_LIMIT as a
    warning and keeps passed=True; only PROVIDER_SIDE_LOAD_ENFORCEMENT_ENABLED zeroes the
    score, and only for the CPU half — a disk figure above the limit is reported, never
    scored, because it is read once and carries every docker category nobody enumerated."""

    LOAD_OK = MessageTemplate(
        event="Provider-side load within limits",
        reason="PROVIDER_SIDE_LOAD_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    LOAD_ABOVE_LIMIT = MessageTemplate(
        event="Provider-side workload consumes the machine's CPU or disk",
        reason="PROVIDER_SIDE_LOAD_ABOVE_LIMIT",
        severity="error",
        category="policy",
        impact="Score set to 0",
        remediation=(
            "A workload outside Lium containers holds this machine's CPU or disk "
            "(another subnet's miner, or host-side jobs). Stop it, then re-run your node. "
            "Capacity you sell on Lium must be free for the renter."
        ),
    )
    NOT_MEASURABLE = MessageTemplate(
        event="Provider-side load not measurable this cycle",
        reason="PROVIDER_SIDE_LOAD_UNKNOWN",
        severity="info",
        category="policy",
        impact="Proceed without penalty",
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
    """DAH-2265 — cached-template verification.

    Confirms whether the executor has the recommended default image pre-pulled so
    DOCKER_PULL is a no-op for default-template rentals. Templates carry severity="info"
    (advisory) by default; before ``settings.CACHED_TEMPLATE_CUTOFF`` the check is purely
    observational. On/after the cutoff the check renders the two bad signals (NOT_CACHED,
    DIGEST_MISMATCH) with a severity="error" override and fails verification — see
    ``CachedTemplateVerificationCheck``.
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
            "Pull the recommended image named in this event on the host to see why the "
            "executor's pre-pull could not fetch it; the executor retries on its own."
        ),
    )
    PENDING = MessageTemplate(
        event="Recommended default image not cached yet on a newly seen executor",
        reason="RECOMMENDED_IMAGE_PENDING",
        severity="info",
        category="runtime",
        impact=(
            "None this cycle — a newly seen node is not failed on this check until its first "
            "pre-pull sweep completes or the grace window ends"
        ),
        remediation=(
            "No action needed yet: the executor's pre-pull may still be fetching the image, or "
            "the executor does not report its pre-pull state and the node gets the time bound "
            "only. If the image is still missing when the grace ends, this check fails and "
            "quotes the executor's pull error when it reported one."
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


class LocalVerifyMessages:
    """liumd phase 1: one signed `POST /verify` in place of the SSH-driven matmul and VerifyX."""

    DISABLED = MessageTemplate(
        event="Local verification disabled",
        reason="LOCAL_VERIFY_DISABLED",
        severity="info",
        category="transport",
        impact="None — every check runs over SSH as before",
    )
    SKIPPED = MessageTemplate(
        event="Local verification skipped",
        reason="LOCAL_VERIFY_SKIPPED",
        severity="info",
        category="transport",
        impact="None — nothing for the local path to run this cycle",
    )
    NOT_ADVERTISED = MessageTemplate(
        event="Executor does not advertise local verification",
        reason="LOCAL_VERIFY_NOT_ADVERTISED",
        severity="info",
        category="transport",
        impact="None — the matmul and VerifyX run over SSH as before",
        remediation="Update the executor image and set EXECUTOR_LOCAL_VERIFY_ENABLED=true to take the one-call path.",
    )
    FALLBACK = MessageTemplate(
        event="Local verification not used, SSH path taken",
        reason="LOCAL_VERIFY_FALLBACK",
        severity="info",
        category="transport",
        impact="None on the verdict — the SSH-driven checks decide; only the time saved is lost",
    )
    CONSUMED = MessageTemplate(
        event="Local verification answered in one call",
        reason="LOCAL_VERIFY_OK",
        severity="info",
        category="transport",
        impact="The matmul and VerifyX verdicts below come from the executor's local run, judged by the validator",
    )


class RentedPodResourcesMessages:
    """DAH-4564 — a host script `docker update --cpuset-cpus`-pins a rented pod, or cuts its CPU quota,
    below 95% of the cores rented: RESOURCES_REDUCED zeroes the score. A raised oom_score_adj is recorded
    in the event and never scored: a renter running as root can raise it himself."""

    RESOURCES_OK = MessageTemplate(
        event="Rented pod CPU resources match the deployment",
        reason="RENTED_POD_RESOURCES_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    RESOURCES_REDUCED = MessageTemplate(
        event="Host pinned or throttled a rented pod below 95% of its advertised CPUs",
        reason="RENTED_POD_RESOURCES_REDUCED",
        severity="error",
        category="policy",
        impact="Score set to 0",
        remediation=(
            "A host-side script changed the renter's container with `docker update` "
            "(--cpuset-cpus, --cpus or --cpu-quota). A rental must keep at least 95% of its advertised "
            "cores in both its cpuset and its CPU quota; remove the script or loosen it to that."
        ),
    )
    RESOURCES_UNKNOWN = MessageTemplate(
        event="Rented pod CPU resources could not be read",
        reason="RENTED_POD_RESOURCES_UNKNOWN",
        severity="info",
        category="policy",
        impact="Proceed without penalty",
    )
    NOT_RENTED = MessageTemplate(
        event="Rented pod resources check skipped: no running rental",
        reason="RENTED_POD_RESOURCES_NOT_RENTED",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    SKIPPED = MessageTemplate(
        event="Rented pod resources check disabled",
        reason="RENTED_POD_RESOURCES_DISABLED",
        severity="info",
        category="policy",
        impact="Proceed",
    )


class GpuAttestationIdentityMessages:
    # DAH-4551: warning in shadow; fails the node under GPU_ATTESTATION_IDENTITY_ENFORCEMENT_ENABLED.
    MISMATCH = MessageTemplate(
        event="GPU identity does not match the NVIDIA-signed card identity",
        reason="GPU_IDENTITY_MISMATCH",
        severity="warning",
        category="policy",
        impact="Score set to 0 while the finding stands",
        remediation="Report the GPUs the host really has: each GPU UUID must stay on the same physical card, "
        "and nothing may intercept NVML or nvidia-smi on the host or in the executor container.",
    )
    # The fallback for cards that cannot attest: the same card under a new UUID (serial or slots kept).
    SWAPPED = MessageTemplate(
        # Neutral on purpose: "every UUID new in the same slots" is also what replacing every card looks like.
        event="GPU UUIDs changed: new UUIDs in the same slots, or a known card under a new UUID",
        reason="GPU_UUID_SWAPPED",
        severity="warning",
        category="policy",
        impact="Score set to 0 while the finding stands",
        remediation="Report each GPU under its real UUID; a GPU UUID must not change while the card stays.",
    )
    # A UUID whose model, memory or serial changed, or one host listing several models.
    INCONSISTENT = MessageTemplate(
        event="GPU details are inconsistent with what this UUID reported before",
        reason="GPU_IDENTITY_INCONSISTENT",
        severity="warning",
        category="policy",
        impact="Score set to 0 while the finding stands",
        remediation="Each GPU UUID must keep its model, memory size and serial; a node lists one GPU model.",
    )
    # Expected on most hosts: NVML only attests Hopper/Blackwell cards in Confidential Computing mode.
    UNAVAILABLE = MessageTemplate(
        event="GPU attestation not available on this node",
        reason="GPU_ATTESTATION_UNAVAILABLE",
        severity="info",
        category="policy",
        impact="Proceed",
    )
    OK = MessageTemplate(
        event="GPU identity verified against NVIDIA's root",
        reason="GPU_IDENTITY_OK",
        severity="info",
        category="policy",
        impact="Proceed",
    )


SSHD_BOOTSTRAP_REMEDIATION = (
    "sshd does not start inside containers on this node, so renters cannot SSH in. Common cause: the host's "
    "container runtime leaves /run/sshd not owned by root or group/world-writable (sshd refuses to start: "
    "\"/run/sshd must be owned by root and not group or world-writable\"). Fix the runtime/image setup so "
    "a fresh container gets a root-owned /run/sshd with mode 755, then check that `docker run --rm ubuntu sh -c "
    "'apt-get update -qq && apt-get install -y -qq openssh-server && mkdir -p /run/sshd && /usr/sbin/sshd -t'` "
    "succeeds. The node is validated again once a new container on it starts sshd."
)


class SshdBootstrapMessages:
    FAILED = MessageTemplate(
        event="sshd does not start in containers on this node",
        reason="SSHD_BOOTSTRAP_FAILED",
        severity="warning",
        category="runtime",
        impact="Score set to 0 and Lium's filler on the node removed until sshd starts in a container on this node",
        remediation=SSHD_BOOTSTRAP_REMEDIATION,
    )
    # the same finding while SSHD_BOOTSTRAP_ENFORCEMENT_ENABLED is off: logged, score unchanged
    FAILED_OBSERVED = MessageTemplate(
        event="sshd does not start in containers on this node (not enforced)",
        reason="SSHD_BOOTSTRAP_FAILED_OBSERVED",
        severity="warning",
        category="runtime",
        impact="None: SSHD_BOOTSTRAP_ENFORCEMENT_ENABLED is off",
        remediation=SSHD_BOOTSTRAP_REMEDIATION,
    )
    OK = MessageTemplate(
        event="No sshd start failure recorded for this node",
        reason="SSHD_BOOTSTRAP_OK",
        severity="info",
        category="runtime",
        impact="Proceed",
    )


class SshdAuthLogMessages:
    LOGINS = MessageTemplate(
        event="Executor sshd logins since the last cycle",
        reason="SSHD_AUTH_LOGINS",
        severity="info",
        category="runtime",
        impact="Proceed; observation only",
    )
    UNREAD = MessageTemplate(
        event="Executor sshd logins not read",
        reason="SSHD_AUTH_LOG_UNREAD",
        severity="info",
        category="runtime",
        impact="Proceed; observation only",
    )
