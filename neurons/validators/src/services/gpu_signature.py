"""Validator-side logic for the on-host GPU hardware-signature challenge (DAH-3137).

Interface-only. This module knows the prober binary's contract — its CLI args
and its one-line response JSON schema — and the per-class performance envelope.
It does NOT know how the signature is computed: the seal is authenticated by the
compiled, obfuscated ``libgpusig.so`` (built in the private celium-gpu-verifier
repo) through the thin :mod:`gpusig_validator` ctypes wrapper. Everything here
runs on the AUTHENTICATED fields that verifier returns.

Response JSON schema emitted by ``bin/gpu_sig`` (one line on stdout):
    {"gpusig": 2, "ok": true, "scheme": "...", "nonce": "...", "device": 0,
     "kernel_uuid": "...", "kernel_model": "...", "cuda_name": "...",
     "pci": "...", "vram_mb": N, "mm_n": N, "mm_iters": N,
     "tflops": F, "gbps": F, "ms": F, "have_kernel": bool,
     "msg": "<opaque signed record>", "sig": "<opaque hex>"}
On failure it prints ``{"gpusig": 2, "ok": false, ..., "error": "<code>"}``.

Trust model (DAH-3137 design doc): the prober ships in the executor image
and is driven with a fresh per-card nonce; the seal is bound to that nonce and to
the card identity, so a precomputed/replayed answer for a different nonce fails
authentication. This closes cheap/scalable spoofs (NVML shims, canned/replayed
answers, wrong-class numbers, count serialisation); it does not defeat a root
provider who extracts the image key and forges fast numbers — that needs the
derived key folded into liumd and/or NVIDIA CC attestation. Kept observe-only
until the envelope is calibrated on real hardware.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

# The prober's schema: the response object carries "gpusig": <version> and
# "scheme": <tag>. A line is a result only when both match, so a cross-repo rename
# of either is reported (no_result with the matched/parsed counts), never silently
# parsed as something else.
GPUSIG_VERSION = 2
GPUSIG_SCHEME = "lium.gpusig.v2"

# The JSON marker the prober prints; only a line carrying it is a candidate. The
# schema check above decides whether a candidate is a result.
RESULT_MARKER = '"gpusig"'

# Hard caps on peer-controlled input (§5 "bounded input"): the real response line is
# ~500 bytes and the signed record ~200; anything far larger is a malformed/hostile
# response, capped before it reaches json.loads or the native verifier.
MAX_RESULT_LINE = 65536
MAX_SEAL_FIELD = 16384


@dataclass(frozen=True)
class SignatureEnvelope:
    """Per-class ground-truth floor for the measured signature.

    tflops_min / gbps_min are the tiled-SGEMM + device-to-device-copy numbers the
    prober produces (a fixed fraction of peak, not vendor peak), so they MUST be
    calibrated with the exact shipped binary on real cards. None = not gated.
    calibrated=False = the whole class is not gated. Margins are deliberately
    generous.
    """

    tflops_min: float | None = None
    gbps_min: float | None = None
    calibrated: bool = False


# Empty on purpose: no class is calibrated yet, so nothing gates. The earlier
# provisional placeholders were dropped rather than left as calibrated=False, so a
# future entry can only be one measured with the shipped binary (DAH-3137 design §9).
#
# When floors are added, key them on an AUTHENTICATED field. ctx.state.gpu_model is
# the NVML-claimed class, which a shim controls, so a wrong-class claim would pick
# its own (lower) row. The signed record authenticates kernel_uuid and vram_mb but
# NOT a model string, so the calibrated envelope must key on the authenticated
# capacity (vram_mb) or a class derived from it, never on the NVML model.
GPU_SIGNATURE_ENVELOPE: dict[str, SignatureEnvelope] = {}


@dataclass
class CardVerdict:
    slot: int  # the CUDA_VISIBLE_DEVICES index the validator pinned this run to
    ok: bool
    reasons: list[str] = field(default_factory=list)
    kernel_uuid: str = ""
    tflops: float | None = None
    gbps: float | None = None
    have_kernel: bool = False


@dataclass
class SignatureVerdict:
    passed: bool
    reasons: list[str]
    per_card: list[dict[str, Any]]
    elapsed_seconds: float
    over_wall_clock: bool
    claimed_count: int
    verified_count: int


@dataclass(frozen=True)
class ParsedResult:
    """Outcome of scanning one prober stdout.

    ``result`` is the newest line that parsed AND matched the schema (or None).
    ``matched`` counts lines carrying the marker, ``parsed`` those that were JSON
    objects; on no_result they tell a renamed key apart from an empty stdout.
    """

    result: dict[str, Any] | None
    matched: int
    parsed: int

    @property
    def no_result_reason(self) -> str:
        return f"no_result:matched={self.matched},parsed={self.parsed}"


def wall_clock_seconds(gpu_count: int, timeout_seconds: float, max_concurrent: int) -> float:
    """Aggregate ceiling for ``gpu_count`` per-card runs, each bounded by
    ``timeout_seconds``, at most ``max_concurrent`` at a time.

    N honest cards answer concurrently in ~single-card time per wave; a count lie
    serialises onto fewer real cards and takes longer. The ceiling is timeout x waves,
    so an honest 14-card host at 8 concurrent (two waves) is never flagged for
    needing its second wave. This is a COARSE backstop: the robust count signals are
    device-selection failure + duplicate kernel UUID.
    """
    waves = max(1, math.ceil(max(1, gpu_count) / max(1, max_concurrent)))
    return float(timeout_seconds) * waves


def parse_result(stdout: str) -> ParsedResult:
    """Scan miner-controlled stdout for the prober's result line.

    The image may print other lines, so scan from the end for the marker and take
    the first that parses AND matches the schema (``gpusig == 2`` and
    ``scheme == "lium.gpusig.v2"``).
    """
    matched = parsed = 0
    result: dict[str, Any] | None = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if RESULT_MARKER not in line or len(line) > MAX_RESULT_LINE:
            continue
        matched += 1
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001 — any parse failure (incl. RecursionError) skips the line
            continue
        if not isinstance(obj, dict):
            continue
        parsed += 1
        if (
            result is None
            and obj.get("gpusig") == GPUSIG_VERSION
            and obj.get("scheme") == GPUSIG_SCHEME
        ):
            result = obj
    return ParsedResult(result=result, matched=matched, parsed=parsed)


def parse_result_line(stdout: str) -> dict[str, Any] | None:
    """Return the newest schema-matching gpu_sig JSON object on stdout, or None."""
    return parse_result(stdout).result


def seal_within_bounds(msg: str, sig: str) -> bool:
    """§5 bounded-input guard before msg/sig cross into the native verifier."""
    return len(msg) <= MAX_SEAL_FIELD and len(sig) <= MAX_SEAL_FIELD


def kernel_uuid_mismatch(kernel_uuids: list[str], claimed_uuids: list[str]) -> bool:
    """True when the kernel reported a GPU UUID that NVML never advertised.

    That one-directional check is the userspace-NVML-shim signal (DAH-2662): a shim
    rewrites the NVML-claimed set, so a `/proc`-sourced UUID outside it is suspicious.
    A card that merely failed (empty UUID, so absent here) or a genuine subset — some
    cards down — is NOT a mismatch; that is the per-card verdict's job.
    """
    claimed = {u for u in claimed_uuids if u}
    return bool({u for u in kernel_uuids if u} - claimed)


def check_envelope(model: str | None, tflops: float, gbps: float) -> list[str]:
    """Return a list of below-floor reasons for the claimed class (empty = pass).

    Unknown or uncalibrated models are not gated: the signature is logged for
    calibration, never used to fail an honest host on a guess.
    """
    env = GPU_SIGNATURE_ENVELOPE.get(model or "")
    if env is None or not env.calibrated:
        return []
    reasons: list[str] = []
    if env.tflops_min is not None and tflops < env.tflops_min:
        reasons.append(f"tflops {tflops:.1f} < floor {env.tflops_min:.1f}")
    if env.gbps_min is not None and gbps < env.gbps_min:
        reasons.append(f"gbps {gbps:.1f} < floor {env.gbps_min:.1f}")
    return reasons


def evaluate_card(
    seal_verdict: dict[str, Any],
    claimed_model: str | None,
    slot: int,
    have_kernel: bool = False,
) -> CardVerdict:
    """Turn one libgpusig.so verdict into a scored per-card verdict.

    ``seal_verdict`` is what ``libgpusig.so`` returned (or a synthesised
    ``{"sealed": False, "reason": ...}`` for a prober error / missing result).
    When sealed, its ``tflops``/``gbps``/``kernel_uuid``/``device`` are the
    AUTHENTICATED values; a replayed nonce, tampered numbers or a wrong key all
    arrive here as ``sealed=False`` with a reason. ``slot`` is the
    CUDA_VISIBLE_DEVICES index the validator pinned; the prober always runs as
    its own ``--device 0`` inside that masked view.
    """
    if not isinstance(seal_verdict, dict) or not seal_verdict.get("sealed"):
        reason = (
            seal_verdict.get("reason") if isinstance(seal_verdict, dict) else None
        ) or "unsealed"
        return CardVerdict(slot=slot, ok=False, reasons=[reason])

    # The verifier .so is separately versioned; treat any shape drift (a non-numeric
    # metric, a non-string uuid) as an unsealed card, never an exception into the pipeline.
    try:
        tflops = float(seal_verdict.get("tflops") or 0.0)
        gbps = float(seal_verdict.get("gbps") or 0.0)
    except (TypeError, ValueError):
        return CardVerdict(slot=slot, ok=False, reasons=["verifier_bad_output"])
    kernel_uuid = seal_verdict.get("kernel_uuid")
    kernel_uuid = kernel_uuid if isinstance(kernel_uuid, str) else ""
    # `device` must be present and integral before it can be compared: a missing or
    # non-int device is schema drift (bad output), not a card answering for another slot.
    device = seal_verdict.get("device")
    if isinstance(device, bool) or not isinstance(device, int):
        return CardVerdict(slot=slot, ok=False, reasons=["verifier_bad_output"])

    reasons = check_envelope(claimed_model, tflops, gbps)
    if device != 0:
        reasons.append("device_mismatch")
    # The kernel-reported UUID (/proc/driver/nvidia, outside NVML) is the identity anchor.
    # An empty one means the prober could not read /proc — a provider that hides it from
    # the executor container would otherwise pass every count signal (summarize and
    # kernel_uuid_mismatch both skip empty UUIDs). Make it a per-card reason. Key it on
    # the SEALED kernel_uuid, not only on have_kernel: have_kernel is a plain stdout
    # field the miner can flip to true, while kernel_uuid is inside the signed record.
    if not have_kernel or not kernel_uuid:
        reasons.append("no_kernel_identity")

    return CardVerdict(
        slot=slot,
        ok=not reasons,
        reasons=reasons,
        kernel_uuid=kernel_uuid,
        tflops=tflops,
        gbps=gbps,
        have_kernel=bool(have_kernel),
    )


def summarize(
    verdicts: list[CardVerdict],
    elapsed_seconds: float,
    claimed_count: int,
    wall_clock_seconds: float,
) -> SignatureVerdict:
    """Aggregate per-card verdicts into a node verdict (the count-spoof gate).

    ``claimed_count`` is the NVML-claimed card count the check challenged; it is
    reported, not compared: the check issues exactly one challenge per claimed slot,
    so a count lie surfaces as failed slots (device-selection error), duplicate
    kernel UUIDs or a blown wall clock, never as a shorter verdict list.
    ``wall_clock_seconds`` comes from :func:`wall_clock_seconds` (timeout x waves).
    """
    verified = [v for v in verdicts if v.ok]
    failed = [v for v in verdicts if not v.ok]
    over_wall_clock = elapsed_seconds > wall_clock_seconds

    # One physical card answering for many claimed cards shows up as a repeated
    # kernel UUID across the (sealed, /proc-sourced) per-card results.
    seen: dict[str, int] = {}
    for v in verdicts:
        if v.kernel_uuid:
            seen[v.kernel_uuid] = seen.get(v.kernel_uuid, 0) + 1
    duplicate_uuids = sorted(u for u, n in seen.items() if n > 1)

    reasons: list[str] = []
    if failed:
        reasons.append(f"{len(failed)}/{len(verdicts)} card(s) failed signature")
    if duplicate_uuids:
        reasons.append(f"duplicate kernel UUID across cards: {','.join(duplicate_uuids)}")
    if over_wall_clock:
        reasons.append(
            f"aggregate wall-clock {elapsed_seconds:.1f}s over {wall_clock_seconds:.0f}s "
            "(serialisation on fewer real cards than claimed)"
        )

    return SignatureVerdict(
        passed=(not failed and not over_wall_clock and not duplicate_uuids),
        reasons=reasons,
        per_card=[
            {
                "slot": v.slot,
                "ok": v.ok,
                "reasons": v.reasons,
                "kernel_uuid": v.kernel_uuid,
                "tflops": v.tflops,
                "gbps": v.gbps,
                "have_kernel": v.have_kernel,
            }
            for v in verdicts
        ],
        elapsed_seconds=round(elapsed_seconds, 3),
        over_wall_clock=over_wall_clock,
        claimed_count=claimed_count,
        verified_count=len(verified),
    )
