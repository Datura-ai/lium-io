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

Trust model (DAH-3137 design doc): the prober ships in the signed executor image
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
from dataclasses import dataclass, field
from typing import Any

# Aggregate wall-clock ceiling (seconds) across all claimed cards. N honest cards
# answer N nonce-bound challenges concurrently in ~single-card time; a count lie
# serialises onto fewer real cards and takes ~N× as long. This is a COARSE backstop:
# it only trips a serialisation slow enough to exceed the ceiling (fast cards may not),
# so the robust count signals are device-selection failure + duplicate kernel UUID.
# Wide on purpose (honest thermal throttle / a card busy with a filler must not trip it)
# and left uncalibrated until real per-class timings are collected.
GPU_SIGNATURE_WALL_CLOCK_SECONDS = 120.0

# The JSON marker the prober prints; only a line carrying it is parsed from stdout.
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


# Keyed on the canonical GPU model the pipeline already stores in ctx.state.gpu_model.
# Numbers below are PROVISIONAL placeholders seeded from public FP32/HBM specs
# (roughly halved to cover tiled-SGEMM efficiency + slow hosts); every class is
# calibrated=False, so nothing gates on them until a real pod run with the shipped
# binary replaces them. See the DAH-3137 design doc §9 and the PR's pod-run log.
GPU_SIGNATURE_ENVELOPE: dict[str, SignatureEnvelope] = {
    "NVIDIA B300 SXM6 AC": SignatureEnvelope(gbps_min=3000.0),
    "NVIDIA B200": SignatureEnvelope(gbps_min=3000.0),
    "NVIDIA H200": SignatureEnvelope(gbps_min=2000.0),
    "NVIDIA H200 NVL": SignatureEnvelope(gbps_min=2000.0),
    "NVIDIA H100 80GB HBM3": SignatureEnvelope(gbps_min=1500.0),
    "NVIDIA H100 PCIe": SignatureEnvelope(gbps_min=1000.0),
    "NVIDIA GeForce RTX 5090": SignatureEnvelope(gbps_min=800.0),
    "NVIDIA GeForce RTX 4090": SignatureEnvelope(gbps_min=500.0),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": SignatureEnvelope(gbps_min=700.0),
    "NVIDIA RTX A6000": SignatureEnvelope(gbps_min=350.0),
    "NVIDIA A100 80GB PCIe": SignatureEnvelope(gbps_min=900.0),
    "NVIDIA A100-SXM4-80GB": SignatureEnvelope(gbps_min=1000.0),
}


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


def parse_result_line(stdout: str) -> dict[str, Any] | None:
    """Return the last parseable gpu_sig JSON object printed on stdout, or None.

    Stdout is miner-controlled and the image may print other lines, so scan from
    the end for the marker and take the first that parses.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if RESULT_MARKER not in line or len(line) > MAX_RESULT_LINE:
            continue
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001 — any parse failure (incl. RecursionError) skips the line
            continue
        if isinstance(obj, dict) and "gpusig" in obj:
            return obj
    return None


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

    Unknown or uncalibrated models gate nothing (fail-open): the signature is
    logged for calibration, never used to fail an honest host on a guess.
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
        reason = (seal_verdict.get("reason") if isinstance(seal_verdict, dict) else None) or "unsealed"
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

    reasons = check_envelope(claimed_model, tflops, gbps)
    if seal_verdict.get("device") != 0:
        reasons.append("device_mismatch")

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
    wall_clock_seconds: float = GPU_SIGNATURE_WALL_CLOCK_SECONDS,
) -> SignatureVerdict:
    """Aggregate per-card verdicts into a node verdict (the count-spoof gate)."""
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
    if len(verdicts) != claimed_count:
        reasons.append(f"card_count {len(verdicts)} != claimed {claimed_count}")
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
        passed=(
            not failed
            and not over_wall_clock
            and not duplicate_uuids
            and len(verdicts) == claimed_count
        ),
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
