"""Verification logic for the on-host GPU hardware-signature challenge (DAH-3137).

Pure standard-library so it unit-tests without a GPU or the bittensor stack. The
matching prober is ``neurons/executor/gpu_sig/gpu_sig.cu``; the seal format is
kept byte-identical (see ``test_seal.c`` and ``tests/test_gpu_signature.py``).

Trust model (see ~/lium-ads/verification-binary/DESIGN.md):
  - The prober ships in the signed executor image and is driven with a fresh
    per-call nonce, so a precomputed/replayed answer for a different nonce fails.
  - The result is HMAC-sealed with a key derived from the card's kernel-reported
    UUID; the validator derives the same key independently, so a fabricated
    identity yields a seal it cannot reproduce.
  - This closes cheap/scalable spoofs (NVML shims, canned/replayed answers,
    wrong-class numbers, count serialisation). It does NOT defeat a root
    provider who fully reverse-engineers the binary and forges plausible fast
    numbers — that needs the derived-key seal folded into liumd and/or NVIDIA CC
    attestation. Kept observe-only until the envelope is calibrated on real
    hardware and the score gate is wired (follow-up).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any

# Aggregate wall-clock ceiling (seconds) across all claimed cards. N honest cards
# answer N nonce-bound challenges concurrently in ~single-card time; a count lie
# serialises onto fewer real cards and blows past this. Wide on purpose so honest
# slowdowns (thermal throttle, a card already busy with a filler) do not trip it.
GPU_SIGNATURE_WALL_CLOCK_SECONDS = 120.0

# The JSON marker the prober prints; only this line is parsed out of stdout.
RESULT_MARKER = '"gpu_sig"'

# Number of "|"-joined tokens in the signed message:
# nonce|device|kernel_uuid|pci|tflops|gbps|ms
_MSG_FIELDS = 7


@dataclass(frozen=True)
class SignatureEnvelope:
    """Per-class ground-truth floor for the measured signature.

    tflops_min / gbps_min are the tiled-SGEMM + device-to-device-copy numbers the
    prober produces (a fixed fraction of peak, not vendor peak), so they MUST be
    calibrated with this exact binary on real cards. None = not yet calibrated =
    that metric is not gated. Margins are deliberately generous.
    """

    tflops_min: float | None = None
    gbps_min: float | None = None
    calibrated: bool = False


# Keyed on the canonical (normalised) GPU model the pipeline already stores in
# ctx.state.gpu_model. Numbers below are PROVISIONAL placeholders seeded from
# public FP32/HBM specs (roughly halved to cover tiled-SGEMM efficiency + slow
# hosts); they are marked uncalibrated so nothing gates on them until a real pod
# run with this binary replaces them. See DESIGN.md §9 and the PR's pod-run log.
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


def derive_card_key(master_key: bytes, kernel_uuid: str) -> bytes:
    """Per-card seal key = HMAC(master, kernel_uuid). The validator derives this
    from the UUID it independently expects, binding the seal to identity."""
    return hmac.new(master_key, kernel_uuid.encode("utf-8"), hashlib.sha256).digest()


def parse_result_line(stdout: str) -> dict[str, Any] | None:
    """Return the last parseable gpu_sig JSON object printed on stdout, or None.

    Stdout is miner-controlled and the image may print other lines, so scan from
    the end for the marker and take the first that parses.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if RESULT_MARKER not in line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and "gpu_sig" in obj:
            return obj
    return None


def _split_msg(msg: str) -> list[str] | None:
    parts = msg.split("|")
    return parts if len(parts) == _MSG_FIELDS else None


def verify_seal(master_key: bytes, result: dict[str, Any]) -> tuple[bool, str]:
    """Authenticate the sealed message. Returns (ok, reason).

    The signed ``msg`` string is authoritative; the cosmetic numeric JSON fields
    are never trusted for scoring.
    """
    msg = result.get("msg")
    sig = result.get("sig")
    if not isinstance(msg, str) or not isinstance(sig, str):
        return False, "missing_seal"
    parts = _split_msg(msg)
    if parts is None:
        return False, "malformed_msg"
    kernel_uuid = parts[2]
    key = derive_card_key(master_key, kernel_uuid)
    expected = hmac.new(key, msg.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig.strip().lower()):
        return False, "seal_mismatch"
    return True, ""


def fields_from_msg(msg: str) -> dict[str, Any] | None:
    parts = _split_msg(msg)
    if parts is None:
        return None
    try:
        return {
            "nonce": parts[0],
            "device": int(parts[1]),
            "kernel_uuid": parts[2],
            "pci": parts[3],
            "tflops": float(parts[4]),
            "gbps": float(parts[5]),
            "ms": float(parts[6]),
        }
    except (ValueError, TypeError):
        return None


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
    master_key: bytes,
    expected_nonce: str,
    claimed_model: str | None,
    slot: int,
    result: dict[str, Any] | None,
) -> CardVerdict:
    """Verdict for one card: run present, sealed, fresh, and within the class envelope.

    ``slot`` is the CUDA_VISIBLE_DEVICES index the validator pinned; the prober
    always runs as its own ``--device 0`` inside that masked view.
    """
    if result is None:
        return CardVerdict(slot=slot, ok=False, reasons=["no_result"])
    if result.get("ok") is not True:
        return CardVerdict(
            slot=slot, ok=False, reasons=[f"prober_error:{result.get('error', 'unknown')}"]
        )

    sealed, seal_reason = verify_seal(master_key, result)
    if not sealed:
        return CardVerdict(slot=slot, ok=False, reasons=[seal_reason])

    fields = fields_from_msg(result["msg"])
    if fields is None:
        return CardVerdict(slot=slot, ok=False, reasons=["malformed_msg"])

    reasons: list[str] = []
    if not hmac.compare_digest(fields["nonce"], expected_nonce):
        reasons.append("nonce_mismatch")  # stale/replayed
    if fields["device"] != 0:
        reasons.append("device_mismatch")
    reasons.extend(check_envelope(claimed_model, fields["tflops"], fields["gbps"]))

    return CardVerdict(
        slot=slot,
        ok=not reasons,
        reasons=reasons,
        kernel_uuid=fields["kernel_uuid"],
        tflops=fields["tflops"],
        gbps=fields["gbps"],
        have_kernel=bool(result.get("have_kernel")),
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
