"""Unit tests for the GPU hardware-signature verification (DAH-3137).

Pure Python, no GPU: they construct sealed results exactly as gpu_sig.cu does
(so they also pin the on-the-wire seal format) and exercise the validator's
verify / envelope / aggregate logic, including the count-spoof signals.
"""
import hashlib
import hmac

from services import gpu_signature as gs

MASTER = b"lium-gpu-sig-dev-key-do-not-use-in-prod"
NONCE = "a" * 64


def _seal(master: bytes, nonce: str, uuid: str, pci: str, tflops: str, gbps: str, ms: str, device: int = 0):
    """Reproduce gpu_sig.cu: msg = nonce|device|uuid|pci|tflops|gbps|ms,
    key = HMAC(master, uuid), sig = HMAC(key, msg)."""
    msg = f"{nonce}|{device}|{uuid}|{pci}|{tflops}|{gbps}|{ms}"
    key = hmac.new(master, uuid.encode(), hashlib.sha256).digest()
    sig = hmac.new(key, msg.encode(), hashlib.sha256).hexdigest()
    return msg, sig


def _result(uuid="GPU-1111", pci="0000:65:00.0", tflops="40.000", gbps="900.000", ms="800.000",
            nonce=NONCE, device=0, master=MASTER, ok=True, have_kernel=True):
    msg, sig = _seal(master, nonce, uuid, pci, tflops, gbps, ms, device)
    return {
        "gpu_sig": 1, "ok": ok, "nonce": nonce, "device": device,
        "kernel_uuid": uuid, "pci": pci, "tflops": float(tflops), "gbps": float(gbps),
        "ms": float(ms), "have_kernel": have_kernel, "msg": msg, "sig": sig,
    }


# ---- seal --------------------------------------------------------------------

def test_verify_seal_ok():
    ok, reason = gs.verify_seal(MASTER, _result())
    assert ok and reason == ""


def test_verify_seal_rejects_tampered_numbers():
    r = _result(tflops="40.000")
    # miner edits the displayed msg to inflate TFLOPS but cannot recompute the seal
    r["msg"] = r["msg"].replace("40.000", "999.000")
    ok, reason = gs.verify_seal(MASTER, r)
    assert not ok and reason == "seal_mismatch"


def test_verify_seal_rejects_wrong_key():
    r = _result(master=b"some-other-key")
    ok, reason = gs.verify_seal(MASTER, r)
    assert not ok and reason == "seal_mismatch"


def test_verify_seal_missing_fields():
    assert gs.verify_seal(MASTER, {"gpu_sig": 1})[0] is False


# ---- per-card evaluation -----------------------------------------------------

def test_evaluate_card_ok():
    v = gs.evaluate_card(MASTER, NONCE, "NVIDIA GeForce RTX 4090", 0, _result())
    assert v.ok and not v.reasons and v.kernel_uuid == "GPU-1111"


def test_evaluate_card_replayed_nonce_fails():
    # a sealed result from a previous challenge (different nonce) must not pass
    stale = _result(nonce="b" * 64)
    v = gs.evaluate_card(MASTER, NONCE, None, 0, stale)
    assert not v.ok and "nonce_mismatch" in v.reasons


def test_evaluate_card_prober_error():
    v = gs.evaluate_card(MASTER, NONCE, None, 3, {"ok": False, "error": "set_device:cudaErrorInvalidDevice"})
    assert not v.ok and v.reasons[0].startswith("prober_error")


def test_evaluate_card_no_result_is_failure():
    v = gs.evaluate_card(MASTER, NONCE, None, 5, None)
    assert not v.ok and v.reasons == ["no_result"]


# ---- envelope ----------------------------------------------------------------

def test_envelope_uncalibrated_does_not_gate():
    # shipped entries are calibrated=False -> never fail an honest host on a guess
    assert gs.check_envelope("NVIDIA H200", 0.1, 1.0) == []
    assert gs.check_envelope("totally-unknown-model", 0.0, 0.0) == []


def test_envelope_calibrated_flags_underperformer(monkeypatch):
    monkeypatch.setitem(
        gs.GPU_SIGNATURE_ENVELOPE, "NVIDIA H200",
        gs.SignatureEnvelope(tflops_min=30.0, gbps_min=2000.0, calibrated=True),
    )
    assert gs.check_envelope("NVIDIA H200", 45.0, 3000.0) == []          # healthy
    reasons = gs.check_envelope("NVIDIA H200", 5.0, 500.0)               # relabelled slow card
    assert any("tflops" in r for r in reasons) and any("gbps" in r for r in reasons)


# ---- aggregate / count spoof -------------------------------------------------

def test_summarize_all_good():
    verdicts = [
        gs.evaluate_card(MASTER, NONCE, None, i, _result(uuid=f"GPU-{i}")) for i in range(4)
    ]
    out = gs.summarize(verdicts, elapsed_seconds=10.0, claimed_count=4)
    assert out.passed and out.verified_count == 4 and not out.reasons


def test_summarize_count_lie_device_selection():
    # claims 4, only card 0 real: cards 1..3 return device-selection errors
    verdicts = [gs.evaluate_card(MASTER, NONCE, None, 0, _result(uuid="GPU-0"))]
    for i in range(1, 4):
        verdicts.append(gs.evaluate_card(MASTER, NONCE, None, i, {"ok": False, "error": "set_device"}))
    out = gs.summarize(verdicts, elapsed_seconds=10.0, claimed_count=4)
    assert not out.passed and out.verified_count == 1


def test_summarize_duplicate_uuid_is_count_spoof():
    # one physical card answering for four claimed cards -> same kernel UUID
    verdicts = [gs.evaluate_card(MASTER, NONCE, None, i, _result(uuid="GPU-SAME")) for i in range(4)]
    out = gs.summarize(verdicts, elapsed_seconds=10.0, claimed_count=4)
    assert not out.passed and any("duplicate kernel UUID" in r for r in out.reasons)


def test_summarize_over_wall_clock():
    verdicts = [gs.evaluate_card(MASTER, NONCE, None, i, _result(uuid=f"GPU-{i}")) for i in range(2)]
    out = gs.summarize(verdicts, elapsed_seconds=999.0, claimed_count=2, wall_clock_seconds=120.0)
    assert not out.passed and out.over_wall_clock


# ---- stdout parsing ----------------------------------------------------------

def test_parse_result_line_picks_json_amid_noise():
    r = _result()
    import json as _json
    stdout = "some MOTD banner\nWARNING: whatever\n" + _json.dumps(r) + "\n"
    parsed = gs.parse_result_line(stdout)
    assert parsed is not None and parsed["msg"] == r["msg"]


def test_parse_result_line_none_when_absent():
    assert gs.parse_result_line("no json here\njust text") is None
