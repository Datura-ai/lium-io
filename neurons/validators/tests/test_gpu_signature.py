"""Unit tests for the GPU hardware-signature validator logic (DAH-3137).

Pure Python, no GPU and no libgpusig.so: the seal itself is authenticated by the
compiled verifier (covered by celium-gpu-verifier's interop test and the PR's
GPU-pod run). These exercise the validator's business logic on the AUTHENTICATED
fields the verifier returns — envelope gating, per-card scoring, and the
count-spoof aggregation — by feeding synthesised verifier verdicts.
"""

from services import gpu_signature as gs


def _sealed(uuid="GPU-1111", tflops=40.0, gbps=900.0, device=0):
    """A verdict shaped exactly like libgpusig.so returns on success."""
    return {
        "sealed": True,
        "reason": "",
        "scheme": "lium.gpusig.v2",
        "nonce": "a" * 64,
        "kernel_uuid": uuid,
        "pci": "0000:65:00.0",
        "device": device,
        "vram_mb": 81920,
        "tflops": tflops,
        "gbps": gbps,
        "ms": 800.0,
    }


# ---- parse_result_line -------------------------------------------------------


def test_parse_result_line_picks_json_amid_noise():
    # msg/sig are opaque to the validator; the check hands them to libgpusig.so.
    line = '{"gpusig": 2, "ok": true, "nonce": "a", "msg": "<opaque>", "sig": "<opaque>"}'
    stdout = "MOTD banner\nWARNING: whatever\n" + line + "\n"
    parsed = gs.parse_result_line(stdout)
    assert parsed is not None and parsed["gpusig"] == 2 and parsed["ok"] is True


def test_parse_result_line_none_when_absent():
    assert gs.parse_result_line("no json here\njust text") is None


# ---- envelope ----------------------------------------------------------------


def test_envelope_uncalibrated_does_not_gate():
    # shipped entries are calibrated=False -> never fail an honest host on a guess
    assert gs.check_envelope("NVIDIA H200", 0.1, 1.0) == []
    assert gs.check_envelope("totally-unknown-model", 0.0, 0.0) == []


def test_envelope_calibrated_flags_underperformer(monkeypatch):
    monkeypatch.setitem(
        gs.GPU_SIGNATURE_ENVELOPE,
        "NVIDIA H200",
        gs.SignatureEnvelope(tflops_min=30.0, gbps_min=2000.0, calibrated=True),
    )
    assert gs.check_envelope("NVIDIA H200", 45.0, 3000.0) == []  # healthy
    reasons = gs.check_envelope("NVIDIA H200", 5.0, 500.0)  # relabelled slow card
    assert any("tflops" in r for r in reasons) and any("gbps" in r for r in reasons)


# ---- per-card evaluation -----------------------------------------------------


def test_evaluate_card_ok():
    v = gs.evaluate_card(_sealed(), "NVIDIA GeForce RTX 4090", 0, have_kernel=True)
    assert v.ok and not v.reasons and v.kernel_uuid == "GPU-1111" and v.have_kernel


def test_evaluate_card_unsealed_carries_reason():
    # a replayed nonce / tampered numbers / wrong key all arrive as sealed=False
    v = gs.evaluate_card({"sealed": False, "reason": "nonce_mismatch"}, None, 0)
    assert not v.ok and v.reasons == ["nonce_mismatch"]


def test_evaluate_card_prober_error():
    v = gs.evaluate_card({"sealed": False, "reason": "prober_error:set_device"}, None, 3)
    assert not v.ok and v.reasons[0].startswith("prober_error")


def test_evaluate_card_malformed_verifier_output_never_raises():
    # the verifier .so is separately versioned: a shape drift must become an unsealed
    # verdict, never an exception into the pipeline
    bad = {"sealed": True, "tflops": "not-a-number", "gbps": 1.0, "device": 0, "kernel_uuid": "x"}
    v = gs.evaluate_card(bad, None, 0)
    assert not v.ok and v.reasons == ["verifier_bad_output"]
    assert gs.evaluate_card("not-a-dict", None, 0).ok is False
    # a non-string kernel_uuid is coerced to "" so summarize()'s sort/join never raises
    coerced = gs.evaluate_card({**_sealed(), "kernel_uuid": 123}, None, 0)
    assert coerced.kernel_uuid == ""
    gs.summarize([coerced], elapsed_seconds=1.0, claimed_count=1)  # must not raise


def test_parse_result_line_bounded_against_hostile_stdout():
    # a deeply nested line UNDER the length cap must still be skipped, never raise
    # (RecursionError from json.loads is not a JSONDecodeError) — exercises the broad except
    assert gs.parse_result_line('[' * 20000 + '"gpusig"') is None
    # a line over the length cap is dropped before json.loads
    assert gs.parse_result_line('{"gpusig":1}' + 'x' * gs.MAX_RESULT_LINE) is None
    assert gs.seal_within_bounds("m", "s") is True
    assert gs.seal_within_bounds("m" * (gs.MAX_SEAL_FIELD + 1), "s") is False


def test_evaluate_card_device_mismatch():
    v = gs.evaluate_card(_sealed(device=2), None, 0)
    assert not v.ok and "device_mismatch" in v.reasons


def test_evaluate_card_calibrated_below_floor_fails(monkeypatch):
    monkeypatch.setitem(
        gs.GPU_SIGNATURE_ENVELOPE,
        "NVIDIA H200",
        gs.SignatureEnvelope(tflops_min=30.0, gbps_min=2000.0, calibrated=True),
    )
    v = gs.evaluate_card(_sealed(tflops=5.0, gbps=500.0), "NVIDIA H200", 0)
    assert not v.ok and any("gbps" in r for r in v.reasons)


# ---- aggregate / count spoof -------------------------------------------------


def test_summarize_all_good():
    verdicts = [gs.evaluate_card(_sealed(uuid=f"GPU-{i}"), None, i) for i in range(4)]
    out = gs.summarize(verdicts, elapsed_seconds=10.0, claimed_count=4)
    assert out.passed and out.verified_count == 4 and not out.reasons


def test_summarize_count_lie_device_selection():
    # claims 4, only card 0 real: cards 1..3 come back unsealed (device-selection error)
    verdicts = [gs.evaluate_card(_sealed(uuid="GPU-0"), None, 0)]
    for i in range(1, 4):
        verdicts.append(
            gs.evaluate_card({"sealed": False, "reason": "prober_error:set_device"}, None, i)
        )
    out = gs.summarize(verdicts, elapsed_seconds=10.0, claimed_count=4)
    assert not out.passed and out.verified_count == 1


def test_summarize_duplicate_uuid_is_count_spoof():
    # one physical card answering for four claimed cards -> same kernel UUID
    verdicts = [gs.evaluate_card(_sealed(uuid="GPU-SAME"), None, i) for i in range(4)]
    out = gs.summarize(verdicts, elapsed_seconds=10.0, claimed_count=4)
    assert not out.passed and any("duplicate kernel UUID" in r for r in out.reasons)


def test_summarize_over_wall_clock():
    verdicts = [gs.evaluate_card(_sealed(uuid=f"GPU-{i}"), None, i) for i in range(2)]
    out = gs.summarize(verdicts, elapsed_seconds=999.0, claimed_count=2, wall_clock_seconds=120.0)
    assert not out.passed and out.over_wall_clock


# ---- kernel-vs-NVML UUID cross-check (the 4th count/type-spoof signal) -------


def test_kernel_uuid_mismatch_flags_uuid_nvml_never_claimed():
    # a userspace NVML shim: the kernel reports a card NVML did not advertise
    assert gs.kernel_uuid_mismatch(["GPU-real", "GPU-shimmed"], ["GPU-real"]) is True


def test_kernel_uuid_mismatch_ignores_failed_cards_and_subsets():
    # some cards failed (empty, so absent from kernel_uuids) -> not a shim signal
    assert gs.kernel_uuid_mismatch(["GPU-a"], ["GPU-a", "GPU-b"]) is False
    # every kernel UUID is claimed -> clean
    assert gs.kernel_uuid_mismatch(["GPU-a", "GPU-b"], ["GPU-b", "GPU-a"]) is False
    # nothing authenticated -> nothing to flag
    assert gs.kernel_uuid_mismatch([], ["GPU-a"]) is False
