"""Unit tests for the GPU hardware-signature validator logic (DAH-3137).

Pure Python, no GPU and no libgpusig.so: the seal itself is authenticated by the
compiled verifier (covered by celium-gpu-verifier's interop test and, here, by
test_gpu_signature_check.py's fixture tests against the committed .so). These
exercise the validator's business logic on the AUTHENTICATED fields the verifier
returns — envelope gating, per-card scoring, and the count-spoof aggregation — by
feeding synthesised verifier verdicts. One behaviour per test function.
"""

import json

import pytest

from services import gpu_signature as gs

WALL = 120.0


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


def _result_line(**over) -> str:
    obj = {
        "gpusig": 2,
        "scheme": "lium.gpusig.v2",
        "ok": True,
        "nonce": "a",
        "msg": "<m>",
        "sig": "<s>",
    }
    obj.update(over)
    import json

    return json.dumps(obj)


# ---- parse_result / parse_result_line ---------------------------------------


def test_parse_result_line_picks_json_amid_noise():
    stdout = "MOTD banner\nWARNING: whatever\n" + _result_line() + "\n"
    parsed = gs.parse_result_line(stdout)
    assert parsed is not None and parsed["gpusig"] == 2 and parsed["ok"] is True


def test_parse_result_line_none_when_absent():
    assert gs.parse_result_line("no json here\njust text") is None


def test_parse_result_requires_schema_version_and_scheme():
    # The regression: a prober rename (a different version or scheme tag) used to be
    # parsed as a result because only the literal '"gpusig"' marker was matched.
    assert gs.parse_result_line(_result_line(gpusig=3)) is None
    assert gs.parse_result_line(_result_line(scheme="lium.gpusig.v3")) is None
    renamed = '{"gpusig_v3": 3, "ok": true, "gpusig": "yes"}'
    assert gs.parse_result_line(renamed) is None


def test_parse_result_reports_matched_and_parsed_counts_on_no_result():
    out = gs.parse_result("noise\n" + _result_line(gpusig=3) + "\n" + '{"gpusig": broken')
    assert out.result is None
    assert out.matched == 2 and out.parsed == 1
    assert out.no_result_reason == "no_result:matched=2,parsed=1"


def test_parse_result_empty_stdout_counts_zero():
    out = gs.parse_result("")
    assert out.result is None and out.no_result_reason == "no_result:matched=0,parsed=0"


def test_parse_result_line_skips_deeply_nested_line_without_raising():
    # RecursionError from json.loads is not a JSONDecodeError — exercises the broad except
    assert gs.parse_result_line("[" * 20000 + '"gpusig"') is None


def test_parse_result_line_drops_line_over_length_cap():
    # The same valid, schema-matching line parses just under the cap and is dropped
    # just over it: only the length decides.
    def line(pad: int) -> str:
        return json.dumps({"gpusig": 2, "scheme": gs.GPUSIG_SCHEME, "pad": "x" * pad})

    under = line(gs.MAX_RESULT_LINE - 60)
    assert len(under) <= gs.MAX_RESULT_LINE
    assert gs.parse_result_line(under)["gpusig"] == 2
    over = line(gs.MAX_RESULT_LINE)
    assert len(over) > gs.MAX_RESULT_LINE
    assert gs.parse_result_line(over) is None


def test_seal_within_bounds_caps_each_field():
    assert gs.seal_within_bounds("m", "s") is True
    assert gs.seal_within_bounds("m" * (gs.MAX_SEAL_FIELD + 1), "s") is False
    assert gs.seal_within_bounds("m", "s" * (gs.MAX_SEAL_FIELD + 1)) is False


# ---- wall clock ------------------------------------------------------------------


def test_wall_clock_is_timeout_times_waves():
    # 14 cards at 8 concurrent = 2 waves: an honest host needs up to 2 x timeout
    assert gs.wall_clock_seconds(14, 120, 8) == 240.0
    assert gs.wall_clock_seconds(8, 120, 8) == 120.0
    assert gs.wall_clock_seconds(1, 120, 8) == 120.0
    assert gs.wall_clock_seconds(17, 60, 8) == 180.0


def test_wall_clock_never_below_one_wave():
    assert gs.wall_clock_seconds(0, 120, 8) == 120.0
    assert gs.wall_clock_seconds(4, 120, 0) == 120.0 * 4  # a zero cap means one at a time


# ---- envelope ----------------------------------------------------------------


def test_envelope_uncalibrated_does_not_gate():
    # the envelope ships empty (no class calibrated) -> never fail an honest host on a guess
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


def test_evaluate_card_non_numeric_metric_is_bad_output():
    # the verifier .so is separately versioned: a shape drift must become an unsealed
    # verdict, never an exception into the pipeline
    bad = {"sealed": True, "tflops": "not-a-number", "gbps": 1.0, "device": 0, "kernel_uuid": "x"}
    v = gs.evaluate_card(bad, None, 0)
    assert not v.ok and v.reasons == ["verifier_bad_output"]


def test_evaluate_card_non_dict_verdict_fails_closed():
    assert gs.evaluate_card("not-a-dict", None, 0).ok is False


def test_evaluate_card_coerces_non_string_uuid():
    # a non-string kernel_uuid is coerced to "" so summarize()'s sort/join never raises
    coerced = gs.evaluate_card({**_sealed(), "kernel_uuid": 123}, None, 0)
    assert coerced.kernel_uuid == ""
    gs.summarize([coerced], elapsed_seconds=1.0, claimed_count=1, wall_clock_seconds=WALL)


def test_evaluate_card_device_mismatch():
    v = gs.evaluate_card(_sealed(device=2), None, 0, have_kernel=True)
    assert not v.ok and "device_mismatch" in v.reasons


@pytest.mark.parametrize("device", [None, "0", 0.0, True, [0]])
def test_evaluate_card_missing_or_non_int_device_is_bad_output(device):
    # The regression: a verdict with no `device` (schema drift) scored device_mismatch
    # on every card; it is bad output, like a non-numeric tflops.
    verdict = _sealed()
    if device is None:
        verdict.pop("device")
    else:
        verdict["device"] = device
    v = gs.evaluate_card(verdict, None, 0, have_kernel=True)
    assert not v.ok and v.reasons == ["verifier_bad_output"]


def test_evaluate_card_blind_card_is_a_reason():
    # A provider that hides /proc/driver/nvidia from the executor container makes the
    # prober emit an empty kernel_uuid (have_kernel False). That must be a per-card
    # reason, not a silent pass.
    blind = gs.evaluate_card(_sealed(uuid=""), None, 0, have_kernel=False)
    assert not blind.ok and "no_kernel_identity" in blind.reasons


def test_evaluate_card_forged_have_kernel_over_empty_uuid_still_flagged():
    # have_kernel is a plain stdout field the miner can flip; the sealed kernel_uuid is
    # the one the verifier authenticated.
    forged = gs.evaluate_card(_sealed(uuid=""), None, 0, have_kernel=True)
    assert not forged.ok and "no_kernel_identity" in forged.reasons


def test_summarize_fails_node_when_every_card_is_blind():
    forged_node = gs.summarize(
        [gs.evaluate_card(_sealed(uuid=""), None, i, have_kernel=True) for i in range(4)],
        elapsed_seconds=1.0,
        claimed_count=4,
        wall_clock_seconds=WALL,
    )
    assert forged_node.passed is False


def test_evaluate_card_with_identity_has_no_blind_reason():
    seen = gs.evaluate_card(_sealed(uuid="GPU-real"), None, 0, have_kernel=True)
    assert seen.ok and "no_kernel_identity" not in seen.reasons


def test_evaluate_card_calibrated_below_floor_fails(monkeypatch):
    monkeypatch.setitem(
        gs.GPU_SIGNATURE_ENVELOPE,
        "NVIDIA H200",
        gs.SignatureEnvelope(tflops_min=30.0, gbps_min=2000.0, calibrated=True),
    )
    v = gs.evaluate_card(_sealed(tflops=5.0, gbps=500.0), "NVIDIA H200", 0, have_kernel=True)
    assert not v.ok and any("gbps" in r for r in v.reasons)


# ---- aggregate / count spoof -------------------------------------------------


def _good(n):
    return [gs.evaluate_card(_sealed(uuid=f"GPU-{i}"), None, i, have_kernel=True) for i in range(n)]


def test_summarize_all_good():
    out = gs.summarize(_good(4), elapsed_seconds=10.0, claimed_count=4, wall_clock_seconds=WALL)
    assert out.passed and out.verified_count == 4 and not out.reasons
    assert out.claimed_count == 4


def test_summarize_count_lie_device_selection():
    # claims 4, only card 0 real: cards 1..3 come back unsealed (device-selection error)
    verdicts = [gs.evaluate_card(_sealed(uuid="GPU-0"), None, 0, have_kernel=True)]
    for i in range(1, 4):
        verdicts.append(
            gs.evaluate_card({"sealed": False, "reason": "prober_error:set_device"}, None, i)
        )
    out = gs.summarize(verdicts, elapsed_seconds=10.0, claimed_count=4, wall_clock_seconds=WALL)
    assert not out.passed and out.verified_count == 1


def test_summarize_duplicate_uuid_is_count_spoof():
    # one physical card answering for four claimed cards -> same kernel UUID
    verdicts = [
        gs.evaluate_card(_sealed(uuid="GPU-SAME"), None, i, have_kernel=True) for i in range(4)
    ]
    out = gs.summarize(verdicts, elapsed_seconds=10.0, claimed_count=4, wall_clock_seconds=WALL)
    assert not out.passed and any("duplicate kernel UUID" in r for r in out.reasons)


def test_summarize_over_wall_clock():
    out = gs.summarize(_good(2), elapsed_seconds=999.0, claimed_count=2, wall_clock_seconds=120.0)
    assert not out.passed and out.over_wall_clock


def test_summarize_two_wave_host_within_derived_ceiling_passes():
    # 14 cards at 8 concurrent legitimately take two waves; with the derived ceiling
    # (2 x 120 s) an elapsed 200 s is not a serialisation signal.
    ceiling = gs.wall_clock_seconds(14, 120, 8)
    out = gs.summarize(
        _good(14), elapsed_seconds=200.0, claimed_count=14, wall_clock_seconds=ceiling
    )
    assert out.passed and not out.over_wall_clock


# ---- kernel-vs-NVML UUID cross-check (the 4th count/type-spoof signal) -------


def test_kernel_uuid_mismatch_flags_uuid_nvml_never_claimed():
    # a userspace NVML shim: the kernel reports a card NVML did not advertise
    assert gs.kernel_uuid_mismatch(["GPU-real", "GPU-shimmed"], ["GPU-real"]) is True


def test_kernel_uuid_mismatch_ignores_failed_cards_and_subsets():
    # some cards failed (empty, so absent from kernel_uuids) -> not a shim signal
    assert gs.kernel_uuid_mismatch(["GPU-a"], ["GPU-a", "GPU-b"]) is False
    # every kernel UUID is claimed -> clean
    assert gs.kernel_uuid_mismatch(["GPU-a", "GPU-b"], ["GPU-b", "GPU-a"]) is False


def test_kernel_uuid_mismatch_nothing_authenticated_is_clean():
    assert gs.kernel_uuid_mismatch([], ["GPU-a"]) is False
