"""Orchestration tests for GpuSignatureCheck (DAH-3137) plus fixture tests of the
committed libgpusig.so.

The check is driven with a FakeRunner (the test_gpu_fault_probe_check pattern) and a
recording fake verifier, so the skip reasons, the binary probe, the per-slot command,
the nonce handed to the verifier, the runner error path, the schema-mismatch reason,
the derived wall clock and the enforcement wording are all pinned without a GPU.

The seal fixture tests load the COMMITTED neurons/validators/libgpusig.so (Linux
x86_64 only, skipped elsewhere) with a genuine dev-key seal captured from the
private prober harness: good verifies, a forged digit fails, a replay under another
nonce fails, a sealed `nan` is refused as a bad number. The fixture reveals nothing
about the algorithm (it is an HMAC output), and the .so is what the image ships.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from neurons.validators.src.services import gpu_signature as gs
from neurons.validators.src.services.task.checks import gpu_signature as module
from neurons.validators.src.services.task.checks.gpu_signature import GpuSignatureCheck
from neurons.validators.src.services.task.messages import GpuSignatureMessages as Msg
from neurons.validators.src.services.task.runner import SSHCommandResult

from tests.helpers import build_context_config, build_services, build_state

CHECK = "neurons.validators.src.services.task.checks.gpu_signature"
VALIDATORS_DIR = Path(__file__).resolve().parents[1]
RELEASE_SO = VALIDATORS_DIR / "libgpusig.so"
DEV_KEY = "lium-gpu-sig-dev-key-do-not-use-in-prod"

# Genuine dev-key seal captured from the private prober harness (celium-gpu-verifier
# `test_seal seal ...`). Nonce "c"*64, uuid GPU-fixture-..., pci 0000:41:00.0.
FIX_NONCE = "c" * 64
FIX_MSG = (
    "lium.gpusig.v2;" + FIX_NONCE + ";GPU-fixture-0000-1111-2222-333344445555;0000:41:00.0;"
    "0;81920;42.500;1800.250;800.000"
)
FIX_SIG = "f75cdfe0da433183e5e22f79d4bf66faaba49f30db38e374906b076868e812d5"


@contextmanager
def gate(*, enabled=True, enforce=False, timeout=120, max_concurrent=8):
    with patch(f"{CHECK}.settings") as s:
        s.ENABLE_GPU_SIGNATURE_CHECK = enabled
        s.GPU_SIGNATURE_ENFORCEMENT_ENABLED = enforce
        s.GPU_SIGNATURE_KEY = DEV_KEY
        s.GPU_SIGNATURE_BINARY_RELATIVE = "bin/gpu_sig"
        s.GPU_SIGNATURE_MAX_CONCURRENT = max_concurrent
        s.GPU_SIGNATURE_TIMEOUT_SECONDS = timeout
        yield s


class FakeRunner:
    """Answers the binary probe and every per-slot prober command from canned data."""

    def __init__(
        self,
        slot_stdout="",
        *,
        exit_code=0,
        error_type=None,
        probe_stdout="GPUSIG_PRESENT",
        probe_error=None,
    ):
        self.slot_stdout = slot_stdout
        self.exit_code = exit_code
        self.error_type = error_type
        self.probe_stdout = probe_stdout
        self.probe_error = probe_error
        self.calls: list[dict] = []

    async def run(self, cmd, *, timeout=60, check=False, retryable=True, stdin_text=None):
        self.calls.append({"cmd": cmd, "timeout": timeout, "retryable": retryable})
        probe = cmd.startswith("test -x ")
        now = datetime.now(UTC)
        err = self.probe_error if probe else self.error_type
        return SSHCommandResult(
            command=cmd,
            command_id="cid",
            exit_code=0 if probe else self.exit_code,
            stdout=self.probe_stdout if probe else self.slot_stdout,
            stderr="",
            duration_ms=10,
            started_at=now,
            finished_at=now,
            success=err is None,
            error_type=err,
            error_message=err,
        )

    @property
    def slot_cmds(self):
        return [c["cmd"] for c in self.calls if not c["cmd"].startswith("test -x ")]


class FakeVerifier:
    """Records what the check hands to libgpusig.so and answers with a canned verdict."""

    calls: list[tuple] = []
    verdict: dict = {}

    def __init__(self, lib_path=None):
        self.lib_path = lib_path

    def verify_seal(self, master_key, expected_nonce, msg, sig):
        FakeVerifier.calls.append((master_key, expected_nonce, msg, sig))
        return dict(FakeVerifier.verdict)


def sealed_verdict(uuid="GPU-1", device=0):
    return {
        "sealed": True,
        "reason": "",
        "scheme": "lium.gpusig.v2",
        "nonce": "x",
        "kernel_uuid": uuid,
        "pci": "0000:65:00.0",
        "device": device,
        "vram_mb": 81920,
        "tflops": 40.0,
        "gbps": 900.0,
        "ms": 800.0,
    }


def prober_line(**over) -> str:
    obj = {
        "gpusig": 2,
        "ok": True,
        "scheme": "lium.gpusig.v2",
        "have_kernel": True,
        "msg": "<m>",
        "sig": "<s>",
    }
    obj.update(over)
    return "driver noise\n" + json.dumps(obj) + "\n"


def make_ctx(context_factory, runner, *, gpu_count=1, model="NVIDIA H200"):
    state = build_state(
        specs={"gpu": {"count": gpu_count}},
        gpu_count=gpu_count,
        gpu_model=model,
        gpu_details=[{"uuid": f"GPU-{i}"} for i in range(gpu_count)],
    )
    return context_factory(
        services=build_services(), config=build_context_config(), state=state, runner=runner
    )


@pytest.fixture(autouse=True)
def _fake_verifier():
    FakeVerifier.calls = []
    FakeVerifier.verdict = sealed_verdict()
    with patch(f"{CHECK}.GpuSigVerifier", FakeVerifier):
        yield


# ---- skips -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_runs_nothing(context_factory):
    runner = FakeRunner()
    ctx = make_ctx(context_factory, runner)
    with gate(enabled=False):
        result = await GpuSignatureCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == Msg.SKIPPED.reason
    assert result.event.what_we_saw["skipped"] == "disabled"
    assert runner.calls == []


@pytest.mark.asyncio
async def test_no_gpus_skips(context_factory):
    runner = FakeRunner()
    ctx = make_ctx(context_factory, runner, gpu_count=0)
    with gate():
        result = await GpuSignatureCheck().run(ctx)
    assert result.event.what_we_saw["skipped"] == "no_gpus"
    assert runner.calls == []


@pytest.mark.asyncio
async def test_binary_absent_skips_and_names_the_path(context_factory):
    runner = FakeRunner(probe_stdout="")
    ctx = make_ctx(context_factory, runner)
    with gate():
        result = await GpuSignatureCheck().run(ctx)
    assert result.event.what_we_saw["skipped"] == "binary_absent"
    assert result.event.what_we_saw["binary_path"] == "/root/app/bin/gpu_sig"
    assert runner.slot_cmds == []
    (probe,) = runner.calls
    assert probe["cmd"] == "test -x /root/app/bin/gpu_sig && echo GPUSIG_PRESENT"
    assert probe["retryable"] is False


@pytest.mark.asyncio
async def test_binary_probe_channel_error_is_not_absent(context_factory):
    # The regression: a dead channel used to read as binary_absent.
    runner = FakeRunner(probe_error="timeout")
    ctx = make_ctx(context_factory, runner)
    with gate():
        result = await GpuSignatureCheck().run(ctx)
    assert result.event.what_we_saw["skipped"] == "binary_probe_failed"
    assert runner.slot_cmds == []


@pytest.mark.asyncio
async def test_verifier_unavailable_skips(context_factory):
    def boom(lib_path=None):
        raise OSError("libgpusig.so: cannot open shared object file")

    runner = FakeRunner()
    ctx = make_ctx(context_factory, runner)
    with gate(), patch(f"{CHECK}.GpuSigVerifier", boom):
        result = await GpuSignatureCheck().run(ctx)
    assert result.event.what_we_saw["skipped"] == "verifier_unavailable"
    assert "libgpusig.so" in result.event.what_we_saw["error"]


# ---- per-slot command and the nonce binding ------------------------------------


@pytest.mark.asyncio
async def test_one_pinned_command_per_claimed_slot_with_fresh_nonces(context_factory):
    runner = FakeRunner(prober_line())
    ctx = make_ctx(context_factory, runner, gpu_count=3)
    with gate(timeout=77):
        result = await GpuSignatureCheck().run(ctx)
    cmds = sorted(runner.slot_cmds)
    assert len(cmds) == 3
    nonces = []
    for slot, cmd in enumerate(cmds):
        m = re.fullmatch(
            rf"CUDA_VISIBLE_DEVICES={slot} /root/app/bin/gpu_sig --nonce ([0-9a-f]{{64}}) --device 0",
            cmd,
        )
        assert m, cmd
        nonces.append(m.group(1))
    assert len(set(nonces)) == 3, "every slot gets its own nonce"
    slot_calls = [c for c in runner.calls if not c["cmd"].startswith("test -x ")]
    assert all(c["timeout"] == 77 and c["retryable"] is False for c in slot_calls)
    assert result.event.what_we_saw["claimed_count"] == 3


@pytest.mark.asyncio
async def test_verifier_receives_the_slot_nonce_and_the_raw_seal(context_factory):
    # The regression a copy/paste swap would cause: the nonce handed to verify_seal must
    # be the one in that slot's command, and msg/sig must be passed through untouched.
    runner = FakeRunner(prober_line(msg="record-bytes", sig="ab" * 32))
    ctx = make_ctx(context_factory, runner, gpu_count=2)
    with gate():
        await GpuSignatureCheck().run(ctx)
    issued = {re.search(r"--nonce ([0-9a-f]{64})", c).group(1) for c in runner.slot_cmds}
    assert len(FakeVerifier.calls) == 2
    for master_key, nonce, msg, sig in FakeVerifier.calls:
        assert master_key == DEV_KEY
        assert nonce in issued
        assert msg == "record-bytes" and sig == "ab" * 32
    assert {c[1] for c in FakeVerifier.calls} == issued


@pytest.mark.asyncio
async def test_runner_error_type_becomes_ssh_reason(context_factory):
    runner = FakeRunner("", error_type="timeout")
    ctx = make_ctx(context_factory, runner)
    with gate():
        result = await GpuSignatureCheck().run(ctx)
    assert result.passed is True  # observe-only
    assert result.event.reason_code == Msg.FAILED.reason
    (card,) = result.event.what_we_saw["per_card"]
    assert card["reasons"] == ["ssh:timeout"]
    assert FakeVerifier.calls == []


@pytest.mark.asyncio
async def test_schema_mismatch_reports_matched_and_parsed_counts(context_factory):
    # A renamed prober (different version or tag) is a no_result WITH counts, so the
    # log tells "printed nothing" from "printed something we no longer recognise".
    runner = FakeRunner(prober_line(gpusig=3))
    ctx = make_ctx(context_factory, runner)
    with gate():
        result = await GpuSignatureCheck().run(ctx)
    (card,) = result.event.what_we_saw["per_card"]
    assert card["reasons"] == ["no_result:matched=1,parsed=1"]
    assert FakeVerifier.calls == []


@pytest.mark.asyncio
async def test_prober_error_line_becomes_prober_error_reason(context_factory):
    runner = FakeRunner(
        prober_line(ok=False, error="set_device:cudaErrorInvalidDevice"), exit_code=3
    )
    ctx = make_ctx(context_factory, runner)
    with gate():
        result = await GpuSignatureCheck().run(ctx)
    (card,) = result.event.what_we_saw["per_card"]
    assert card["reasons"] == ["prober_error:set_device:cudaErrorInvalidDevice"]


@pytest.mark.asyncio
async def test_oversized_seal_never_reaches_the_verifier(context_factory):
    runner = FakeRunner(prober_line(msg="m" * (gs.MAX_SEAL_FIELD + 1)))
    ctx = make_ctx(context_factory, runner)
    with gate():
        result = await GpuSignatureCheck().run(ctx)
    (card,) = result.event.what_we_saw["per_card"]
    assert card["reasons"] == ["oversized_seal"]
    assert FakeVerifier.calls == []


# ---- node verdict, wall clock, enforcement wording -----------------------------


@pytest.mark.asyncio
async def test_all_cards_sealed_passes_and_reports_ceiling(context_factory):
    runner = FakeRunner(prober_line())
    FakeVerifier.verdict = sealed_verdict(uuid="GPU-0")
    ctx = make_ctx(context_factory, runner, gpu_count=1)
    with gate(timeout=120, max_concurrent=8):
        result = await GpuSignatureCheck().run(ctx)
    what = result.event.what_we_saw
    assert result.event.reason_code == Msg.OK.reason
    assert what["passed"] is True and what["verified_count"] == 1
    assert what["wall_clock_ceiling_seconds"] == 120.0


@pytest.mark.asyncio
async def test_wall_clock_ceiling_follows_timeout_and_waves(context_factory):
    runner = FakeRunner(prober_line())
    ctx = make_ctx(context_factory, runner, gpu_count=14)
    with gate(timeout=100, max_concurrent=8):
        result = await GpuSignatureCheck().run(ctx)
    assert result.event.what_we_saw["wall_clock_ceiling_seconds"] == 200.0


@pytest.mark.asyncio
async def test_failure_in_observe_mode_is_a_warning_and_passes(context_factory):
    runner = FakeRunner(prober_line())
    FakeVerifier.verdict = {"sealed": False, "reason": "seal_mismatch"}
    ctx = make_ctx(context_factory, runner)
    with gate(enforce=False):
        result = await GpuSignatureCheck().run(ctx)
    assert result.passed is True
    assert result.event.reason_code == Msg.FAILED.reason
    assert result.event.severity == "warning"
    assert "until enforcement is wired" in result.event.impact
    assert result.event.what_we_saw["would_enforce"] is False


@pytest.mark.asyncio
async def test_failure_under_enforcement_flag_is_an_error_with_honest_impact(context_factory):
    # The regression: severity went to error while the impact text still described the
    # flag-off state. The event now says the flag is on and the score gate is not wired.
    runner = FakeRunner(prober_line())
    FakeVerifier.verdict = {"sealed": False, "reason": "seal_mismatch"}
    ctx = make_ctx(context_factory, runner)
    with gate(enforce=True):
        result = await GpuSignatureCheck().run(ctx)
    assert result.passed is True  # still observe-only: the score gate is a follow-up
    assert result.event.severity == "error"
    assert result.event.impact == Msg.FAILED_ENFORCEMENT_FLAG_IMPACT
    assert "NOT affected" in result.event.impact
    assert result.event.what_we_saw["would_enforce"] is True


@pytest.mark.asyncio
async def test_kernel_uuid_nvml_never_claimed_flags_the_node(context_factory):
    runner = FakeRunner(prober_line())
    FakeVerifier.verdict = sealed_verdict(uuid="GPU-shimmed")
    ctx = make_ctx(context_factory, runner, gpu_count=1)  # NVML claims GPU-0
    with gate():
        result = await GpuSignatureCheck().run(ctx)
    what = result.event.what_we_saw
    assert what["kernel_vs_nvml_uuid_mismatch"] is True
    assert "kernel_vs_nvml_uuid_mismatch" in what["reasons"]
    assert what["passed"] is False


# ---- the committed libgpusig.so (Linux x86_64 only) ---------------------------


def _release_verifier():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        pytest.skip("the committed libgpusig.so is a Linux x86_64 ELF")
    from neurons.validators.src.services.gpusig_validator import GpuSigVerifier

    return GpuSigVerifier(str(RELEASE_SO))


def test_committed_so_matches_recorded_digest():
    # Provenance: the Dockerfile's `sha256sum -c` uses this same file; a blob-only
    # change without a matching digest line fails here too.
    recorded = (VALIDATORS_DIR / "libgpusig.so.sha256").read_text().split()[0]
    assert hashlib.sha256(RELEASE_SO.read_bytes()).hexdigest() == recorded


def test_committed_so_accepts_a_genuine_seal():
    v = _release_verifier()
    r = v.verify_seal(DEV_KEY, FIX_NONCE, FIX_MSG, FIX_SIG)
    assert r["sealed"] is True and r["reason"] == ""
    assert r["kernel_uuid"] == "GPU-fixture-0000-1111-2222-333344445555"
    assert r["device"] == 0 and r["vram_mb"] == 81920
    assert abs(r["tflops"] - 42.5) < 1e-6 and abs(r["gbps"] - 1800.25) < 1e-6


def test_committed_so_rejects_a_forged_number():
    v = _release_verifier()
    forged = FIX_MSG.replace("42.500", "942.500")
    r = v.verify_seal(DEV_KEY, FIX_NONCE, forged, FIX_SIG)
    assert r["sealed"] is False and r["reason"] == "seal_mismatch" and "tflops" not in r


def test_committed_so_rejects_a_replay_under_another_nonce():
    v = _release_verifier()
    r = v.verify_seal(DEV_KEY, "d" * 64, FIX_MSG, FIX_SIG)
    assert r["sealed"] is False and r["reason"] == "nonce_mismatch"


def test_committed_so_rejects_a_wrong_key():
    v = _release_verifier()
    r = v.verify_seal("not-the-key", FIX_NONCE, FIX_MSG, FIX_SIG)
    assert r["sealed"] is False and r["reason"] == "seal_mismatch"


def test_committed_so_refuses_non_finite_token_with_valid_json():
    # Sealed under the right key, but the tflops token is "nan": refused as a bad number
    # (the wrapper never sees non-JSON, never raises).
    v = _release_verifier()
    msg = FIX_MSG.replace("42.500", "nan")
    r = v.verify_seal(DEV_KEY, FIX_NONCE, msg, FIX_SIG)
    assert r["sealed"] is False and r["reason"] == "bad_number"


def test_wrapper_output_read_is_bounded_and_stops_at_nul():
    # The regression: `ctypes.string_at(ptr)` read until a NUL with no size. The wrapper
    # now reads byte by byte, stops at the first NUL and raises past OUT_MAX bytes; it
    # frees the buffer either way. Any platform: a stub library stands in for the .so.
    import ctypes

    from neurons.validators.src.services import gpusig_validator as gv

    class StubLib:
        def __init__(self):
            self.freed = []

        def gpusig_str_free(self, ptr):
            self.freed.append(ptr)

    v = gv.GpuSigVerifier.__new__(gv.GpuSigVerifier)
    v.lib = StubLib()

    good = ctypes.create_string_buffer(b'{"sealed": false, "reason": "x"}')
    assert (
        v._take(ctypes.cast(good, ctypes.POINTER(ctypes.c_char)))
        == '{"sealed": false, "reason": "x"}'
    )

    unterminated = (ctypes.c_char * (gv.OUT_MAX + 64))(*([b"a"] * (gv.OUT_MAX + 64)))
    with pytest.raises(ValueError):
        v._take(ctypes.cast(unterminated, ctypes.POINTER(ctypes.c_char)))
    assert len(v.lib.freed) == 2


def test_evaluate_card_scores_the_committed_so_verdict():
    v = _release_verifier()
    r = v.verify_seal(DEV_KEY, FIX_NONCE, FIX_MSG, FIX_SIG)
    card = module.evaluate_card(r, "NVIDIA H200", 0, have_kernel=True)
    assert card.ok and card.kernel_uuid == "GPU-fixture-0000-1111-2222-333344445555"
