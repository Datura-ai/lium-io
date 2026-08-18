import json
import logging
import time
from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.payload_models.payloads import GpuPowerLimit
from neurons.validators.src.services.gpu_power_limit import (
    GpuPowerReadback,
    GpuPowerRestoreRecord,
    GpuPowerState,
    _clamp_watts,
    _parse_power_readback_csv,
    _parse_power_state_csv,
    _pod_index_key,
    _restore_key,
    apply_filler_gpu_power_limits,
    raise_low_power_limits_to_default,
    read_gpu_power_restore_records,
    restore_all_host_gpu_power_limits,
    restore_filler_pod_gpu_power_limits,
    restore_tracked_gpu_power_limits,
)


def _limits(**watts_by_uuid: int) -> list[GpuPowerLimit]:
    return [GpuPowerLimit(gpu_uuid=uuid.replace("_", "-"), watts=watts) for uuid, watts in watts_by_uuid.items()]


def _record(gpu_uuid: str, watts: int, executor_id: str = "executor-1") -> str:
    return GpuPowerRestoreRecord(
        gpu_uuid=gpu_uuid, watts=watts, pod_id=POD_ID, executor_id=executor_id, capped_at=time.time()
    ).model_dump_json()


# uuid, current, default, min, max
STATE_CSV = "GPU-a, 130, 400, 100, 400\nGPU-b, 250, 250, 100, 250\n"
POD_ID = "pod-1"
EXECUTOR_ID = "executor-1"


class FakeRun:
    def __init__(self, stdout: str = "", stderr: str = "", exit_status: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_status = exit_status


def fake_ssh(*responses: FakeRun) -> AsyncMock:
    ssh = AsyncMock()
    ssh.run.side_effect = responses
    return ssh


def _commands(ssh: AsyncMock) -> list[str]:
    return [call.args[0] for call in ssh.run.call_args_list]


def _logged_field(caplog: pytest.LogCaptureFixture, field: str) -> list[object]:
    """Values of one structured field across every log record that carries it."""
    return [
        record.msg.extra[field]
        for record in caplog.records
        if hasattr(record.msg, "extra") and field in record.msg.extra
    ]


def _set_ok(readback_watts: int, persistence: str = "Enabled") -> list[FakeRun]:
    """SSH responses for one successful verified set: -pm 1, -pl, readback confirming the target."""
    return [FakeRun(), FakeRun(), FakeRun(stdout=f"{readback_watts}.00, {persistence}\n")]


def _set_commands(gpu_uuid: str, watts: int) -> list[str]:
    """The exact command triple one verified set issues."""
    return [
        f"nvidia-smi -i {gpu_uuid} -pm 1",
        f"nvidia-smi -i {gpu_uuid} -pl {watts}",
        f"nvidia-smi -i {gpu_uuid} --query-gpu=power.limit,persistence_mode --format=csv,noheader,nounits",
    ]


class FakeRedis:
    def __init__(self, initial: dict[str, str] | None = None):
        self.store: dict[str, str] = dict(initial or {})

    async def set(self, key: str, value: str) -> None:
        self.store[key] = value

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)


# ---------------------------- _parse_power_state_csv (pure) ----------------------------


def test_parse_power_state_csv_reads_all_fields() -> None:
    state = _parse_power_state_csv(STATE_CSV)
    assert state == {
        "GPU-a": GpuPowerState(current_watts=130, min_watts=100, max_watts=400, default_watts=400),
        "GPU-b": GpuPowerState(current_watts=250, min_watts=100, max_watts=250, default_watts=250),
    }


def test_parse_power_state_csv_rounds_fractional_watts() -> None:
    state = _parse_power_state_csv("GPU-a, 127.40, 300.00, 100.00, 209.30\n")
    assert state["GPU-a"] == GpuPowerState(
        current_watts=127, min_watts=100, max_watts=209, default_watts=300
    )


def test_parse_power_state_csv_skips_line_without_current() -> None:
    # current is required to record a pre-cap value; a GPU reporting "[N/A]" current is dropped.
    state = _parse_power_state_csv("GPU-a, 130, 400, 100, 400\ngarbage\nGPU-x, [N/A], 400, 100, 400\n")
    assert list(state) == ["GPU-a"]


def test_parse_power_state_csv_keeps_gpu_with_na_bounds() -> None:
    # H100-style: default/min/max can be "[N/A]" — the GPU stays usable, bounds become None.
    state = _parse_power_state_csv("GPU-h, 350, [N/A], [N/A], [N/A]\n")
    assert state["GPU-h"] == GpuPowerState(
        current_watts=350, min_watts=None, max_watts=None, default_watts=None
    )


# ---------------------------- _clamp_watts (pure) ----------------------------


def test_clamp_watts_within_bounds_unchanged() -> None:
    state = GpuPowerState(current_watts=400, min_watts=100, max_watts=400)
    assert _clamp_watts(209, state) == 209


def test_clamp_watts_below_min_pinned_to_min() -> None:
    state = GpuPowerState(current_watts=140, min_watts=100, max_watts=140)
    assert _clamp_watts(98, state) == 100


def test_clamp_watts_above_max_pinned_to_max() -> None:
    state = GpuPowerState(current_watts=250, min_watts=100, max_watts=250)
    assert _clamp_watts(300, state) == 250


def test_clamp_watts_skips_na_bounds() -> None:
    # bounds that came back "[N/A]" (None) are not applied — the target passes through unclamped.
    state = GpuPowerState(current_watts=350, min_watts=None, max_watts=None)
    assert _clamp_watts(217, state) == 217


# ---------------------------- _parse_power_readback_csv (pure) ----------------------------


def test_parse_power_readback_reads_watts_and_enabled_persistence() -> None:
    assert _parse_power_readback_csv("315.00, Enabled\n") == GpuPowerReadback(
        watts=315, persistence_enabled=True
    )


def test_parse_power_readback_reads_disabled_persistence() -> None:
    assert _parse_power_readback_csv("315.00, Disabled\n") == GpuPowerReadback(
        watts=315, persistence_enabled=False
    )


def test_parse_power_readback_keeps_watts_when_persistence_unreported() -> None:
    # A GPU that does not expose persistence_mode must still yield a usable readback.
    assert _parse_power_readback_csv("315.00, [N/A]\n") == GpuPowerReadback(
        watts=315, persistence_enabled=None
    )


def test_parse_power_readback_unreadable_watts_is_none() -> None:
    assert _parse_power_readback_csv("[N/A], Enabled\n") == GpuPowerReadback(
        watts=None, persistence_enabled=True
    )


# ---------------------------- persistence-mode verdict (DAH-2702) ----------------------------


@pytest.mark.asyncio
async def test_cap_logs_persistence_enabled_and_does_not_warn(caplog) -> None:
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(209, persistence="Enabled"))

    with caplog.at_level(logging.DEBUG):
        ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209), FakeRedis(), POD_ID, EXECUTOR_ID)

    assert ok is True
    assert set(_logged_field(caplog, "persistence_enabled")) == {True}
    assert not [record for record in caplog.records if record.levelno == logging.WARNING]


@pytest.mark.asyncio
async def test_cap_with_persistence_off_still_succeeds_but_warns(caplog) -> None:
    # Not fail-closed on purpose: a host that cannot hold persistence mode would lose PEARL entirely.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(209, persistence="Disabled"))

    with caplog.at_level(logging.DEBUG):
        ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209), FakeRedis(), POD_ID, EXECUTOR_ID)

    assert ok is True
    # ONE event, raised to WARNING — not a second log line, so counting capped GPUs still works.
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].msg.extra["persistence_enabled"] is False
    assert warnings[0].msg.extra["gpu_power_action"] == "cap"
    assert warnings[0].msg.extra["status"] == "ok"
    assert "persistence mode is off" in str(warnings[0].msg)


@pytest.mark.asyncio
async def test_cap_does_not_warn_when_persistence_is_unreported(caplog) -> None:
    # Unreported ("[N/A]", no column) is not the same as off: only an explicit Disabled warns,
    # otherwise every GPU that cannot report the field would look like a revert risk.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(209, persistence="[N/A]"))

    with caplog.at_level(logging.DEBUG):
        ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209), FakeRedis(), POD_ID, EXECUTOR_ID)

    assert ok is True
    assert _logged_field(caplog, "persistence_enabled") == [None]
    assert not [record for record in caplog.records if record.levelno == logging.WARNING]


@pytest.mark.asyncio
async def test_restore_with_persistence_off_does_not_warn(caplog) -> None:
    # Only a cap is at risk from a lost persistence mode: a restore sets the limit back UP, and a
    # driver unload lands on the default anyway. Warning here would inflate the cap-side signal.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(400, persistence="Disabled"))
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})

    with caplog.at_level(logging.DEBUG):
        restored = await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert restored == 1
    assert not [record for record in caplog.records if record.levelno == logging.WARNING]


# ---------------------------- apply_filler_gpu_power_limits (fail-closed) ----------------------------


@pytest.mark.asyncio
async def test_apply_stores_frozen_records_pod_index_sets_clamped_and_returns_true() -> None:
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(209), *_set_ok(250))
    redis = FakeRedis()

    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209, GPU_b=300), redis, POD_ID, EXECUTOR_ID)

    assert ok is True
    # pre-cap current limits are stored per gpu_uuid for later restore
    record_a = GpuPowerRestoreRecord.model_validate_json(redis.store[_restore_key("GPU-a")])
    record_b = GpuPowerRestoreRecord.model_validate_json(redis.store[_restore_key("GPU-b")])
    assert (record_a.watts, record_a.pod_id, record_a.executor_id) == (130, POD_ID, EXECUTOR_ID)
    assert (record_b.watts, record_b.pod_id, record_b.executor_id) == (250, POD_ID, EXECUTOR_ID)
    assert record_a.capped_at > 0
    # the pod index remembers which GPUs this pod capped, for delete-time restore
    assert json.loads(redis.store[_pod_index_key(POD_ID)]) == ["GPU-a", "GPU-b"]
    # targets are set (persistence mode first, readback verify after), clamped to hw max (GPU-b 300 -> 250)
    assert _commands(ssh)[1:] == _set_commands("GPU-a", 209) + _set_commands("GPU-b", 250)


@pytest.mark.asyncio
async def test_apply_never_overwrites_existing_record() -> None:
    # A leftover record means an earlier restore failed: it holds the TRUE original limit, while the
    # GPU's current limit is the old cap. Overwriting would ratchet the "original" down forever.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(209))
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})

    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209), redis, "pod-2", EXECUTOR_ID)

    assert ok is True
    assert json.loads(redis.store[_restore_key("GPU-a")])["watts"] == 400  # frozen, not 130
    assert _commands(ssh)[1:] == _set_commands("GPU-a", 209)


@pytest.mark.asyncio
async def test_apply_settles_at_hw_min_when_target_below_min() -> None:
    # 70% of a small card's TDP can fall below its hardware minimum limit -> settle at the minimum.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(100))
    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=90), FakeRedis(), POD_ID, EXECUTOR_ID)  # min is 100
    assert ok is True
    assert _commands(ssh)[1:] == _set_commands("GPU-a", 100)


@pytest.mark.asyncio
async def test_apply_fails_closed_when_gpu_missing() -> None:
    # A requested GPU not in nvidia-smi output -> refuse: no store, no -pl set, returns False.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV))
    redis = FakeRedis()

    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_missing=200, GPU_a=200), redis, POD_ID, EXECUTOR_ID)

    assert ok is False
    assert ssh.run.call_count == 1  # only the state query ran
    assert redis.store == {}


@pytest.mark.asyncio
async def test_apply_fails_closed_on_state_query_failure() -> None:
    ssh = fake_ssh(FakeRun(exit_status=1, stderr="no gpu"))
    redis = FakeRedis()
    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=200), redis, POD_ID, EXECUTOR_ID)
    assert ok is False
    assert ssh.run.call_count == 1
    assert redis.store == {}


@pytest.mark.asyncio
async def test_apply_fails_closed_when_record_cannot_be_stored() -> None:
    # Never lower a GPU without a persisted way to raise it back: if Redis can't store the pre-cap
    # record, set no cap and return False.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV))
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.side_effect = RuntimeError("redis down")
    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=200), redis, POD_ID, EXECUTOR_ID)
    assert ok is False
    assert ssh.run.call_count == 1  # only the state query ran; no -pl set


@pytest.mark.asyncio
async def test_apply_undoes_partial_cap_when_set_fails() -> None:
    # GPU-a gets capped, GPU-b's -pl hangs -> apply must uncap GPU-a and clear all state before
    # returning False (the filler never starts, so nothing else would restore the node).
    ssh = AsyncMock()
    ssh.run.side_effect = [
        FakeRun(stdout=STATE_CSV),   # state query
        *_set_ok(209),               # cap GPU-a ok (pm, -pl, readback)
        FakeRun(),                   # cap GPU-b: pm ok
        TimeoutError("nvidia-smi hung"),  # cap GPU-b: -pl fails
        FakeRun(stdout=STATE_CSV),   # undo: state query for before-values
        *_set_ok(130),               # undo: restore GPU-a
        *_set_ok(250),               # undo: restore GPU-b (record was stored before capping)
    ]
    redis = FakeRedis()

    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209, GPU_b=200), redis, POD_ID, EXECUTOR_ID)

    assert ok is False
    assert _commands(ssh)[7:] == _set_commands("GPU-a", 130) + _set_commands("GPU-b", 250)
    assert redis.store == {}  # records and pod index all cleared


@pytest.mark.asyncio
async def test_apply_fails_closed_when_limit_does_not_stick() -> None:
    # Live-repro bug (H100, 2026-07-13): with persistence mode off `nvidia-smi -pl` exits 0 but the
    # limit silently reverts to the old value. The readback verify must catch it: undo + False.
    ssh = AsyncMock()
    ssh.run.side_effect = [
        FakeRun(stdout=STATE_CSV),      # state query
        FakeRun(),                      # pm GPU-a ok
        FakeRun(),                      # -pl 209 reports success (lying)
        FakeRun(stdout="130.00\n"),     # readback: still 130 -> the cap did not stick
        FakeRun(stdout=STATE_CSV),      # undo: state query
        *_set_ok(130),                  # undo: restore GPU-a to its pre-cap 130 (sticks)
    ]
    redis = FakeRedis()

    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209), redis, POD_ID, EXECUTOR_ID)

    assert ok is False
    assert redis.store == {}  # undo cleared the record and pod index


@pytest.mark.asyncio
async def test_apply_succeeds_when_persistence_mode_fails_but_limit_sticks() -> None:
    # -pm can fail on exotic setups; the readback verify is the gate, not persistence mode itself.
    ssh = fake_ssh(
        FakeRun(stdout=STATE_CSV),
        FakeRun(exit_status=3, stderr="pm not supported"),  # -pm 1 fails
        FakeRun(),                                          # -pl ok
        FakeRun(stdout="209.00\n"),                         # readback confirms
    )
    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209), FakeRedis(), POD_ID, EXECUTOR_ID)
    assert ok is True


@pytest.mark.asyncio
async def test_apply_fails_closed_when_readback_fails() -> None:
    # An unverifiable cap counts as no cap: readback error -> undo + False.
    ssh = AsyncMock()
    ssh.run.side_effect = [
        FakeRun(stdout=STATE_CSV),           # state query
        FakeRun(),                           # pm ok
        FakeRun(),                           # -pl ok
        FakeRun(exit_status=1, stderr="driver hung"),  # readback fails
        FakeRun(stdout=STATE_CSV),           # undo: state query
        *_set_ok(130),                       # undo: restore GPU-a
    ]
    redis = FakeRedis()
    ok = await apply_filler_gpu_power_limits(ssh, _limits(GPU_a=209), redis, POD_ID, EXECUTOR_ID)
    assert ok is False
    assert redis.store == {}


# ---------------------------- restore_tracked_gpu_power_limits ----------------------------


@pytest.mark.asyncio
async def test_restore_sets_stored_values_and_clears_keys() -> None:
    # state query (for before-values) + one verified set per tracked GPU
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(400), *_set_ok(250))
    redis = FakeRedis({
        _restore_key("GPU-a"): _record("GPU-a", 400),
        _restore_key("GPU-b"): _record("GPU-b", 250),
    })

    restored = await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a", "GPU-b"])

    assert restored == 2
    assert _commands(ssh)[1:] == _set_commands("GPU-a", 400) + _set_commands("GPU-b", 250)
    assert redis.store == {}  # keys cleared only after successful restore


@pytest.mark.asyncio
async def test_restore_noop_without_records() -> None:
    # No records for these GPUs -> zero SSH traffic (the common create-path case).
    ssh = fake_ssh()
    restored = await restore_tracked_gpu_power_limits(ssh, FakeRedis(), ["GPU-a", "GPU-b"])
    assert restored == 0
    ssh.run.assert_not_called()


@pytest.mark.asyncio
async def test_restore_keeps_record_when_set_fails() -> None:
    # The record is deleted ONLY after the -pl restore succeeded — a failed restore keeps it for retry.
    ssh = AsyncMock()
    ssh.run.side_effect = [FakeRun(stdout=STATE_CSV), FakeRun(), TimeoutError("nvidia-smi hung")]
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})

    restored = await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert restored == 0
    assert _restore_key("GPU-a") in redis.store


@pytest.mark.asyncio
async def test_restore_keeps_record_when_readback_mismatches() -> None:
    # A restore that silently did not stick (persistence mode off) must not delete the record.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), FakeRun(), FakeRun(), FakeRun(stdout="130.00\n"))
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})

    restored = await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert restored == 0
    assert _restore_key("GPU-a") in redis.store


@pytest.mark.asyncio
async def test_restore_proceeds_without_before_values_when_state_query_fails() -> None:
    # The state query is only for logging before-values; its failure must not block the restore.
    ssh = fake_ssh(FakeRun(exit_status=1, stderr="driver hung"), *_set_ok(400))
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})

    restored = await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert restored == 1
    assert _commands(ssh)[1:] == _set_commands("GPU-a", 400)
    assert redis.store == {}


@pytest.mark.asyncio
async def test_restore_ignores_corrupt_record() -> None:
    # A corrupt record can't be restored and can't grant a check pass — it is logged and left alone.
    ssh = fake_ssh()
    redis = FakeRedis({_restore_key("GPU-a"): "not-json"})

    restored = await restore_tracked_gpu_power_limits(ssh, redis, ["GPU-a"])

    assert restored == 0
    ssh.run.assert_not_called()


# ---------------------------- restore_all_host_gpu_power_limits ----------------------------


@pytest.mark.asyncio
async def test_restore_all_host_enumerates_gpus_and_restores_tracked() -> None:
    # Whole-node create path: enumerate the host's GPUs, restore any tracked ones.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(400))
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})

    restored = await restore_all_host_gpu_power_limits(ssh, redis)

    assert restored == 1
    assert _commands(ssh)[1:] == _set_commands("GPU-a", 400)
    assert redis.store == {}


@pytest.mark.asyncio
async def test_restore_all_host_survives_state_query_failure() -> None:
    ssh = fake_ssh(FakeRun(exit_status=1, stderr="driver hung"))
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})

    restored = await restore_all_host_gpu_power_limits(ssh, redis)

    assert restored == 0
    assert _restore_key("GPU-a") in redis.store  # record survives for a later retry


# ---------------------------- restore_filler_pod_gpu_power_limits ----------------------------


@pytest.mark.asyncio
async def test_pod_restore_restores_only_this_pods_gpus_and_drops_index() -> None:
    # Another pod's fresh record (GPU-b) must survive a concurrent filler replacement.
    ssh = fake_ssh(FakeRun(stdout=STATE_CSV), *_set_ok(400))
    redis = FakeRedis({
        _restore_key("GPU-a"): _record("GPU-a", 400),
        _restore_key("GPU-b"): _record("GPU-b", 250),
        _pod_index_key(POD_ID): json.dumps(["GPU-a"]),
    })

    restored = await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID)

    assert restored == 1
    assert _commands(ssh)[1:] == _set_commands("GPU-a", 400)
    assert _restore_key("GPU-a") not in redis.store
    assert _restore_key("GPU-b") in redis.store  # untouched: belongs to another pod
    assert _pod_index_key(POD_ID) not in redis.store


@pytest.mark.asyncio
async def test_pod_restore_noop_without_index() -> None:
    # A filler we never capped (miner default job, DPHN): one Redis read, zero SSH traffic.
    ssh = fake_ssh()
    redis = FakeRedis()
    restored = await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID)
    assert restored == 0
    ssh.run.assert_not_called()


@pytest.mark.asyncio
async def test_pod_restore_keeps_record_but_drops_index_when_set_fails() -> None:
    # The per-GPU record (not the index) is the source of truth: a failed restore keeps the record
    # for the safety nets, while the index is dropped either way.
    ssh = AsyncMock()
    ssh.run.side_effect = [FakeRun(stdout=STATE_CSV), FakeRun(), TimeoutError("nvidia-smi hung")]
    redis = FakeRedis({
        _restore_key("GPU-a"): _record("GPU-a", 400),
        _pod_index_key(POD_ID): json.dumps(["GPU-a"]),
    })

    restored = await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID)

    assert restored == 0
    assert _restore_key("GPU-a") in redis.store
    assert _pod_index_key(POD_ID) not in redis.store


@pytest.mark.asyncio
async def test_pod_restore_drops_corrupt_index() -> None:
    ssh = fake_ssh()
    redis = FakeRedis({_pod_index_key(POD_ID): "not-json"})
    restored = await restore_filler_pod_gpu_power_limits(ssh, redis, POD_ID)
    assert restored == 0
    ssh.run.assert_not_called()
    assert redis.store == {}


# ---------------------------- read_gpu_power_restore_records (read_failed flag) ----------------------------


class BrokenRedis(FakeRedis):
    async def get(self, key: str) -> str | None:
        raise ConnectionError("redis down")


@pytest.mark.asyncio
async def test_read_records_reports_clean_read() -> None:
    redis = FakeRedis({_restore_key("GPU-a"): _record("GPU-a", 400)})
    result = await read_gpu_power_restore_records(redis, ["GPU-a", "GPU-b"])
    assert result.read_failed is False
    assert [record.gpu_uuid for record in result.records] == ["GPU-a"]


@pytest.mark.asyncio
async def test_read_records_flags_failed_read() -> None:
    result = await read_gpu_power_restore_records(BrokenRedis(), ["GPU-a"])
    assert result.read_failed is True
    assert result.records == []


# ---------------------------- raise_low_power_limits_to_default (state-free net) ----------------------------

# GPU-low is below 90% of default, GPU-band sits in the allowed 90-100% band,
# GPU-exact is exactly at the floor, GPU-nodefault exposes no default limit.
RAISE_STATE_CSV = (
    "GPU-low, 130, 400, 100, 400\n"
    "GPU-band, 380, 400, 100, 400\n"
    "GPU-exact, 360, 400, 100, 400\n"
    "GPU-nodefault, 130, [N/A], 100, 400\n"
)


@pytest.mark.asyncio
async def test_raise_lifts_only_below_floor_gpu_to_default() -> None:
    ssh = fake_ssh(FakeRun(stdout=RAISE_STATE_CSV), *_set_ok(400))
    raised = await raise_low_power_limits_to_default(
        ssh, EXECUTOR_ID, ["GPU-low", "GPU-band", "GPU-exact", "GPU-nodefault"]
    )
    assert raised == 1
    assert _commands(ssh)[1:] == _set_commands("GPU-low", 400)


@pytest.mark.asyncio
async def test_raise_covers_all_host_gpus_when_uuids_missing() -> None:
    ssh = fake_ssh(FakeRun(stdout=RAISE_STATE_CSV), *_set_ok(400))
    raised = await raise_low_power_limits_to_default(ssh, EXECUTOR_ID, None)
    assert raised == 1
    assert _commands(ssh)[1:] == _set_commands("GPU-low", 400)


@pytest.mark.asyncio
async def test_raise_noop_when_all_gpus_healthy() -> None:
    ssh = fake_ssh(FakeRun(stdout="GPU-band, 380, 400, 100, 400\n"))
    raised = await raise_low_power_limits_to_default(ssh, EXECUTOR_ID, ["GPU-band"])
    assert raised == 0
    assert len(_commands(ssh)) == 1  # only the state query, no -pl


@pytest.mark.asyncio
async def test_raise_survives_state_query_failure() -> None:
    ssh = fake_ssh(FakeRun(exit_status=1, stderr="nvml boom"))
    raised = await raise_low_power_limits_to_default(ssh, EXECUTOR_ID, ["GPU-low"])
    assert raised == 0


@pytest.mark.asyncio
async def test_raise_counts_only_verified_sets() -> None:
    # -pl reports success but the readback shows the limit did not stick -> not counted.
    ssh = fake_ssh(
        FakeRun(stdout="GPU-low, 130, 400, 100, 400\n"),
        FakeRun(),
        FakeRun(),
        FakeRun(stdout="130.00\n"),
    )
    raised = await raise_low_power_limits_to_default(ssh, EXECUTOR_ID, ["GPU-low"])
    assert raised == 0
