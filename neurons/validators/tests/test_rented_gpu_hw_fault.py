"""DAH-5869: a rented card that stays visible but fails (a new hardware Xid, or a PCIe link
downgrade read twice in a row) is named in the renter's incident, and a rented node whose
scrape keeps timing out posts gpu_unverified instead of nothing. The pod is never touched."""

import json
import subprocess
import sys

import pytest
from neurons.validators.src.services.task.checks import MachineSpecScrapeCheck, RentedGpuDropCheck
from neurons.validators.src.services.task.checks import gpu_memory_health
from neurons.validators.src.services.task.checks.machine_spec_scrape import SCRAPE_TIMEOUT_CYCLES
from neurons.validators.src.services.task.checks.rented_gpu_drop import (
    FAULT_GPU_HW,
    FAULT_GPU_UNVERIFIED,
)
from neurons.validators.src.services.task.messages import RentedGpuDropMessages as Msg

from core.config import settings
from tests.helpers import FakeRedis, build_context_config, build_state, default_executor
from tests.test_gpu_fault_probe_check import FakeRunner, _probe_namespace
from tests.test_gpu_memory_health_check import PROBE_SOURCE, _report
from tests.test_machine_spec_scrape_check import DummySSHCommandRunner, make_command_result
from tests.test_rented_gpu_drop_check import UUIDS, _clear_local_marks, _ctx, _rented, _services, _states  # noqa: F401

# --- the probe (runs on the executor) ---------------------------------------------------------------------------


def _suspects(**gpu):
    fn = _probe_namespace("nvml_card_suspects")["nvml_card_suspects"]
    return fn({"available": True, "gpus": [{"index": 1, "uuid": "GPU-1", **gpu}]})


def test_a_clock_slowdown_bit_is_not_a_suspect():
    # clocks are GpuThermalThrottleCheck's alone (review)
    assert _suspects(clock_event_reasons=0x8 | 0x40 | 0x80) == []


def test_a_narrow_or_slow_link_is_a_suspect_only_at_full_performance():
    assert _suspects(pcie_width=8, pcie_width_max=16, pstate=0) == ["gpu 1 (GPU-1): PCIe link x8 of x16"]
    # a card below P0 (a rented one between kernels too) narrows and slows its link by design (review)
    assert _suspects(pcie_width=8, pcie_width_max=16, pstate=8) == []
    assert _suspects(pcie_gen=1, pcie_gen_max=4, pstate=8) == []
    assert _suspects(pcie_gen=3, pcie_gen_max=4, pstate=0) == ["gpu 1 (GPU-1): PCIe link gen 3 of gen 4"]
    assert _suspects(pcie_width=16, pcie_width_max=16, pcie_gen=4, pcie_gen_max=4, pstate=0) == []
    # not supported (None) is not judged
    assert _suspects(pcie_width=None, pcie_width_max=16, pstate=0) == []


XID_GPU = {"index": 2, "uuid": "GPU-2", "pci_bus_id": "00000000:81:00.0"}
OLD = "NVRM: Xid (PCI:0000:81:00): 79, pid=1, GPU has fallen off the bus."
HW = "NVRM: Xid (PCI:0000:81:00): 119, pid=1, Timeout waiting for RPC from GSP!"
APP = "NVRM: Xid (PCI:0000:81:00): 13, pid=1, Graphics Exception"
# an application Xid outside SOFTWARE_XIDS: a renter's out-of-range access, not the card (review)
APP_UNLISTED = "NVRM: Xid (PCI:0000:81:00): 69, pid=1, Graphics Exception: Class error"
OTHER_CARD = "NVRM: Xid (PCI:0000:c1:00): 79, pid=1, GPU has fallen off the bus."


def _card_xids(lines, xid_seen, available=True):
    ns = _probe_namespace("card_xid_faults", "xid_faults", "pci_key", "SOFTWARE_XIDS", "HARDWARE_XIDS")
    ns["re"] = __import__("re")
    xid = {"available": available, "count": len(lines), "lines": lines}
    return ns["card_xid_faults"]({"available": True, "gpus": [XID_GPU]}, xid, xid_seen)


def test_only_a_new_hardware_xid_on_this_nodes_card_is_a_fault():
    faults = _card_xids([OLD, HW, APP, APP_UNLISTED, OTHER_CARD], xid_seen=1)

    assert faults == [f"gpu 2 (GPU-2): new NVRM Xid on a probed GPU: {HW}"]


@pytest.mark.parametrize(
    "xid_seen, available",
    [
        pytest.param(None, True, id="no-baseline-yet"),
        pytest.param(9, True, id="log-shorter-than-baseline"),
        pytest.param(0, False, id="kernel-log-not-readable"),
    ],
)
def test_no_baseline_a_rotated_log_or_an_unreadable_one_flags_nothing(xid_seen, available):
    assert _card_xids([OLD, HW], xid_seen=xid_seen, available=available) == []


def test_nvml_only_reports_the_xid_read_and_the_card_lists():
    run = subprocess.run(
        [sys.executable, "-I", "-", "--nvml-only", "--xid-seen", "0"],
        input=PROBE_SOURCE,
        capture_output=True,
        text=True,
        timeout=60,
    )

    line = next(line for line in run.stdout.splitlines() if line.startswith("GPU_FAULT_PROBE_JSON:"))
    report = json.loads(line.split(":", 1)[1])
    assert run.returncode == 0
    assert set(report["xid"]) <= {"available", "count", "error"}
    assert "available" in report["xid"]
    assert report["card_suspects"] == [] and report["xid_faults"] == []
    assert report["boot_id"] is None or isinstance(report["boot_id"], str)


# --- the validator's memory read --------------------------------------------------------------------------------

NARROW_LINK = "gpu 3 (%s): PCIe link x8 of x16" % UUIDS[3]


FULL_LINK_AT_P0 = {"pstate": 0, "pcie_width": 16, "pcie_width_max": 16, "pcie_gen": 4, "pcie_gen_max": 4}


def _ok(gpu=None, **extra):
    card = {"index": 3, "uuid": UUIDS[3], "pci_bus_id": "00000000:03:00.0", "ecc_uncorrected": 0}
    nvml = {"available": True, "gpus": [{**card, **(gpu or {})}]}
    return _report("ok", flags=[], nvml=nvml, **extra)


async def _read(context_factory, services, stdout, listed=UUIDS):
    ctx = _ctx(context_factory, services, listed=listed).model_copy(update={"runner": FakeRunner(stdout)})
    read = await gpu_memory_health.read_memory_health(ctx)
    return read, ctx.runner.calls[0]["cmd"]


def _stored(redis):
    [key] = [k for k in redis.store if k.startswith("gpu_card_health:")]
    return json.loads(redis.store[key])


@pytest.mark.asyncio
async def test_the_xid_baseline_is_kept_and_passed_to_the_next_read(context_factory):
    services = _services()
    read, cmd = await _read(context_factory, services, _ok(xid={"available": True, "count": 4}))
    assert cmd.endswith("--nvml-only")
    assert _stored(services.redis)["xid_seen"] == 4

    _, cmd = await _read(context_factory, services, _ok(xid={"available": False, "error": "denied"}))
    assert cmd.endswith("--nvml-only --xid-seen 4")
    # a log not readable this time keeps the old baseline
    assert _stored(services.redis)["xid_seen"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("unread", ["uuid", "pci_bus_id"])
async def test_a_card_nvml_could_not_name_keeps_the_xid_baseline(context_factory, unread):
    services = _services()
    await _read(context_factory, services, _ok(xid={"available": True, "count": 4}))
    # the probe cannot tie that card's new Xid lines to it this cycle, so they must be judged next time
    await _read(context_factory, services, _ok(gpu={unread: None}, xid={"available": True, "count": 6}))
    assert _stored(services.redis)["xid_seen"] == 4


@pytest.mark.asyncio
async def test_a_first_read_with_an_untied_card_still_takes_the_xid_baseline(context_factory):
    services = _services()
    # no baseline yet: the probe judged no lines, so the count is the baseline even if a card is untied (review)
    await _read(context_factory, services, _ok(gpu={"pci_bus_id": None}, xid={"available": True, "count": 4}))
    assert _stored(services.redis)["xid_seen"] == 4


@pytest.mark.asyncio
async def test_a_reboot_heals_a_held_xid_even_when_a_card_is_missing_from_a_partial_read(context_factory):
    services = _services()
    xid_fault = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW}"
    await _read(context_factory, services, _ok(xid={"available": True, "count": 4}, boot_id="boot-a"))
    await _read(
        context_factory, services,
        _ok(xid={"available": True, "count": 5}, xid_faults=[xid_fault], boot_id="boot-a"),
    )
    # the card is not named on this read (no UUID), but the host booted again: the reboot proof must count
    rebooted, _ = await _read(
        context_factory, services, _ok(gpu={"uuid": None}, xid={"available": True, "count": 1}, boot_id="boot-b")
    )
    assert rebooted.report["card_faults"] == []


@pytest.mark.asyncio
async def test_a_reboot_takes_the_new_xid_baseline_even_with_an_untied_card(context_factory):
    services = _services()
    await _read(context_factory, services, _ok(xid={"available": True, "count": 9}, boot_id="boot-a"))
    # the new boot's log is shorter than the old baseline: keeping 9 would hide its next Xids (review)
    await _read(
        context_factory, services, _ok(gpu={"pci_bus_id": None}, xid={"available": True, "count": 2}, boot_id="boot-b")
    )
    assert _stored(services.redis)["xid_seen"] == 2


@pytest.mark.asyncio
async def test_a_held_fault_closes_on_absence_only_when_the_scrape_also_misses_the_card(context_factory):
    services = _services()
    other = {"uuid": UUIDS[4], "index": 4, "pci_bus_id": "00000000:04:00.0"}
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    # the card dropped out after this cycle's scrape listed it: no missing-card fault yet, so keep it (review)
    still_scraped, _ = await _read(context_factory, services, _ok(gpu=other))
    gone, _ = await _read(context_factory, services, _ok(gpu=other), listed=[u for u in UUIDS if u != UUIDS[3]])

    assert still_scraped.report["card_faults"] == [NARROW_LINK]
    assert gone.report["card_faults"] == []


@pytest.mark.asyncio
async def test_a_downgrade_counts_from_its_second_read_and_holds_until_a_full_link_at_p0(context_factory):
    services = _services()
    first, _ = await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    second, _ = await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    # a read that merely shows no downgrade (an idle card is not judged) proves nothing (review)
    idle, _ = await _read(context_factory, services, _ok(gpu={"pstate": 8}, card_suspects=[]))
    healed, _ = await _read(context_factory, services, _ok(gpu=FULL_LINK_AT_P0, card_suspects=[]))
    once_again, _ = await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))

    assert first.report["card_faults"] == []
    assert second.report["card_faults"] == [NARROW_LINK]
    assert idle.report["card_faults"] == [NARROW_LINK]
    assert healed.report["card_faults"] == []
    assert once_again.report["card_faults"] == []


@pytest.mark.asyncio
async def test_an_xid_fault_holds_until_the_host_boots_again(context_factory):
    services = _services()
    xid_fault = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW}"
    boot = {"boot_id": "boot-a"}
    await _read(context_factory, services, _ok(xid={"available": True, "count": 4}, **boot))
    new, _ = await _read(
        context_factory, services, _ok(xid={"available": True, "count": 5}, xid_faults=[xid_fault], **boot)
    )
    quiet, _ = await _read(context_factory, services, _ok(xid={"available": True, "count": 5}, **boot))
    unread, _ = await _read(context_factory, services, _ok(xid={"available": False, "error": "denied"}, **boot))
    # a cleared or wrapped kernel log shrinks the count on the same boot: no proof
    cleared, _ = await _read(context_factory, services, _ok(xid={"available": True, "count": 1}, **boot))
    no_boot_id, _ = await _read(context_factory, services, _ok(xid={"available": True, "count": 1}))
    rebooted, _ = await _read(context_factory, services, _ok(xid={"available": True, "count": 1}, boot_id="boot-b"))

    assert new.report["card_faults"] == [xid_fault]
    assert quiet.report["card_faults"] == [xid_fault]
    assert unread.report["card_faults"] == [xid_fault]
    assert cleared.report["card_faults"] == [xid_fault]
    assert no_boot_id.report["card_faults"] == [xid_fault]
    assert rebooted.report["card_faults"] == []


@pytest.mark.asyncio
async def test_a_card_state_redis_cannot_read_is_unread_and_not_overwritten(context_factory, monkeypatch):
    services = _services()
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    stored = _stored(services.redis)

    async def failing_get(key):
        raise ConnectionError("redis down")

    monkeypatch.setattr(services.redis, "get", failing_get)
    read, _ = await _read(context_factory, services, _ok(gpu=FULL_LINK_AT_P0, card_suspects=[]))

    assert gpu_memory_health.card_fault_uuids(read.report) is None
    assert _stored(services.redis) == stored


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gpu",
    [
        pytest.param({"uuid": None}, id="card-without-uuid-in-the-read"),
        pytest.param({"pstate": 0, "pcie_width": None, "pcie_width_max": None}, id="pcie-values-unread"),
    ],
)
async def test_a_read_missing_values_does_not_heal_a_held_downgrade(context_factory, gpu):
    services = _services()
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    read, _ = await _read(context_factory, services, _ok(gpu=gpu, card_suspects=[]))

    assert read.report["card_faults"] == [NARROW_LINK]


@pytest.mark.asyncio
async def test_a_link_that_changes_width_between_reads_still_confirms(context_factory):
    services = _services()
    narrower = f"gpu 3 ({UUIDS[3]}): PCIe link x4 of x16"
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    second, _ = await _read(context_factory, services, _ok(card_suspects=[narrower]))

    assert second.report["card_faults"] == [narrower]


@pytest.mark.asyncio
async def test_repeated_xids_on_one_card_keep_one_held_fault(context_factory):
    services = _services()
    boot = {"boot_id": "boot-a"}
    await _read(context_factory, services, _ok(xid={"available": True, "count": 4}, **boot))
    for count in range(5, 9):
        faults = [f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW} #{n}" for n in range(2)]
        await _read(context_factory, services, _ok(xid={"available": True, "count": count}, xid_faults=faults, **boot))

    assert len(_stored(services.redis)["held"]) == 1


@pytest.mark.asyncio
async def test_a_held_xid_survives_a_later_downgrade_on_the_same_card_healing(context_factory):
    services = _services()
    boot = {"boot_id": "boot-a"}
    xid_fault = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW}"
    await _read(context_factory, services, _ok(xid={"available": True, "count": 4}, **boot))
    await _read(context_factory, services, _ok(xid={"available": True, "count": 5}, xid_faults=[xid_fault], **boot))
    narrow = _ok(xid={"available": True, "count": 5}, card_suspects=[NARROW_LINK], **boot)
    await _read(context_factory, services, narrow)
    both, _ = await _read(context_factory, services, narrow)
    link_healed, _ = await _read(
        context_factory, services, _ok(gpu=FULL_LINK_AT_P0, xid={"available": True, "count": 5}, **boot)
    )

    assert sorted(both.report["card_faults"]) == sorted([NARROW_LINK, xid_fault])
    assert link_healed.report["card_faults"] == [xid_fault]


@pytest.mark.asyncio
async def test_a_confirmed_downgrade_redis_could_not_store_is_not_closed_by_an_idle_read(context_factory, monkeypatch):
    services = _services()
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    real_set = services.redis.set

    async def failing_set(*args, **kwargs):
        raise ConnectionError("redis down")

    monkeypatch.setattr(services.redis, "set", failing_set)
    confirmed, _ = await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    monkeypatch.setattr(services.redis, "set", real_set)
    idle, _ = await _read(context_factory, services, _ok(gpu={"pstate": 8}, card_suspects=[]))

    assert confirmed.report["card_faults"] == [NARROW_LINK]
    assert idle.report["card_faults"] == [NARROW_LINK]


@pytest.mark.asyncio
async def test_an_unread_cycle_keeps_a_held_fault_from_expiring(context_factory):
    services = _services()
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    await _read(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    [key] = [k for k in services.redis.store if k.startswith("gpu_card_health:")]
    services.redis.ttl[key] = 5
    await _read(context_factory, services, _report("error", error="NVML unavailable"))

    assert services.redis.ttl[key] == gpu_memory_health.CARD_HEALTH_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_dry_run_writes_no_card_state(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DRY_RUN", True)
    services = _services()
    await _read(context_factory, services, _ok(xid={"available": True, "count": 4}))

    assert not any(k.startswith("gpu_card_health:") for k in services.redis.store)


# --- the renter's incident ---------------------------------------------------------------------------------------

XID_FAULT = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW}"


async def _drop(context_factory, services, stdout):
    ctx = _ctx(context_factory, services, listed=UUIDS).model_copy(update={"runner": FakeRunner(stdout)})
    return await RentedGpuDropCheck().run(ctx)


@pytest.mark.asyncio
async def test_a_new_hardware_xid_on_a_listed_card_is_posted_naming_it(context_factory):
    services = _services()

    result = await _drop(context_factory, services, _ok(xid_faults=[XID_FAULT]))

    assert (result.passed, result.event.reason_code) == (True, Msg.DROP.reason)
    call = services.backend.report_rented_gpu_drop.await_args
    assert call.kwargs["faults"] == [FAULT_GPU_HW]
    assert call.kwargs["missing_uuids"] == [UUIDS[3]]
    # the pod is left alone: no verified-job clear, nothing but the memory read in the state update
    assert set(result.updates) == {"state"}


@pytest.mark.asyncio
async def test_a_link_downgrade_is_posted_from_its_second_cycle_and_recovers_on_a_full_link(context_factory):
    services = _services()

    await _drop(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    assert _states(services) == []
    await _drop(context_factory, services, _ok(card_suspects=[NARROW_LINK]))
    assert _states(services) == ["fault"]
    # an NVML read that failed proves nothing: the incident stays open
    await _drop(context_factory, services, _report("error", flags=[], nvml={"available": False}))
    assert _states(services) == ["fault"]
    # no downgrade read is no proof either: only the full link at P0 is
    await _drop(context_factory, services, _ok(card_suspects=[]))
    assert _states(services) == ["fault"]
    await _drop(context_factory, services, _ok(gpu=FULL_LINK_AT_P0, card_suspects=[]))
    assert _states(services) == ["fault", "recovered"]


# --- a rented node whose scrape keeps timing out ----------------------------------------------------------------


def _scrape_ctx(context_factory, services, run):
    executor = default_executor()
    return context_factory(
        services=services,
        config=build_context_config(machine_scrape_filename="scrape.sh", machine_scrape_timeout=300),
        state=build_state(remote_dir="/remote/path", rented_data=_rented(executor.uuid)),
        runner=DummySSHCommandRunner(result=run),
        executor=executor,
        verified={"uuids": ",".join(UUIDS)},
    )


TIMEOUT = make_command_result(success=False, exit_code=-1, duration_ms=300_000, error_type="timeout")
FAILED = make_command_result(success=False, exit_code=127, stderr="scrape.sh: No such file")


@pytest.mark.asyncio
async def test_a_rented_node_timing_out_n_cycles_in_a_row_posts_gpu_unverified(context_factory):
    services = _services(redis=FakeRedis())
    check = MachineSpecScrapeCheck()

    for _ in range(SCRAPE_TIMEOUT_CYCLES - 1):
        await check.run(_scrape_ctx(context_factory, services, TIMEOUT))
    # another failure ends the run
    await check.run(_scrape_ctx(context_factory, services, FAILED))
    for _ in range(SCRAPE_TIMEOUT_CYCLES - 1):
        await check.run(_scrape_ctx(context_factory, services, TIMEOUT))
    assert _states(services) == []

    ctx = _scrape_ctx(context_factory, services, TIMEOUT)
    result = await check.run(ctx)

    assert result.event.reason_code == "SCRAPE_TIMEOUT"
    call = services.backend.report_rented_gpu_drop.await_args
    assert call.kwargs["state"] == "fault"
    assert call.kwargs["faults"] == [FAULT_GPU_UNVERIFIED]
    assert (call.kwargs["expected_gpu_count"], call.kwargs["visible_gpu_count"]) == (8, 0)
    # no memory read was run on the node: the scrape was the only command
    assert len(ctx.runner.calls) == 1

    # the next clean scrape is no timeout: the run ends, and RentedGpuDropCheck posts the recovery as usual
    await check.run(_scrape_ctx(context_factory, services, FAILED))
    assert not any(k.startswith("rented_scrape_timeouts:") for k in services.redis.store)
