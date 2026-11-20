"""DAH-5869: a rented card that stays visible but logs a new hardware Xid is named in the
renter's incident until a read proves it healed. The pod is never touched."""

import json
import subprocess
import sys

import pytest
from neurons.validators.src.services.task.checks import RentedGpuDropCheck
from neurons.validators.src.services.task.checks import gpu_memory_health
from neurons.validators.src.services.task.checks.rented_gpu_drop import FAULT_GPU_HW
from neurons.validators.src.services.task.messages import RentedGpuDropMessages as Msg

from core.config import settings
from tests.test_gpu_fault_probe_check import FakeRunner, _probe_namespace
from tests.test_gpu_memory_health_check import PROBE_SOURCE, _report
from tests.test_rented_gpu_drop_check import UUIDS, _clear_local_marks, _ctx, _services, _states  # noqa: F401

# --- the probe (runs on the executor) ---------------------------------------------------------------------------


XID_GPU = {"index": 2, "uuid": "GPU-2", "pci_bus_id": "00000000:81:00.0"}
OLD = "[  100.000001] NVRM: Xid (PCI:0000:81:00): 79, pid=1, GPU has fallen off the bus."
HW = "[  200.5] NVRM: Xid (PCI:0000:81:00): 119, pid=1, Timeout waiting for RPC from GSP!"
APP = "[  201.0] NVRM: Xid (PCI:0000:81:00): 13, pid=1, Graphics Exception"
# an application Xid outside SOFTWARE_XIDS: a renter's out-of-range access, not the card (review)
APP_UNLISTED = "[  202.0] NVRM: Xid (PCI:0000:81:00): 69, pid=1, Graphics Exception: Class error"
OTHER_CARD = "[  203.0] NVRM: Xid (PCI:0000:c1:00): 79, pid=1, GPU has fallen off the bus."


def _card_xids(lines, xid_after, available=True):
    ns = _probe_namespace("card_xid_faults", "xid_faults", "kernel_ts", "pci_key", "SOFTWARE_XIDS", "HARDWARE_XIDS")
    ns["re"] = __import__("re")
    xid = {"available": available, "count": len(lines), "lines": lines}
    return ns["card_xid_faults"]({"available": True, "gpus": [XID_GPU]}, xid, xid_after)


def test_only_a_new_hardware_xid_on_this_nodes_card_is_a_fault():
    faults = _card_xids([OLD, HW, APP, APP_UNLISTED, OTHER_CARD], xid_after=100.000001)

    assert faults == [f"gpu 2 (GPU-2): new NVRM Xid on a probed GPU: {HW}"]


def test_a_wrapped_ring_buffer_still_shows_the_new_hardware_xid():
    # the buffer dropped two judged lines while two new ones came in: the count stayed at 3 (review)
    judged = [f"[  {t}.0] NVRM: Xid (PCI:0000:81:00): 13, pid=1, Graphics Exception" for t in (90, 95, 100)]
    wrapped = [judged[2], APP, HW.replace("200.5", "204.0")]

    assert _card_xids(wrapped, xid_after=100.0) == [
        f"gpu 2 (GPU-2): new NVRM Xid on a probed GPU: {wrapped[2]}"
    ]


@pytest.mark.parametrize(
    "lines, xid_after, available",
    [
        pytest.param([OLD, HW], None, True, id="no-baseline-yet"),
        pytest.param([OLD, HW], 300.0, True, id="nothing-after-the-baseline"),
        pytest.param([OLD.split("] ", 1)[1], HW.split("] ", 1)[1]], 0.0, True, id="lines-without-kernel-time"),
        pytest.param([OLD, HW], 0.0, False, id="kernel-log-not-readable"),
    ],
)
def test_no_baseline_no_new_line_or_an_unreadable_log_flags_nothing(lines, xid_after, available):
    assert _card_xids(lines, xid_after=xid_after, available=available) == []


@pytest.mark.parametrize(
    "xid_boot, this_boot, expected",
    [
        pytest.param("boot-a", "boot-a", 500.0, id="same-boot-keeps-the-baseline"),
        pytest.param("boot-a", "boot-b", 0.0, id="new-boot-judges-every-line"),
        pytest.param(None, "boot-b", 500.0, id="no-stored-boot"),
        pytest.param("boot-a", None, 500.0, id="boot-not-readable"),
    ],
)
def test_a_baseline_from_another_boot_judges_the_new_boot_from_its_first_line(xid_boot, this_boot, expected):
    ns = _probe_namespace("xid_after_for_this_boot")

    assert ns["xid_after_for_this_boot"](500.0, xid_boot, this_boot) == expected


def test_nvml_only_reports_the_xid_read_and_the_card_lists():
    run = subprocess.run(
        [sys.executable, "-I", "-", "--nvml-only", "--xid-after", "0"],
        input=PROBE_SOURCE,
        capture_output=True,
        text=True,
        timeout=60,
    )

    line = next(line for line in run.stdout.splitlines() if line.startswith("GPU_FAULT_PROBE_JSON:"))
    report = json.loads(line.split(":", 1)[1])
    assert run.returncode == 0
    assert set(report["xid"]) <= {"available", "count", "last_ts", "error"}
    assert "available" in report["xid"]
    assert report["xid_faults"] == []
    assert report["boot_id"] is None or isinstance(report["boot_id"], str)


# --- the validator's memory read --------------------------------------------------------------------------------

XID_FAULT = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW}"


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
    read, cmd = await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 4}))
    assert cmd.endswith("--nvml-only")
    assert _stored(services.redis)["xid_after"] == 4

    _, cmd = await _read(context_factory, services, _ok(xid={"available": False, "error": "denied"}))
    assert cmd.endswith("--nvml-only --xid-after 4")
    # a log not readable this time keeps the old baseline
    assert _stored(services.redis)["xid_after"] == 4


@pytest.mark.asyncio
async def test_the_baseline_carries_its_boot_and_a_reboot_without_a_log_read_starts_over(context_factory):
    services = _services()
    await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 900}, boot_id="boot-a"))
    _, cmd = await _read(context_factory, services, _ok(xid={"available": False, "error": "denied"}, boot_id="boot-b"))
    assert cmd.endswith("--nvml-only --xid-after 900 --xid-boot boot-a")
    # the old boot's 900 would skip the new boot's first lines on the next read (review)
    assert _stored(services.redis)["xid_after"] == 0.0
    assert _stored(services.redis)["boot_id"] == "boot-b"


@pytest.mark.asyncio
@pytest.mark.parametrize("unread", ["uuid", "pci_bus_id"])
async def test_a_card_nvml_could_not_name_keeps_the_xid_baseline(context_factory, unread):
    services = _services()
    await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 4}))
    # the probe cannot tie that card's new Xid lines to it this cycle, so they must be judged next time
    await _read(context_factory, services, _ok(gpu={unread: None}, xid={"available": True, "last_ts": 6}))
    assert _stored(services.redis)["xid_after"] == 4


@pytest.mark.asyncio
async def test_a_first_read_with_an_untied_card_still_takes_the_xid_baseline(context_factory):
    services = _services()
    # no baseline yet: the probe judged no lines, so the last line's time is the baseline even if a card is untied (review)
    await _read(context_factory, services, _ok(gpu={"pci_bus_id": None}, xid={"available": True, "last_ts": 4}))
    assert _stored(services.redis)["xid_after"] == 4


@pytest.mark.asyncio
async def test_a_reboot_heals_a_held_xid_even_when_a_card_is_missing_from_a_partial_read(context_factory):
    services = _services()
    xid_fault = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW}"
    await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 4}, boot_id="boot-a"))
    await _read(
        context_factory, services,
        _ok(xid={"available": True, "last_ts": 5}, xid_faults=[xid_fault], boot_id="boot-a"),
    )
    # the card is not named on this read (no UUID), but the host booted again: the reboot proof must count
    rebooted, _ = await _read(
        context_factory, services, _ok(gpu={"uuid": None}, xid={"available": True, "last_ts": 1}, boot_id="boot-b")
    )
    assert rebooted.report["card_faults"] == []


@pytest.mark.asyncio
async def test_a_reboot_takes_the_new_xid_baseline_even_with_an_untied_card(context_factory):
    services = _services()
    await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 9}, boot_id="boot-a"))
    # the new boot's kernel clock starts over below the old baseline: keeping 9 would hide its next Xids (review)
    await _read(
        context_factory, services, _ok(gpu={"pci_bus_id": None}, xid={"available": True, "last_ts": 2}, boot_id="boot-b")
    )
    assert _stored(services.redis)["xid_after"] == 2


@pytest.mark.asyncio
async def test_a_card_missing_from_the_read_keeps_its_hold(context_factory):
    services = _services()
    other = {"uuid": UUIDS[4], "index": 4, "pci_bus_id": "00000000:04:00.0"}
    await _read(context_factory, services, _ok(xid_faults=[XID_FAULT]))
    # the card dropped out after this cycle's scrape listed it
    missing, _ = await _read(context_factory, services, _ok(gpu=other))

    assert missing.report["card_faults"] == [XID_FAULT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "listed",
    [
        pytest.param([u for u in UUIDS if u != UUIDS[3]], id="scrape-misses-the-card-too"),
        pytest.param([], id="scrape-listed-nothing"),
    ],
)
async def test_a_card_gone_from_the_read_and_the_scrape_keeps_its_hold_when_it_comes_back(
    context_factory, listed, caplog
):
    # 0d0c64ef, 18 Nov 12:22Z: one cycle without the held card closed the hold, with no reboot or reset;
    # the card came back on the next cycle, took new rentals and faulted again at 14:07Z
    services = _services()
    other = {"uuid": UUIDS[4], "index": 4, "pci_bus_id": "00000000:04:00.0"}
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    await _read(context_factory, services, _ok(xid_faults=[XID_FAULT], **boot))
    caplog.set_level("INFO")
    gone, _ = await _read(context_factory, services, _ok(gpu=other, **boot), listed=listed)
    back, _ = await _read(context_factory, services, _ok(**boot))

    assert gone.report["card_faults"] == [XID_FAULT]
    assert back.report["card_faults"] == [XID_FAULT]
    assert "GPU card hold judged" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "now"),
    [
        pytest.param("boot-a", "", id="empty-now"),
        pytest.param("boot-a", "  ", id="blank-now"),
        pytest.param("", "boot-b", id="empty-stored"),
        pytest.param(None, "boot-b", id="none-stored"),
        pytest.param("boot-a", None, id="none-now"),
    ],
)
async def test_a_missing_or_empty_boot_id_is_never_a_reboot(context_factory, stored, now):
    services = _services()
    await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 5}))
    key = next(k for k in services.redis.store if k.startswith("gpu_card_health:"))
    services.redis.store[key] = json.dumps({"xid_after": 5, "boot_id": stored, "held": [XID_FAULT], "reset_marks": {}})

    read, _ = await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 5}, boot_id=now))

    assert read.report["card_faults"] == [XID_FAULT]


@pytest.mark.asyncio
async def test_an_xid_fault_holds_until_the_host_boots_again(context_factory):
    services = _services()
    xid_fault = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW}"
    boot = {"boot_id": "boot-a"}
    await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 4}, **boot))
    new, _ = await _read(
        context_factory, services, _ok(xid={"available": True, "last_ts": 5}, xid_faults=[xid_fault], **boot)
    )
    quiet, _ = await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 5}, **boot))
    unread, _ = await _read(context_factory, services, _ok(xid={"available": False, "error": "denied"}, **boot))
    # a cleared or wrapped kernel log moves the last line's time back on the same boot: no proof
    cleared, _ = await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 1}, **boot))
    no_boot_id, _ = await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 1}))
    rebooted, _ = await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 1}, boot_id="boot-b"))

    assert new.report["card_faults"] == [xid_fault]
    assert quiet.report["card_faults"] == [xid_fault]
    assert unread.report["card_faults"] == [xid_fault]
    assert cleared.report["card_faults"] == [xid_fault]
    assert no_boot_id.report["card_faults"] == [xid_fault]
    assert rebooted.report["card_faults"] == []


@pytest.mark.asyncio
async def test_a_reset_of_the_card_heals_its_held_xid_without_a_reboot(context_factory):
    services = _services()
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    needs_reset = {"ecc_uncorrected": 1, "recovery_action": 1}
    await _read(context_factory, services, _ok(gpu=needs_reset, **boot))
    held, _ = await _read(context_factory, services, _ok(gpu=needs_reset, xid_faults=[XID_FAULT], **boot))
    reset, _ = await _read(context_factory, services, _ok(gpu={"recovery_action": 0}, **boot))

    assert held.report["card_faults"] == [XID_FAULT]
    assert reset.report["card_faults"] == []
    assert _stored(services.redis)["reset_marks"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "after"),
    [
        ({}, {"recovery_action": 0}),  # nothing a reset clears was ever set: a quiet read proves nothing
        ({"ecc_uncorrected": 1}, {"ecc_uncorrected": 1}),
        ({"recovery_action": 1}, {"recovery_action": None}),  # an unreadable counter is not a cleared one
        ({"ecc_uncorrected": 1, "recovery_action": 1}, {"recovery_action": 1}),
    ],
)
async def test_a_held_xid_stays_without_proof_of_a_reset(context_factory, before, after):
    services = _services()
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    await _read(context_factory, services, _ok(gpu=before, xid_faults=[XID_FAULT], **boot))
    later, _ = await _read(context_factory, services, _ok(gpu=after, **boot))

    assert later.report["card_faults"] == [XID_FAULT]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inconclusive",
    [
        pytest.param({"gpu": {"ecc_uncorrected": None, "recovery_action": None}}, id="counters-unread"),
        pytest.param({"gpu": {"ecc_uncorrected": 1, "recovery_action": None}}, id="one-counter-unread"),
    ],
)
async def test_an_inconclusive_read_keeps_the_reset_baseline_of_a_held_card(context_factory, inconclusive):
    """A read that cannot read the card's reset-only counters neither lifts the hold nor forgets what the card
    last read: a reset after it is still proven by the next conclusive read."""
    services = _services()
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    needs_reset = {"ecc_uncorrected": 1, "recovery_action": 1}
    await _read(context_factory, services, _ok(gpu=needs_reset, xid_faults=[XID_FAULT], **boot))
    unread, _ = await _read(context_factory, services, _ok(**inconclusive, **boot))
    reset, _ = await _read(context_factory, services, _ok(gpu={"recovery_action": 0}, **boot))

    assert unread.report["card_faults"] == [XID_FAULT]
    assert reset.report["card_faults"] == []


@pytest.mark.asyncio
async def test_zeroed_counters_lift_nothing_while_the_kernel_log_is_unreadable(context_factory):
    """Zeroed counters prove a reset only in a read that also sees the kernel log: a card logging Xids the read
    cannot see stays held until a read that can see them."""
    services = _services()
    boot = {"boot_id": "boot-a"}
    needs_reset = {"ecc_uncorrected": 1, "recovery_action": 1}
    log = {"available": True, "last_ts": 5}
    await _read(context_factory, services, _ok(gpu=needs_reset, xid_faults=[XID_FAULT], xid=log, **boot))
    blind, _ = await _read(context_factory, services, _ok(gpu={"recovery_action": 0}, xid={"available": False}, **boot))
    seen, _ = await _read(context_factory, services, _ok(gpu={"recovery_action": 0}, xid=log, **boot))

    assert blind.report["card_faults"] == [XID_FAULT]
    assert seen.report["card_faults"] == []


def _held_before_reset_marks(services, xid: int) -> str:
    """A card state stored by a validator that kept no reset marks: one held Xid on card 3."""
    fault = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: NVRM: Xid (PCI:0000:03:00): {xid}, pid=1"
    key = next(k for k in services.redis.store if k.startswith("gpu_card_health:"))
    services.redis.store[key] = json.dumps({"xid_after": 5, "boot_id": "boot-a", "held": [fault]})
    return fault


@pytest.mark.asyncio
@pytest.mark.parametrize("xid", [48, 79, 95])
async def test_a_hold_stored_before_reset_marks_heals_on_a_zeroed_read_for_a_reset_only_xid(context_factory, xid):
    services = _services()
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    await _read(context_factory, services, _ok(**boot))
    _held_before_reset_marks(services, xid)

    read, _ = await _read(context_factory, services, _ok(gpu={"recovery_action": 0}, **boot))

    assert read.report["card_faults"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("xid", "gpu"),
    [
        (74, {"recovery_action": 0}),  # NVLink: a healthy-looking read says nothing about it
        (119, {"recovery_action": 0}),
        (48, {"recovery_action": None}),  # an unreadable counter is not a cleared one
        (48, {"ecc_uncorrected": 1, "recovery_action": 0}),
    ],
)
async def test_a_hold_stored_before_reset_marks_stays_without_proof(context_factory, xid, gpu):
    services = _services()
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    await _read(context_factory, services, _ok(**boot))
    fault = _held_before_reset_marks(services, xid)

    read, _ = await _read(context_factory, services, _ok(gpu=gpu, **boot))

    assert read.report["card_faults"] == [fault]


@pytest.mark.asyncio
async def test_a_reset_only_xid_logged_just_before_a_reset_heals_on_the_next_zeroed_read(context_factory):
    """An Xid 48 logged after one read and reset away before the next is judged on a read whose counters are
    already 0, so the card holds it with no marks: the zeroed-read rule for a reset-only Xid still lifts it."""
    services = _services()
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    dbe = f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: [  210.0] NVRM: Xid (PCI:0000:03:00): 48, pid=1, DBE"
    await _read(context_factory, services, _ok(gpu={"ecc_uncorrected": 1, "recovery_action": 0}, **boot))
    await _read(context_factory, services, _ok(gpu={"ecc_uncorrected": 2, "recovery_action": 0}, **boot))
    held, _ = await _read(context_factory, services, _ok(gpu={"recovery_action": 0}, xid_faults=[dbe], **boot))
    clean, _ = await _read(context_factory, services, _ok(gpu={"recovery_action": 0}, **boot))

    assert held.report["card_faults"] == [dbe]
    assert clean.report["card_faults"] == []


@pytest.mark.asyncio
async def test_a_hold_stored_before_reset_marks_still_heals_after_a_read_without_the_kernel_log(context_factory):
    services = _services()
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    await _read(context_factory, services, _ok(**boot))
    fault = _held_before_reset_marks(services, 48)

    no_log = _ok(gpu={"recovery_action": 0}, boot_id="boot-a", xid={"available": False})
    blind, _ = await _read(context_factory, services, no_log)
    seen, _ = await _read(context_factory, services, _ok(gpu={"recovery_action": 0}, **boot))

    assert blind.report["card_faults"] == [fault]
    assert seen.report["card_faults"] == []


@pytest.mark.asyncio
async def test_a_card_missing_from_one_read_keeps_its_reset_marks(context_factory):
    services = _services()
    boot = {"boot_id": "boot-a", "xid": {"available": True, "last_ts": 5}}
    other = {"uuid": UUIDS[4], "index": 4, "pci_bus_id": "00000000:04:00.0"}
    await _read(context_factory, services, _ok(gpu={"ecc_uncorrected": 2}, xid_faults=[XID_FAULT], **boot))
    await _read(context_factory, services, _ok(gpu=other, **boot))
    reset, _ = await _read(context_factory, services, _ok(**boot))

    assert reset.report["card_faults"] == []


@pytest.mark.asyncio
async def test_a_card_state_redis_cannot_read_is_unread_and_not_overwritten(context_factory, monkeypatch):
    services = _services()
    await _read(context_factory, services, _ok(xid_faults=[XID_FAULT]))
    stored = _stored(services.redis)

    async def failing_get(key):
        raise ConnectionError("redis down")

    monkeypatch.setattr(services.redis, "get", failing_get)
    read, _ = await _read(context_factory, services, _ok())

    assert gpu_memory_health.card_fault_uuids(read.report) is None
    assert _stored(services.redis) == stored


@pytest.mark.asyncio
async def test_a_read_naming_no_card_does_not_heal_a_held_fault(context_factory):
    services = _services()
    await _read(context_factory, services, _ok(xid_faults=[XID_FAULT]))
    read, _ = await _read(context_factory, services, _ok(gpu={"uuid": None}), listed=[u for u in UUIDS if u != UUIDS[3]])

    assert read.report["card_faults"] == [XID_FAULT]


@pytest.mark.asyncio
async def test_repeated_xids_on_one_card_keep_one_held_fault(context_factory):
    services = _services()
    boot = {"boot_id": "boot-a"}
    await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 4}, **boot))
    for count in range(5, 9):
        faults = [f"gpu 3 ({UUIDS[3]}): new NVRM Xid on a probed GPU: {HW} #{n}" for n in range(2)]
        await _read(context_factory, services, _ok(xid={"available": True, "last_ts": count}, xid_faults=faults, **boot))

    assert len(_stored(services.redis)["held"]) == 1


@pytest.mark.asyncio
async def test_a_fault_redis_could_not_store_is_not_closed_by_a_quiet_read(context_factory, monkeypatch):
    services = _services()
    real_set = services.redis.set

    async def failing_set(*args, **kwargs):
        raise ConnectionError("redis down")

    monkeypatch.setattr(services.redis, "set", failing_set)
    confirmed, _ = await _read(context_factory, services, _ok(xid_faults=[XID_FAULT]))
    monkeypatch.setattr(services.redis, "set", real_set)
    quiet, _ = await _read(context_factory, services, _ok())

    assert confirmed.report["card_faults"] == [XID_FAULT]
    assert quiet.report["card_faults"] == [XID_FAULT]


@pytest.mark.asyncio
async def test_an_unread_cycle_keeps_a_held_fault_from_expiring(context_factory):
    services = _services()
    await _read(context_factory, services, _ok(xid_faults=[XID_FAULT]))
    [key] = [k for k in services.redis.store if k.startswith("gpu_card_health:")]
    services.redis.ttl[key] = 5
    await _read(context_factory, services, _report("error", error="NVML unavailable"))

    assert services.redis.ttl[key] == gpu_memory_health.CARD_HEALTH_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_dry_run_writes_no_card_state(context_factory, monkeypatch):
    monkeypatch.setattr(settings, "DRY_RUN", True)
    services = _services()
    await _read(context_factory, services, _ok(xid={"available": True, "last_ts": 4}))

    assert not any(k.startswith("gpu_card_health:") for k in services.redis.store)


# --- the renter's incident ---------------------------------------------------------------------------------------

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
async def test_an_xid_incident_stays_open_until_the_host_boots_again(context_factory):
    services = _services()

    await _drop(context_factory, services, _ok(xid_faults=[XID_FAULT], boot_id="boot-a"))
    assert _states(services) == ["fault"]
    # an NVML read that failed proves nothing: the incident stays open
    await _drop(context_factory, services, _report("error", flags=[], nvml={"available": False}))
    assert _states(services) == ["fault"]
    # a quiet read on the same boot is no proof either: only a new boot is
    await _drop(context_factory, services, _ok(boot_id="boot-a"))
    assert _states(services) == ["fault"]
    await _drop(context_factory, services, _ok(boot_id="boot-b"))
    assert _states(services) == ["fault", "recovered"]
