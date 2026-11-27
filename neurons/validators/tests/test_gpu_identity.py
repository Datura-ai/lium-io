"""Hardware identity facts from the machine scrape: what is a mismatch and what is not."""

from __future__ import annotations

import pytest

from services.task.checks.gpu_identity import CARDS_MAX, evaluate_gpu_identity


def _card(
    index=0,
    *,
    uuid=None,
    pci=None,
    model="NVIDIA L4",
    vram_mb=22565,
    dev_node_present=True,
    nvml=True,
    proc=True,
    sysfs=True,
):
    uuid = uuid or f"GPU-{index}"
    pci = pci or f"0000:{31 + index}:00.0"
    sources = {}
    if nvml:
        sources["nvml"] = {
            "status": "agree",
            "uuid": uuid,
            "pci": pci,
            "model": model,
            "vram_mb": vram_mb,
            "error": None,
        }
    if proc:
        sources["proc"] = {"status": "ok", "uuid": uuid, "pci": pci, "model": model, "irq": "0"}
    if sysfs:
        sources["sysfs"] = {
            "status": "ok",
            "vendor": "0x10de",
            "device": "0x27b8",
            "class": None,
            "subsystem_vendor": "0x10de",
            "subsystem_device": "0x16ca",
            "link_speed": "16.0 GT/s PCIe",
            "link_width": "16",
        }
    return {
        "index": index,
        "uuid": uuid,
        "pci": pci,
        "model": model,
        "vram_mb": vram_mb,
        "driver": "595.112.07",
        "dev_node_present": dev_node_present,
        "minor": index,
        "sources": sources,
        "mismatches": [],
    }


def _report(*cards, module=True, proc_driver=True):
    return {
        "cards": list(cards),
        "nvidia_module_present": module,
        "proc_driver_present": proc_driver,
    }


def _reasons(report, claimed=1, **kwargs):
    return evaluate_gpu_identity(report, claimed, **kwargs)["reasons"]


def test_a_consistent_report_has_no_reasons():
    judged = evaluate_gpu_identity(_report(_card()), 1)

    assert (judged["present"], judged["claimed_count"], judged["evidence_count"]) == (True, 1, 1)
    assert judged["reasons"] == [] and judged["would_fail"] is False


@pytest.mark.parametrize("value", [None, "junk", 7, [], {}, {"cards": "nope"}])
def test_a_missing_identity_block_is_a_mismatch_not_an_exception(value):
    judged = evaluate_gpu_identity(value, 1)

    assert judged["present"] is False
    assert judged["reasons"] == ["identity_missing"] and judged["would_fail"] is True


def test_a_missing_driver_or_procfs_is_a_mismatch():
    assert "nvidia_module_missing" in _reasons(_report(_card(), module=False))
    assert "proc_driver_missing" in _reasons(_report(_card(), proc_driver=False))


def test_the_card_count_must_match_the_scrape_claim():
    assert "missing_card" in _reasons(_report(_card()), 2)
    assert "extra_card" in _reasons(_report(_card(0), _card(1)), 1)


def test_a_card_only_the_kernel_names_is_the_accepted_superset():
    block = _report(_card(0), _card(1, nvml=False))

    assert _reasons(block, 1, kernel_uuids=["GPU-0", "GPU-1"]) == []


def test_a_card_only_the_kernel_names_needs_the_kernel_read():
    block = _report(_card(0), _card(1, nvml=False))

    assert "extra_card" not in _reasons(block, 1)
    assert "nvml_source_missing" in _reasons(block, 1)


def test_a_card_the_kernel_does_not_name_is_still_a_missing_source():
    block = _report(_card(0), _card(1, nvml=False))

    assert "nvml_source_missing" in _reasons(block, 1, kernel_uuids=["GPU-0"])


def test_a_container_without_the_driver_procfs_is_accepted_on_the_kernel_view():
    block = _report(_card(proc=False), proc_driver=False)

    assert {"proc_driver_missing", "proc_source_missing"} <= set(_reasons(block, 1))
    assert _reasons(block, 1, kernel_uuids=["GPU-0"]) == []


def test_a_masked_module_and_procfs_are_accepted_on_the_kernel_view():
    block = _report(_card(), module=False, proc_driver=False)

    assert _reasons(block, 1, kernel_uuids=["GPU-0"]) == []


def test_a_non_text_claimed_model_is_not_an_exception():
    judged = evaluate_gpu_identity(_report(_card()), 1, claimed_model=7)

    assert judged["reasons"] == []


def test_an_identity_block_beyond_the_card_limit_is_rejected():
    cards = [_card(index) for index in range(CARDS_MAX + 1)]

    judged = evaluate_gpu_identity(_report(*cards), CARDS_MAX + 1)

    assert judged["reasons"] == ["cards_over_limit"]
    assert judged["evidence_count"] == CARDS_MAX + 1
    assert judged["would_fail"] is True


def test_an_identity_block_at_the_card_limit_is_judged():
    cards = [_card(index) for index in range(CARDS_MAX)]

    judged = evaluate_gpu_identity(_report(*cards), CARDS_MAX)

    assert judged["reasons"] == []


def test_a_dev_node_is_judged_only_when_it_is_known():
    card = _card(dev_node_present=False)
    card["minor"] = None
    assert "dev_node_missing" not in _reasons(_report(card))

    card = _card(dev_node_present=False)
    card["sources"]["nvml"] = {}
    assert "dev_node_missing" not in _reasons(_report(card))

    assert "dev_node_missing" in _reasons(_report(_card(dev_node_present=False)))


def test_a_duplicate_uuid_or_pci_is_a_mismatch():
    assert "duplicate_uuid" in _reasons(
        _report(_card(0, uuid="GPU-1"), _card(1, uuid="GPU-1")), 2
    )
    assert "duplicate_pci" in _reasons(
        _report(_card(0, pci="0000:31:00.0"), _card(1, pci="0000:31:00.0")), 2
    )


def test_a_card_without_a_uuid_or_bus_is_a_mismatch():
    card = _card()
    card["uuid"] = None
    card["pci"] = ""
    assert {"uuid_missing", "pci_missing"} <= set(_reasons(_report(card)))


def test_a_missing_source_is_a_mismatch():
    reasons = _reasons(_report(_card(nvml=False, proc=False, sysfs=False)))
    assert {"nvml_source_missing", "proc_source_missing", "sysfs_source_missing"} <= set(reasons)


def test_a_source_that_disagrees_with_the_card_is_a_mismatch():
    card = _card()
    card["sources"]["nvml"]["uuid"] = "GPU-OTHER"
    assert "nvml_uuid_mismatch" in _reasons(_report(card))

    card = _card()
    card["sources"]["nvml"]["vram_mb"] = 1024
    assert "nvml_vram_mismatch" in _reasons(_report(card))

    card = _card()
    card["sources"]["proc"]["pci"] = "0000:99:00.0"
    assert "proc_pci_mismatch" in _reasons(_report(card))

    card = _card()
    card["sources"]["proc"]["model"] = "NVIDIA H100"
    assert "proc_model_mismatch" in _reasons(_report(card))


def test_a_sysfs_vendor_that_is_not_nvidia_is_a_mismatch():
    card = _card()
    card["sources"]["sysfs"]["vendor"] = "0x1234"
    assert "sysfs_vendor_mismatch" in _reasons(_report(card))


def test_a_missing_kernel_device_is_a_mismatch():
    assert "dev_node_missing" in _reasons(_report(_card(dev_node_present=False)))


def test_a_model_that_contradicts_the_claim_is_a_mismatch():
    judged = evaluate_gpu_identity(_report(_card()), 1, claimed_model="NVIDIA H100")
    assert "claimed_model_mismatch" in judged["reasons"]
    assert judged["would_fail"] is True


def test_timing_and_benchmark_fields_are_measurements_not_reasons():
    # The scrape may carry clocks, TFLOPS, bandwidth and wall clock; the identity check never
    # reads them. Only the source facts decide.
    card = _card()
    card["timing"] = {"started_at_ms": 0, "first_result_at_ms": 1, "finished_at_ms": 2}
    card["tflops"] = 0.01
    card["bandwidth_gbps"] = 0.01
    report = _report(card)
    report["wall_clock_ms"] = 1
    report["first_result_spread_ms"] = 999_999

    assert _reasons(report) == []


def test_details_name_the_card_and_the_two_values():
    judged = evaluate_gpu_identity(
        _report(_card(0, uuid="GPU-1"), _card(1, uuid="GPU-1")), 2
    )
    assert any("duplicate uuid" in detail and "cards [0, 1]" in detail for detail in judged["details"])

    card = _card()
    card["sources"]["nvml"]["uuid"] = "GPU-OTHER"
    judged = evaluate_gpu_identity(_report(card), 1)
    assert any("card 0" in detail and "nvml uuid" in detail and "GPU-OTHER" in detail for detail in judged["details"])


def test_clean_details_are_empty():
    assert evaluate_gpu_identity(_report(_card()), 1)["details"] == []


def test_an_nvml_read_error_is_a_reason_even_when_the_cards_parse():
    report = _report(_card())
    report["nvml_error"] = "nvidia-smi exit Some(15)"
    judged = evaluate_gpu_identity(report, 1)
    assert "nvml_error" in judged["reasons"]
    assert any("nvidia-smi exit" in detail for detail in judged["details"])


def test_kernel_named_cards_do_not_judge_model_or_sysfs():
    card = _card(uuid="GPU-a", sysfs=False, model="NVIDIA L4")
    judged = evaluate_gpu_identity(
        _report(card), 1, claimed_model="NVIDIA H100", kernel_uuids=["GPU-a"]
    )
    assert "sysfs_source_missing" not in judged["reasons"]
    assert "claimed_model_mismatch" not in judged["reasons"]
