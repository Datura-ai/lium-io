"""Hardware identity from the machine scrape: NVML, the driver's procfs, and PCI sysfs.

The scrape collects this on every run. The validator judges only facts that do not move with
clocks, load or power: source disagreement, duplicate cards, a card count that differs from the
scrape's own claim, a missing kernel device, and a model that contradicts the claim. Timing is
measured on the matmul step and is never judged here.

`VALIDATOR_LIUMD_EVIDENCE_MODE` has two values. `shadow` logs a warning for a mismatch and passes.
`enforce` fails the scrape check, which stops the run at score 0.
"""

from __future__ import annotations

import logging
from typing import Any

from core.utils import _m, get_extra_info

from ..pipeline import Context

logger = logging.getLogger(__name__)

GPU_IDENTITY_EVENT = "[liumd_evidence]"
SOURCE_FIELDS: dict[str, tuple[str, ...]] = {
    "nvml": ("uuid", "pci", "model", "vram_mb"),
    "proc": ("uuid", "pci", "model"),
}
SYSFS_VENDOR = "0x10de"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _field_agrees(field: str, declared: Any, reported: Any) -> bool:
    """A source agrees with the card's declared field; a value either side did not report is not
    a mismatch."""
    if field == "vram_mb":
        left, right = _as_int(declared), _as_int(reported)
        return left is None or right is None or left == right
    left, right = _as_text(declared), _as_text(reported)
    return left is None or right is None or left.lower() == right.lower()


def _has_duplicates(values: list[str | None]) -> bool:
    present = [value.lower() for value in values if value]
    return len(present) != len(set(present))


def evaluate_gpu_identity(
    identity: Any,
    claimed_count: Any,
    *,
    claimed_model: str | None = None,
) -> dict[str, Any]:
    """The validator's own verdict on the scrape's identity block. Defensive: junk is a mismatch,
    never an exception."""
    block = _as_dict(identity)
    cards_value = block.get("cards")
    claimed = _as_int(claimed_count)
    judged: dict[str, Any] = {
        "present": isinstance(cards_value, list),
        "claimed_count": claimed,
        "evidence_count": 0,
        "nvidia_module_present": block.get("nvidia_module_present") is True,
        "proc_driver_present": block.get("proc_driver_present") is True,
        "reasons": [],
        "would_fail": False,
    }
    if not isinstance(cards_value, list):
        judged["reasons"] = ["identity_missing"]
        judged["would_fail"] = True
        return judged

    cards = [_as_dict(card) for card in cards_value if isinstance(card, dict)]
    judged["evidence_count"] = len(cards)
    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    if not judged["nvidia_module_present"]:
        add("nvidia_module_missing")
    if not judged["proc_driver_present"]:
        add("proc_driver_missing")
    if claimed is not None and len(cards) < claimed:
        add("missing_card")
    if claimed is not None and len(cards) > claimed:
        add("extra_card")
    if _has_duplicates([_as_text(card.get("uuid")) for card in cards]):
        add("duplicate_uuid")
    if _has_duplicates([_as_text(card.get("pci")) for card in cards]):
        add("duplicate_pci")
    for card in cards:
        if not _as_text(card.get("uuid")):
            add("uuid_missing")
        if not _as_text(card.get("pci")):
            add("pci_missing")
        sources = _as_dict(card.get("sources"))
        nvml = _as_dict(sources.get("nvml"))
        proc = _as_dict(sources.get("proc"))
        sysfs = _as_dict(sources.get("sysfs"))
        if not nvml:
            add("nvml_source_missing")
        if not proc:
            add("proc_source_missing")
        if not sysfs:
            add("sysfs_source_missing")
        for name, fields in SOURCE_FIELDS.items():
            source = _as_dict(sources.get(name))
            if not source:
                continue
            for field in fields:
                if not _field_agrees(field, card.get(field), source.get(field)):
                    label = "vram" if field == "vram_mb" else field
                    add(f"{name}_{label}_mismatch")
        vendor = _as_text(sysfs.get("vendor"))
        if vendor is not None and vendor.lower() != SYSFS_VENDOR:
            add("sysfs_vendor_mismatch")
        if card.get("dev_node_present") is False:
            add("dev_node_missing")
        if card.get("char_dev_present") is False:
            add("char_dev_missing")
        model = _as_text(card.get("model"))
        if claimed_model and model and model.lower() != claimed_model.strip().lower():
            add("claimed_model_mismatch")

    judged["reasons"] = reasons
    judged["would_fail"] = bool(reasons)
    return judged


def log_gpu_identity(ctx: Context, judged: dict[str, Any], mode: str) -> None:
    """One line per node. A mismatch is a warning in shadow so it can be watched before enforce."""
    fields = {
        **ctx.default_extra,
        "transport": "liumd_evidence",
        "executor_uuid": ctx.executor.uuid,
        "address": ctx.executor.address,
        "mode": mode,
        "present": judged["present"],
        "claimed_count": judged["claimed_count"],
        "evidence_count": judged["evidence_count"],
        "nvidia_module_present": judged["nvidia_module_present"],
        "proc_driver_present": judged["proc_driver_present"],
        "reasons": judged["reasons"],
        "would_fail": judged["would_fail"],
    }
    try:
        log = logger.warning if judged["would_fail"] else logger.info
        log(_m(GPU_IDENTITY_EVENT, extra=get_extra_info(fields)))
    except Exception:  # noqa: BLE001 — a log line must not fail the scrape
        pass


def enforced(mode: str, judged: dict[str, Any]) -> bool:
    return mode == "enforce" and bool(judged["reasons"])


def event_fields(judged: dict[str, Any]) -> dict[str, Any]:
    """What the failed check's event carries. The reasons name the verdict."""
    return {
        "claimed_count": judged["claimed_count"],
        "evidence_count": judged["evidence_count"],
        "nvidia_module_present": judged["nvidia_module_present"],
        "proc_driver_present": judged["proc_driver_present"],
        "reasons": judged["reasons"],
    }
