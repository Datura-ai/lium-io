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


DETAIL_MAX = 32
DETAIL_CHARS = 240
# no host has this many cards; a longer block is junk and is rejected before it is walked
CARDS_MAX = 128


def _card_label(index: int, card: dict[str, Any]) -> str:
    uuid = _as_text(card.get("uuid")) or "<no uuid>"
    pci = _as_text(card.get("pci")) or "<no pci>"
    return f"card {index} {uuid} {pci}"


def _duplicate_groups(cards: list[dict[str, Any]], field: str) -> list[tuple[str, list[int]]]:
    groups: dict[str, tuple[str, list[int]]] = {}
    for index, card in enumerate(cards):
        value = _as_text(card.get(field))
        if not value:
            continue
        _original, indexes = groups.setdefault(value.lower(), (value, []))
        indexes.append(index)
    return [(value, indexes) for value, indexes in groups.values() if len(indexes) > 1]


def dev_node_missing(card: dict[str, Any]) -> int | None:
    """The card's minor when the block says its /dev node is missing, else None.

    A block that could not read the driver's procfs has no minor, so liumd leaves
    `dev_node_present` false: unknown, not missing. A card NVML does not name is not the
    node's to offer, so its dev node is not judged either.
    """
    minor = _as_int(card.get("minor"))
    if minor is None or card.get("dev_node_present") is not False:
        return None
    return minor if _as_dict(_as_dict(card.get("sources")).get("nvml")) else None


def evaluate_gpu_identity(
    identity: Any,
    claimed_count: Any,
    *,
    claimed_model: str | None = None,
    kernel_uuids: list[str] | None = None,
) -> dict[str, Any]:
    """The validator's own verdict on the scrape's identity block. Defensive: junk is a mismatch,
    never an exception.

    `reasons` is the stable code set used by the check and the event. `details` is the
    operator log: card index, UUID, PCI and the two values that disagreed. It is never put in
    the event, so a miner cannot read our internals from the backend.

    `kernel_uuids` is the validator's own kernel card list, read when the scrape's container
    could not supply one (DAH-6273). A card it names is real whether or not the block's NVML or
    procfs source saw it, so those missing sources are not reasons; a card it does not name is
    the only kind that can be an extra card.
    """
    block = _as_dict(identity)
    cards_value = block.get("cards")
    claimed = _as_int(claimed_count)
    kernel = {str(uuid).strip().lower() for uuid in kernel_uuids or [] if str(uuid).strip()}

    def kernel_names(card: dict[str, Any]) -> bool:
        uuid = _as_text(card.get("uuid"))
        return bool(kernel) and uuid is not None and uuid.lower() in kernel

    claimed_model = _as_text(claimed_model)
    nvml_error = _as_text(block.get("nvml_error"))
    judged: dict[str, Any] = {
        "present": isinstance(cards_value, list),
        "claimed_count": claimed,
        "evidence_count": 0,
        "nvidia_module_present": block.get("nvidia_module_present") is True,
        "proc_driver_present": block.get("proc_driver_present") is True,
        "nvml_error": nvml_error,
        "reasons": [],
        "details": [],
        "would_fail": False,
    }
    if not isinstance(cards_value, list):
        judged["reasons"] = ["identity_missing"]
        judged["details"] = ["identity block missing or cards is not a list"]
        judged["would_fail"] = True
        return judged
    if len(cards_value) > CARDS_MAX:
        judged["evidence_count"] = len(cards_value)
        judged["reasons"] = ["cards_over_limit"]
        judged["details"] = [f"identity block names {len(cards_value)} cards, limit {CARDS_MAX}"]
        judged["would_fail"] = True
        return judged

    cards = [_as_dict(card) for card in cards_value if isinstance(card, dict)]
    judged["evidence_count"] = len(cards)
    kernel_names_all = bool(kernel) and bool(cards) and all(kernel_names(card) for card in cards)
    reasons: list[str] = []
    details: list[str] = []

    def add(reason: str, detail: str) -> None:
        if reason not in reasons:
            reasons.append(reason)
        detail = detail[:DETAIL_CHARS]
        if detail not in details:
            details.append(detail)

    if not judged["nvidia_module_present"] and not kernel_names_all:
        add("nvidia_module_missing", "nvidia kernel module is not present")
    if not judged["proc_driver_present"] and not kernel_names_all:
        add("proc_driver_missing", "/proc/driver/nvidia/gpus is not readable")
    if nvml_error:
        add("nvml_error", f"nvidia-smi: {nvml_error}")
    if claimed is not None and len(cards) < claimed:
        add("missing_card", f"claimed {claimed} cards, evidence has {len(cards)}")
    nvml_cards = sum(1 for card in cards if _as_dict(_as_dict(card.get("sources")).get("nvml")))
    if claimed is not None and nvml_cards > claimed:
        add("extra_card", f"claimed {claimed} cards, nvml names {nvml_cards}")
    for value, indexes in _duplicate_groups(cards, "uuid"):
        add("duplicate_uuid", f"duplicate uuid {value} on cards {indexes}")
    for value, indexes in _duplicate_groups(cards, "pci"):
        add("duplicate_pci", f"duplicate pci {value} on cards {indexes}")
    for index, card in enumerate(cards):
        label = _card_label(index, card)
        if not _as_text(card.get("uuid")):
            add("uuid_missing", f"{label}: no uuid")
        if not _as_text(card.get("pci")):
            add("pci_missing", f"{label}: no pci")
        sources = _as_dict(card.get("sources"))
        nvml = _as_dict(sources.get("nvml"))
        proc = _as_dict(sources.get("proc"))
        sysfs = _as_dict(sources.get("sysfs"))
        if not nvml and not kernel_names(card):
            add("nvml_source_missing", f"{label}: nvml source missing")
        if not proc and not kernel_names(card):
            add("proc_source_missing", f"{label}: proc source missing")
        if not sysfs and not kernel_names(card):
            add("sysfs_source_missing", f"{label}: sysfs source missing")
        for name, fields in SOURCE_FIELDS.items():
            source = _as_dict(sources.get(name))
            if not source:
                continue
            for field in fields:
                declared, reported = card.get(field), source.get(field)
                if not _field_agrees(field, declared, reported):
                    label_field = "vram" if field == "vram_mb" else field
                    add(
                        f"{name}_{label_field}_mismatch",
                        f"{label}: {name} {field} {reported!r}, card says {declared!r}",
                    )
        vendor = _as_text(sysfs.get("vendor"))
        if vendor is not None and vendor.lower() != SYSFS_VENDOR:
            add("sysfs_vendor_mismatch", f"{label}: sysfs vendor {vendor}, expected {SYSFS_VENDOR}")
        minor = dev_node_missing(card)
        if minor is not None:
            add("dev_node_missing", f"{label}: /dev/nvidia{minor} missing")
        model = _as_text(card.get("model"))
        if (
            claimed_model
            and model
            and not kernel_names(card)
            and model.lower() != claimed_model.lower()
        ):
            add("claimed_model_mismatch", f"{label}: model {model!r}, claimed {claimed_model!r}")

    judged["reasons"] = reasons
    judged["details"] = details[:DETAIL_MAX]
    if len(details) > DETAIL_MAX:
        judged["details"].append(f"... {len(details) - DETAIL_MAX} more")
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
        "nvml_error": judged.get("nvml_error"),
        "reasons": judged["reasons"],
        "details": judged.get("details", []),
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
        "nvml_error": judged.get("nvml_error"),
        "reasons": judged["reasons"],
    }
