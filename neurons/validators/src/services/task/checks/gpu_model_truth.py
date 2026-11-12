"""DAH-6241: the GPU model a node reports must match the card the matmul ran on.

The model name is the node's own claim; the matmul's sealed metrics carry the compute capability
and SM count the CUDA runtime read off the card it ran on (device 0, the first reported card). A
name whose card has another generation (cc) or another SM count is not that model: an A100
reporting itself as an H100 (8.0/108 against 9.0/132), an A6000 as an RTX 6000 Ada (8.6/84
against 8.9/142). Only a card below the reported model fails: a lower generation or fewer SMs.
A bigger card under a smaller name (an SXM H100 listed as an H100 PCIe) is not paid more for it,
so it passes. Only names in the table are judged; an unknown name, or a run without both
numbers, never fails.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# reported name -> (compute capability, SM count), NVIDIA's values; each matches what the
# sealed matmul read on the fleet's honest nodes of that name (12 Nov 2026, 12 h of
# MATMUL_SPEED_SHADOW). Exact names: one name is one SKU, so its SM count is fixed.
EXPECTED_CC_AND_SM: dict[str, tuple[str, int]] = {
    "NVIDIA B200": ("10.0", 148),
    "NVIDIA B300 SXM6 AC": ("10.3", 148),
    "NVIDIA B300 SXM6 PC": ("10.3", 148),
    "NVIDIA GeForce RTX 3070": ("8.6", 46),
    "NVIDIA GeForce RTX 3070 Ti": ("8.6", 48),
    "NVIDIA GeForce RTX 3090": ("8.6", 82),
    "NVIDIA GeForce RTX 3090 Ti": ("8.6", 84),
    "NVIDIA GeForce RTX 4060 Ti": ("8.9", 34),
    "NVIDIA GeForce RTX 4070 Ti": ("8.9", 60),
    "NVIDIA GeForce RTX 4070 Ti SUPER": ("8.9", 66),
    "NVIDIA GeForce RTX 4080": ("8.9", 76),
    "NVIDIA GeForce RTX 4080 SUPER": ("8.9", 80),
    "NVIDIA GeForce RTX 4090": ("8.9", 128),
    "NVIDIA GeForce RTX 5060 Ti": ("12.0", 36),
    "NVIDIA GeForce RTX 5070": ("12.0", 48),
    "NVIDIA GeForce RTX 5070 Ti": ("12.0", 70),
    "NVIDIA GeForce RTX 5080": ("12.0", 84),
    "NVIDIA GeForce RTX 5090": ("12.0", 170),
    "NVIDIA H100 80GB HBM3": ("9.0", 132),
    "NVIDIA H100 PCIe": ("9.0", 114),
    "NVIDIA H200": ("9.0", 132),
    "NVIDIA H200 NVL": ("9.0", 132),
    "NVIDIA L40": ("8.9", 142),
    "NVIDIA L40S": ("8.9", 142),
    "NVIDIA RTX 6000 Ada Generation": ("8.9", 142),
    "NVIDIA RTX A4000": ("8.6", 48),
    "NVIDIA RTX A4500": ("8.6", 56),
    "NVIDIA RTX A6000": ("8.6", 84),
    "NVIDIA RTX PRO 4000 Blackwell": ("12.0", 70),
    "NVIDIA RTX PRO 5000 Blackwell": ("12.0", 110),
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": ("12.0", 188),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": ("12.0", 188),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": ("12.0", 188),
}


@dataclass(frozen=True)
class GpuModelMismatch:
    card_index: int
    reported_gpu_model: str
    expected_compute_capability: str
    expected_sm_count: int
    measured_compute_capability: str
    measured_sm_count: int
    measured_device_name: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def reported_gpu_model(machine_spec: Any, card_index: int) -> str | None:
    gpu = machine_spec.get("gpu") if isinstance(machine_spec, dict) else None
    details = gpu.get("details") if isinstance(gpu, dict) else None
    card = details[card_index] if isinstance(details, list) and 0 <= card_index < len(details) else None
    name = card.get("name") if isinstance(card, dict) else None
    return name if isinstance(name, str) else None


def _sm_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        return None
    return int(value)


def _cc_version(value: Any) -> tuple[int, int] | None:
    major, dot, minor = value.partition(".") if isinstance(value, str) else ("", "", "")
    return (int(major), int(minor)) if dot and major.isdigit() and minor.isdigit() else None


def gpu_model_mismatch(machine_spec: Any, metrics: Any, card_index: int = 0) -> GpuModelMismatch | None:
    """What was reported and measured when the card (the reported card at `card_index`, the one
    the matmul ran on) is below the reported model; None otherwise (the model or a bigger card,
    an unknown name, or a run that did not give both numbers)."""
    model = reported_gpu_model(machine_spec, card_index)
    expected = EXPECTED_CC_AND_SM.get(model) if model else None
    if expected is None or not isinstance(metrics, dict):
        return None
    cc, sm_count = metrics.get("cc"), _sm_count(metrics.get("sm_count"))
    measured_cc = _cc_version(cc)
    if measured_cc is None or sm_count is None:
        return None
    if measured_cc >= _cc_version(expected[0]) and sm_count >= expected[1]:
        return None
    device = metrics.get("device")
    return GpuModelMismatch(
        card_index=card_index,
        reported_gpu_model=model,
        expected_compute_capability=expected[0],
        expected_sm_count=expected[1],
        measured_compute_capability=cc,
        measured_sm_count=sm_count,
        measured_device_name=device if isinstance(device, str) else None,
    )


def gpu_model_mismatches(
    machine_spec: Any, metrics: Any, card_metrics: dict[int, dict] | None = None
) -> list[GpuModelMismatch]:
    """Every card below its reported model: device 0 from the capability matmul, then each card
    the all-cards probe measured under its own seal."""
    found = [gpu_model_mismatch(machine_spec, metrics)]
    found += [gpu_model_mismatch(machine_spec, m, i) for i, m in sorted((card_metrics or {}).items())]
    return list({m.card_index: m for m in found if m is not None}.values())


def judged_cards(metrics: Any, card_metrics: dict[int, dict] | None = None) -> set[int]:
    """The cards a run gave both numbers for: only these can clear a stored mismatch."""
    runs = {0: metrics, **(card_metrics or {})}
    return {
        i
        for i, m in runs.items()
        if isinstance(m, dict) and _cc_version(m.get("cc")) and _sm_count(m.get("sm_count")) is not None
    }
