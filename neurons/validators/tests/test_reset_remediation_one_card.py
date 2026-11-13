"""Provider remediation text resets one card: `nvidia-smi -r -i <index>`.

A bare `nvidia-smi -r` resets every GPU on the host and fails while any card is busy. Only the
NVLink fault resets the whole board on purpose (workloads stopped, Fabric Manager restarted).
"""
import re

from services.task import messages
from services.task.messages import MessageTemplate

WHOLE_BOARD_RESET = {"NVLINK_FAULT"}
# these only describe a reset (the latch cure, the executor losing its GPUs); they do not ask for one
DESCRIBES_A_RESET = {"GPU_WEDGED", "RENTAL_NOT_VERIFIED"}
BARE_RESET = re.compile(r"nvidia-smi -r(?! -i)[`\s]")


def _templates() -> list[MessageTemplate]:
    return [
        value
        for cls in vars(messages).values()
        if isinstance(cls, type)
        for value in vars(cls).values()
        if isinstance(value, MessageTemplate)
    ]


def test_reset_advice_names_one_card():
    reasons_with_bare_reset = [
        t.reason
        for t in _templates()
        if t.reason not in WHOLE_BOARD_RESET | DESCRIBES_A_RESET
        and t.remediation
        and BARE_RESET.search(t.remediation)
    ]
    assert reasons_with_bare_reset == []


def test_gpu_fault_probe_advice_resets_only_the_card():
    probe = [t for t in _templates() if t.reason == "GPU_FAULT_PROBE_FAILED"]
    assert probe and all("nvidia-smi -r -i <index>" in t.remediation for t in probe)
