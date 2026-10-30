"""DAH-5088: the reference model of an unknown NVIDIA card.

The same file, byte for byte, is in lium-platform
(apps/lium/backend/apps/server/src/core/gpu_reference_table.py) and lium-io
(neurons/validators/src/services/gpu_reference_table.py), so the backend lists and the validator
pays an unknown card as one model. Edit both copies together: a test in each repo pins its sha256.
"""

# a reported VRAM within this ratio of a nominal size is the same memory
REFERENCE_VRAM_BAND = (0.90, 1.10)

# The candidates: every model the validator has a rate for (GPU_MODEL_RATES) and the backend a price
# for (MACHINE_PRICES): model -> (machine price $/GPU/h, nominal VRAM sizes in GB). The price is the
# 24 Oct 2026 MACHINE_PRICES and only ranks the candidates; payout uses the reference's live tables.
GPU_REFERENCE_TABLE: dict[str, tuple[float, tuple[int, ...]]] = {
    "NVIDIA A10 Tensor Core GPU": (0.2, (24,)),
    "NVIDIA A100 80GB PCIe": (0.45, (80,)),
    "NVIDIA A100-SXM4-80GB": (0.7, (80,)),
    "NVIDIA A100-PCIE-40GB": (0.37, (40,)),
    "NVIDIA A100-SXM4-40GB": (0.47, (40,)),
    "NVIDIA A800 80GB PCIe": (0.31, (80,)),
    "NVIDIA B200": (5.6, (192,)),
    "NVIDIA B300 SXM6 AC": (8.0, (288,)),
    "NVIDIA B300 SXM6 PC": (8.0, (288,)),
    "NVIDIA CMP 170HX": (0.3, (8, 40, 64)),
    "NVIDIA GB300": (8.0, (288,)),
    "NVIDIA GeForce GTX 1060": (0.09, (3, 6)),
    "NVIDIA GeForce GTX 1070": (0.37, (8,)),
    "NVIDIA GeForce GTX 1070 Ti": (0.04, (8,)),
    "NVIDIA GeForce GTX 1080": (0.03, (8,)),
    "NVIDIA GeForce GTX 1080 Ti": (0.04, (11,)),
    "NVIDIA GeForce GTX 1660": (0.07, (6,)),
    "NVIDIA GeForce GTX 1660 SUPER": (0.05, (6,)),
    "NVIDIA GeForce GTX 1660 Ti": (0.07, (6,)),
    "NVIDIA GeForce RTX 2060": (0.04, (6, 12)),
    "NVIDIA GeForce RTX 2060 SUPER": (0.06, (8,)),
    "NVIDIA GeForce RTX 2070 SUPER": (0.07, (8,)),
    "NVIDIA GeForce RTX 2080 SUPER": (0.74, (8,)),
    "NVIDIA GeForce RTX 2080 Ti": (0.08, (11,)),
    "NVIDIA GeForce RTX 3050": (0.05, (6, 8)),
    "NVIDIA GeForce RTX 3060": (0.06, (8, 12)),
    "NVIDIA GeForce RTX 3060 Laptop GPU": (0.08, (6,)),
    "NVIDIA GeForce RTX 3060 Ti": (0.06, (8,)),
    "NVIDIA GeForce RTX 3070": (0.07, (8,)),
    "NVIDIA GeForce RTX 3070 Ti": (0.09, (8,)),
    "NVIDIA GeForce RTX 3080": (0.09, (10, 12)),
    "NVIDIA GeForce RTX 3080 Ti": (0.1, (12,)),
    "NVIDIA GeForce RTX 3090": (0.16, (24,)),
    "NVIDIA GeForce RTX 3090 Ti": (0.16, (24,)),
    "NVIDIA GeForce RTX 4060": (0.06, (8,)),
    "NVIDIA GeForce RTX 4060 Ti": (0.11, (8, 16)),
    "NVIDIA GeForce RTX 4070": (0.08, (12,)),
    "NVIDIA GeForce RTX 4070 SUPER": (0.09, (12,)),
    "NVIDIA GeForce RTX 4070 Ti": (0.11, (12,)),
    "NVIDIA GeForce RTX 4070 Ti SUPER": (0.18, (16,)),
    "NVIDIA GeForce RTX 4080": (0.16, (16,)),
    "NVIDIA GeForce RTX 4080 SUPER": (0.16, (16,)),
    "NVIDIA GeForce RTX 4090": (0.3, (24,)),
    "NVIDIA GeForce RTX 4090 D": (0.11, (24,)),
    "NVIDIA GeForce RTX 5060": (0.11, (8,)),
    "NVIDIA GeForce RTX 5060 Ti": (0.12, (8, 16)),
    "NVIDIA GeForce RTX 5070": (0.09, (12,)),
    "NVIDIA GeForce RTX 5070 Ti": (0.12, (16,)),
    "NVIDIA GeForce RTX 5080": (0.15, (16,)),
    "NVIDIA GeForce RTX 5090": (0.4, (32,)),
    "NVIDIA GeForce RTX 5090 D": (0.15, (32,)),
    "NVIDIA H100 80GB HBM3": (1.39, (80,)),
    "NVIDIA H100 NVL": (1.11, (94,)),
    "NVIDIA H100 PCIe": (1.3, (80,)),
    "NVIDIA H200": (3.65, (141,)),
    "NVIDIA H200 NVL": (2.9, (141,)),
    "NVIDIA H800 80GB HBM3": (0.88, (80,)),
    "NVIDIA H800 NVL": (0.8, (94,)),
    "NVIDIA H800 PCIe": (0.8, (80,)),
    "NVIDIA L4": (0.11, (24,)),
    "NVIDIA L40": (0.33, (48,)),
    "NVIDIA L40S": (0.38, (48,)),
    "NVIDIA Quadro P4000": (0.05, (8,)),
    "NVIDIA Quadro RTX 5000": (0.07, (16,)),
    "NVIDIA Quadro RTX 6000": (0.13, (24,)),
    "NVIDIA Quadro RTX 8000": (0.21, (48,)),
    "NVIDIA RTX 4500 Ada Generation": (0.2, (24,)),
    "NVIDIA RTX 5000 Ada Generation": (0.27, (32,)),
    "NVIDIA RTX 5880 Ada Generation": (0.34, (48,)),
    "NVIDIA RTX 6000 Ada Generation": (0.75, (48,)),
    "NVIDIA RTX 6000D": (0.82, (84,)),
    "NVIDIA RTX A2000": (0.06, (6, 12)),
    "NVIDIA RTX A4000": (0.12, (16,)),
    "NVIDIA RTX A4500": (0.13, (20,)),
    "NVIDIA RTX A5000": (0.16, (24,)),
    "NVIDIA RTX A6000": (0.42, (48,)),
    "NVIDIA RTX PRO 2000 Blackwell": (0.17, (16,)),
    "NVIDIA RTX PRO 4000 Blackwell": (0.2, (24,)),
    "NVIDIA RTX PRO 4500 Blackwell": (0.4, (32,)),
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": (0.4, (32,)),
    "NVIDIA RTX PRO 5000 Blackwell": (0.67, (48, 72)),
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": (1.19, (96,)),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": (1.19, (96,)),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": (1.19, (96,)),
    "NVIDIA RTX PRO 6000D Blackwell Workstation Edition": (0.82, (84,)),
    "NVIDIA T4 Tensor Core GPU": (0.15, (16,)),
    "NVIDIA TITAN RTX": (0.12, (24,)),
    "NVIDIA TITAN V": (0.1, (12,)),
    "NVIDIA TITAN Xp": (0.04, (12,)),
    "NVIDIA Tesla M40": (0.08, (12, 24)),
    "NVIDIA Tesla P100": (0.08, (12, 16)),
    "NVIDIA Tesla P40": (0.07, (24,)),
    "NVIDIA Tesla V100 Tensor Core GPU": (0.17, (16, 32)),
}


def pick_reference(vram_mib: float | None) -> str | None:
    """The cheapest candidate with a nominal size within REFERENCE_VRAM_BAND of `vram_mib` (MiB, as
    nvidia-smi reports it), else the cheapest with the largest size below it; a price tie goes to
    the first name. None for an unknown VRAM or one below every candidate."""
    if not vram_mib or vram_mib <= 0:
        return None
    low, high = REFERENCE_VRAM_BAND
    sizes = [
        (size_gb * 1024, price, model)
        for model, (price, sizes_gb) in GPU_REFERENCE_TABLE.items()
        for size_gb in sizes_gb
    ]
    in_band = [(price, model) for size, price, model in sizes if low <= size / vram_mib <= high]
    if in_band:
        return min(in_band)[1]
    below = [(-size, price, model) for size, price, model in sizes if size < vram_mib]
    return min(below)[2] if below else None
