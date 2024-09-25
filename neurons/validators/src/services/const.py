GPU_MAX_SCORES = {
    # Latest Gen NVIDIA GPUs (Averaged if applicable)
    "NVIDIA H100": (3.49 + 3.29 + 2.89) / 3,  # Average of H100 SXM, H100 NVL, and H100 PCIe
    "NVIDIA RTX 4090": 0.69,
    "NVIDIA RTX 4000 Ada": 0.38,
    "NVIDIA RTX 6000 Ada": 1.03,
    "NVIDIA L4": 0.43,
    "NVIDIA L40": (0.99 + 1.03) / 2,  # Average of L40 and L40S
    "NVIDIA RTX 2000 Ada": 0.28,

    # Previous Gen NVIDIA GPUs (Averaged if applicable)
    "NVIDIA A100": (1.64 + 1.89) / 2,  # Average of A100 PCIe and A100 SXM
    "NVIDIA RTX A6000": 0.76,
    "NVIDIA RTX A5000": 0.43,
    "NVIDIA RTX A4000": 0.32,
    "NVIDIA A40": 0.39,
    "NVIDIA RTX 3090": 0.43,
}
