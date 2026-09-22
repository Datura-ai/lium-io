from lium_core.shared_config.model import SharedConfig

B300_SXM6_AC = "NVIDIA B300 SXM6 AC"
# Provider-observed on real hardware, 21 Sep 2026 (nvidia-smi: name NVIDIA B300 SXM6 PC, 275040 MiB, all
# 8 GPUs of the host); not in NVIDIA's public chip list, which has only the AC spelling. Listed as the AC
# card's alias and never a row of its own: each table below derives it from the AC entry, so a re-price
# of the AC card moves both names.
B300_SXM6_PC = "NVIDIA B300 SXM6 PC"


def _with_pc_alias(table: dict[str, float]) -> dict[str, float]:
    return {**table, B300_SXM6_PC: table[B300_SXM6_AC]}


DEFAULT_SHARED_CONFIG = SharedConfig(
    # The base listing price per GPU model (USD per GPU-hour): a listing sits between machine_min_price_rate x and
    # machine_max_price_rate x of it. Rule (owner, 2026-09-18): every model with >= 50 paid rentals in the trailing
    # 30 days is its 30-day paid median, GPU-hour-weighted, floored to the cent -- the platform's
    # gpu_price_stat.lium_median_30d (rental_history.price_per_gpu over rentals started in the window, each row
    # weighted gpu_count x rental_hours, lower weighted median); models under 50 rentals or with none keep their value.
    # lium-platform#558 moves the backend's MACHINE_PRICES (core/constants.py) to the same 16 values.
    machine_prices=_with_pc_alias({
        "NVIDIA B300 SXM6 AC": 8.00,
        "NVIDIA B200": 5.60,
        "NVIDIA H200": 3.65,
        "NVIDIA H200 NVL": 2.90,
        "NVIDIA H100 80GB HBM3": 1.39,
        "NVIDIA H100 NVL": 1.11,
        "NVIDIA H100 PCIe": 1.30,
        "NVIDIA H800 80GB HBM3": 0.88,
        "NVIDIA H800 NVL": 0.80,
        "NVIDIA H800 PCIe": 0.80,
        "NVIDIA GeForce RTX 5090": 0.40,
        "NVIDIA GeForce RTX 4090": 0.30,
        "NVIDIA GeForce RTX 4090 D": 0.11,
        "NVIDIA RTX 4000 Ada Generation": 0.16,
        "NVIDIA RTX 6000 Ada Generation": 0.75,
        # same card for a renter; anchored at parity (owner's rule, 2026-09-08) at the Server Edition's weighted
        # median (942 of the 1,041 rentals; the Workstation Edition alone reads 1.00)
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1.19,
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 1.19,
        "NVIDIA L4": 0.11,
        "NVIDIA L40S": 0.38,
        "NVIDIA L40": 0.33,
        "NVIDIA RTX 2000 Ada Generation": 0.07,
        "NVIDIA A100 80GB PCIe": 0.45,
        "NVIDIA A100-SXM4-80GB": 0.70,
        "NVIDIA RTX A6000": 0.42,
        "NVIDIA RTX A5000": 0.16,
        "NVIDIA RTX A4500": 0.13,
        "NVIDIA RTX A4000": 0.12,
        "NVIDIA A40": 0.12,
        "NVIDIA A30": 0.10,
        "NVIDIA GeForce RTX 3090": 0.16,
    }),
    required_deposit_amount=_with_pc_alias({
        "NVIDIA B300 SXM6 AC": 0.274,
        "NVIDIA B200": 0.223,
        "NVIDIA H200": 0.158,
        "NVIDIA H200 NVL": 0.131,
        "NVIDIA H100 80GB HBM3": 0.103,
        "NVIDIA H100 NVL": 0.086,
        "NVIDIA H100 PCIe": 0.086,
        "NVIDIA H800 80GB HBM3": 0.051,
        "NVIDIA H800 NVL": 0.045,
        "NVIDIA H800 PCIe": 0.045,
        "NVIDIA GeForce RTX 5090": 0.014,
        "NVIDIA GeForce RTX 4090": 0.010,
        "NVIDIA GeForce RTX 4090 D": 0.008,
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": 0.0425,
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 0.0459,
        "NVIDIA RTX 6000 Ada Generation": 0.017,
        "NVIDIA L4": 0.008,
        "NVIDIA L40S": 0.027,
        "NVIDIA L40": 0.024,
        "NVIDIA A100 80GB PCIe": 0.027,
        "NVIDIA A100-SXM4-80GB": 0.031,
        "NVIDIA RTX A6000": 0.018,
        "NVIDIA RTX A5000": 0.009,
        "NVIDIA RTX A4500": 0.008,
        "NVIDIA RTX A4000": 0.008,
        "NVIDIA GeForce RTX 3090": 0.008,
    }),
    gpu_architectures={
        # Blackwell (sm_100/120)
        "NVIDIA B200": {"arch": "blackwell", "min_cuda": 12.8, "compute_cap": "sm_100"},
        "NVIDIA GeForce RTX 5090": {"arch": "blackwell", "min_cuda": 12.8, "compute_cap": "sm_120"},
        # Hopper (sm_90)
        "NVIDIA H100 80GB HBM3": {"arch": "hopper", "min_cuda": 11.8, "optimal_cuda": 12.2, "compute_cap": "sm_90"},
        "NVIDIA H100 NVL": {"arch": "hopper", "min_cuda": 11.8, "optimal_cuda": 12.2, "compute_cap": "sm_90"},
        "NVIDIA H100 PCIe": {"arch": "hopper", "min_cuda": 11.8, "optimal_cuda": 12.2, "compute_cap": "sm_90"},
        "NVIDIA H200": {"arch": "hopper", "min_cuda": 11.8, "optimal_cuda": 12.2, "compute_cap": "sm_90"},
        "NVIDIA H800 80GB HBM3": {"arch": "hopper", "min_cuda": 11.8, "optimal_cuda": 12.2, "compute_cap": "sm_90"},
        "NVIDIA H800 NVL": {"arch": "hopper", "min_cuda": 11.8, "optimal_cuda": 12.2, "compute_cap": "sm_90"},
        "NVIDIA H800 PCIe": {"arch": "hopper", "min_cuda": 11.8, "optimal_cuda": 12.2, "compute_cap": "sm_90"},
        # Ada Lovelace (sm_89)
        "NVIDIA GeForce RTX 4090": {"arch": "ada", "min_cuda": 11.8, "compute_cap": "sm_89"},
        "NVIDIA GeForce RTX 4090 D": {"arch": "ada", "min_cuda": 11.8, "compute_cap": "sm_89"},
        "NVIDIA L4": {"arch": "ada", "min_cuda": 11.8, "compute_cap": "sm_89"},
        "NVIDIA L40": {"arch": "ada", "min_cuda": 11.8, "compute_cap": "sm_89"},
        "NVIDIA L40S": {"arch": "ada", "min_cuda": 11.8, "compute_cap": "sm_89"},
        "NVIDIA RTX 2000 Ada Generation": {"arch": "ada", "min_cuda": 11.8, "compute_cap": "sm_89"},
        "NVIDIA RTX 4000 Ada Generation": {"arch": "ada", "min_cuda": 11.8, "compute_cap": "sm_89"},
        "NVIDIA RTX 6000 Ada Generation": {"arch": "ada", "min_cuda": 11.8, "compute_cap": "sm_89"},
        # Ampere (sm_86)
        "NVIDIA A100 80GB PCIe": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
        "NVIDIA A100-SXM4-80GB": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
        "NVIDIA A30": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
        "NVIDIA A40": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
        "NVIDIA RTX A4000": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
        "NVIDIA RTX A4500": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
        "NVIDIA RTX A5000": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
        "NVIDIA RTX A6000": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
        "NVIDIA GeForce RTX 3090": {"arch": "ampere", "min_cuda": 11.0, "compute_cap": "sm_86"},
    },
    driver_cuda_map={
        450: 11.0,
        470: 11.4,
        495: 11.5,
        510: 11.6,
        515: 11.7,
        525: 12.0,
        530: 12.1,
        535: 12.2,
        545: 12.3,
        550: 12.4,
        555: 12.5,
        560: 12.6,
        565: 12.7,
        570: 12.8,
        575: 12.8,
        580: 13.0,
        590: 13.1,
    },
    machine_max_price_rate=3.0,
    machine_min_price_rate=0.5,
    soft_limit_price_rate=1.1,
    rental_fees_rate=0.9,
    collateral_days=7,
    collateral_contract_address="0x7DCCb5659c70Ce2104A9bb79E9E257473ECbe628",
    bittensor_netuid=51,
    volume_gb_hour_price_usd=0.00005,
    max_initial_port_count=200,
    total_burn_emission=0.91,
    require_storage_limit_supported=False,
    payout_delay_days=2,
    payout_processing_hour_utc=17,
)
