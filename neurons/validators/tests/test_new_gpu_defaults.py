from incentive.config import BASE_GPU_MAP, IncentiveConfig
from services.const import GPU_MODEL_RATES, MACHINE_PRICES, REQUIRED_DEPOSIT_AMOUNT


NEW_GPU_MODELS = {
    "NVIDIA GeForce RTX 3080": "RTX 3080",
    "NVIDIA GeForce RTX 3080 Ti": "RTX 3080 Ti",
    "NVIDIA GeForce RTX 3090 Ti": "RTX 3090 Ti",
    "NVIDIA GeForce RTX 4070 SUPER": "RTX 4070 SUPER",
    "NVIDIA GeForce RTX 4070 Ti": "RTX 4070 Ti",
    "NVIDIA GeForce RTX 4080": "RTX 4080",
    "NVIDIA GeForce RTX 4080 SUPER": "RTX 4080 SUPER",
    "NVIDIA GeForce RTX 5070": "RTX 5070",
    "NVIDIA GeForce RTX 5070 Ti": "RTX 5070 Ti",
    "NVIDIA GeForce RTX 5080": "RTX 5080",
    "NVIDIA A10 Tensor Core GPU": "A10",
    "NVIDIA T4 Tensor Core GPU": "T4",
    "NVIDIA Tesla V100 Tensor Core GPU": "V100",
    "NVIDIA RTX 5000 Ada Generation": "RTX 5000 Ada Generation",
    "NVIDIA RTX 5880 Ada Generation": "RTX 5880 Ada Generation",
}


def test_new_gpu_models_are_supported_with_zero_default_portion():
    for model in NEW_GPU_MODELS:
        assert model in MACHINE_PRICES
        assert model in REQUIRED_DEPOSIT_AMOUNT
        assert model in GPU_MODEL_RATES
        assert GPU_MODEL_RATES[model] == 0.0


def test_new_gpu_models_are_excluded_from_unrented_pool_by_default():
    config = IncentiveConfig()

    for model, base_model in NEW_GPU_MODELS.items():
        assert BASE_GPU_MAP[model] == base_model
        assert base_model not in config.rental_incentive_gpu_types
        assert config.max_unrented_gpus[base_model] == {}
