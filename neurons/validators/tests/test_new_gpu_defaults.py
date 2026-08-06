from incentive.config import BASE_GPU_MAP, IncentiveConfig
from services.const import GPU_MODEL_RATES

NEW_GPU_MODELS = {
    "NVIDIA GeForce GTX 1060": "GTX 1060",
    "NVIDIA GeForce GTX 1070": "GTX 1070",
    "NVIDIA GeForce GTX 1070 Ti": "GTX 1070 Ti",
    "NVIDIA GeForce GTX 1080": "GTX 1080",
    "NVIDIA GeForce GTX 1080 Ti": "GTX 1080 Ti",
    "NVIDIA GeForce GTX 1660": "GTX 1660",
    "NVIDIA GeForce GTX 1660 SUPER": "GTX 1660 SUPER",
    "NVIDIA GeForce GTX 1660 Ti": "GTX 1660 Ti",
    "NVIDIA GeForce RTX 2060": "RTX 2060",
    "NVIDIA GeForce RTX 2060 SUPER": "RTX 2060 SUPER",
    "NVIDIA GeForce RTX 2070 SUPER": "RTX 2070 SUPER",
    "NVIDIA GeForce RTX 2080 SUPER": "RTX 2080 SUPER",
    "NVIDIA GeForce RTX 2080 Ti": "RTX 2080 Ti",
    "NVIDIA GeForce RTX 3050": "RTX 3050",
    "NVIDIA GeForce RTX 3060": "RTX 3060",
    "NVIDIA GeForce RTX 3060 Laptop GPU": "RTX 3060 Laptop",
    "NVIDIA GeForce RTX 3060 Ti": "RTX 3060 Ti",
    "NVIDIA GeForce RTX 3070": "RTX 3070",
    "NVIDIA GeForce RTX 3070 Ti": "RTX 3070 Ti",
    "NVIDIA GeForce RTX 3080": "RTX 3080",
    "NVIDIA GeForce RTX 3080 Ti": "RTX 3080 Ti",
    "NVIDIA GeForce RTX 3090 Ti": "RTX 3090 Ti",
    "NVIDIA GeForce RTX 4060": "RTX 4060",
    "NVIDIA GeForce RTX 4060 Ti": "RTX 4060 Ti",
    "NVIDIA GeForce RTX 4070": "RTX 4070",
    "NVIDIA GeForce RTX 4070 SUPER": "RTX 4070 SUPER",
    "NVIDIA GeForce RTX 4070 Ti": "RTX 4070 Ti",
    "NVIDIA GeForce RTX 4070 Ti SUPER": "RTX 4070 Ti SUPER",
    "NVIDIA GeForce RTX 4080": "RTX 4080",
    "NVIDIA GeForce RTX 4080 SUPER": "RTX 4080 SUPER",
    "NVIDIA GeForce RTX 5060": "RTX 5060",
    "NVIDIA GeForce RTX 5060 Ti": "RTX 5060 Ti",
    "NVIDIA GeForce RTX 5070": "RTX 5070",
    "NVIDIA GeForce RTX 5070 Ti": "RTX 5070 Ti",
    "NVIDIA GeForce RTX 5080": "RTX 5080",
    "NVIDIA A10 Tensor Core GPU": "A10",
    "NVIDIA Quadro P4000": "Quadro P4000",
    "NVIDIA Quadro RTX 5000": "Quadro RTX 5000",
    "NVIDIA Quadro RTX 6000": "Quadro RTX 6000",
    "NVIDIA Quadro RTX 8000": "Quadro RTX 8000",
    "NVIDIA RTX A2000": "RTX A2000",
    "NVIDIA T4 Tensor Core GPU": "T4",
    "NVIDIA Tesla V100 Tensor Core GPU": "V100",
    "NVIDIA Tesla M40": "Tesla M40",
    "NVIDIA Tesla P100": "Tesla P100",
    "NVIDIA Tesla P40": "Tesla P40",
    "NVIDIA TITAN RTX": "TITAN RTX",
    "NVIDIA TITAN V": "TITAN V",
    "NVIDIA TITAN Xp": "TITAN Xp",
    "NVIDIA RTX 5000 Ada Generation": "RTX 5000 Ada Generation",
    "NVIDIA RTX 5880 Ada Generation": "RTX 5880 Ada Generation",
    "NVIDIA RTX PRO 4000 Blackwell": "RTX PRO 4000",
    "NVIDIA RTX PRO 5000 Blackwell": "RTX PRO 5000",
    "NVIDIA RTX PRO 2000 Blackwell": "RTX PRO 2000",
    "NVIDIA RTX PRO 4500 Blackwell": "RTX PRO 4500",
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": "RTX PRO 4500",
    "NVIDIA RTX PRO 6000D Blackwell Workstation Edition": "RTX PRO 6000D",
    "NVIDIA RTX 6000D": "RTX PRO 6000D",
    "NVIDIA RTX 4500 Ada Generation": "RTX 4500 Ada Generation",
    "NVIDIA A800 80GB PCIe": "A800",
    "NVIDIA CMP 170HX": "CMP 170HX",
}


def test_new_gpu_models_are_supported_with_zero_default_portion():
    for model in NEW_GPU_MODELS:
        assert model in GPU_MODEL_RATES
        assert GPU_MODEL_RATES[model] == 0.0


def test_new_gpu_models_are_excluded_from_unrented_pool_by_default():
    config = IncentiveConfig()

    for model, base_model in NEW_GPU_MODELS.items():
        assert BASE_GPU_MAP[model] == base_model
        assert base_model not in config.rental_incentive_gpu_types
        assert config.max_unrented_gpus[base_model] == {}
