from core.const import REQUIRED_DEPOSIT_AMOUNT


NEW_GPU_MODELS = {
    "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA GeForce RTX 4070 SUPER",
    "NVIDIA GeForce RTX 4070 Ti",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA GeForce RTX 5070",
    "NVIDIA GeForce RTX 5070 Ti",
    "NVIDIA GeForce RTX 5080",
    "NVIDIA A10 Tensor Core GPU",
    "NVIDIA T4 Tensor Core GPU",
    "NVIDIA Tesla V100 Tensor Core GPU",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX 5880 Ada Generation",
}


def test_new_gpu_models_have_deposit_constants():
    for model in NEW_GPU_MODELS:
        assert model in REQUIRED_DEPOSIT_AMOUNT
        assert REQUIRED_DEPOSIT_AMOUNT[model] > 0
