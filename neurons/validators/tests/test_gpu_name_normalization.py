import pytest

from services.gpu_spec_table import normalize_gpu_model

V100 = "NVIDIA Tesla V100 Tensor Core GPU"


@pytest.mark.parametrize(
    "raw",
    [
        "Tesla V100-SXM2-16GB",
        "Tesla V100-SXM2-32GB",
        "Tesla V100-PCIE-16GB",
        "Tesla V100-PCIE-32GB",
        # 18 Nov: a DGX Station V100 was refused as an unsupported model
        "Tesla V100-DGXS-16GB",
        "Tesla V100-DGXS-32GB",
        "Tesla V100-SXM3-32GB",
        "Tesla V100-FHHL-16GB",
        "Tesla V100-SXM2-32GB-LS",
        " Tesla V100-DGXS-32GB ",
    ],
)
def test_every_v100_board_name_is_a_v100(raw):
    assert normalize_gpu_model(raw) == V100


@pytest.mark.parametrize("raw", ["Tesla V100S-PCIE-32GB", "Tesla V100-PCIE-24GB", "Tesla V100-PCIE-32GB-extra"])
def test_names_outside_the_v100_boards_pass_through(raw):
    # V100S is its own card; a size no V100 has is not one
    assert normalize_gpu_model(raw) == raw
