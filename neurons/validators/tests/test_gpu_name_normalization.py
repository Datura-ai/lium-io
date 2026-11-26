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


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        # ticket-1398 (25 Nov): the add-node check refused a P100 the driver names by its board
        ("Tesla P100-PCIE-16GB", "NVIDIA Tesla P100"),
        ("Tesla P100-PCIE-12GB", "NVIDIA Tesla P100"),
        ("Tesla P100-SXM2-16GB", "NVIDIA Tesla P100"),
        ("Tesla P40", "NVIDIA Tesla P40"),
        ("Tesla M40", "NVIDIA Tesla M40"),
        ("Tesla M40 24GB", "NVIDIA Tesla M40"),
        (" Tesla P100-PCIE-16GB ", "NVIDIA Tesla P100"),
    ],
)
def test_every_tesla_board_name_is_its_rate_key(raw, canonical):
    assert normalize_gpu_model(raw) == canonical


@pytest.mark.parametrize("raw", ["Tesla K80", "Tesla P4", "Tesla P100-PCIE-16GB-extra"])
def test_a_tesla_with_no_rate_key_passes_through(raw):
    assert normalize_gpu_model(raw) == raw


def test_the_preflight_and_reference_lookup_accept_a_p100_board_name():
    from services.gpu_spec_table import reference_gpu_model

    assert reference_gpu_model("Tesla P100-PCIE-16GB", 16384) == "NVIDIA Tesla P100"
