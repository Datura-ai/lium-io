"""The matmul work-proof must fit next to what a confidential-computing card already holds.

DAH-2850: get_max_matrix_dimensions sized the challenge to capacity minus 2 GB regardless of
memory in use. A B200 in CC mode holds ~1.8 GB at idle, so d_B could not be allocated and the
node was zeroed by gpu.validate.capability. Allocation shape mirrors lium-gpu-verifier
DMCompVerify.cu processChallengeResult: d_A and d_B (dim_n x dim_k doubles each) plus d_C
(dim_n x dim_n doubles).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import services.matrix_validation_service as mvs
from services.gpu_spec_table import GPU_VRAM_SIZES_MB

B200_CAPACITY_MB = 183359  # NVML total on a 192 GB B200
CC_MODE_IDLE_USED_MB = 1766  # 146.88.195.16 with --set-cc-mode=on, 2026-09-03
CUDA_CONTEXT_MB = 600  # what the probe process itself takes on a Blackwell card
OLD_HEADROOM_MB = 2048  # the value that OOMed d_B


def _matmul_bytes(dim_n: int, dim_k: int) -> int:
    return (2 * dim_n * dim_k + dim_n * dim_n) * 8


def _cc_mode_b200_free_bytes() -> int:
    return (B200_CAPACITY_MB - CC_MODE_IDLE_USED_MB - CUDA_CONTEXT_MB) * 1024**2


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(mvs, "DMCompVerifyWrapper", lambda *_a, **_kw: MagicMock())
    return mvs.ValidationService()


@pytest.mark.parametrize("dim_n", [1900, 2000])
def test_matmul_fits_beside_cc_mode_reserve_on_b200(service, dim_n):
    # Arrange
    free_bytes = _cc_mode_b200_free_bytes()

    # Act
    dim_k = int(service.get_max_matrix_dimensions(B200_CAPACITY_MB, dim_n))

    # Assert
    assert dim_k > 0
    assert _matmul_bytes(dim_n, dim_k) <= free_bytes


@pytest.mark.parametrize("dim_n", [1900, 2000])
def test_old_headroom_did_not_fit_beside_cc_mode_reserve(service, monkeypatch, dim_n):
    # Arrange: pin the regression so nobody lowers the constant back
    monkeypatch.setattr(mvs, "MATMUL_VRAM_HEADROOM_MB", OLD_HEADROOM_MB)
    free_bytes = _cc_mode_b200_free_bytes()

    # Act
    dim_k = int(service.get_max_matrix_dimensions(B200_CAPACITY_MB, dim_n))

    # Assert
    assert _matmul_bytes(dim_n, dim_k) > free_bytes


def test_headroom_still_leaves_a_real_challenge_on_b200(service):
    # Arrange
    dim_n = 2000

    # Act
    dim_k = int(service.get_max_matrix_dimensions(B200_CAPACITY_MB, dim_n))

    # Assert: the proof still touches the bulk of the card, not a token slice of it
    assert _matmul_bytes(dim_n, dim_k) >= 0.95 * B200_CAPACITY_MB * 1024**2


@pytest.mark.parametrize(
    "capacity_mb", sorted({size for sizes in GPU_VRAM_SIZES_MB.values() for size in sizes})
)
def test_every_registered_card_keeps_a_positive_challenge(service, capacity_mb):
    # Arrange
    dim_n = 2000

    # Act
    dim_k = int(service.get_max_matrix_dimensions(capacity_mb, dim_n))

    # Assert: never a non-positive dim_k, and at least half of the card stays in the proof
    assert dim_k > 0
    assert _matmul_bytes(dim_n, dim_k) >= 0.5 * capacity_mb * 1024**2
