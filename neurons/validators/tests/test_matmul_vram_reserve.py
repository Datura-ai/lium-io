"""The matmul work-proof must fit next to what a confidential-computing card already holds.

Allocation shape mirrors lium-gpu-verifier DMCompVerify.cu processChallengeResult: d_A and d_B
(dim_n x dim_k doubles each) plus d_C (dim_n x dim_n doubles). Why the reserve is what it is:
see MATMUL_VRAM_RESERVE_MAX_MB in matrix_validation_service.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import services.matrix_validation_service as mvs
from services.const import GPU_HELD_VRAM_MB_LIMIT
from services.gpu_spec_table import GPU_VRAM_SIZES_MB

B200_CAPACITY_MB = 183359  # NVML total on a 192 GB B200
CC_MODE_IDLE_USED_MB = 1766  # 146.88.195.16 with --set-cc-mode=on, 2026-09-03
PROBE_CUDA_CONTEXT_MB = 600  # measured on a Blackwell card
RESERVE_BEFORE_DAH2850_MB = 2048
RESERVE_ROUNDING_SLACK_MB = 64  # dim_k is floored, so the reserve overshoots by ~30 MB


def _matmul_allocation_bytes(dim_n: int, dim_k: int) -> int:
    return (2 * dim_n * dim_k + dim_n * dim_n) * 8


def _cc_mode_b200_free_bytes() -> int:
    return (B200_CAPACITY_MB - CC_MODE_IDLE_USED_MB - PROBE_CUDA_CONTEXT_MB) * 1024**2


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
    assert _matmul_allocation_bytes(dim_n, dim_k) <= free_bytes


@pytest.mark.parametrize("dim_n", [1900, 2000])
def test_old_flat_reserve_did_not_fit_beside_cc_mode(service, monkeypatch, dim_n):
    # Arrange: pin the regression so nobody lowers the constant back
    monkeypatch.setattr(mvs, "MATMUL_VRAM_RESERVE_MAX_MB", RESERVE_BEFORE_DAH2850_MB)
    free_bytes = _cc_mode_b200_free_bytes()

    # Act
    dim_k = int(service.get_max_matrix_dimensions(B200_CAPACITY_MB, dim_n))

    # Assert
    assert _matmul_allocation_bytes(dim_n, dim_k) > free_bytes


def test_reserve_clears_the_vram_an_idle_card_may_legally_hold(service):
    # Arrange: gpu_usage lets an idle card hold up to GPU_HELD_VRAM_MB_LIMIT without penalty,
    # so the matmul has to survive a card sitting right at that limit
    dim_n = 2000

    # Act
    dim_k = int(service.get_max_matrix_dimensions(B200_CAPACITY_MB, dim_n))
    reserved_mb = B200_CAPACITY_MB - _matmul_allocation_bytes(dim_n, dim_k) / 1024**2

    # Assert
    assert reserved_mb >= GPU_HELD_VRAM_MB_LIMIT + PROBE_CUDA_CONTEXT_MB


@pytest.mark.parametrize(
    "capacity_mb", sorted({size for sizes in GPU_VRAM_SIZES_MB.values() for size in sizes})
)
def test_every_registered_card_keeps_a_positive_challenge_and_bounded_reserve(service, capacity_mb):
    # Arrange
    dim_n = 2000

    # Act
    dim_k = int(service.get_max_matrix_dimensions(capacity_mb, dim_n))
    reserved_mb = capacity_mb - _matmul_allocation_bytes(dim_n, dim_k) / 1024**2

    # Assert
    assert dim_k > 0
    assert RESERVE_BEFORE_DAH2850_MB <= reserved_mb
    assert reserved_mb <= mvs.MATMUL_VRAM_RESERVE_MAX_MB + RESERVE_ROUNDING_SLACK_MB
