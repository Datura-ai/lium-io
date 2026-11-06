"""G33: challenge seeds, matrix dimensions and obfuscation names come from the OS random source, not the clock or
the Mersenne Twister whose state a node can recover from the seeds it is sent."""

import random
import secrets

from services import file_encrypt_service, matrix_validation_service as mvs
from services.matrix_validation_service import VerifierParams


def _params() -> VerifierParams:
    params = VerifierParams(dim_n=0, dim_k=0, seed=0, uuid="")
    params.generate()
    return params


def test_matmul_challenges_do_not_follow_the_seeded_global_generator():
    random.seed(1234)
    first = _params()
    random.seed(1234)
    second = _params()

    assert (first.dim_n, first.dim_k, first.seed) != (second.dim_n, second.dim_k, second.seed)


def test_matmul_seed_and_dimensions_stay_in_range():
    params = _params()

    assert 1 <= params.seed < 2**31
    assert 1900 <= params.dim_n <= 2000
    assert 2000000 <= params.dim_k <= 2586932


def test_generators_are_the_os_source():
    assert isinstance(mvs._rng, secrets.SystemRandom)
    assert isinstance(file_encrypt_service._rng, secrets.SystemRandom)
