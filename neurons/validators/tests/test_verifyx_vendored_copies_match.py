"""The validator and executor images each vendor libverifyx; the validator fails a hash mismatch hard and
library refresh is off by default, so both copies of each arch must be the same bytes."""

from __future__ import annotations

import hashlib
import pathlib

import pytest

NEURONS = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["libverifyx.so", "libverifyx-aarch64.so"])
def test_validator_and_executor_copies_are_identical(name):
    validator = NEURONS / "validators" / name
    executor = NEURONS / "executor" / name
    assert validator.is_file() and executor.is_file(), (validator, executor)
    v = hashlib.sha256(validator.read_bytes()).hexdigest()
    e = hashlib.sha256(executor.read_bytes()).hexdigest()
    assert v == e, f"{name}: validator {v} != executor {e}; re-vendor both copies together"
