"""Test-only constants.

The LIVE burn emission split is sourced from shared config at runtime via
core.config.get_total_burn_emission() (DAH-2274) — the backend is the source of
truth. This is the validator's expected production value: it is what the tests
assert against and what the test harness serves into the hermetic shared config
(see conftest.py). (1 - TOTAL_BURN_EMISSION) is the rented mining pool.
"""

TOTAL_BURN_EMISSION = 0.87
