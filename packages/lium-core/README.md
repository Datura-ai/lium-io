# lium-core

Shared library for the Lium platform, published to PyPI as [`lium-core`](https://pypi.org/project/lium-core/).
Imported from `Datura-ai/lium-core` with its history (DAH-3135); that repository is archived once this lands and PyPI's trusted publisher points here.

- `lium_core.shared_config` — `SharedConfigClient`, `SharedConfig` and the defaults every service agrees on.
  Consumers: the validator and miner in this repository (`neurons/validators`, `neurons/miners`), and the
  lium backend, portal backend and support bot in `Datura-ai/lium-platform`. Every consumer installs the
  PyPI release, not this directory: the validator lock pins `lium-core` 0.1.8, the miner lock 0.1.6.

## Develop

```bash
cd packages/lium-core
pip install -e . pytest
pytest -q tests
```

CI: `.github/workflows/lium-core-ci.yml` runs the tests and builds the wheel on every change under this directory.

## Release

`.github/workflows/lium-core-release.yml` (manual `workflow_dispatch`) builds from this directory and publishes
through PyPI trusted publishing (`release` environment). Bump `version` in `pyproject.toml` first.

### Release notes

- **0.1.13** (DAH-3648, 2026-09-19): `machine_prices` re-anchored on the 30-day GPU-hour-weighted paid median (the
  platform's `gpu_price_stat.lium_median_30d`) for the 16 models with at least 50 paid rentals in the window; 14 move
  (B300 6.40 → 8.00, B200 4.25 → 5.60, H200 2.85 → 3.65, H100 HBM3 1.494 → 1.39, H100 PCIe 1.1988 → 1.30,
  RTX 5090 0.65 → 0.40, RTX PRO 6000 SE and WE 1.0 → 1.19, RTX 6000 Ada 0.69 → 0.75, L40S 0.35 → 0.38, L40 0.36 → 0.33,
  A100 PCIe 0.36 → 0.45, A100 SXM 0.6923 → 0.70, RTX A6000 0.32 → 0.42), RTX 4090 0.30 and RTX 3090 0.16 already sit
  on theirs. A consumer that bumps its lock to this release changes what the validator's `RENTAL_PRICES_PER_HOUR`
  spreads from it.
- **0.1.12** (DAH-3230, unreleased): RTX PRO 6000 Server Edition at parity with the Workstation Edition; B300 6.40 (DAH-3542).
