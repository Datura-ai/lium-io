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

- **0.1.13** (DAH-3648, 2026-09-18): `machine_prices` re-anchored on the 30-day paid median for the 15 models with
  at least 50 paid rentals in the window (B300 6.40 → 8.00, B200 4.25 → 5.60, H200 2.85 → 3.25, H100 HBM3 1.494 → 1.30,
  H100 PCIe 1.1988 → 1.50, RTX 5090 0.65 → 0.60, RTX 4090 0.30 → 0.32, RTX PRO 6000 SE and WE 1.0 → 1.25, L40S 0.35 → 0.38,
  L40 0.36 → 0.33, A100 PCIe 0.36 → 0.30, A100 SXM 0.6923 → 0.68, RTX A6000 0.32 → 0.42, RTX 3090 0.16 → 0.18).
  A consumer that bumps its lock to this release changes what the validator's `RENTAL_PRICES_PER_HOUR` spreads from it.
- **0.1.12** (DAH-3230, unreleased): RTX PRO 6000 Server Edition at parity with the Workstation Edition; B300 6.40 (DAH-3542).
