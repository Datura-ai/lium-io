# lium-core

Shared library for the Lium platform, published to PyPI as [`lium-core`](https://pypi.org/project/lium-core/).
Imported from `Datura-ai/lium-core` with its history (DAH-3135); that repository is history only.

- `lium_core.shared_config` — `SharedConfigClient`, `SharedConfig` and the defaults every service agrees on.
  Consumers: the validator and miner in this repository (`neurons/validators`, `neurons/miners`), and the
  lium backend, portal backend and support bot in `Datura-ai/lium-platform`.

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
