# e2e — the subnet loop on one machine

A real executor (`neurons/executor`, this checkout) on its own docker daemon, a real miner (`neurons/miners`) that
owns it, and the validator's own services (`neurons/validators/src`) driving them the way `Validator.sync` and the
connector do — sign in over REST, install the cycle's SSH key through the miner, SSH in, upload and run the
obfuscated scrape, walk the check pipeline, publish the verdict, rent a container, tear it down. The only thing
missing is the chain: the subtensor client is never constructed, the miner never syncs, and the hotkeys are three
throwaway keys derived from BIP-39 test vectors (`stack.env`). Everything else is the code that runs in production.

CI runs it on every PR that touches a neuron, `datura/` or `e2e/` (`.github/workflows/test.yml`, job **`e2e-gate`**).
The lium-platform repo has the same gate for the platform side (its stub validator/executor stand in for this
repo); together they cover the loop from the renter's `lium up` to the container on the provider's host.

## The stack (`docker-compose.e2e.yml`)

| service | image | role |
|---|---|---|
| `dind` | `docker:27-dind` + a no-op `nvidia-container-runtime-hook` (`dind/`) | the executor host's dockerd. Rental containers land here and publish their ports on this address — which is also the executor's, because the executor shares this network namespace like an executor shares its host's. The hook lets `docker run --gpus …` start a GPU-less container on a runner with no GPU |
| `executor` | built from `neurons/executor` | the real executor: FastAPI on :8001 (`/version`, `/upload_ssh_key`, …), sshd on :22, `DOCKER_HOST` → dind. Trusts the e2e validator through `executor/trust_anchor.py` (mounted as `core/config_override.py`) (the mechanism the `:dev` images use) and the e2e miner through `MINER_HOTKEY_SS58_ADDRESS` |
| `miner` + `miner-db` | built from `neurons/miners`, Postgres 15 | the real miner: wallet from the stack mnemonic (`run.sh`), `alembic upgrade head`, uvicorn. `DEBUG_SKIP_SYNC_FLOW` keeps it off the chain; `DEFAULT_VALIDATOR_HOTKEY` is the e2e validator (the real registration check, not the debug bypass). `miner/seed.sql` gives it two executors: the stack's and one nobody answers on |
| `redis`, `val-db` | Redis 7, Postgres 15 | the validator's stores (`MACHINE_SPEC_CHANNEL` is where verdicts go) |
| `tester` | `tester/Dockerfile` = `neurons/validators/Dockerfile` + pytest | the validator. Each suite is one `docker compose run tester pytest tests/<suite>`; the suites import `services.ioc` and call `MinerService` / `TaskService` / `DockerService` directly |

Addresses are fixed on the `172.30.0.0/24` compose network (`stack.env`) because the miner stores executors by IP.

## Run it

Needs Docker with compose v2 (`additional_contexts` → v2.17+), `make`, ~8 GB of disk for the images. Docker-only:
`dind` is privileged. A CI runner, a dev box (`tools/aws_run.sh` in the loop) or a Lium DinD pod; not a laptop.

```sh
cd e2e
make e2e          # build, up (+seed), the three suites, down
make e2e-full     # the merge gate exactly as CI runs it (gate.sh, below)
make build up     # keep the stack for test-writing; `make test-protocol` etc.; `make logs`; `make down`
E2E_GPU=1 make e2e-full   # on a GPU host: the executor uses this machine's dockerd + GPUs (docker-compose.gpu.yml)
```

`make up` is idempotent (`--wait` on every healthcheck, seed is `ON CONFLICT DO NOTHING`). Cold build ≈ 4 min on
16 vCPU, warm ≈ 10 s; the whole gate ≈ 10 min cold.

## The merge gate (`make e2e-full` = `gate.sh`, CI job `e2e-gate`)

Same script as lium-platform's: build → up → every `tests/<name>` suite → logs → down, each under GNU `timeout`
(`T_BUILD` 25m, `T_UP` 8m, `T_SUITE` 20m per suite, SIGKILL 30 s after SIGTERM), every suite run even after one
fails, `artifacts/` always holding `timings.txt`, `summary.md`, `<suite>-junit.xml`, `compose.log`, `compose-ps.txt`,
`executor-docker-ps.txt` and the suites' JSON dumps. CI uploads the directory on every run and posts `summary.md` as
one sticky PR comment. A new suite = `tests/<name>/` + nothing else (`test-%` in the Makefile).

## What the suites prove

**`tests/protocol`** — the wire contract, accepted for the right keys and refused for everything else:
executor `/version`; the miner lists both seeded executors for the validator's signed `POST /executors` and refuses a
stranger's signature under the validator's hotkey (401); the four REST headers (`AuthenticationPayload`) are
accepted, the miner installs the key on the executor and the key opens it over SSH, `ssh-pubkey-remove` closes it
again; a stranger's signature → 401, an unregistered validator → 403, a stale or future timestamp → 401, headers for
another miner refused; on the executor, the miner+validator double signature is accepted (`/upload_ssh_key` →
SSH coordinates → SSH works → `/remove_ssh_key`), a spoofed validator signature → 401, a spoofed miner signature →
401/403, a substituted public key (#744) → 400.

**`tests/cycle`** — `MinerService.request_job_to_miner` (the REST path `lium mine` providers are on):
the pyarmor+PyInstaller scrape build, key install through the miner, SSH, scrape upload + run, the check pipeline,
a `JobResult` for OUR executor (not the synthetic `1111…` failure), with the deterministic verdict for the host — on
a GPU-less runner the scrape reports `gpu.count == 0` and the pipeline refuses the node with **`GPU_COUNT_ZERO`**,
score 0 (a provider with a dead driver gets exactly this); with `E2E_GPU=1` `gpu_count ≥ 1` and a model. Then
`publish_machine_specs` → the `MACHINE_SPEC_CHANNEL` message the connector relays to the platform, with the fields
the platform reads (`executor_uuid`, `score`, `log_text`, `incentive_reasons`, …). Failure paths: the seeded
offline executor is dropped by the miner within its timeout and never scored while the live one is; an unreachable
miner yields the synthetic failed result in seconds, not a hung cycle; `SysboxRequiredCheck` is in the pipeline
and its refusal is `SYSBOX_REQUIRED_MISSING` (the stack runs with `REQUIRE_SYSBOX_FOR_UNRENTED=false` — neither a
CI box nor a Lium pod can run sysbox; bare metal is the only place the flag-on AVAILABLE verdict can be observed).

**`tests/rental`** — `MinerService.handle_container`: `ContainerCreateRequest` for the executor → a container from
`pod/Dockerfile` (ubuntu + openssh-server; `seed` builds it onto the executor's dockerd) with the renter's key,
`ContainerCreated` with the port map; the renter's key opens the pod on the executor's address (what `lium exec`
does); no `/dev/nvidia0` inside on a GPU-less host, `nvidia-smi -L` lists GPUs with `E2E_GPU=1`;
`ContainerDeleteRequest` → `ContainerDeleted` and the port is closed. A create for an executor the miner does not own
→ `FailedContainerRequest`, fast.

## What it cannot prove (and where that is proven)

Weights on chain, the AVAILABLE verdict of an unrented executor behind `REQUIRE_SYSBOX_FOR_UNRENTED` (needs sysbox on
bare metal), TDX/CVM attestation, the real GPU checks (matmul, VerifyX, nvml digest, fingerprint — `E2E_GPU=1` on a
GPU host runs them), the portal → central-miner sync (the executor row is seeded), Watchtower. Those stay on the
team's staging (testnet-37, the staging A4000).
