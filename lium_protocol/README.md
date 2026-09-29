# lium_protocol

The validator↔backend wire, once. Pydantic v2 models for:

- `lium_protocol.validator_to_backend` — every message the validator sends the backend over its WebSocket (`ValidatorMessageType`, 27 types: cycle results and scores, the per-cycle pod-states report, and the answers to container requests).
- `lium_protocol.backend_to_validator` — every message the backend sends down that socket (`BackendMessageType`, 16 types: container lifecycle, ssh keys, backups, Jupyter, the estimate request, the staging-only forced cycle), plus the three typeless replies it sends there (`SOCKET_REPLIES`: `Response`, `RentedMachineResponse`, `RevenuePerGpuTypeResponse` — no `message_type`, told apart by the request the validator is waiting for).
- `lium_protocol.http` — the bodies of the backend HTTP API the validator reads between cycles (`HTTP_MODELS`, 10 bodies).

No application code: nothing here imports the validator or the backend. `PROTOCOL_VERSION` (`1.4.0`) is semver over the wire.

## Using it

```python
from lium_protocol import VALIDATOR_MESSAGES, BACKEND_MESSAGES, ProtocolError

message = VALIDATOR_MESSAGES.parse(raw_text)      # → ContainerCreated | ExecutorSpecRequest | … by message_type
request = BACKEND_MESSAGES.parse(raw_text)        # → ContainerCreateRequest | … ; ProtocolError when it is not one
```

A consumer that wants a stricter type for one message subclasses it and registers the subclass in its own registry; every other type is inherited:

```python
from lium_protocol import Registry, ValidatorMessageType, VALIDATOR_MESSAGES
from lium_protocol.validator_to_backend import ExecutorSpecRequest

class TypedExecutorSpecRequest(ExecutorSpecRequest):
    specs: MachineSpecs | None = None

MY_MESSAGES = Registry(ValidatorMessageType, base=VALIDATOR_MESSAGES)
MY_MESSAGES.register(TypedExecutorSpecRequest)
```

Field types are what a receiver must accept: a field one peer sends and the other ignores is present and optional; a field an older peer omits is optional; unknown keys are ignored, never an error — with one exception, `Response` (the auth reply), which forbids them because the validator's copy does. `ProtocolError.msg` is pydantic's error list without the input, so a refused message never copies its credentials into a log.

## The contract

`snapshots/lium_protocol.v1.json` is the JSON Schema of every model above, committed. `python -m lium_protocol.schema --check` exits 1 with a unified diff when the models no longer match it; CI runs it on every pull request (`.github/workflows/test.yml`, job `protocol`). A wire change is therefore a diff in that file, reviewed as such — regenerate it with `python -m lium_protocol.schema --write` and commit it next to the model change.

Compatibility rules for a change:

- add an optional field or an enum member → minor version bump (`1.0.0` → `1.1.0`, as `FailedContainerRequest.build_log_tail` did);
- make a field required, remove or rename one, change a type → major version bump and a new `snapshots/lium_protocol.v2.json`.

Consumers pin a tag `lium-protocol-v<PROTOCOL_VERSION>` of this repository. The plan for lium-platform: vendor the tree — models, snapshot and recordings — under `apps/lium/backend/apps/server/src/lium_protocol/`, with a `PROTOCOL_PIN.json` naming the tag, the commit and the tree's sha256, and a CI check that compares the vendored copy with that commit.

## Versions

- `1.1.0` — `FailedContainerRequest.build_log_tail` (optional; the last lines a failed custom-Dockerfile build printed) and `FailedContainerRequest.step_detail` (optional; the Docker daemon's reason for a failed volume step, or the dead-transport hint).
- `1.2.0` — `PodStatesReport`, `ExecutorSpecRequest.pod_states` and `FailedContainerErrorCodes.ExecutorUnreachable` (per-cycle pod container states, and "unreachable" told apart from "unknown id").
- `1.3.0` — `RentedPod.ssh_port` and `RentedPod.status` (optional), and the `PodSshUnreachableResponse` HTTP body (the renter-side SSH probe's report).
- `1.4.0` — `RentedExecutorsResponse.filler_revenue_by_gpu_config` (optional): per (base model, GPU count), `usd_per_gpu_hour` = filler revenue / filler GPU-hours over the trailing 24 hours, with the `gpu_hours` it covers; the spot-node pay and the secure floor read it. `RentedExecutorsResponse.provider_spot_executor_ids` (optional, default empty): the subset of `spot_executor_ids` whose provider chose the Spot tier, with no demotion, force-spot hotkey, pin or no-incentive rental on it; only these may take spot-node pay, so the backend must send it before spot-node pay pays anyone. The validator's `incentive_formula_version` stays `rental_price_v2`, which now has two terms when `incentive_formula_inputs` carries `unbucketed_share`: the rental term `rental_share × gpu_count × effective_rate / total_rental_cost` plus the unbucketed term `unbucketed_share × gpu_count × floor_top_up_rate / unbucketed_rental_cost`. A floored secure node is paid both; a spot node (`spot_pay: true`) only the unbucketed term, with `floor_top_up_rate` = `effective_rate`.
- `1.0.0` — first release.

## Tests

```bash
pip install . pytest "pydantic==2.13.*"   # the renderer CI pins: the snapshot is pydantic's JSON Schema
python -m lium_protocol.schema --check
python -m pytest tests -v
```

`lium_protocol/recorded/*.json` hold one message per wire type, one per socket reply and one per HTTP body, as the peers' models serialise them today. `tests/test_recorded_messages.py` parses each through this package and checks no field is dropped; `neurons/validators/tests/test_protocol_compat.py` parses the same files through the validator's own models (`payload_models`, `vc_protocol`) and requires both views to agree on every recorded field. When a message changes on one side, the recording changes with it and the other side's test says what broke. A change under `lium_protocol/` runs both (`.github/actions/changed-packages`: `lium_protocol/` selects `protocol` and `validators`).
