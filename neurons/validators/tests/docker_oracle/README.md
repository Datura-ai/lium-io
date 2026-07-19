# Docker service characterization oracle (DAH-2382)

Characterization suites that freeze the observable boundary of
`src/services/docker_service.py` before the docker-service-refactor extraction
moves any code. Internal control flow is free to change; these goldens are not.

- **Baseline:** lium-io `origin/main` `6be5649f` (`docker_service.py` = 4,777 lines).
  PR-1a touches tests only — every `docker_service.py:<line>` below is valid on
  both the baseline and this branch.
- **Scope (PR-1a):** groups B (create success / profiler / redis), C (volumes /
  vloopback / ports), D (delete / ssh keys / jupyter / security sinks) and
  E (ordering / cancellation / idempotency) of the plan in
  `openspec/changes/docker-service-refactor/characterization-oracle-DRAFT.md`.
  Group A (the create-container failure matrix) lands in PR-1b.

## Normalization rules

- **Byte-for-byte** for every `msg` template and every
  `error_type` / `error_code` / `failure_step` enum value — these are the
  classifier and backend surface. `msg` is the bare template
  (`_StructuredMessage.__str__` returns only `.message`); `str(exc)` lives in
  the log extra, not in `msg`.
- **Normalize only interpolated `str(exc)` tails.** Two places interpolate the
  exception into `msg`: the host-key special case
  (`current_step == "docker_sdk_ssh_host_key"` + `RentalDockerConnectionError`
  → `"Failed create_container: {e}"`) and `delete_container`'s generic
  `"Unknown Error delete_container: {e}"`. Assert the prefix byte-for-byte plus
  a substring of the tail; never freeze a full interpolated string whose tail
  is environment-dependent. When the test itself controls the raised exception
  the full string is deterministic and MAY be pinned `==` (the D5 delete test
  does — the SDK wrapper re-raises the original exception unwrapped).
- **Timestamps / durations are masked.** Assert type and relation
  (e.g. `total == sum(...)`), never literals.
- **An absent key is not `null`.** The profiler serializer omits keys
  (`skipped=False` is dropped; the anchor has no `duration`) — assert the exact
  key set of wire dicts, not just values.
- **Monkeypatch-target migration rule:** an import/patch target may move in a
  later PR, or the asserted behavior may change — never both in the same PR
  (exception: a module constant changing namespace, tracked in PR-4).

## Repo-reality constraints

- **pytest-asyncio runs in STRICT mode** — `pyproject.toml` sets no
  `asyncio_mode`, so every async test needs an explicit `@pytest.mark.asyncio`.
  (Do not copy a miner test: `neurons/miners` uses `asyncio_mode="auto"`.)
- **Goldens use the repo's own `--update-snapshot` flag** (conftest.py
  `update_snapshot` fixture) with the write-on-missing-then-byte-equal pattern
  of `test_custom_dockerfile_build.py::test_A1_golden_snapshot_image_pull_wire`.
  Golden files live in `tests/fixtures/docker_oracle/`. Do NOT add syrupy /
  pytest-regressions — a second snapshot-update workflow is forbidden.
- **One shared harness.** All suites build on
  `tests/docker_oracle/harness.py` (`FakeRentalDockerClient`,
  `RecordingRedisService`, `patch_create_happy_path`, `patch_delete_happy_path`,
  `patch_rental_ssh_connect`, `make_create_request` / `make_delete_request` /
  `make_executor_info` / `make_docker_service`). Four near-copies of the fake
  client already existed before this package; do not add a fifth — extend the
  harness.
- AAA layout with explicit `# Arrange / # Act / # Assert` comments and a
  one-line WHY comment before each assert block.

## Gap analysis (2.1)

Contract → coverage map. "Baseline :N" = line number at `6be5649f` (the three
modified test files shifted a few lines on this branch; names are durable).

| Oracle area | Sketches | Pinned by |
|---|---|---|
| create FAILURE matrix | A1–A14 | **PR-1b (pending).** Baseline already covers: `docker_pull` (test_docker_service.py baseline :2997), `set_environment` (:3077), host-key substring (:2405), custom-build `docker_build`/`build_timeout` (test_custom_dockerfile_build.py) |
| create success / profiler / redis / DAH-2265 | B1–B9 new; B10/B11 folded; B12 dropped | `test_create_success.py` (9 tests: `test_container_created_full_golden`, `test_profiler_shape_anchor_timestamp` / `_duration_only` / `_duration_skipped`, `test_profiler_name_sequence_golden`, `test_loki_profile_steps_format`, `test_redis_operation_order_customer_rental`, `test_redis_remove_pending_for_filler`, `test_streaming_log_publish_shape`) + 2 goldens; folds in `tests/test_deploy_optimizations.py` |
| volume sizing (C1–C9) | finer pins over an existing suite (see stale-[N] below) | `test_volume_sizing_and_ports.py::test_sizing_*` (8) + `::test_parse_volume_size_formats`; baseline DAH-2183 suite `test_docker_service.py::test_resolve_volume_sizing_*` (baseline :4281–:4458) |
| SDK retry loop (C10–C17) | all new — live `_run_rental_docker_create_with_port_retry` had zero tests (only the dead SSH twin was tested) | `test_sdk_retry_loop.py` (11: C10–C17 as `test_sdk_*` + 3 extra: `test_sdk_partial_mount_phrases_do_not_trigger_repair`, `test_sdk_port_retry_sleeps_then_removes_then_reruns`, `test_sdk_port_retry_second_phrase_address_in_use`) |
| port planner (C18–C20) | C18/C19 dropped (pre-covered); C20 new | `test_docker_service.py::test_min_port_count_validation` (baseline :623), `::test_generate_portMappings_offsets_filler_custom_external_port` (baseline :673); `test_volume_sizing_and_ports.py::test_ports_lock_error_returns_empty[LockError\|LockNotOwnedError]` |
| delete flow (D1–D11) | D1, D6–D11 new; D2–D5 extended in place | `test_delete_flow.py` (7 fn / 9 items); extensions: `test_docker_service.py::test_delete_container_stops_gracefully_before_forced_removal` (D2, grace `[30]`), `::test_delete_filler_stops_with_reduced_grace` (D3, `[15]`), `::test_delete_container_stop_failure_still_removes` (D4, byte WARNING), `::test_delete_container_remove_container_error_fails_undeploy` (D5) |
| ssh keys (D12–D17 + attestation) | D12/D13 folded; D14–D17 + attestation ×2 new | folds in `tests/test_docker_service_rental_security.py`; `test_ssh_keys_and_jupyter.py::test_add_ssh_key_no_keys` / `::test_remove_ssh_keys_no_keys` / `::test_add_ssh_key_generic_exception` / `::test_remove_ssh_keys_generic_exception` / `::test_ssh_key_methods_attestation_error_returns_addsskeyfailed[add\|remove]` |
| jupyter (D18–D20) | all new — method had zero tests | `test_ssh_keys_and_jupyter.py::test_install_jupyter_success_url_shape` / `::test_install_jupyter_attestation_returns_failed_container_request` / `::test_install_jupyter_runtime_exception_returns_jupyter_installation_failed` |
| argv / security (D21–D24) | D21 case 1 dropped; rest new (live builder had no unit test) | `test_startup_argv.py::test_startup_commands_argv_neutralizes_injection` / `::test_startup_commands_argv_preserves_quoted_metacharacters` / `::test_startup_commands_unbalanced_quotes_fall_back` / `::test_prepare_known_hosts_policy_fail_open_returns_none` / `::test_prepare_known_hosts_policy_fail_closed_raises_attestation_error` |
| behavior invariants (E1–E6) | all new (E1 cleanup half is a SPEC xfail) | `test_behavior_invariants.py` (9 fn / 13 items — see Suite stats) |
| E7 differential | harness + self-consistency test | `tests/integration/test_docker_service_differential.py::test_e7_differential_probe_self_consistency` (env-gated) + `Makefile` `e7` / `e7-record` |

### §0.5 audit: sketch → fate

| Sketch | Fate | Where it landed |
|---|---|---|
| B12 | DROP | pre-covered: `test_deploy_optimizations.py::test_ships_sshd_with_startup_commands_runs_bootstrap_and_run_jupyter` (baseline :454) |
| C18 | DROP | `test_docker_service.py::test_min_port_count_validation` (baseline :623) |
| C19 | DROP | `test_docker_service.py::test_generate_portMappings_offsets_filler_custom_external_port` (baseline :673) |
| D21 case 1 | DROP case | `test_docker_service_rental_security.py::test_create_container_keeps_hostile_fields_out_of_host_shell_commands` (baseline :362, full-boundary, stronger); case 2 kept → `test_startup_argv.py::test_startup_commands_argv_neutralizes_injection` |
| A14 | RETAG [E] → PR-1b | will extend `test_custom_dockerfile_build.py::test_A11_empty_dockerfile_content_*` (only byte `msg` + `remove_pending_pod` assert are new) |
| B10 / B11 | FOLD | asserts added to `test_deploy_optimizations.py::test_ships_sshd_true_forwards_jupyter_password_instead_of_run_jupyter` / `::test_ships_sshd_none_with_jupyter_uses_run_jupyter` (baseline :382/:435); plus `_CREDS` so the login asserts are non-vacuous |
| D6 | NARROW | `test_delete_flow.py::test_delete_teardown_step_failure_nonfatal` parametrized over exactly the 3 uncovered steps: `restore_filler_gpu_power`, `sweep_wedged_gpus`, `inspector_stop` |
| D12 / D13 | FOLD | return-shape + key-echo asserts added to `test_docker_service_rental_security.py::test_add_ssh_key_writes_public_keys_as_stdin_data` / `::test_remove_ssh_key_writes_public_keys_as_stdin_data` (baseline :608/:645) |

### Stale-[N] discoveries (baseline coverage the DRAFT missed)

- **C1–C8 were tagged [N] but a baseline suite already existed**:
  `test_docker_service.py` baseline :4281–:4458 (DAH-2183,
  `test_resolve_volume_sizing_*`) covers legacy, pool-bound, storage_opt
  unsupported, request_cap, df_guard, below-min raise, severe-shrink, fallback,
  1 GB floor and base parse formats. The new file adds only the finer pins:
  the exact 4-command SSH sequence + `--filter driver=` on the volume listing;
  `docker volume inspect` skipped when no volumes exist; the byte-exact
  `VolumeMinSizeError` msg (`"Fresh vloopback sizing produced {v}GB volume,
  below required minimum {m}GB"`); the `capped_by` tie-break (first in
  candidate list order); the full `None`-field set on fallback/legacy; parse
  edge formats. C9's [E] tag was correct.
- **B10/B11 asserts partially pre-existed**: the login-skip matrix tests at
  `test_deploy_optimizations.py` baseline :761/:796 already asserted
  `login_calls == []` / `DOCKER_LOGIN.skipped` — folds were needed only at
  :382/:435.

### Known uncovered, low risk (deliberate)

- `_cleanup_custom_build_artifacts` failure case (delete :4208): not routed
  through `_best_effort_delete_step` — self-guarding WARNING `"Custom build
  artifact cleanup failed (non-fatal)"` with no `step=` key. Still non-fatal by
  its own try/except; exercising it adds no boundary information.
- `remove_volume_external` failure case (:4249, incl.
  `disable_s3fs_volume_plugin`): only the hostile-name guard is covered. The
  step uses the same `_best_effort_delete_step` wrapper whose semantics D6 pins
  on three representative steps.
- Retry-loop repair-returns-False branch (`repair_stale_vloopback_mountpoint`
  → `False` → immediate propagate, once-flag already consumed): one-line
  branch sharing the C17 propagate path.

## Failure-site matrix (2.2)

Scope honestly: PR-1a pins the **delete / ssh-key / jupyter** failure sites and
the create **SUCCESS** surface. The create failure funnel step-map (5
early-return sites + 1 funnel across 29 `current_step` labels) is **PR-1b
group A** — below is only what PR-1a already pinned, on baseline `6be5649f`.

### delete_container (`error_type=ContainerDeletionFailed`, `failure_step` always `None`)

| Site (src) | Trigger | error_code / msg (byte) | Pinned by |
|---|---|---|---|
| volume-name guard (:4153) | hostile `local_volume`, checked BEFORE `decrypt_payload` | UnknownError / `"Invalid Docker volume name"` (wire literal; descriptive ValueError text only in the log), no SDK calls | `test_delete_flow.py::test_delete_unsafe_volume_name_rejected` (D10); complements `rental_security.py::test_delete_container_rejects_unsafe_volume_names_before_shell` (baseline :770) |
| attestation (after decrypt/`import_private_key`, before SSH/SDK) | `attestation_service.prepare_host_policy` raises `AttestationError` | UnknownError (NOT the unused `AttestationError` enum member) / `"Attestation failed"` | `test_delete_flow.py::test_delete_attestation_failure` (D11) |
| graceful stop, non-fatal (:4094) | `stop` raises non-missing | still ContainerDeleted; WARNING `"Graceful container stop failed; proceeding to forced removal"`, force-remove still runs | `test_docker_service.py::test_delete_container_stop_failure_still_removes` (D4) |
| **fatal force-remove** (funnel :4296–:4299) | `_force_remove_container` (:4107) raises non-missing | UnknownError / `"Unknown Error delete_container: {e}"` (full `==` pinned — tail deterministic), `remove_rented_machine` NOT awaited | `test_docker_service.py::test_delete_container_remove_container_error_fails_undeploy` (D5) |
| missing container (idempotency) | `"No such container"` substring (:294), incl. RetryError unwrap | swallowed → 2nd delete returns ContainerDeleted; graceful stop on missing = INFO skip (:1156) | `test_behavior_invariants.py::test_delete_container_is_idempotent_when_container_already_missing` (E3) + baseline `::test_delete_filler_container_treats_missing_container_as_deleted` / `::test_delete_customer_rental_treats_missing_container_as_deleted` (baseline :851/:917) |

Post-teardown best-effort steps, in execution order. Wrapper =
`_best_effort_delete_step` (:234–:246): swallows, logs **ERROR**
`"delete_container post-teardown step failed (non-fatal)"` with extra keys
`step`, `error` (+`volume_name` where applicable). Result stays
ContainerDeleted.

| # | Step (src) | Gate | Covered by |
|---|---|---|---|
| 1 | `sweep_wedged_gpus_after_failed_remove` (:4201) | FILLER, remove-except only, then re-raise | `test_delete_flow.py::test_delete_filler_sweeps_before_propagating_failure` (D8, was uncovered) |
| 2 | `_cleanup_custom_build_artifacts` (:4208) | all | NOT wrapper-routed (own WARNING, no `step=`); failure case uncovered (low risk, above) |
| 3 | `restore_filler_gpu_power` (:4217) | FILLER | `test_delete_flow.py::test_delete_teardown_step_failure_nonfatal[restore_filler_gpu_power]` (D6, was uncovered) |
| 4 | `sweep_wedged_gpus` (:4224) | FILLER | D6 `[sweep_wedged_gpus]` + `::test_delete_filler_sweeps_wedged_gpus_on_success_nonfatal` (D7) + `::test_delete_customer_rental_does_not_sweep` (D9) — call-site only; internals stay in `test_gpu_wedge_teardown_sweep.py` |
| 5 | `prune_images` (:4227) | all | pre-covered `test_docker_service.py::test_delete_container_prune_images_failure_still_deleted` (baseline :1507) |
| 6 | `remove_volume_local` (:4235) | local volume | pre-covered `::test_delete_container_volume_read_timeout_still_deleted` / `::test_delete_container_missing_volume_still_deleted` (baseline :1407/:1461) |
| 7 | `remove_volume_external` (:4249) | external volume | hostile-name guard only; failure case uncovered (low risk, above) |
| 8 | `remove_rented_machine` (:4269) | all | pre-covered `::test_delete_container_redis_failures_still_deleted` (baseline :1286) — IS best-effort |
| 9 | `inspector_stop` (:4275) | all | failure pre-covered (baseline :1286); D6 `[inspector_stop]` adds the `step=` log pin |

FILLER gate (:4216) is ONE `if` wrapping steps 3–4. Sweep ordering (docker
remove before sweep) pinned via the journal in D8.

### add_ssh_key / remove_ssh_keys (`error_type=AddSSkeyFailed` on BOTH paths, `failure_step` always `None`)

| Trigger | error_code / msg (byte, src) | Pinned by |
|---|---|---|
| add: empty keys | NoSshKeys / `"ssh key Add error: no public key"` (:4591) | `test_ssh_keys_and_jupyter.py::test_add_ssh_key_no_keys` (D14) |
| remove: empty keys | NoSshKeys / `"ssh key Remove error: no public key"` (:4458) | `::test_remove_ssh_keys_no_keys` (D15) |
| add: generic exception | UnknownError / `"Failed add_ssh_key"` (:4657) | `::test_add_ssh_key_generic_exception` (D16) |
| remove: generic exception | UnknownError / `"Unknown Error remove_ssh_keys"` (:4525) | `::test_remove_ssh_keys_generic_exception` (D17) |
| add/remove: `AttestationError` — the attestation gate runs BEFORE the empty-keys check | UnknownError / `"Attestation failed"` | `::test_ssh_key_methods_attestation_error_returns_addsskeyfailed[add\|remove]` |
| success acks | `SshPubKeyAdded` / `SshPubKeyRemoved`, `user_public_keys` echoed | D12/D13 folds in `rental_security.py` (the remove ack is silently dropped by the backend — frozen anyway, DRAFT §8) |

### install_jupyter_server (asymmetric two-shape failure contract)

| Trigger | Result | Pinned by |
|---|---|---|
| success | `JupyterServerInstalled(jupyter_url="http://{executor address}:{external port}/lab?token=<32 lowercase hex>")`; token = `secrets.token_hex(16)`; `run_jupyter` receives the INTERNAL port + token | `::test_install_jupyter_success_url_shape` (D18) |
| `AttestationError` | `FailedContainerRequest(error_type=ContainerCreationFailed` — surprising, create's type — `, UnknownError, "Attestation failed")` (:4347) | `::test_install_jupyter_attestation_returns_failed_container_request` (D19) |
| `run_jupyter` raises | `JupyterInstallationFailed(msg="Failed install jupyter server")` — payload (payloads.py:598) carries ONLY `msg`, no error_type/error_code/failure_step fields | `::test_install_jupyter_runtime_exception_returns_jupyter_installation_failed` (D20) |

### Create-side failure sites already pinned via PR-1a groups

| Site | Behavior (byte values) | Pinned by |
|---|---|---|
| retry-loop exhaustion in `_run_rental_docker_create_with_port_retry` | deadline `time.monotonic()+90` checked `>=` after each failure; final non-retryable error logged `"Docker SDK run container failed"` ERROR `exc_info=True` + `stream_log` before re-raise; propagates to the boundary as `failure_step="docker_run"` | `test_sdk_retry_loop.py::test_sdk_port_retry_deadline_expired` / `::test_sdk_non_port_error_propagates`; boundary step + orphan cleanup: `test_behavior_invariants.py::test_create_failure_at_docker_run_cleans_up_container_and_volume` (E5) |
| health check false after successful run | sticky step: boundary reports `failure_step="container_health_check"` (not `docker_run`); container + volume cleaned up | `::test_create_failure_at_health_check_cleans_up_container_and_volume` (E5); byte `msg` lands in PR-1b A10 |
| `LockError` / `LockNotOwnedError` in `generate_portMappings` (:968) | `([], None)` → boundary NoPortMappings / `port_mapping` (boundary half in PR-1b A1) | `test_volume_sizing_and_ports.py::test_ports_lock_error_returns_empty[...]` (C20) |
| `VolumeMinSizeError` from `resolve_volume_sizing` | raises with byte msg `"Fresh vloopback sizing produced {v}GB volume, below required minimum {m}GB"` → boundary `failure_step="volume_sizing"` in PR-1b A8 | `::test_sizing_below_min_raises` (C4) |
| retry-loop log events | `VLOOPBACK_STALE_MOUNTPOINT_RETRY` (INFO); `PORT_ALREADY_ALLOCATED_RETRY` (INFO, extras `attempt`/`remaining_sec`/`sleep_seconds`/`port_allocation_phrase`) then sleep 5 s then SDK `remove_container(container_name=..., force=True, remove_volumes=True)` as operation `remove_failed_container_for_retry`; `PORT_RETRY_STALE_RM_FAILED` (WARNING, rm failure non-fatal, extras `container_name`/`rm_error`) | `test_sdk_retry_loop.py` (C10–C17 + extras) |

## Observed-vs-sketch corrections

Places where baseline reality differed from the DRAFT sketches. PR-1b/PR-2+
authors: trust this list over the DRAFT, do not re-derive.

1. **`"Finished create_container"` does not exist.** The Loki summary line is
   `logger.info("Deployment profile summary")` (:3747).
2. **`jupyter_url` host is the executor address** (`executor_info.address`,
   :3654), for BOTH DAH-2265 branches — the sketch's `127.0.0.1` was just the
   test executor's address.
3. **`_sweep_wedged_gpus_after_teardown` is module-level** (:4749), not an
   instance method — patch `services.docker_service._sweep_wedged_gpus_after_teardown`;
   same for `restore_filler_pod_gpu_power_limits` (module import :66).
4. **Best-effort post-teardown log is ERROR, not warning**
   (`"delete_container post-teardown step failed (non-fatal)"`, extra keys
   `step`, `error`, +`volume_name`).
5. **Every attestation branch uses `error_code=UnknownError`** (delete D11,
   add/remove keys, jupyter D19) — `FailedContainerErrorCodes.AttestationError`
   exists but is unused on these paths (mirror of the A4 surprise).
6. **`ContainerDeleted` has no optional fields**: exact wire =
   `{message_type, miner_hotkey, executor_id, pod_id, workload_kind}`.
7. **Stale-mount matching requires ALL 3 phrases** (`"VolumeDriver.Mount"`,
   `"cannot create mount point dir"`, `"file exists"` — `all()` at :306); port
   phrases are ANY-match, first match fills the log field.
8. **Repair-retry does not increment the SDK-operation `attempt`** (both runs
   log `attempt=1`; a port retry goes 1→2). Repair-retry has NO sleep and NO
   rm; port-retry sleeps 5 s then SDK-removes the stale container.
9. **Sizing tie-break**: winner = `min(candidates)`; on a tie the FIRST in
   candidate list order wins — `pool` > `request_cap` > `df_guard` (pinned
   inside C5).
10. **Gating precedence**: `storage_limit_gb is None` →
    `storage_opt_unsupported` is checked BEFORE `disk_share is None` → legacy.
11. **`df_guard = max((df − 10 GiB) × 1.5, 0)`** — the clamp-to-0 was omitted
    in the sketch; `request_cap` candidate is ABSENT when `volume_limit_gb` is
    None.
12. **`VolumeSizingResult` has no `warnings` field** —
    `vloopback_fresh_sizing_fallback` surfaces only via `logger.warning`. The
    dataclass `path` comment (:272) omits `storage_opt_unsupported`
    (code-comment drift; fix in a later PR, not here).
13. **C20's catch tuple is redundant**: `(LockError, LockNotOwnedError)` where
    `LockNotOwnedError` subclasses `LockError`; `UUID(executor_id)` runs BEFORE
    the try (:968 region).
14. **D24 fail-closed msg interpolates `executor.port`** (the API port, e.g.
    `8080`), NOT the ssh port: `"Unable to prepare host-key policy for executor
    127.0.0.1:8080; refusing unpinned rental SSH"`. Fail-open returns `None` +
    warning `"Unable to prepare known_hosts policy"`. `known_hosts_policy` is
    unused in `add_ssh_key` (SDK path ignores it), not even assigned in
    remove; only `install_jupyter_server` consumes it.
15. **DAH-2265 skip_login asymmetry**: `skip_login = not has_credentials or
    bool(ships_sshd)` (:3100) gates on `ships_sshd` ALONE — `ships_sshd` + a
    non-blank `startup_commands` still skips docker login while falling back to
    validator-managed bootstrap/`run_jupyter` (bootstrap skip gates on
    `image_manages_services` :3608, `run_jupyter` skip on
    `image_managed_jupyter` :3636).
16. **`JUPYTER_PASSWORD` is injected into `custom_options.environment` BEFORE
    the run-spec build** (:3279) — it travels as `docker run -e`, not
    post-create.
17. **E1 cancellation propagates but leaks**: `CancelledError` escapes the
    except-Exception funnel (green test), but NO cleanup runs and the pending
    pod leaks — documented as strict xfail ×3 seams (`docker_run`,
    `ssh_bootstrap`, `finalize_add_rented_pod`); fix lands in PR-8. The
    executor lock cannot leak at these seams (scoped inside
    `generate_portMappings`, released before them).
18. **B9 latent shape**: `handle_stream_logs` has no guard against starting
    after `finish_stream_logs` — improbable with real I/O; candidate note for
    PR-2 (LogStreamer). (An all-AsyncMock version of this race deadlocked the
    B9 test until a real `await asyncio.sleep(0)` was added.)
19. **Test-authoring trap**: a bare `Mock()` `attestation_service` makes
    `await prepare_host_policy(...)` raise TypeError → the fail-open branch →
    silently `None`. Stub it as AsyncMock (see harness) — PR-1b A4 must
    override `harness.mocks["_prepare_known_hosts_policy"]`.
20. **Live argv builder**: `build_container_command_argv` at
    `src/services/rental_docker_sdk.py:560`, sole live call site
    `docker_service.py:702` (`_build_rental_container_run_spec` →
    `ContainerRunSpec.command`); `None`/empty/whitespace → `()`, shlex ValueError
    → `()`. `build_startup_command_args` is dead (PR-3 decision, DRAFT §8).

## E7 differential gate

```bash
cd neurons/validators
make e7          # run against a real local docker daemon (byte-equal vs golden)
make e7-record   # re-record after an INTENDED behavior change
```

D-B duty: extraction-PR authors (PR-2..PR-12) run `make e7` locally **before
push** and record `E7: ran locally, N passed @ <sha>` in the PR description;
snapshot drift = behavior change (fix it, or `make e7-record` + call the diff
out in the PR). Full prerequisites, the macOS `DOCKER_HOST` gotcha, the PR-1a
probe scope and the PR-2 old-vs-new evolution: `tests/integration/README_E7.md`.

## Suite stats

- **68 new tests** in `tests/docker_oracle/` — harness smoke 1,
  `test_create_success.py` 9, `test_volume_sizing_and_ports.py` 11,
  `test_sdk_retry_loop.py` 11, `test_delete_flow.py` 9,
  `test_ssh_keys_and_jupyter.py` 9, `test_startup_argv.py` 5,
  `test_behavior_invariants.py` 13 — plus **1 env-gated E7 test**
  (`tests/integration/test_docker_service_differential.py`, skipped without
  `RUN_RENTAL_DOCKER_SDK_INTEGRATION=1`) and assert-folds in 3 pre-existing
  files (`test_deploy_optimizations.py`, `test_docker_service.py`,
  `test_docker_service_rental_security.py`).
- **The 3 xfails are by design** (strict):
  `test_create_cancelled_midflight_runs_cleanup_and_removes_pending_pod` ×3
  seams — the D1 deviation #1 SPEC test (cancellation cleanup, lands PR-8).
- **Goldens** in `tests/fixtures/docker_oracle/`:
  `container_created_success_golden.json` (durations masked `"<duration_ms>"`),
  `profiler_name_sequence_golden.json` (the frozen 15-step sequence),
  `e7_differential_probe_surface.json` (recorded from a real daemon). Review
  any diff there as a behavior change, never as noise.
- Full validators suite at PR-1a head: 1179 passed, 9 skipped, 3 xfailed;
  3 pre-existing env-dependent failures in `test_sshd_bootstrap_script.py`
  (unrelated to this PR).
