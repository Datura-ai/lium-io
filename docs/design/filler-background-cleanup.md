# DAH-4978: two-phase filler cleanup (draft)

Status: RFC with a read-only evidence prototype. No rental/delete path uses the
prototype yet; this PR does not enable background removal in production.
Keep this PR in draft for discussion. Evaluate proactive cache eviction before
implementing the two-phase protocol; the protocol is not the selected fix yet.

## Problem and evidence

On Raff, Docker's volume-plugin Unmount took 50.55 and 100.94 seconds; a second
Raff host had a 15.25-second Unmount after TaskDelete. Docker can print its
10-second SIGKILL warning while the process is already dead and cleanup is
running. Waiting for Docker's stopped/removed state therefore also waits for
filesystem cleanup.

A dedicated Shadeform experiment with the stock loopback plugin and 104.86 GB
written reproduced the delay: 0.61 seconds for container removal after 1 MiB
writes versus 10.80 seconds after 4 KiB writes. The slow Unmount profile contains
`truncate_inode_pages_final`, `delete_from_page_cache_batch` and `release_pages`.
The inner file had 73.70 GB resident before the slow removal. A new container
allocated/wrote 512 MiB on the same GPU and wrote to a new volume 0.827 seconds
after the old processes disappeared, while the old Unmount was still running.
This tests the mechanism and handoff principle, not a complete customer startup.
A kernel profile of the slow production Raff event is still missing.

The plugin has no effective global mutex across its value-receiver methods.
Different volumes can progress concurrently. Docker still locks each old
container/volume: inspecting it during cleanup can block. A follow-up experiment
excluded exact retiring IDs before inspection, completed the remaining checks
and `df` in 0.138 seconds, then ran a new container while the old Unmount continued.
Moving only `docker rm` to the background would move the wait to the next inspect.

## First candidate: evict unused cache before preemption

The earlier control experiment is also evidence for a potentially cheaper fix:
after 1 MiB writes, targeted `POSIX_FADV_DONTNEED` took 0.31 seconds for the inner
file and 14.81 seconds for the backing file, followed by container removal in
0.19 seconds and volume removal in 0.04 seconds. Calling this only a causal control
was too narrow. It demonstrates moving work before deletion; it does not yet
demonstrate safe eviction while a real ENGY_TEE model is serving, nor the slow
4 KiB-folio case with a live model.

Prefer testing targeted eviction of the completed weight files and their exact
loop backing file after verified model readiness. Both cache layers matter.
Do not infer that all weights are disposable from GPU allocation or a download
completion message: measure continued reads, mapped/locked pages, CPU offload,
reloads and cache growth under representative inference. Preserve active/shared
cache ownership; do not delete or truncate files. DONTNEED is advisory and does
not guarantee eviction of dirty pages. Any necessary flush also costs time.

| Candidate | Potential benefit | Missing evidence / risk |
|---|---|---|
| Targeted file DONTNEED | Preserve the existing synchronous removal contract; pay eviction cost before rental | Both cache layers, stable model readiness, repeated reads, serving latency, preemption during eviction |
| Cgroup `memory.reclaim` | Explicit proactive reclaim without a persistent memory limit | Kernel support, actual cache charging (including the backing file), incomplete reclaim and workload impact |
| Cgroup `memory.high` | Limit future cache accumulation | Throttles the workload under reclaim pressure; includes other charged memory and is not a file-only eviction operation |
| Host `drop_caches` | Useful comparison on an isolated lab machine | Host-wide scope affects unrelated caches; an earlier no-customer check does not prevent a new rental racing the operation |
| Two-phase cleanup | Avoid waiting for filesystem cleanup even when cache remains | Durable protocol, caller changes, GPU visibility, pending disk and recovery complexity |

The first comparison must use the same ready ENGY_TEE workload and configuration:
baseline; inner-file eviction only; both layers; then, if needed, cgroup reclaim.
Record per-file residency and folio sizes, dirty bytes, disk reads/writes, CPU/PSI,
serving throughput/tail latency and errors before, during and after eviction.
Recheck cache residency after sustained inference and a model reload. Measure
process exit, Unmount, separate volume removal and complete customer readiness
individually. Include preemption during loading, during eviction and after cache
regrowth. A completed pre-eviction run alone does not cover early preemption.

Prefer this approach if repeated real customer handoffs meet the five-second
target without material serving regression or weakening existing checks. If it
only works after warmup, state that coverage explicitly; do not claim the loading
case is fixed. No ready-model A/B result or fleet coverage estimate exists yet.

References: [DONTNEED semantics](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html)
and [cgroup memory controls](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory).

## Execution boundary and TEE capability gate

This module currently runs on a manually prepared lab host. Importing it in the
validator would read the validator's own `/proc` and run local commands; that is
not a production integration. The validator already uploads other helper scripts
through SFTP (for example `_run_legacy_bootstrap_restore`), but this helper is not
packaged, uploaded or called by any such path.

If the protocol is selected, the proposed transport is a standalone versioned
helper shipped with the validator, uploaded through the authenticated executor
SSH connection, and invoked with `executor_info.python_path`. Use a private remote
directory, an atomic upload and verified content identity; pass validated arguments
with shell quoting and bounded JSON output. Verify the execution environment sees
the host Docker process PIDs, boot identity, cgroup mount and NVIDIA devices. Merely
running inside an executor container does not establish those namespace properties.
Persisted snapshots require a schema/helper version. The detached cleanup worker
and recovery have a different lifecycle from this read-only probe and are still
unimplemented. Moving the helper into an executor image is an alternative delivery
choice requiring a separate release, not something this PR already does.

Before designing admission around this evidence, capture the exact GPU UUID and
compute-client query outputs on a B300 in the same confidential-computing mode and
driver as Raff: while a known workload is active, after natural process release,
and during a controlled replacement allocation on a dedicated matching host.
Confirm visibility of the known active client, not just an empty idle response.
Do not infer support from non-confidential L40/A100/A6000 experiments. Unsupported
or redacted `N/A` output must remain unknown, never become an empty client list.
If clients are invisible, this probe cannot authorize handoff; an independently
validated health/allocation check is required or the ordinary path remains.
MIG instance UUIDs are intentionally unsupported; only physical GPU UUIDs are
accepted. Actual B300/TEE output has not been validated by this PR.

## Proposed state contract

| State | Durable evidence | Customer admission |
|---|---|---|
| Captured | Full container ID, host boot ID, PID/start ticks, cgroup path/inode, authoritative GPU allocation, owned volume names and sizes | Wait |
| Removing | Removal submitted by exact ID; durable worker owns this cleanup | Wait |
| Processes released | Captured init is gone, original cgroup is empty/removed including descendants; selected GPU checks pass; old volume bytes remain reserved through Unmount and separate volume removal | May continue only after disk admission and replacement checks |
| Cleanup complete | Old container and owned volumes removed, worker outcome recorded | Ordinary path |
| Unknown/failed | Missing evidence, blocked process, worker failure or host identity mismatch | Fail closed or await ordinary cleanup; never infer success |

The evidence prototype implements only process/cgroup observation and the
selected GPUs' compute-client query. It does not implement this state machine,
a worker, a deletion-success response, GPU health checks or disk admission.
Empty NVML compute-client output alone does not prove that no orphan GPU memory
or wedged kernel remains. Keep the existing GPU health/wedge checks and verify
actual allocation in integration tests. A snapshot is not authorization to kill
its PID; removal remains a Docker operation by full container ID.

## Integration points in current main

1. `DockerService.delete_container`: use the same retirement protocol for backend
   filler-stop requests, before the current graceful-stop/force-remove awaits.
   Customer deletes keep their existing completed-deletion semantics. Decide
   explicitly whether a filler reply means resource release or physical deletion;
   audit backend preemption, retries and filler state transitions before changing it.
2. `remove_fillers_at_ssh_connect`, `clean_existing_containers` and the post-start
   `_remove_live_fillers_beside_customer_container` check: share the durable record
   by host and exact ID. A newly created filler under an old name is a different
   target and must never inherit the old release evidence. Record our removals for
   the existing own-sweep accounting. Do not relax the customer/filler exclusion.
   `_confirm_fillers_removed` currently classifies fillers still present in
   `docker ps -a` as `FILLER_STILL_RUNNING`. An exact-ID match with fresh verified
   process-release evidence must instead be reported as cleanup pending, with
   separate cleanup duration/failure events. Unknown evidence, a live process or
   a replacement ID retains the failure path. Audit callers and backend consumers
   before changing event semantics; never suppress all surviving filler names.
3. `prerun_host_probe_command`, `clean_stale_vloopback_volumes` and fallback
   commands: filter retiring IDs **before** inspecting mounts. Protect their
   volume names from concurrent GC. Cached probe results and failed-probe
   fallbacks must follow the same rule, including early parallel probes.
4. `_inspect_vloopback_volumes_bytes`, `probe_volume_host` and
   `resolve_volume_sizing`: avoid inspecting a deleting volume, retain its known
   capacity in accounting, and read current free bytes. A volume can disappear
   from `volume ls` before its storage is freed. Never credit those bytes twice,
   remove `df_guard`, or silently reduce the customer's promised disk allocation.
   When space is needed from that volume, wait for cleanup or reject admission.
   Quantify coverage before rollout: for candidate filler-to-customer transitions,
   compare actual free bytes, pending allocated bytes and the promised customer
   volume against the unchanged sizing/guard logic. Report sample size, time,
   host class and eligible fraction. No fleet fraction is known yet; hosts that
   need the old volume's space cannot benefit from early admission.
5. Keep PEARL power restoration, inspector lifecycle and GPU wedge handling in
   the correct order. A later cleanup worker must never reset GPUs or power
   settings after the new customer's workload starts.

ENGY checkpoint caching landed in main in #1822 (DAH-5014). The new cache volumes
must remain distinct from the old filler's disposable root volume. Cache eviction
can itself cost time when disk is needed; this proposal does not promise a fast
handoff in that case. Existing ENGY_TEE deployments may still have weights inside
their root volume, so the new cache code does not close this investigation.

## Worker and recovery requirements

Persist the record on the host outside the old container, before submitting its
removal. A per-ID worker lock prevents duplicate ownership. Detach the worker
from the SSH session, preserve errors and reconcile it after reconnect or
validator/executor restart. Never use only an in-memory task or a TTL as evidence
that cleanup succeeded. Do not mark the record complete merely because a Docker
listing omits the volume; wait for the worker's confirmed result.

The worker must own both `docker rm -fv <full-id>` and the subsequent explicit
`docker volume rm <owned-volume>` used by `clear_volume=True`. The latter took
about 10 seconds in our large lab run and 32–35 seconds in the parallel lab run;
process release removes neither this cost nor the occupied disk. Track each phase
and errors separately, retain reservations until physical cleanup is confirmed,
and prevent the existing synchronous caller or GC from repeating either step.

Validate host boot identity, captured process start ticks and cgroup identity.
Capture before removal: Docker inspect may be blocked after it begins. A legacy
removal already in progress without a snapshot cannot safely take the early
handoff path. A new create using a retiring volume name must wait; it cannot race
the old worker's unlink. Failed/restarted workers must not delete a replacement
container or an active/shared cache volume.

## Prototype

On a dedicated Linux test host with host PID visibility, cgroup v2, Docker and
`nvidia-smi`, from `neurons/validators` with `PYTHONPATH=src`:

```sh
python scripts/probe_filler_release.py capture FULL_CONTAINER_ID --gpu GPU-UUID > snapshot.json
# Start the existing, authorized removal of that exact lab container separately.
python scripts/probe_filler_release.py observe snapshot.json
```

The target must have a `filler_` name. Supply GPUs from the workload's authoritative
allocation, not a guessed subset. Both commands are read-only. Capture uses a
bounded Docker inspect; observation uses only `/proc`, cgroup events and bounded
NVIDIA queries. Failed or unsupported reads raise instead of returning release.
The snapshot stores identifiers only, not Docker environment variables or secrets.
The returned observation is transient evidence, not a durable completed state.

The prototype was exercised on the existing Shadeform Linux host: before removal
it reported the captured init alive; after removal submission it reported process
release and empty compute-client state in 0.194 seconds, while `docker rm` returned
in 2.922 seconds. A separate smoke test of the current prototype against a real
`sysbox-runc` container also observed the live-to-released transition. These tests
do not implement or validate the complete asynchronous rental workflow.

Interface references: [cgroup v2 population events](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#un-populated-notification),
[NVIDIA compute-process queries](https://docs.nvidia.com/deploy/nvidia-smi/index.html#selective-query-options),
and [Docker's exit-handler cleanup ordering](https://github.com/moby/moby/blob/v27.5.1/daemon/monitor.go#L85).

## Gates before enabling the change

- [ ] Compare proactive eviction against baseline on ready ENGY_TEE; include both
  cache layers, serving impact, regrowth, reload and early/concurrent preemption.
- [ ] Select the approach using measured handoff latency and disk-eligible coverage.
- [ ] Validate GPU client visibility on matching B300/TEE hardware; unsupported
  evidence must not authorize a rental.
- [ ] Implement and test delivery/execution in the correct host namespaces.
- [ ] Implement the durable protocol and both caller paths above; review the
  backend meaning of a successful filler deletion.
- [ ] Test SSH loss and validator/executor restart in each state; no leaked job,
  premature success, retry by name or deletion of a replacement/shared volume.
- [ ] Cover every probe fallback and post-start filler check while old inspect
  calls are deliberately blocked; confirm no indirect wait returns.
- [ ] Test sufficient/insufficient free disk and early disappearance from volume
  listings; preserve the promised volume size and disk admission checks.
- [ ] Run the real sysbox/customer path with GPU allocation, networking, ports,
  power restoration and cache handling, not only an Alpine/GPU probe.
- [ ] Capture a natural slow Raff Unmount with stack, cache and disk counters.
- [ ] Agree an opt-in rollout, cleanup observability and rollback/drain behavior.

The target is resource handoff within five seconds, ideally two, when process
termination and available disk permit it. Background cleanup does not accelerate
a process actually stuck in D-state or make occupied disk space available early.
