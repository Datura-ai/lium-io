# Debugging VerifyX validation failures

This doc is for miners and support engineers triaging a `VerifyX validation failed` event.
The validator emits the event whenever the challenge response coming back from your
executor cannot be verified. Every failure event now carries a `failure_class` field in
`what_we_saw` — match it to the section below.

## 1. Read the validator log line

On each failure the validator emits a single structured ERROR log line:

```
VerifyX validation failed
  failure_class=...
  exit_status=...
  stdout_len=...
  stderr_tail=...
  transport_error=...
  executor_uuid=...
  miner_hotkey=...
  seed=...
  cipher_text=...
```

The `seed` and `cipher_text` come from the preceding `VerifyX Python Script Command`
info line emitted moments earlier for the same `pipeline_id` — use `pipeline_id` to
join them in Loki/Grafana. The two values are all you need to reproduce the run.

## 2. Failure classes

### `SSH_TRANSPORT`

The validator could not complete the SSH command. `exit_status` is `null` and
`transport_error` holds the asyncssh exception class and message.

Check:
- Executor is up: `docker compose ps`
- SSH port reachable from the validator IP
- Validator public key present in the executor user's `~/.ssh/authorized_keys`

### `EXECUTOR_CRASH`

`verifyx_executor.py` exited non-zero and produced no valid stdout. `stderr_tail`
contains the last 2 KB of stderr from the process (usually a Python traceback or a
CUDA/NVML error).

Check:
- Read the full executor container logs: `docker logs <executor-container>`
- Verify GPU drivers and `nvidia-smi` work inside the container
- Reproduce locally: see section 3 below

### `EMPTY_RESPONSE`

The SSH command exited `0` but stdout was empty or shorter than 64 bytes. Usually
an OOM-killer or disk-full condition midway through the probe.

Check:
- `dmesg | tail -n 200` for `Out of memory: Killed process ... verifyx_executor`
- `df -h` for disk full on the executor data mount
- Recent memory pressure on the host

### `CIPHER_REJECTED`

The executor returned a structurally plausible cipher, but the validator's
`libverifyx.so::verify` returned null. Most common cause: executor running a
different version of `libverifyx.so` than the validator.

Check:
- SHA256 match between validator and executor (section 4 below)
- If mismatched: `docker compose pull && docker compose up -d` on the executor

### `UNKNOWN`

Classification fell through. Treat as a validator bug — include the full event
payload and the validator log line when reporting.

## 3. Reproduce the executor run locally

SSH into the executor host, then run `verifyx_executor.py` with the `seed` and
`cipher_text` from the validator log:

```bash
cd <executor-repo-root>
python src/verifyx_executor.py \
  --seed <seed-from-log> \
  --cipher_text <cipher-from-log>
```

Expected: the process prints a long hex cipher response to stdout and exits `0`.
Any other outcome matches the failure class you saw.

## 4. Verify `libverifyx.so` SHA256

On the executor:

```bash
sha256sum /usr/lib/libverifyx.so
```

On the validator (contact the validator operator, or check the release artifact
for the deployed validator image). The two SHA256 values MUST match. A mismatch
means the executor image is out of date — pull the latest:

```bash
docker compose pull executor
docker compose up -d executor
```

## 5. Still stuck?

Collect and share:
- The full failure event (JSON payload from the miner dashboard)
- The validator log line from section 1 (with `pipeline_id`)
- The preceding `VerifyX Python Script Command` log line (for the `seed` and
  `cipher_text`)
- `docker logs` from the executor around the timestamp of the failure
