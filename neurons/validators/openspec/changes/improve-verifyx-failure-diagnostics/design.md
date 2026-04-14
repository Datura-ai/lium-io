## Context

`VerifyXValidationService.validate_verifyx_and_process_job` (`src/services/verifyx_validation_service.py`) runs a challenge over SSH on each executor:

1. Build a cipher_text from a 64-bit seed + machine_info via `libverifyx.so::generate`.
2. SSH-execute `verifyx_executor.py --seed <s> --cipher_text <c>` on the executor.
3. Take `result.stdout.strip()` as the response cipher, feed it to `lib.verify(...)`.
4. If `verify` returns a null pointer, raise `RuntimeError("Failed to verify challenge response")`.
5. Wrap exceptions as `VerifyXResponse(error=f"challenge verification failed ({str(e)})")`.

The downstream `VerifyXCheck` (`src/services/task/checks/verifyx.py`) then emits the static `VERIFY_FAILED` event from `messages.py` with a single hardcoded `remediation` and `help_uri=None`.

Several real failure causes collapse into the same opaque message:
- SSH connect/auth/timeout (`asyncssh` exception in `shell.ssh_client.run`)
- Executor process crashed (non-zero exit, traceback on `stderr`)
- Executor returned empty/truncated stdout (e.g. OOM-killed mid-output)
- Executor returned a structurally valid cipher that the lib rejects (mismatched lib version, tampered cipher, seed mismatch)

The validator already has the data needed to disambiguate — it just throws it away. `result` from `asyncssh.SSHClientProcess.run` exposes `.stdout`, `.stderr`, `.exit_status`. None of the latter two is logged today.

Production data (Loki, last 24h, prod validator):
- 48 raw events of the opaque error (16 unique failures after dedup × 3 log lines)
- 9 distinct miner hotkeys, 14 distinct executor uuids
- One spike of 13 failures from a single cluster on 2026-04-13 13:00 UTC

## Goals / Non-Goals

**Goals:**
- A miner reading a `VERIFY_FAILED` event can tell which of the four failure classes occurred and what to do next, without contacting support.
- The validator log line for a VerifyX failure contains stderr tail + exit status + stdout length, sufficient for support to triage in under a minute.
- `help_uri` on the failure event resolves to a short, actionable doc.
- No regression in success path latency; no changes to `libverifyx.so` or the cipher protocol.

**Non-Goals:**
- Exposing executor stderr to miners through a public dashboard or API (it goes into validator-side logs and into the failure event, which already contains miner/executor identity).
- Restructuring the existing single-binary VerifyX library to return error codes (would require coordinating a Rust/C library release).
- Re-running the challenge on transient errors (a separate concern).
- Building a self-test mode of `verifyx_executor.py` (rejected scope option — kept for a follow-up if doc + diagnostics aren't enough).

## Decisions

### 1. Capture stderr/stdout/exit code at the SSH boundary, not inside `verify_response`

The C-library wrapper `VerifyXValidator.verify_response` only sees the cipher string — by the time we're inside it, we've already lost the SSH context. The natural seam is in `validate_verifyx_and_process_job` right after `await shell.ssh_client.run(command)`. We extract `(stdout, stderr, exit_status)` once and pass them down as a structured dict.

Alternative considered: return a richer `VerifyXResponse` object with all three fields. Rejected — `VerifyXResponse` already has `data`/`error`; adding three more fields couples it to the SSH transport. Better: add a single optional `diagnostics: dict | None` field carrying the bag.

### 2. Classify failure with a small enum, not free-form remediation strings

A `VerifyXFailureClass` enum with four values:
- `SSH_TRANSPORT` — `shell.ssh_client.run` raised, or returned `None`, or `result` missing attributes
- `EXECUTOR_CRASH` — `exit_status != 0` OR stderr non-empty AND stdout empty
- `EMPTY_RESPONSE` — `exit_status == 0` AND `len(stdout.strip()) < MIN_CIPHER_LEN`
- `CIPHER_REJECTED` — otherwise (this is the current `Failed to verify challenge response` case)

Each class maps to a `MessageTemplate` constant in `messages.py`, with its own `remediation`, `help_uri`, and event text. This keeps the message catalog statically inspectable (matches existing pattern — every other check uses static templates) and makes it easy to localize/iterate later.

Alternative considered: pass the remediation as a runtime string into a single `VERIFY_FAILED` template. Rejected — breaks the convention that templates are the source of truth, and makes it harder to grep prod logs for a specific failure class.

### 3. Truncate stderr; do not redact

The stderr from `verifyx_executor.py` is a Python traceback or NVML/driver error. There is no known case where it can contain miner secrets (no API keys are passed in). Cap the captured stderr at 2 KB (last 2 KB if longer) to bound log volume, but do not regex-redact. If a future change introduces secrets into the executor process env, the redaction should happen at the source, not in this logging path.

### 4. `help_uri` points to a doc in the lium-io repo, not lium-docs

The doc is for technical miner debugging (read systemd logs, run a command), not user-facing. It belongs alongside the security and architecture docs in `docs/lium-io/`. The `help_uri` value will be a stable GitHub blob URL: `https://github.com/Datura-ai/lium-io/blob/main/docs/lium-io/verifyx-debug.md`.

Alternative: host on lium-docs.lium.io. Rejected for now — adds a publishing step and a dependency on the docs deploy pipeline. Can be moved later by changing one constant.

## Risks / Trade-offs

- **[Risk] Misclassification** — e.g., an executor that prints a warning to stderr but exits 0 with a valid cipher would currently be classified `EXECUTOR_CRASH` if we used `stderr non-empty` alone. **Mitigation**: classification key is `(exit_status, len(stdout), bool(stderr))`, with the order above so that a non-empty valid stdout always wins over stderr noise.

- **[Risk] Log-volume growth** — capturing stderr on every failure adds ~1-2 KB per event. At current 16 failures/day this is negligible (<50 KB/day per validator). At a 100x spike it's 5 MB/day per validator — still well within Loki budget.

- **[Trade-off] Static enum vs. dynamic remediation** — the four classes will not cover every future failure shape. We accept that and add a 5th class (`UNKNOWN`) reserving today's generic remediation as the fallback. Adding a 6th class is a one-line change.

- **[Risk] Doc rot** — `help_uri` could 404 if the doc is moved. **Mitigation**: the URL is stored in one constant in `messages.py`; PRs that move/rename the doc must update it. No hard guard, but easy to grep.

## Migration Plan

No migration needed. This is additive in the failure path; the success path is untouched. Rollback is a single revert.

Deploy order:
1. Merge doc + code together (so the `help_uri` is live the moment the new event ships).
2. Validators auto-update via watchtower; new diagnostics appear on next sync cycle.

## Open Questions

- Should we also forward the diagnostic dict into the `VerifyXResponse` so the connector / compute-app can show it in the miner portal? Currently the failure event reaches there via the same pipeline event, so probably yes for free — confirm with a quick check of `result_handler.py` during implementation.
