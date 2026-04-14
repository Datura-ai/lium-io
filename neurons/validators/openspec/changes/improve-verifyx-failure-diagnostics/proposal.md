## Why

When VerifyX validation fails, miners receive an opaque error `"challenge verification failed (Failed to verify challenge response)"` with a remediation `"Run VerifyX locally to debug network, disk, and RAM probes"` that has no documented procedure. Production logs show ~16 unique miners/day hitting this path (48 raw events / 24h, with spikes of 13 failures/hour from a single cluster on 2026-04-13), and miners contact support unable to act. The validator-side code already discards the data needed to diagnose the failure (`result.stderr` from the SSH command is never logged), so this is fixable without changes to the C library.

## What Changes

- Capture and log `result.stderr`, exit code, and stdout length/preview from the executor SSH command in `VerifyXValidationService.validate_verifyx_and_process_job`.
- Classify the failure into a small set of buckets (SSH transport error, executor process crash, empty/truncated response, cipher-verify rejection) and produce a class-specific remediation string instead of the single opaque sentence.
- Surface the stderr tail (truncated, secrets-stripped) inside `VERIFY_FAILED.what_we_saw`, so the failure event seen by the miner contains actionable information.
- Add a `help_uri` value pointing to a new short executor-side debug doc (covers: how to read executor logs, how to run `verifyx_executor.py` against a captured cipher, library checksum check). Doc lives under `docs/lium-io/`.

Non-breaking. No changes to the C library, the cipher protocol, or the scoring path.

## Capabilities
nfr
### New Capabilities
- `verifyx-failure-diagnostics`: Defines what diagnostic information the validator MUST capture and surface when a VerifyX challenge fails, and what remediation guidance the failure event MUST carry.

### Modified Capabilities
None — no existing spec exists for VerifyX yet.

## Impact

- Code:
  - `neurons/validators/src/services/verifyx_validation_service.py` — capture stderr/stdout/exit code, classify failures
  - `neurons/validators/src/services/task/checks/verifyx.py` — pass diagnostic fields into `what_we_saw`
  - `neurons/validators/src/services/task/messages.py` — replace single `VERIFY_FAILED` template with class-specific templates (or accept dynamic remediation), set `help_uri`
- Tests:
  - `neurons/validators/tests/test_verifyx_check.py` — new cases for each failure class
- Docs:
  - New `docs/lium-io/verifyx-debug.md` — miner-facing local-debug procedure (the `help_uri` target)
- No DB / migration / API impact. No subnet protocol change.
