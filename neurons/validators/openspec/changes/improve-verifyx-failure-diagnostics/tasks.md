## 1. Capture diagnostics in VerifyXValidationService

- [x] 1.1 Introduce `VerifyXFailureClass` enum (`SSH_TRANSPORT`, `EXECUTOR_CRASH`, `EMPTY_RESPONSE`, `CIPHER_REJECTED`, `UNKNOWN`) in `verifyx_validation_service.py`
- [x] 1.2 Add an internal `_classify_failure(exit_status, stdout, stderr, transport_error) -> VerifyXFailureClass` helper
- [x] 1.3 Replace the bare `await shell.ssh_client.run(command)` block with a wrapper that captures `(stdout, stderr, exit_status, transport_error)` into a dict with stderr truncated to last 2 KB
- [x] 1.4 Extend `VerifyXResponse` with optional `diagnostics: dict | None` field (do not break callers that read `.error` / `.data`)
- [x] 1.5 In every failure return point, populate `diagnostics` with the captured fields and the classification

## 2. Update VerifyXCheck to surface diagnostics

- [x] 2.1 In `checks/verifyx.py`, when `result.data` is missing, read `result.diagnostics` and merge `failure_class`, `exit_status`, `stdout_len`, `stderr_tail` into `what_we_saw`
- [x] 2.2 Pick the per-class `MessageTemplate` based on `result.diagnostics["failure_class"]`, falling back to the existing `VERIFY_FAILED` template for `UNKNOWN`

## 3. Add per-class message templates and help_uri

- [x] 3.1 In `messages.py` `VerifyXMessages`, add `VERIFY_FAILED_SSH_TRANSPORT`, `VERIFY_FAILED_EXECUTOR_CRASH`, `VERIFY_FAILED_EMPTY_RESPONSE`, `VERIFY_FAILED_CIPHER_REJECTED` templates with class-specific `remediation` strings
- [x] 3.2 Define a single `VERIFYX_DEBUG_DOC_URL` constant in `messages.py` and set it as `help_uri` on all five VerifyX failure templates
- [x] 3.3 Keep the legacy `VERIFY_FAILED` template populated with a generic remediation for the `UNKNOWN` fallback

## 4. Validator-side structured log line

- [x] 4.1 In `validate_verifyx_and_process_job`, emit one `logger.error(_m(...))` per failure with `failure_class`, `exit_status`, `stdout_len`, `stderr_tail`, `transport_error`, `executor_uuid`, `miner_hotkey`
- [x] 4.2 Confirm via test that the success path emits zero new log lines

## 5. Documentation

- [x] 5.1 Create `docs/lium-io/verifyx-debug.md` covering: how to read executor container logs, how to find the seed/cipher in the validator log line, how to invoke `verifyx_executor.py` directly with those values on the executor host, how to verify `libverifyx.so` SHA256 against the validator's
- [x] 5.2 Confirm `VERIFYX_DEBUG_DOC_URL` matches the deployed doc path on the `main` branch

## 6. Tests

- [x] 6.1 Add `tests/test_verifyx_failure_classification.py` with one test per failure class driving the `_classify_failure` helper
- [x] 6.2 Extend `tests/test_verifyx_check.py` (or create) with cases asserting `what_we_saw` contains the diagnostic keys for each class and that `help_uri` is populated
- [x] 6.3 Add a regression test asserting the success path produces no `diagnostics` and no extra log lines

## 7. Verification

- [x] 7.1 Run `pdm run pytest tests/test_verifyx_*` from `neurons/validators/`
- [ ] 7.2 Spot-check on staging: trigger a deliberate failure (e.g., stop executor process) and confirm a Loki query for the new log line returns the diagnostic fields
- [ ] 7.3 Verify the `help_uri` URL in the rendered event opens the new doc on GitHub
