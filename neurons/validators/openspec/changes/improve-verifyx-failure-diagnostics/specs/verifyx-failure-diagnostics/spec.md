## ADDED Requirements

### Requirement: Validator MUST capture executor SSH process diagnostics on VerifyX failure

When the VerifyX SSH command on an executor fails to produce a verifiable response, the validator SHALL capture the following from the SSH process result and attach them to the failure context: exit status (or `null` if the SSH transport itself failed), stdout length in bytes, the last 2 KB of stderr, and the exception type/message if the SSH call raised. These fields are required so support and miners can distinguish transport failures from executor crashes from cipher rejection.

#### Scenario: SSH transport raises

- **WHEN** `shell.ssh_client.run` raises any exception
- **THEN** the failure context SHALL include `exit_status=null`, `stderr=null`, `stdout_len=null`, and `transport_error=<exception class>: <message>`
- **AND** the failure SHALL be classified as `SSH_TRANSPORT`

#### Scenario: Executor process exits non-zero

- **WHEN** the SSH command completes with `exit_status != 0` and stdout is empty
- **THEN** the failure context SHALL include the actual exit code, the stderr tail (last 2 KB), and `stdout_len=0`
- **AND** the failure SHALL be classified as `EXECUTOR_CRASH`

#### Scenario: Executor produces empty or truncated stdout

- **WHEN** the SSH command completes with `exit_status == 0` but `len(stdout.strip()) < 64`
- **THEN** the failure context SHALL include the exit code, the stderr tail, and the actual stdout length
- **AND** the failure SHALL be classified as `EMPTY_RESPONSE`

#### Scenario: Cipher response is rejected by the verify library

- **WHEN** stdout is non-empty and `lib.verify(...)` returns a null pointer
- **THEN** the failure context SHALL include `exit_status=0`, `stdout_len=<actual length>`, and the stderr tail (which is normally empty in this case but recorded for completeness)
- **AND** the failure SHALL be classified as `CIPHER_REJECTED`

### Requirement: Failure event MUST surface diagnostics to the miner

The `VerifyX validation failed` event emitted by `VerifyXCheck` SHALL include the captured diagnostic fields under `what_we_saw`, so that miners reading the event can act on the failure without contacting support.

#### Scenario: Diagnostics appear in event payload

- **WHEN** any VerifyX failure class is emitted
- **THEN** the event's `what_we_saw` dict SHALL contain the keys `failure_class`, `exit_status`, `stdout_len`, and `stderr_tail` (any of which MAY be `null` per the capture rules above)
- **AND** the existing `errors` key SHALL remain populated with a human-readable summary

### Requirement: Remediation guidance MUST be specific to the failure class

Each failure class SHALL have its own `MessageTemplate` in `messages.py` with a `remediation` string that names a concrete next step the miner can take. The single generic `VERIFY_FAILED` template MUST NOT be used as the catch-all for all four classes.

#### Scenario: SSH_TRANSPORT remediation

- **WHEN** the failure class is `SSH_TRANSPORT`
- **THEN** the remediation text SHALL instruct the miner to verify the executor is reachable on its SSH port and that the validator's key is authorized

#### Scenario: EXECUTOR_CRASH remediation

- **WHEN** the failure class is `EXECUTOR_CRASH`
- **THEN** the remediation text SHALL instruct the miner to read the executor container logs (or systemd unit) and reproduce by running `verifyx_executor.py` with the seed and cipher captured in the validator log line

#### Scenario: EMPTY_RESPONSE remediation

- **WHEN** the failure class is `EMPTY_RESPONSE`
- **THEN** the remediation text SHALL instruct the miner to check for OOM / disk-full conditions on the executor host

#### Scenario: CIPHER_REJECTED remediation

- **WHEN** the failure class is `CIPHER_REJECTED`
- **THEN** the remediation text SHALL instruct the miner to verify the executor `libverifyx.so` checksum matches the validator's, and to restart the executor container if it does not

### Requirement: Failure event MUST carry a help_uri to the debug doc

Every VerifyX failure template SHALL set a non-null `help_uri` pointing to a stable URL for the miner-facing VerifyX debug doc.

#### Scenario: help_uri is populated on every failure class

- **WHEN** any of the five failure templates renders an event
- **THEN** the event's `help_uri` SHALL be a non-null string URL

### Requirement: Diagnostics MUST be logged at the validator-side log

The validator SHALL log a single structured log line per VerifyX failure containing the diagnostic dict, before the `VerifyXResponse(error=...)` is returned. This log line is the support-side primary triage source.

#### Scenario: Log line is emitted on every failure

- **WHEN** any VerifyX failure class is detected inside `validate_verifyx_and_process_job`
- **THEN** a single structured log entry at level `ERROR` SHALL be emitted with the failure class, exit_status, stdout_len, stderr_tail, transport_error (if any), executor_uuid, and miner_hotkey

### Requirement: Existing success path MUST be unchanged

The diagnostic-capture logic SHALL be inert on the success path. No new fields, log lines, or per-call work SHALL be introduced when `lib.verify(...)` returns a non-null cipher pointer.

#### Scenario: Success path has no overhead

- **WHEN** VerifyX validation succeeds
- **THEN** no diagnostic dict SHALL be constructed, no extra log line SHALL be emitted, and the returned `VerifyXResponse` SHALL contain only the existing `data` field
