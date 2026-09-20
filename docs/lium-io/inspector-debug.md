# Debugging an `INSPECTOR_UNREADABLE` inspector event

This doc is for support and reliability engineers reading `inspector_events` rows (and the
validator's `Inspector validation failed` log line) whose reason is `INSPECTOR_UNREADABLE`.

## What the reason means

The validator runs the integrity check on a rented executor over SSH:

```
<python_path> <root_dir>/src/inspector_executor.py --interactive
```

The protocol is one JSON line per command on the executor's **stdout**: the validator writes
`{"cmd": ...}\n`, the executor answers `{"ok": true, "result": "..."}\n` or
`{"ok": false, "error": "..."}\n`. Nothing else may appear on that stream — the executor points
its fd 1 at stderr for everything but the protocol (a chatty dependency, a stray `print`, the
collector's status line) and escapes every result with `json.dumps(..., ensure_ascii=True)`.

`INSPECTOR_UNREADABLE` is recorded when a response line is still not one JSON object. The node is
**counted as unreadable**: it is neither `INSPECTOR_CLEAN` nor the generic
`INSPECTOR_VALIDATION_ERROR`, the score is unchanged (nothing acts on this reason yet), and the
other executors of the run are unaffected — each executor's check parses in its own try.

## The fields

Every unreadable event carries, in `what_we_saw` (the check event) and in `error` (the
`inspector_events` row), next to `executor_uuid` and `reason`:

| field | meaning |
|---|---|
| `payload_cmd` | the command whose answer broke: `start-collector`, `handshake-reply`, `execute`, `quit` |
| `payload_bytes` | how many bytes of the line the validator held |
| `payload_terminated` | `true` = the line ended in `\n`; `false` = it was cut at EOF (executor died, channel closed) or at `INSPECTOR_RESPONSE_MAX_BYTES` (64 MiB) |
| `payload_head` | the first 200 characters, `repr()`-escaped — never the whole payload |
| `json_error` | the decoder's message (`Unterminated string starting at`, `Expecting value`, `response is not a JSON object (int)`, `response line exceeds the …-byte cap`) |
| `json_error_pos` / `json_error_lineno` / `json_error_colno` | where the decoder stopped (0-based char offset; 1-based line and column) |
| `executor_stderr` | up to 8 KiB of the process's stderr, when it wrote any |

The log line (JSON formatter; one line, the `extra` object holds the fields):

```
Inspector validation failed >>> {"executor_uuid": "<uuid>", "reason": "INSPECTOR_UNREADABLE",
  "error_type": "InspectorUnreadableError", "payload_cmd": "execute", "payload_bytes": 2097152,
  "payload_head": "'{\"ok\": true, \"result\": \"AAAA…'", "payload_terminated": false,
  "json_error": "Unterminated string starting at", "json_error_pos": 23, "json_error_lineno": 1,
  "json_error_colno": 24, "error": "inspector executor wrote an unreadable response to 'execute': …",
  "command": "... --interactive", "sensor_integrity": "shell_sha256_unattested", ...}
```

## Reading the shapes

- **`Unterminated string starting at` at column 24, `payload_terminated: false`** — column 24 is
  the opening quote of the `result` value in `{"ok": true, "result": "`; the line was cut inside the
  response cipher. Before this fix the cut came from asyncssh's `readline()`, which returns a
  partial line once one line outgrows the 2 MiB channel receive window (the 20 Sep 2026 case: a
  host whose collector report cipher crossed 2 MiB was unreadable on every run). The validator
  now reassembles a line across those partial reads, so this shape today means the executor
  process ended mid-line (`executor_stderr` may say why) or the line passed the 64 MiB cap
  (`json_error` says `cap`; the cap is checked before the newline, and only the first 200 chars
  of such a line are kept).
- **`INSPECTOR_FAILED_INTERACTIVE`** — the executor answered `ok: false`; its `error` text is the
  row's `error`, kept as a string and cut at 2,048 chars.
- **`Expecting value` at column 1, `payload_terminated: true`** — a non-JSON line on stdout: an
  executor image whose `inspector_executor.py` predates the fd-1 claim, or a shell rc file that
  prints on a non-interactive login. `payload_head` is the line.
- **`Expecting ',' delimiter` mid-line** — an unescaped quote inside the result; the executor is
  not emitting through `json.dumps` (an old image or a patched script).
- **`response is not a JSON object`** — a bare value on the stream; same causes as above.

## What to do

1. `payload_terminated: false` with `Unterminated string` — check the executor container's logs
   for a crash or OOM during `--interactive`; look at `payload_bytes` against the cap.
2. `payload_terminated: true` — the executor image is behind: `docker compose pull && docker
   compose up -d` on the provider's host brings the fd-1 claim; the SHA check
   (`INSPECTOR_FAILED_LIB_MISMATCH`) only covers `libinspector.so`, not the script.
3. Count the reason per executor over the window you care about; a node that is unreadable on
   every run is a node the integrity tool cannot see while it is rented.
