from __future__ import annotations

import json
import shlex
from dataclasses import replace
from typing import Any, Literal

from ..messages import MachineSpecMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from ..runner import SSHCommandResult
from services.file_encrypt_service import ORIGINAL_KEYS
from services.gpu_spec_table import normalize_gpu_model
from .network_ema import compute_ema
from .upload_files import UploadFailed, upload_validation_files

# DAH-2794: how long a failed stdin attempt may have taken and still be worth retrying with the
# binary. Above a full scrape (~15 s on a real box, so a payload that will not decrypt is only
# knowable around then), below the point where the retry stops fitting: 60 s here + 300 s upload
# + 300 s scrape = 660 s against the 780 s per-executor timeout. The upload gets one attempt for
# exactly that reason.
FALLBACK_MAX_STDIN_DURATION_MS = 60_000
FALLBACK_UPLOAD_ATTEMPTS = 1

# The scrape says only two things on stdout: a Fernet token when it worked, and a JSON object
# with an "error" key when it ran but had nothing to report. Every Fernet token is base64url of
# a 0x80 version byte, so this prefix separates ours from whatever the miner's image printed.
FERNET_TOKEN_PREFIX = "gAAAAA"

# How much stdout the miner puts in front of the scrape's own line is his choice, not a bounded
# amount, so only the tail is searched and only so many candidates are decrypted — both run
# synchronously on the event loop shared with every other executor in the cycle.
STDOUT_SEARCH_TAIL_BYTES = 256 * 1024
MAX_PAYLOAD_DECRYPT_ATTEMPTS = 20


def _update_keys(data: Any, key_mapping: dict[str, str]) -> Any:
    if isinstance(data, dict):
        updated: dict[str, Any] = {}
        for key, value in data.items():
            original_key = key_mapping.get(key, key)
            updated[original_key] = _update_keys(value, key_mapping)
        return updated
    if isinstance(data, list):
        return [_update_keys(item, key_mapping) for item in data]
    return data


def _deobfuscate(spec: dict[str, Any], obfuscation_keys: dict[str, str] | None) -> dict[str, Any]:
    if not obfuscation_keys:
        return spec
    reverse = {v: k for k, v in obfuscation_keys.items()}
    first_pass = _update_keys(spec, reverse)
    return _update_keys(first_pass, ORIGINAL_KEYS)


def _normalize_gpu_details(gpu_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **detail,
            "name": normalize_gpu_model(detail.get("name")),
        }
        if detail.get("name")
        else detail
        for detail in gpu_details
    ]


def _decrypt_payload(ctx: Context, stdout: str) -> str:
    # decrypt the scrape's Fernet token out of whatever else the executor printed
    #
    # The token is the scrape's own print, but the image decides what surrounds it: a .pth or
    # sitecustomize prints before it, an atexit handler after it. Searching newest-first beats
    # betting on a fixed position.
    if not ctx.encrypt_key:
        raise ValueError("Missing encrypt_key in context")

    last_exc: Exception | None = None
    attempts = 0
    for line in reversed(stdout[-STDOUT_SEARCH_TAIL_BYTES:].splitlines()):
        candidate = line.strip()
        if not candidate.startswith(FERNET_TOKEN_PREFIX):
            continue
        try:
            return ctx.services.ssh.decrypt_payload(ctx.encrypt_key, candidate)
        except Exception as exc:
            last_exc = exc
        attempts += 1
        if attempts == MAX_PAYLOAD_DECRYPT_ATTEMPTS:
            break

    raise last_exc or ValueError("No scrape payload on stdout")


def _scrape_reported_its_own_failure(stdout: str) -> bool:
    # the scrape prints {"error": ...} as its last line and exits non-zero when it ran but found
    # nothing to report; a source the interpreter could not run prints nothing of ours at all
    lines = [line for line in stdout[-STDOUT_SEARCH_TAIL_BYTES:].splitlines() if line.strip()]
    if not lines:
        return False
    try:
        report = json.loads(lines[-1])
    except ValueError:
        return False
    return isinstance(report, dict) and "error" in report


def _binary_command(remote_dir: str, script_filename: str) -> str:
    script_path = f"{remote_dir.rstrip('/')}/{script_filename.lstrip('/')}"
    return f"chmod +x {script_path} && {script_path}"


class MachineSpecScrapeCheck:
    """Run the obfuscated scrape script and unpack the executor's hardware profile.

    This is the backbone for nearly every other check: GPU inventory, UUIDs, process
    lists, and sysbox hints are all extracted here exactly as the legacy flow did.
    Skipping or weakening it would starve later checks of their source data.
    """

    check_id = "gpu.scrape.machine_spec"
    fatal = True

    DEFAULT_TIMEOUT = 300

    async def run(self, ctx: Context) -> CheckResult:
        timeout = ctx.config.machine_scrape_timeout or self.DEFAULT_TIMEOUT

        if ctx.config.machine_scrape_source:
            return await self._scrape_from_source(ctx, timeout)
        return await self._scrape_uploaded_binary(ctx, timeout)

    async def _scrape_from_source(self, ctx: Context, timeout: int) -> CheckResult:
        # pipe the obfuscated source into the executor's own interpreter; binary on failure
        #
        # `-I` implies -E -s -P, which drops PYTHONPATH, the user site-dir and the cwd entry
        # from sys.path, so a file left next to the SSH login shell cannot shadow psutil or
        # json. It does not disable site processing: a .pth or sitecustomize inside the image
        # still runs and prints, which is what `_decrypt_payload` searches through.
        command = f"{shlex.quote(ctx.executor.python_path)} -I -"
        scrape_run = await ctx.runner.run(
            command,
            timeout=timeout,
            retryable=False,
            stdin_text=ctx.config.machine_scrape_source,
        )
        result = self._check_result_from_scrape_run(ctx, scrape_run, delivery="stdin")
        if result.passed:
            return result

        # The source did not run here: wrong interpreter, no psutil, or a cryptography too old
        # to produce a token this validator can read. The binary carries its own interpreter and
        # every dependency, so retry with it — but past FALLBACK_MAX_STDIN_DURATION_MS the scrape
        # hung rather than failed to start, and the binary runs the very same code.
        if (
            scrape_run.duration_ms > FALLBACK_MAX_STDIN_DURATION_MS
            or not ctx.config.machine_scrape_filename
        ):
            return result

        # The scrape reporting its own failure (no GPU details, for one) means the interpreter
        # ran it, so the binary would print the same thing for the price of a 13 MB upload.
        if scrape_run.exit_code != 0 and _scrape_reported_its_own_failure(scrape_run.stdout):
            return result

        # The stdin attempt's own event is discarded with the retry, so carry it whole: a run that
        # exited 0 and produced an unreadable payload says nothing in exit_code or stderr.
        fallback_from = {"reason": result.event.reason_code, **result.event.what_we_saw}

        try:
            remote_dir = await upload_validation_files(ctx, attempts=FALLBACK_UPLOAD_ATTEMPTS)
        except UploadFailed as exc:
            event = render_message(
                Msg.SCRAPE_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "delivery": "stdin",
                    "fallback_from": fallback_from,
                    "fallback_upload_error": str(exc),
                },
            )
            return CheckResult(passed=False, event=event)

        scrape_run = await ctx.runner.run(
            _binary_command(remote_dir, ctx.config.machine_scrape_filename),
            timeout=timeout,
            retryable=False,
        )
        return self._check_result_from_scrape_run(
            ctx, scrape_run, delivery="upload", fallback_from=fallback_from
        )

    async def _scrape_uploaded_binary(self, ctx: Context, timeout: int) -> CheckResult:
        # run the frozen scrape that UploadFilesCheck put on the executor
        remote_dir = ctx.state.remote_dir
        if not remote_dir:
            event = render_message(
                Msg.REMOTE_DIR_MISSING,
                ctx=ctx,
                check_id=self.check_id,
            )
            return CheckResult(passed=False, event=event)

        script_filename = ctx.config.machine_scrape_filename
        if not script_filename:
            event = render_message(
                Msg.CONFIG_MISSING,
                ctx=ctx,
                check_id=self.check_id,
                what={"machine_scrape_filename": script_filename},
            )
            return CheckResult(passed=False, event=event)

        scrape_run = await ctx.runner.run(
            _binary_command(remote_dir, script_filename), timeout=timeout, retryable=False
        )
        return self._check_result_from_scrape_run(ctx, scrape_run, delivery="upload")

    def _check_result_from_scrape_run(
        self,
        ctx: Context,
        scrape_run: SSHCommandResult,
        *,
        delivery: Literal["stdin", "upload"],
        fallback_from: dict[str, Any] | None = None,
    ) -> CheckResult:
        what: dict[str, Any] = {"delivery": delivery}
        if fallback_from:
            what["fallback_from"] = fallback_from

        if not scrape_run.success or not scrape_run.stdout.strip():
            event = render_message(
                Msg.SCRAPE_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    **what,
                    "command_id": scrape_run.command_id,
                    "exit_code": scrape_run.exit_code,
                    "duration_ms": scrape_run.duration_ms,
                    "stderr_tail": scrape_run.stderr[-400:],
                },
            )
            return CheckResult(passed=False, event=event)

        try:
            decrypted = _decrypt_payload(ctx, scrape_run.stdout)
            raw = json.loads(decrypted)
            obfuscation_keys = ctx.config.obfuscation_keys
            specs = _deobfuscate(raw, obfuscation_keys)

            gpu_info = specs.get("gpu", {}) or {}
            gpu_count = gpu_info.get("count", 0) or 0
            raw_gpu_details = gpu_info.get("details", []) or []
            # The native capability challenge derives its key from the raw NVML name in
            # specs. Policy and scoring use the canonicalized state fields below.
            gpu_details = _normalize_gpu_details(raw_gpu_details)
            gpu_model = None
            if gpu_count > 0 and gpu_details:
                gpu_model = gpu_details[0].get("name")

            gpu_model_count = f"{gpu_model}:{gpu_count}" if gpu_model is not None else None
            gpu_uuids = ",".join(detail.get("uuid", "") for detail in gpu_details if detail.get("uuid"))
            sysbox_runtime = specs.get("sysbox_runtime", False)
            hardware_supports = specs.get("storage_limit_supported", False)
            gpu_splitting_config = ctx.state.rented_data.gpu_splitting_config if ctx.state.rented_data else {}
            gpu_splitting_min_count = gpu_splitting_config.get(ctx.executor.uuid)
            supports_gpu_splitting = hardware_supports and gpu_splitting_min_count is not None

            prev_ema = (
                ctx.state.rented_data.network_ema.get(ctx.executor.uuid)
                if ctx.state.rented_data else None
            )
            network = specs.get("network") or {}
            network["ema_download_speed"] = compute_ema(
                prev_ema.ema_download_speed if prev_ema else None,
                network.get("download_speed"),
            )
            network["ema_upload_speed"] = compute_ema(
                prev_ema.ema_upload_speed if prev_ema else None,
                network.get("upload_speed"),
            )
            specs = {**specs, "network": network}

            extra_info = {
                "sysbox_runtime": sysbox_runtime,
                "supports_gpu_splitting": supports_gpu_splitting,
                "gpu_splitting_min_count": gpu_splitting_min_count,
            }

            event = render_message(
                Msg.SCRAPE_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    **what,
                    "gpu_count": gpu_count,
                    "gpu_model": gpu_model,
                    "network": specs.get("network"),
                },
                extra=extra_info,
            )
            updated_state = replace(
                ctx.state,
                specs=specs,
                gpu_model=gpu_model,
                gpu_count=gpu_count,
                gpu_details=gpu_details,
                gpu_processes=specs.get("gpu_processes", []) or [],
                sysbox_runtime=sysbox_runtime,
                supports_gpu_splitting=supports_gpu_splitting,
                gpu_splitting_min_count=gpu_splitting_min_count,
                gpu_model_count=gpu_model_count,
                gpu_uuids=gpu_uuids,
            )
            updates: dict[str, object] = {"state": updated_state, "default_extra": {**ctx.default_extra, **extra_info}}
            return CheckResult(passed=True, event=event, updates=updates)

        except Exception as exc:
            event = render_message(
                Msg.SCRAPE_PARSE_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={**what, "exception": str(exc)[:300], "stdout_head": scrape_run.stdout[:200]},
            )
            return CheckResult(passed=False, event=event)
