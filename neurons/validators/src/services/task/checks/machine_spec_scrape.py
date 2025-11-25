from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from ..messages import MachineSpecMessages as Msg, render_message
from ..pipeline import CheckResult, Context
from ..runner import SSHCommandRunner
from services.file_encrypt_service import ORIGINAL_KEYS


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
        runner: SSHCommandRunner = ctx.runner

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

        script_path = f"{remote_dir.rstrip('/')}/{script_filename.lstrip('/')}"
        timeout = ctx.config.machine_scrape_timeout or self.DEFAULT_TIMEOUT

        res = await runner.run(f"chmod +x {script_path} && {script_path}", timeout=timeout, retryable=False)

        if not res.success or not res.stdout.strip():
            event = render_message(
                Msg.SCRAPE_FAILED,
                ctx=ctx,
                check_id=self.check_id,
                what={
                    "command_id": res.command_id,
                    "exit_code": res.exit_code,
                    "duration_ms": res.duration_ms,
                    "stderr_tail": res.stderr[-400:],
                },
            )
            return CheckResult(passed=False, event=event)

        try:
            line = res.stdout.splitlines()[0].strip()
            if not ctx.encrypt_key:
                raise ValueError("Missing encrypt_key in context")

            decrypted = ctx.services.ssh.decrypt_payload(ctx.encrypt_key, line)
            raw = json.loads(decrypted)
            obfuscation_keys = ctx.config.obfuscation_keys
            specs = _deobfuscate(raw, obfuscation_keys)

            gpu_info = specs.get("gpu", {}) or {}
            gpu_count = gpu_info.get("count", 0) or 0
            gpu_details = gpu_info.get("details", []) or []
            gpu_model = None
            if gpu_count > 0 and gpu_details:
                gpu_model = gpu_details[0].get("name")

            gpu_model_count = f"{gpu_model}:{gpu_count}" if gpu_model is not None else None
            gpu_uuids = ",".join(detail.get("uuid", "") for detail in gpu_details if detail.get("uuid"))
            sysbox_runtime = specs.get("sysbox_runtime", False)
            extra_info = {
                "sysbox_runtime": sysbox_runtime,
            }

            event = render_message(
                Msg.SCRAPE_OK,
                ctx=ctx,
                check_id=self.check_id,
                what={
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
                what={"exception": str(exc)[:300], "stdout_head": res.stdout[:200]},
            )
            return CheckResult(passed=False, event=event)
