from __future__ import annotations

import json
from typing import Any

from dataclasses import replace

from ..models import build_msg
from ..pipeline import CheckResult, Context
from ..runner import SSHCommandRunner
from services.file_encrypt_service import ORIGINAL_KEYS
from services.ssh_service import SSHService


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
            event = build_msg(
                event="Remote directory not set",
                reason="MISSING_REMOTE_DIR",
                severity="error",
                category="prep",
                impact="Cannot locate scrape script",
                remediation="Internal error - UploadFilesCheck must run before MachineSpecScrapeCheck",
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        script_filename = ctx.config.machine_scrape_filename
        if not script_filename:
            event = build_msg(
                event="Machine scrape configuration missing",
                reason="SCRAPE_CONFIG_MISSING",
                severity="error",
                category="prep",
                impact="Validation halted",
                remediation="Validator bug: missing scrape configuration in context",
                what={"machine_scrape_filename": script_filename},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        script_path = f"{remote_dir.rstrip('/')}/{script_filename.lstrip('/')}"
        timeout = ctx.config.machine_scrape_timeout or self.DEFAULT_TIMEOUT

        decrypt_service = ctx.services.ssh
        if not isinstance(decrypt_service, SSHService):
            event = build_msg(
                event="SSH service unavailable",
                reason="SCRAPE_DECRYPT_MISSING",
                severity="error",
                category="prep",
                impact="Validation halted",
                remediation="Validator bug: missing SSH service in context",
                what={},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        res = await runner.run(f"chmod +x {script_path} && {script_path}", timeout=timeout, retryable=False)

        if not res.success or not res.stdout.strip():
            event = build_msg(
                event="Machine specs scrape failed",
                reason="SCRAPE_FAILED",
                severity="error",
                category="env",
                impact="Validation halted — GPU unverified",
                remediation=(
                    f"Ensure the scrape script exists and is executable:\n"
                    f"  chmod +x {script_path}\n"
                    f"Check stderr and environment (Python deps) on the executor."
                ),
                what={
                    "command_id": res.command_id,
                    "exit_code": res.exit_code,
                    "duration_ms": res.duration_ms,
                    "stderr_tail": res.stderr[-400:],
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)

        try:
            line = res.stdout.splitlines()[0].strip()
            if not ctx.encrypt_key:
                raise ValueError("Missing encrypt_key in context")

            decrypted = decrypt_service.decrypt_payload(ctx.encrypt_key, line)
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

            event = build_msg(
                event="Machine specs scraped",
                reason="SCRAPE_OK",
                severity="info",
                category="env",
                impact="Proceed",
                what={
                    "gpu_count": gpu_count,
                    "gpu_model": gpu_model,
                },
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            updated_state = replace(
                ctx.state,
                specs=specs,
                gpu_model=gpu_model,
                gpu_count=gpu_count,
                gpu_details=gpu_details,
                gpu_processes=specs.get("gpu_processes", []) or [],
                sysbox_runtime=specs.get("sysbox_runtime", False) or False,
                gpu_model_count=gpu_model_count,
                gpu_uuids=gpu_uuids,
            )

            updates: dict[str, object] = {"state": updated_state}
            return CheckResult(passed=True, event=event, updates=updates)

        except Exception as exc:
            event = build_msg(
                event="Machine specs parse/decrypt failed",
                reason="SCRAPE_PARSE_FAILED",
                severity="error",
                category="env",
                impact="Validation halted — GPU unverified",
                remediation="Confirm encryption key, payload, and repo versions on both validator and executor.",
                what={"exception": str(exc)[:300], "stdout_head": res.stdout[:200]},
                check_id=self.check_id,
                ctx={"executor_uuid": ctx.executor.uuid, "miner_hotkey": ctx.miner_hotkey},
            )
            return CheckResult(passed=False, event=event)
