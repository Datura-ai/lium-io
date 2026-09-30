from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import redis.exceptions
from datura.requests.miner_requests import ExecutorSSHInfo
from neurons.validators.src.protocol.vc_protocol.compute_requests import (
    FillerRunActiveResponse,
    PodRentalActiveResponse,
)
from neurons.validators.src.services.task.pipeline import (
    Context,
    ContextConfig,
    ContextServices,
    ContextState,
)


def _definition_name(node: ast.stmt) -> str:
    if isinstance(node, ast.FunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return ""


def build_scrape_namespace(
    source_path: Path, helper_names: set[str], seed_namespace: dict[str, Any]
) -> dict[str, Any]:
    # the named top-level definitions of a standalone script, executed in a namespace of their own
    tree = ast.parse(source_path.read_text())
    kept_definitions = [node for node in tree.body if _definition_name(node) in helper_names]
    assert {_definition_name(node) for node in kept_definitions} == helper_names, (
        f"{source_path.name} no longer defines all of {sorted(helper_names)} at module level"
    )

    namespace = dict(seed_namespace)
    kept_module = ast.Module(body=kept_definitions, type_ignores=[])
    exec(compile(kept_module, source_path.stem, "exec"), namespace)
    return namespace


def dict_literal_keys(module: ast.Module, dict_name: str) -> list[str]:
    """Keys of the single dict literal assigned to `dict_name`, in source order."""
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == dict_name
            and isinstance(node.value, ast.Dict)
        ):
            return [key.value for key in node.value.keys]
    raise AssertionError(f"{dict_name} dict literal not found")


# Every Fernet token is base64url of a 0x80 version byte, so all of them start with "gAAAAA" —
# and MachineSpecScrapeCheck only tries to decrypt the stdout lines shaped like that.
FERNET_TOKEN = "gAAAAABscrape-payload"


@dataclass(frozen=True)
class SFTPPutCall:
    local_path: str
    remote_path: str
    recurse: bool


class DummySFTPClient:
    """Mock SFTP client that simulates file upload."""

    def __init__(self, *, should_raise: bool = False, error_message: str = ""):
        self.should_raise = should_raise
        self.error_message = error_message
        self.put_called_with: SFTPPutCall | None = None
        self.put_call_count = 0

    async def put(self, local_path: str, remote_path: str, recurse: bool = False) -> None:
        self.put_call_count += 1
        self.put_called_with = SFTPPutCall(local_path, remote_path, recurse)
        if self.should_raise:
            raise RuntimeError(self.error_message)


class DummySSHClient:
    """Mock SSH client that provides SFTP access."""

    def __init__(self, *, sftp_should_raise: bool = False, sftp_error: str = ""):
        self.sftp_client = DummySFTPClient(
            should_raise=sftp_should_raise,
            error_message=sftp_error,
        )

    def start_sftp_client(self):
        return self

    async def __aenter__(self):
        return self.sftp_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyScoreCalc:
    def __call__(self, *args, **kwargs):  # pragma: no cover
        return 0.0, 0.0, ""


class FakeRedis:
    """Dict-backed stand-in for RedisService's get/set/delete and the hash calls (hset/hgetall/hdel/expire).

    `ttl` records the `ex` of the last set per key (None when set without one), so a
    test can assert that a mark carries an expiry. `failing` makes every call raise the client's
    ConnectionError, the shape of a Redis outage seen through RedisService; `fail_next_set_of`
    makes only the next `set` of those keys raise (one shot each; `fail_set_of_after[key] = n` skips n sets first), the shape of a blip that hits
    one write in the middle of a cycle.
    """

    def __init__(self, *, failing: bool = False):
        self.store: dict[str, str] = {}
        # hset/hgetall/hdel: the per-cycle fleet and due hashes (DAH-2870), kept apart from the
        # string keys so `store` still reads as the per-pod marks alone.
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttl: dict[str, int | None] = {}
        self.failing = failing
        self.fail_next_set_of: set[str] = set()
        # key -> how many `set`s of it to let through before the one that raises (one shot each)
        self.fail_set_of_after: dict[str, int] = {}
        # keys whose next DELETE fails (one shot each): a plain `delete`, or the whole batch when
        # it sits inside write_atomically
        self.fail_delete_of: set[str] = set()
        self.calls = 0

    def _touch(self):
        self.calls += 1
        if self.failing:
            raise redis.exceptions.ConnectionError("Error 111 connecting to redis:6379")

    async def get(self, key: str):
        self._touch()
        return self.store.get(key)

    def _set_hook(self, key: str):
        if key in self.fail_next_set_of:
            self.fail_next_set_of.discard(key)
            raise redis.exceptions.TimeoutError(f"Timeout writing {key}")
        if key in self.fail_set_of_after:
            if self.fail_set_of_after[key] == 0:
                del self.fail_set_of_after[key]
                raise redis.exceptions.TimeoutError(f"Timeout writing {key}")
            self.fail_set_of_after[key] -= 1

    async def set(self, key: str, value: str, ex: int | None = None):
        self._touch()
        self._set_hook(key)
        self.store[key] = value
        self.ttl[key] = ex

    async def write_atomically(self, writes):
        """`RedisService.write_atomically`: all of `writes` or none. `fail_delete_of` names keys whose
        DELETE fails the whole batch (the shape of a connection lost mid-transaction); the set hooks
        above apply to every SET in the batch. A failing batch applies nothing."""
        self._touch()
        for name, args, _kwargs in writes.ops:
            if name == "set":
                self._set_hook(args[0])
            if name == "delete" and args[0] in self.fail_delete_of:
                self.fail_delete_of.discard(args[0])
                raise redis.exceptions.ConnectionError(f"Connection lost deleting {args[0]}")
        # EXEC: applied without the hooks (they were consumed above) and as one round trip
        for name, args, kwargs in writes.ops:
            if name == "set":
                key, value = args
                self.store[key] = value
                self.ttl[key] = kwargs.get("ex")
            elif name == "delete":
                (key,) = args
                self.store.pop(key, None)
                self.hashes.pop(key, None)
                self.ttl.pop(key, None)
            elif name == "hset":
                key, field, value = args
                self.hashes.setdefault(key, {})[field] = value
            elif name == "expire":
                key, seconds = args
                if key in self.store or key in self.hashes:
                    self.ttl[key] = seconds
            else:
                raise AssertionError(f"FakeRedis.write_atomically: unknown write {name}")

    async def delete(self, key: str):
        self._touch()
        if key in self.fail_delete_of:
            self.fail_delete_of.discard(key)
            raise redis.exceptions.ConnectionError(f"Connection lost deleting {key}")
        self.store.pop(key, None)
        self.hashes.pop(key, None)
        self.ttl.pop(key, None)

    async def hset(self, key: str, field: str, value: str):
        self._touch()
        self.hashes.setdefault(key, {})[field] = value

    async def hget(self, key: str, field: str):
        self._touch()
        return self.hashes.get(key, {}).get(field)

    async def hgetall(self, key: str):
        self._touch()
        # redis-py returns bytes for both sides; the module must decode them
        return {k.encode(): v.encode() for k, v in self.hashes.get(key, {}).items()}

    async def hdel(self, key: str, *fields: str):
        self._touch()
        for field in fields:
            self.hashes.get(key, {}).pop(field, None)

    async def expire(self, key: str, seconds: int):
        self._touch()
        if key in self.store or key in self.hashes:
            self.ttl[key] = seconds


def default_executor() -> ExecutorSSHInfo:
    return ExecutorSSHInfo(
        uuid="executor-123",
        address="127.0.0.1",
        port=22,
        ssh_username="root",
        ssh_port=22,
        python_path="/usr/bin/python",
        root_dir="/root/app",
        price_per_gpu=0.5,
    )


def build_context_config(**overrides) -> ContextConfig:
    base = dict(
        executor_root="/root/app",
        compute_rest_app_url="http://validator",
        gpu_monitor_script_relative="src/gpus_utility.py",
        machine_scrape_filename="scrape.sh",
        machine_scrape_timeout=300,
        obfuscation_keys={},
        default_docker_image_digests={},
        validator_keypair=None,
        max_gpu_count=None,
        gpu_model_rates={},
        nvml_digest_map={},
        enable_no_collateral=False,
        verifyx_enabled=False,
        inspector_enabled=False,
        port_private_key=None,
        port_public_key=None,
        job_batch_id="batch-1",
    )
    base.update(overrides)
    return ContextConfig(**base)


def build_services(**overrides) -> ContextServices:
    pod_recovery = AsyncMock()
    # A bare AsyncMock would report every pod as recovered; opt in per test instead.
    pod_recovery.recover_pod_after_stale_vloopback_mount.return_value = False
    backend = AsyncMock()
    # A bare AsyncMock would confirm every Lium-named container as ours (DAH-2757); opt in per test.
    backend.get_filler_run_active.return_value = FillerRunActiveResponse(active=False)
    backend.get_pod_rental_active.return_value = PodRentalActiveResponse(active=False)
    # A bare AsyncMock would report a live Lium workload on every node (DAH-3480); opt in per test.
    backend.get_rented_executors_now.return_value = None
    base = dict(
        ssh=None,
        # DAH-2870: the rented check keeps per-pod marks in Redis on every rented cycle.
        redis=FakeRedis(),
        collateral=None,
        validation=None,
        verifyx=None,
        inspector=None,
        connectivity=None,
        shell=None,
        score_calculator=DummyScoreCalc(),
        backend=backend,
        container_cleanup=None,
        pod_recovery=pod_recovery,
    )
    base.update(overrides)
    return ContextServices(**base)


def build_state(**overrides) -> ContextState:
    base = dict(specs={}, verified_port_count=0)
    base.update(overrides)
    return ContextState(**base)


def make_context(
    *,
    executor: ExecutorSSHInfo | None = None,
    services: ContextServices | None = None,
    config: ContextConfig | None = None,
    state: ContextState | None = None,
    miner_hotkey: str = "miner-hotkey",
    miner_address: str = "127.0.0.1",
    miner_port: int = 8000,
    pipeline_id: str = "test-pipeline-id",
    **extra,
) -> Context:
    executor_obj = executor or default_executor()
    services_obj = services or build_services()
    config_obj = config or build_context_config()
    state_obj = state or build_state()

    base_kwargs = dict(
        pipeline_id=pipeline_id,
        executor=executor_obj,
        miner_hotkey=miner_hotkey,
        miner_address=miner_address,
        miner_port=miner_port,
        ssh=None,
        runner=None,
        verified={},
        settings={},
        encrypt_key=None,
        default_extra={},
        services=services_obj,
        config=config_obj,
        state=state_obj,
        is_rental_succeed=False,
    )
    base_kwargs.update(extra)

    return Context.model_construct(**base_kwargs)


# Incentive logging assertion helpers

def assert_incentive_log_present(log_text: str) -> None:
    """Verify 'Incentive Scores Calculation Logs:' header is present."""
    assert "Incentive Scores Calculation Logs:" in log_text


def assert_executor_has_log(log_text: str, executor_id: str) -> None:
    """Verify executor_id appears in the logs."""
    assert executor_id in log_text


def assert_log_contains_keys(log_text: str, expected_keys: list[str]) -> None:
    """Verify log_text contains all expected key strings."""
    for key in expected_keys:
        assert key in log_text, f"Expected key '{key}' not found in log_text"


def extract_incentive_section(log_text: str) -> str:
    """Extract the incentive logs section from log_text."""
    if "Incentive Scores Calculation Logs:" not in log_text:
        return ""
    return log_text.split("Incentive Scores Calculation Logs:")[1]


def count_incentive_log_entries(log_text: str) -> int:
    """Count incentive log entries by counting 'executor_id:' occurrences."""
    section = extract_incentive_section(log_text)
    return section.count("executor_id:")


# Rental price incentive log (rental_price.py lines 146-165): message + extra keys
RENTAL_PRICE_INCENTIVE_LOG_MESSAGE = (
    "Rental price incentive for executor is calculated successfully. "
    "Formula: rental_share * gpu_count * effective_rate / total_rental_cost"
)
RENTAL_PRICE_INCENTIVE_EXTRA_KEYS = [
    "hotkey",
    "executor_id",
    "gpu_model",
    "gpu_count",
    "hourly_rate",
    "unrented_cap_multiplier",
    "effective_rate",
    "total_unrented_by_gpu_type",
    "max_cap",
    "cap_dilution_applied",
    "rental_share",
    "burn_share",
    "incentive",
    "total_rental_cost",
]


def assert_rental_price_incentive_log_full_content(full_log_text: str) -> None:
    """Assert full_log_text contains the rental price incentive log message and all extra keys.

    Covers the logic in incentive/rental_price.py that appends the rental price
    incentive log (message + get_extra_info(...)). Verifies the message and
    every extra field is present in the log.
    """
    assert_incentive_log_present(full_log_text)
    assert (
        RENTAL_PRICE_INCENTIVE_LOG_MESSAGE in full_log_text
    ), f"Expected rental price incentive message not found in full_log_text"
    assert_log_contains_keys(full_log_text, RENTAL_PRICE_INCENTIVE_EXTRA_KEYS)
