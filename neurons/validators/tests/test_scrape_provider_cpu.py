"""DAH-2734: the scrape's CPU attribution must void itself rather than blame the provider.

The host reading only means something when every container that burned CPU in the SAME window is
subtracted from it. Two things break that pairing on a real host: a container that exits between
`docker stats` and `docker ps`, and a docker daemon too wedged to answer `stats` at all. Both must
end with no host reading in the payload, which makes the gate report NOT_MEASURABLE and withhold
no money.

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers
are compiled out of the source and executed on their own, like test_scrape_disk_breakdown.py does.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace
from neurons.validators.tests.test_scrape_disk_breakdown import _obfuscation_tables

SRC = Path(__file__).resolve().parents[1] / "src"

CPU_HELPERS = {"get_container_cpu_percents", "get_docker_info", "parse_container_net_io", "docker_size_bytes", "_DOCKER_SIZE_UNITS"}

RENTER = "aaaaaaaaaaaa" + "0" * 52
EXECUTOR = "bbbbbbbbbbbb" + "0" * 52


class _Psutil:
    """`cpu_percent(interval=None)` returns the load since the previous call — the reset returns 0."""

    def __init__(self, host_percent: float) -> None:
        self._readings = [0.0, host_percent]

    def cpu_percent(self, interval=None) -> float:
        return self._readings.pop(0)


def _fake_run_cmd(outputs: dict[str, str | Exception]):
    """Match a docker call by the subcommand in it, so the temp binary path does not matter."""

    def run_cmd(cmd: str) -> str:
        for marker, output in outputs.items():
            if marker in cmd:
                if isinstance(output, Exception):
                    raise output
                return output
        return ""

    return run_cmd


def _scrape(outputs: dict[str, str | Exception], host_percent: float = 40.0) -> dict[str, object]:
    namespace = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        CPU_HELPERS,
        {"json": json, "os": os, "tempfile": tempfile, "psutil": _Psutil(host_percent)},
    )
    namespace["run_cmd"] = _fake_run_cmd(outputs)
    return namespace


def _docker_outputs(stats: str | Exception, ps_ids: list[str]) -> dict[str, str | Exception]:
    return {
        "version --format": "24.0.7\n",
        "stats --no-stream": stats,
        "ps --no-trunc": "\n".join(ps_ids) + "\n",
        "{{.Image}}": "sha256:image\n",
        "json .RepoDigests": '["daturaai/pod@sha256:digest"]\n',
        "{{.Name}}": "/pod_renter\n",
    }


def test_container_cpu_is_attached_and_the_host_reading_is_kept():
    # Arrange — one container, one stats row for it
    scrape = _scrape(_docker_outputs(f"{RENTER[:12]}|158.00%\n", [RENTER]), host_percent=40.0)

    # Act
    data = scrape["get_docker_info"](b"#!/bin/sh\n")

    # Assert
    assert data["docker_host_cpu_percent"] == 40.0
    assert data["docker_containers"][0]["each_cpu_percent"] == 158.0


def test_a_container_that_exits_between_stats_and_ps_voids_the_host_reading():
    # Arrange — stats saw two containers, ps lists only one. The missing container's CPU sits
    # inside the host reading with nothing to subtract it, so the whole reading has to go.
    stats = f"{RENTER[:12]}|158.00%\n{EXECUTOR[:12]}|12.00%\n"
    scrape = _scrape(_docker_outputs(stats, [RENTER]))

    # Act
    data = scrape["get_docker_info"](b"#!/bin/sh\n")

    # Assert
    assert "docker_host_cpu_percent" not in data
    assert data["docker_containers"][0]["each_cpu_percent"] == 158.0


def test_a_wedged_docker_daemon_voids_the_reading_and_keeps_the_scrape_alive():
    # Arrange — `timeout 30 docker stats` returns non-zero, so run_cmd raises
    outputs = _docker_outputs(RuntimeError("run_cmd error: timeout"), [RENTER])
    scrape = _scrape(outputs)

    # Act
    data = scrape["get_docker_info"](b"#!/bin/sh\n")

    # Assert — the scrape is fatal for the executor, so it must survive with the CPU part missing
    assert "docker_host_cpu_percent" not in data
    assert "each_cpu_percent" not in data["docker_containers"][0]
    assert data["docker_version"] == "24.0.7"


def test_an_unparsable_stats_row_raises_so_the_caller_voids_the_reading():
    # Arrange — a row the window cannot account for must not be dropped silently
    scrape = _scrape(_docker_outputs(f"{RENTER[:12]}|--\n", [RENTER]))

    # Act / Assert
    with pytest.raises(ValueError):
        scrape["get_container_cpu_percents"]("/tmp/docker")


def test_the_cpu_keys_are_registered_in_both_obfuscation_tables():
    # A key missing from ORIGINAL_KEYS arrives under its raw scrape name and the backend model
    # strips it; missing from generate_key_mappings it ships in the clear. DAH-2514 shipped
    # nulls to 347 executors this way.
    original_keys, mapped_keys = _obfuscation_tables()
    emitted = {"docker_host_cpu_percent", "each_cpu_percent"}

    assert emitted <= original_keys, f"missing from ORIGINAL_KEYS: {sorted(emitted - original_keys)}"
    assert emitted <= mapped_keys, f"missing from generate_key_mappings: {sorted(emitted - mapped_keys)}"


def test_a_container_whose_image_is_gone_is_listed_without_a_digest_and_the_scrape_survives():
    # Arrange — `docker inspect <image id>` answers "no such object" for an image removed under
    # a running container (seen on a provider host 14 Oct: SCRAPE_FAILED_ON_HOST every wave)
    outputs = _docker_outputs(f"{RENTER[:12]}|5.00%\n", [RENTER])
    outputs["json .RepoDigests"] = RuntimeError("run_cmd error: error: no such object: sha256:image")
    scrape = _scrape(outputs)

    # Act
    data = scrape["get_docker_info"](b"#!/bin/sh\n")

    # Assert — the container is still listed, with no digest to excuse its load
    assert data["docker_containers"][0]["each_digest"] == ""
    assert data["docker_containers"][0]["each_cpu_percent"] == 5.0
    assert data["docker_version"] == "24.0.7"


def test_container_traffic_is_attached_in_bytes():
    # Arrange — docker prints NetIO as "<received> / <sent>" since the container started, decimal units
    scrape = _scrape(_docker_outputs(f"{RENTER[:12]}|158.00%|1.23TB / 456MB\n", [RENTER]))

    # Act
    container = scrape["get_docker_info"](b"#!/bin/sh\n")["docker_containers"][0]

    # Assert
    assert container["each_net_rx_bytes"] == 1_230_000_000_000
    assert container["each_net_tx_bytes"] == 456_000_000
    assert container["each_cpu_percent"] == 158.0


def test_an_unparsable_traffic_reading_is_left_out_and_keeps_the_cpu_reading():
    scrape = _scrape(_docker_outputs(f"{RENTER[:12]}|5.00%|-- / --\n", [RENTER]), host_percent=40.0)

    data = scrape["get_docker_info"](b"#!/bin/sh\n")

    assert "each_net_rx_bytes" not in data["docker_containers"][0]
    assert data["docker_containers"][0]["each_cpu_percent"] == 5.0
    assert data["docker_host_cpu_percent"] == 40.0


def test_the_traffic_keys_are_registered_in_both_obfuscation_tables():
    original_keys, mapped_keys = _obfuscation_tables()
    emitted = {"each_net_rx_bytes", "each_net_tx_bytes"}

    assert emitted <= original_keys
    assert emitted <= mapped_keys


def test_the_traffic_and_cpu_readings_survive_the_production_obfuscation():
    # ecrypt_miner_job_files runs obfuscator.py and the key replacement over the whole script before it
    # ships; a nested helper came out of that with its body renamed and its signature not (lium-io#2146).
    import ast
    import contextlib
    import io
    import sys

    sys.path.insert(0, str(SRC))
    from miner_jobs.obfuscator import obfuscate_code
    from services.file_encrypt_service import FileEncryptService

    source_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    with contextlib.redirect_stdout(io.StringIO()):
        obfuscated = obfuscate_code(source_text)
        key_mapping, _ = FileEncryptService.generate_key_mappings(FileEncryptService.__new__(FileEncryptService))
    for original_key, random_name in key_mapping.items():
        obfuscated = obfuscated.replace(original_key, random_name)

    def top_level_name(node: ast.AST) -> str:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            return node.name
        if isinstance(node, ast.Assign):
            return getattr(node.targets[0], "id", "")
        return ""

    source_tree, obfuscated_tree = ast.parse(source_text), ast.parse(obfuscated)
    kept = {
        top_level_name(node): obfuscated_tree.body[index]
        for index, node in enumerate(source_tree.body)
        if top_level_name(node) in CPU_HELPERS
    }
    namespace = {"json": json, "os": os, "tempfile": tempfile, "psutil": _Psutil(40.0)}
    exec(ast.unparse(ast.Module(body=list(kept.values()), type_ignores=[])), namespace)  # noqa: S102
    run_cmd_name = next(
        obfuscated_tree.body[index].name
        for index, node in enumerate(source_tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "run_cmd"
    )
    namespace[run_cmd_name] = _fake_run_cmd({"stats --no-stream": f"{RENTER[:12]}|158.00%|1.23TB / 456MB\n"})

    host, cpu, net_io = namespace[kept["get_container_cpu_percents"].name]("/tmp/docker")

    assert (host, cpu) == (40.0, {RENTER[:12]: 158.0})
    assert net_io == {RENTER[:12]: (1_230_000_000_000, 456_000_000)}
