"""DAH-2922 — get_gpu_interconnect() and cdn_bandwidth_probe() helpers in machine_scrape.py.

Table shapes follow `nvidia-smi topo -m`, `nvidia-smi topo -p2p r` and `nvidia-smi nvlink -s` as
printed by driver 5xx on an HGX H200 host and on the PCIe-only 8x H200 host that motivated the
change (all `SYS`, P2P `NS` everywhere, NCCL unusable without NCCL_P2P_DISABLE=1).

machine_scrape.py is a script, not a module — importing it runs the whole scrape — so the helpers
are extracted by ast and executed in their own namespace (same pattern as test_scrape_infiniband.py).
"""

import ast
import re
from pathlib import Path
from typing import Any

import pytest
from neurons.validators.tests.helpers import build_scrape_namespace, dict_literal_keys

SRC = Path(__file__).resolve().parents[1] / "src"

INTERCONNECT_HELPERS = {
    "NVIDIA_SMI_TOPO_MATRIX_CMD",
    "NVIDIA_SMI_TOPO_P2P_READ_CMD",
    "NVIDIA_SMI_NVLINK_STATUS_CMD",
    "PCIE_LINK_CLASSES_WORST_FIRST",
    "TOPO_GPU_LABEL_PATTERN",
    "TOPO_NVLINK_CELL_PATTERN",
    "NVLINK_ACTIVE_LINK_PATTERN",
    "NVLINK_GPU_HEADER_PATTERN",
    "parse_topology_matrix",
    "count_active_nvlinks_per_gpu",
    "GpuInterconnectObservation",
    "summarize_gpu_interconnect",
    "get_gpu_interconnect",
}

CDN_HELPERS = {"aggregate_curl_throughput_mbps"}

TOPO_LEGEND = """
Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
"""


def topo_table(cells: list[list[str]], *, nic_columns: bool = True, separator: str = "\t") -> str:
    """An `nvidia-smi topo -m` table for the given GPUxGPU cells, with the extra columns nvidia-smi adds."""
    count = len(cells)
    header = [""] + [f"GPU{i}" for i in range(count)]
    if nic_columns:
        header += ["NIC0", "NIC1"]
    header += ["CPU Affinity", "NUMA Affinity", "GPU NUMA ID"]
    lines = [separator.join(header)]
    for i, row in enumerate(cells):
        line = [f"GPU{i}"] + [" X " if i == j else cell for j, cell in enumerate(row)]
        if nic_columns:
            line += ["PIX" if i < count / 2 else "SYS", "SYS" if i < count / 2 else "PIX"]
        # nvidia-smi leaves the NUMA Affinity column empty on many hosts: two tabs in a row
        line += ["0-47,96-143" if i < count / 2 else "48-95,144-191", "0" if i < count / 2 else "1", "", "N/A"]
        lines.append(separator.join(line))
    if nic_columns:
        lines.append(separator.join(["NIC0"] + ["PIX"] * (count // 2) + ["SYS"] * (count - count // 2) + [" X ", "SYS"]))
        lines.append(separator.join(["NIC1"] + ["SYS"] * (count // 2) + ["PIX"] * (count - count // 2) + ["SYS", " X "]))
    return "\n".join(lines) + "\n" + TOPO_LEGEND


def p2p_table(cells: list[list[str]]) -> str:
    """An `nvidia-smi topo -p2p r` table (leading space, tab separated, own legend)."""
    count = len(cells)
    lines = [" \t" + "\t".join(f"GPU{i}" for i in range(count))]
    for i, row in enumerate(cells):
        lines.append(f" GPU{i}\t" + "\t".join("X" if i == j else cell for j, cell in enumerate(row)))
    lines += ["", "Legend:", "", "  X    = Self", "  OK   = Status Ok", "  NS   = Not supported"]
    return "\n".join(lines) + "\n"


def uniform(count: int, cell: str) -> list[list[str]]:
    return [[cell] * count for _ in range(count)]


def nvlink_status(active_per_gpu: list[int], total_links: int = 18) -> str:
    lines = []
    for gpu, active in enumerate(active_per_gpu):
        lines.append(f"GPU {gpu}: NVIDIA H200 (UUID: GPU-0000000{gpu}-0000-0000-0000-000000000000)")
        for link in range(total_links):
            lines.append(f"\t Link {link}: 26.562 GB/s" if link < active else f"\t Link {link}: <inactive>")
    return "\n".join(lines) + "\n"


HGX_H200 = uniform(8, "NV18")

# The PCIe-only host: GPUs 0-3 hang off one CPU, 4-7 off the other; nothing is bonded.
PCIE_ONLY_H200 = [["PHB" if (i < 4) == (j < 4) else "SYS" for j in range(8)] for i in range(8)]

# 2+2 A100 PCIe cards with two NVLink bridges: NV4 inside a pair, SYS across pairs.
BRIDGED_A100 = [["NV4" if i // 2 == j // 2 else "SYS" for j in range(4)] for i in range(4)]

PCIE_CARD_NVLINK_STATUS = (
    "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-00000000-0000-0000-0000-000000000000)\n"
    "\t NVML: Unable to retrieve NVLink information: Not Supported\n"
)


@pytest.fixture
def scrape() -> dict[str, Any]:
    return build_scrape_namespace(SRC / "miner_jobs" / "machine_scrape.py", INTERCONNECT_HELPERS | CDN_HELPERS, {"re": re})


# -- parse_topology_matrix -------------------------------------------------------------------------


def test_topology_matrix_drops_nic_and_affinity_columns(scrape: dict[str, Any]) -> None:
    # Arrange
    output = topo_table(HGX_H200)

    # Act
    labels, rows = scrape["parse_topology_matrix"](output)

    # Assert — square, GPU-only, self cell normalised
    assert labels == [f"GPU{i}" for i in range(8)]
    assert all(len(row) == 8 for row in rows)
    assert rows[0][0] == "X"
    assert rows[0][1] == "NV18"
    assert rows[7][0] == "NV18"


def test_topology_matrix_reads_a_space_aligned_table(scrape: dict[str, Any]) -> None:
    # Arrange — some drivers pad with spaces instead of tabs
    output = topo_table(BRIDGED_A100, nic_columns=False, separator="    ")

    # Act
    labels, rows = scrape["parse_topology_matrix"](output)

    # Assert
    assert labels == ["GPU0", "GPU1", "GPU2", "GPU3"]
    assert rows[0] == ["X", "NV4", "SYS", "SYS"]


def test_topology_matrix_without_a_gpu_header_is_empty(scrape: dict[str, Any]) -> None:
    # Act
    labels, rows = scrape["parse_topology_matrix"]("No devices were found\n")

    # Assert
    assert (labels, rows) == ([], [])


def test_p2p_table_is_read_by_the_same_parser(scrape: dict[str, Any]) -> None:
    # Act
    labels, rows = scrape["parse_topology_matrix"](p2p_table(uniform(8, "NS")))

    # Assert
    assert len(labels) == 8
    assert rows[0][0] == "X"
    assert rows[0][1] == "NS"


# -- summarize_gpu_interconnect ----------------------------------------------------------------------


def test_hgx_host_is_nvlink_all_to_all_with_p2p(scrape: dict[str, Any]) -> None:
    # Act
    payload = scrape["summarize_gpu_interconnect"](topo_table(HGX_H200), p2p_table(uniform(8, "OK")), nvlink_status([18] * 8))

    # Assert
    assert payload["ic_devices"] == 8
    assert payload["ic_gpu_pairs"] == 28
    assert payload["ic_nvlink"] is True
    assert payload["ic_nvlink_pairs"] == 28
    assert payload["ic_nvlink_links"] == 18
    assert payload["ic_nvlink_active_links"] == 18
    assert payload["ic_pcie_class"] is None
    assert payload["ic_p2p"] is True
    assert payload["ic_p2p_pairs"] == 28
    assert payload["ic_p2p_ok_pairs"] == 28
    assert payload["ic_matrix"][0][1] == "NV18"
    assert len(payload["ic_matrix"]) == 8


def test_pcie_only_host_reports_the_worst_class_and_no_p2p(scrape: dict[str, Any]) -> None:
    # Arrange — the 8x H200 that only ran NCCL with NCCL_P2P_DISABLE=1
    topo = topo_table(PCIE_ONLY_H200)
    p2p = p2p_table(uniform(8, "NS"))

    # Act
    payload = scrape["summarize_gpu_interconnect"](topo, p2p, PCIE_CARD_NVLINK_STATUS)

    # Assert — PHB inside a socket, SYS across; the worst one is what a TP job pays for
    assert payload["ic_nvlink"] is False
    assert payload["ic_nvlink_pairs"] == 0
    assert payload["ic_nvlink_links"] is None
    assert payload["ic_pcie_class"] == "SYS"
    assert payload["ic_p2p"] is False
    assert payload["ic_p2p_ok_pairs"] == 0
    assert payload["ic_nvlink_active_links"] == 0


def test_bridged_pairs_are_not_nvlink_for_the_whole_host(scrape: dict[str, Any]) -> None:
    # Arrange — two NVLink bridges on four PCIe cards: fine for TP=2, not for TP=4
    mixed_p2p = [["OK" if i // 2 == j // 2 else "NS" for j in range(4)] for i in range(4)]

    # Act
    payload = scrape["summarize_gpu_interconnect"](topo_table(BRIDGED_A100), p2p_table(mixed_p2p), nvlink_status([4] * 4, 12))

    # Assert
    assert payload["ic_nvlink"] is False
    assert payload["ic_nvlink_pairs"] == 2
    assert payload["ic_gpu_pairs"] == 6
    assert payload["ic_nvlink_links"] == 4
    assert payload["ic_pcie_class"] == "SYS"
    assert payload["ic_p2p"] is False
    assert payload["ic_p2p_ok_pairs"] == 2


def test_single_gpu_has_no_interconnect_to_judge(scrape: dict[str, Any]) -> None:
    # Act
    payload = scrape["summarize_gpu_interconnect"](topo_table([["X"]], nic_columns=False), p2p_table([["X"]]), None)

    # Assert — None, not False: there is no pair a renter could be misled about
    assert payload["ic_devices"] == 1
    assert payload["ic_gpu_pairs"] == 0
    assert payload["ic_nvlink"] is None
    assert payload["ic_p2p"] is None
    assert payload["ic_nvlink_active_links"] is None


def test_missing_optional_tables_leave_their_fields_none(scrape: dict[str, Any]) -> None:
    # Act
    payload = scrape["summarize_gpu_interconnect"](topo_table(HGX_H200), None, None)

    # Assert — the topo table alone still answers the NVLink question
    assert payload["ic_nvlink"] is True
    assert payload["ic_p2p"] is None
    assert payload["ic_p2p_pairs"] is None
    assert payload["ic_nvlink_active_links"] is None


def test_the_gpu_with_the_fewest_active_links_sets_the_figure(scrape: dict[str, Any]) -> None:
    # Act
    counts = scrape["count_active_nvlinks_per_gpu"](nvlink_status([18, 18, 12, 18]))
    payload = scrape["summarize_gpu_interconnect"](topo_table(uniform(4, "NV18"), nic_columns=False), None, nvlink_status([18, 18, 12, 18]))

    # Assert — a GPU with links down is the one the collective waits for
    assert counts == [18, 18, 12, 18]
    assert payload["ic_nvlink_active_links"] == 12
    assert payload["ic_nvlink_links"] == 18


# -- get_gpu_interconnect ----------------------------------------------------------------------------


def _stub_run_cmd(outputs: dict[str, Any]):
    def run_cmd(cmd: str) -> str:
        result = outputs[cmd]
        if isinstance(result, Exception):
            raise result
        return result

    return run_cmd


def test_probe_reports_all_three_tables(scrape: dict[str, Any]) -> None:
    # Arrange
    scrape["run_cmd"] = _stub_run_cmd(
        {
            "nvidia-smi topo -m": topo_table(HGX_H200),
            "nvidia-smi topo -p2p r": p2p_table(uniform(8, "OK")),
            "nvidia-smi nvlink -s": nvlink_status([18] * 8),
        }
    )

    # Act
    observation = scrape["get_gpu_interconnect"]()

    # Assert
    assert observation.scrape_error == ""
    assert observation.payload["ic_nvlink"] is True
    assert observation.payload["ic_p2p"] is True


def test_probe_without_topo_table_reports_why(scrape: dict[str, Any]) -> None:
    # Arrange — nvidia-smi missing or the driver wedged: nothing to summarize
    scrape["run_cmd"] = _stub_run_cmd(
        {
            "nvidia-smi topo -m": RuntimeError("run_cmd error cmd='nvidia-smi topo -m' proc.returncode=127"),
            "nvidia-smi topo -p2p r": p2p_table(uniform(8, "OK")),
            "nvidia-smi nvlink -s": nvlink_status([18] * 8),
        }
    )

    # Act
    observation = scrape["get_gpu_interconnect"]()

    # Assert
    assert observation.payload is None
    assert observation.scrape_error.startswith("nvidia-smi topo -m: RuntimeError")


def test_probe_keeps_the_topo_answer_when_only_optional_tables_fail(scrape: dict[str, Any]) -> None:
    # Arrange — PCIe cards exit non-zero on `nvlink -s`; older drivers on `-p2p r`
    scrape["run_cmd"] = _stub_run_cmd(
        {
            "nvidia-smi topo -m": topo_table(PCIE_ONLY_H200),
            "nvidia-smi topo -p2p r": RuntimeError("run_cmd error proc.returncode=2"),
            "nvidia-smi nvlink -s": RuntimeError("run_cmd error proc.returncode=6"),
        }
    )

    # Act
    observation = scrape["get_gpu_interconnect"]()

    # Assert — the NVLink verdict stands; what could not be read says so
    assert observation.payload["ic_nvlink"] is False
    assert observation.payload["ic_pcie_class"] == "SYS"
    assert observation.payload["ic_p2p"] is None
    assert observation.payload["ic_nvlink_active_links"] is None
    assert "nvidia-smi topo -p2p r" in observation.scrape_error
    assert "nvidia-smi nvlink -s" in observation.scrape_error


# -- aggregate_curl_throughput_mbps ------------------------------------------------------------------


def test_parallel_streams_are_summed_over_the_slowest_stream(scrape: dict[str, Any]) -> None:
    # Arrange — four 50 MB streams, the slowest took 2 s: 200 MB / 2 s = 800 Mbps
    output = "50000000 0 1.900\n50000000 0 2.000\n50000000 0 1.950\n50000000 0 1.800\n"

    # Act
    mbps = scrape["aggregate_curl_throughput_mbps"](output, False)

    # Assert
    assert mbps == 800.0


def test_a_timed_out_stream_still_yields_the_bytes_it_moved(scrape: dict[str, Any]) -> None:
    # Arrange — a 100 Mbps link cannot finish 4x50 MB in 15 s; curl still prints -w on exit 28
    output = "\n".join(["46875000 0 15.001"] * 4) + "\n"

    # Act
    mbps = scrape["aggregate_curl_throughput_mbps"](output, False)

    # Assert
    assert mbps == pytest.approx(100.0, rel=0.01)


def test_upload_uses_the_upload_size_column(scrape: dict[str, Any]) -> None:
    # Arrange — two 25 MB uploads in 1 s: 400 Mbps
    output = "0 26214400 1.0\n0 26214400 0.9\n"

    # Act
    mbps = scrape["aggregate_curl_throughput_mbps"](output, True)

    # Assert
    assert mbps == pytest.approx(419.43, rel=0.001)


def test_no_samples_raise_instead_of_reporting_zero(scrape: dict[str, Any]) -> None:
    # Arrange — curl absent or every stream failed before printing
    with pytest.raises(RuntimeError):
        # Act
        scrape["aggregate_curl_throughput_mbps"]("sh: curl: not found\n", False)


# -- validator-side event summary --------------------------------------------------------------------


def test_scrape_ok_event_carries_the_verdict_without_the_matrix() -> None:
    from services.task.checks.machine_spec_scrape import _interconnect_summary

    # Arrange — specs as they look after de-obfuscation
    specs = {
        "interconnect": {"gpu_count": 8, "nvlink": True, "nvlink_links": 18, "p2p": True, "matrix": [["X"] * 8] * 8},
        "interconnect_scrape_error": "nvidia-smi nvlink -s: RuntimeError('exit 6')",
    }

    # Act
    summary = _interconnect_summary(specs)

    # Assert
    assert summary == {
        "gpu_count": 8,
        "nvlink": True,
        "nvlink_links": 18,
        "p2p": True,
        "scrape_error": "nvidia-smi nvlink -s: RuntimeError('exit 6')",
    }


def test_scrape_ok_event_says_why_when_there_is_no_interconnect() -> None:
    from services.task.checks.machine_spec_scrape import _interconnect_summary

    # Act / Assert — a scrape predating the field says nothing; a failed one says why
    assert _interconnect_summary({}) is None
    assert _interconnect_summary({"interconnect": None, "interconnect_scrape_error": "nvidia-smi topo -m: exit 127"}) == {
        "scrape_error": "nvidia-smi topo -m: exit 127"
    }


# -- obfuscation wiring ------------------------------------------------------------------------------

NEW_KEYS = [
    "data_interconnect",
    "data_interconnect_scrape_error",
    "ic_devices",
    "ic_gpu_pairs",
    "ic_nvlink",
    "ic_nvlink_links",
    "ic_nvlink_pairs",
    "ic_nvlink_active_links",
    "ic_pcie_class",
    "ic_p2p",
    "ic_p2p_pairs",
    "ic_p2p_ok_pairs",
    "ic_matrix",
    "ncdn_down",
    "ncdn_up",
    "ncdn_streams",
    "ncdn_error",
]


def test_interconnect_and_cdn_keys_are_wired_through_both_obfuscation_tables() -> None:
    """A key missing from either table ships un-renamed/un-mapped."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    scrape_text = (SRC / "miner_jobs" / "machine_scrape.py").read_text()

    # Act
    original_keys = dict_literal_keys(service_module, "ORIGINAL_KEYS")
    all_keys = dict_literal_keys(service_module, "all_keys")

    # Assert
    for key in NEW_KEYS:
        assert f'"{key}"' in scrape_text or f"'{key}'" in scrape_text
        assert key in original_keys
        assert key in all_keys


def test_no_key_is_a_substring_of_a_later_key_in_the_rename_table() -> None:
    """ecrypt_miner_job_files renames by sequential str.replace over the whole source: a key that
    appears inside a LATER key corrupts that later key before its own turn comes."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    all_keys = dict_literal_keys(service_module, "all_keys")

    # Act
    offenders = [
        (earlier, later)
        for index, earlier in enumerate(all_keys)
        for later in all_keys[index + 1 :]
        if earlier != later and earlier in later
    ]

    # Assert
    assert offenders == []


def test_the_obfuscated_scrape_still_summarizes_the_topology() -> None:
    """The packaged scrape (obfuscated, keys renamed) must classify an HGX host the way the source does."""
    import contextlib
    import io

    from miner_jobs.obfuscator import obfuscate_code
    from services.file_encrypt_service import FileEncryptService

    # Arrange — same transformation ecrypt_miner_job_files applies, then only the helpers we need
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
    kept = [
        obfuscated_tree.body[index]
        for index, node in enumerate(source_tree.body)
        if top_level_name(node) in INTERCONNECT_HELPERS
    ]
    summarize_name = next(
        obfuscated_tree.body[index].name
        for index, node in enumerate(source_tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "summarize_gpu_interconnect"
    )
    namespace: dict[str, Any] = {"re": re}
    exec(ast.unparse(ast.Module(body=kept, type_ignores=[])), namespace)  # noqa: S102

    # Act
    payload = namespace[summarize_name](topo_table(HGX_H200), p2p_table(uniform(8, "OK")), nvlink_status([18] * 8))

    # Assert — keys are random names by now, so the values carry the check: 11 fields, the NVLink
    # verdict True, 18 links, 28 pairs, the matrix, and no key left in clear text
    assert len(payload) == len(NEW_KEYS) - 6  # the 11 ic_* fields
    values = list(payload.values())
    assert values.count(True) == 2  # nvlink and p2p
    assert 18 in values and 28 in values
    assert not any(key.startswith("ic_") for key in payload)
