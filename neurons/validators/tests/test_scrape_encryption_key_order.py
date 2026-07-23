"""machine_scrape encrypts its payload with a key derived from the LITERAL key order of
gpu_details[0]:

    encryption_key = "".join(machine_specs["data_gpu"]["gpu_details"][0].keys())

The validator rebuilds that key from KEYS_FOR_ENCRYPTION_KEY_GENERATION. If the two ever drift —
a key added to the scrape dict but not to the list, or added in a different position — every
decrypt fails and the whole fleet silently stops reporting specs. Parsed from source rather than
imported: file_encrypt_service pulls in PyInstaller, and machine_scrape needs NVML.
"""

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _gpu_keys_emitted_by_the_scrape() -> list[str]:
    scrape = (SRC / "miner_jobs" / "machine_scrape.py").read_text()
    # The gpu_details entry is the only dict literal built from "gpu.*" keys.
    start = scrape.index('"gpu.name"')
    end = scrape.index('"gpu.memory_utilization"', start)
    return re.findall(r'"(gpu\.[a-z_]+)"', scrape[start:end + len('"gpu.memory_utilization"')])


def _keys_used_for_the_encryption_key() -> list[str]:
    service = (SRC / "services" / "file_encrypt_service.py").read_text()
    for node in ast.walk(ast.parse(service)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "KEYS_FOR_ENCRYPTION_KEY_GENERATION":
            return ast.literal_eval(node.value)
    raise AssertionError("KEYS_FOR_ENCRYPTION_KEY_GENERATION not found")


def test_encryption_key_order_mirrors_the_scrape_dict():
    assert _keys_used_for_the_encryption_key() == _gpu_keys_emitted_by_the_scrape()
