"""Every key of the machine scrape gets its own obfuscated name.

generate_random_name() draws "_" + 3 to 15 random letters with no memory of earlier draws. When two
keys draw the same name, _deobfuscate()'s reverse map keeps only the later key, so the GPU's `name`
can come back as another key and every executor in the cycle fails "GPU model not supported".
"""

import ast
import itertools
import re
import string
from pathlib import Path

from neurons.validators.tests.helpers import dict_literal_keys

SRC = Path(__file__).resolve().parents[1] / "src"


def test_a_repeated_random_name_is_drawn_again_so_every_key_survives_the_reverse_map(monkeypatch) -> None:
    from services.file_encrypt_service import ORIGINAL_KEYS, FileEncryptService
    from services.task.checks.machine_spec_scrape import _deobfuscate

    # Arrange — the first two draws collide, every later draw is fresh
    fresh = ("_Fresh" + "".join(pair) for pair in itertools.product(string.ascii_lowercase, repeat=2))
    draws = itertools.chain(["_Same", "_Same"], fresh)
    service = FileEncryptService.__new__(FileEncryptService)
    monkeypatch.setattr(service, "generate_random_name", lambda: next(draws))

    # Act
    key_mapping, _ = service.generate_key_mappings()
    scrape_output = {"entries": [{name: key} for key, name in key_mapping.items()]}
    deobfuscated = _deobfuscate(scrape_output, key_mapping)

    # Assert — each entry comes back under the key it was written with
    assert len(set(key_mapping.values())) == len(key_mapping)
    assert deobfuscated["entries"] == [{ORIGINAL_KEYS.get(key, key): key} for key in key_mapping]


def test_no_key_can_be_found_inside_a_generated_name() -> None:
    """ecrypt_miner_job_files renames by sequential str.replace: a later key that fits inside
    "_" + letters could match inside a name an earlier replace already wrote."""
    # Arrange
    service_module = ast.parse((SRC / "services" / "file_encrypt_service.py").read_text())
    all_keys = dict_literal_keys(service_module, "all_keys")

    # Act
    offenders = [key for key in all_keys if re.fullmatch(r"_?[A-Za-z]+", key)]

    # Assert
    assert all_keys
    assert offenders == []
