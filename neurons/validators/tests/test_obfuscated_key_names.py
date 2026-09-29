"""A repeated random name must not make _deobfuscate() drop a key of the machine scrape."""

import itertools
import string

import pytest


def test_a_repeated_random_name_is_drawn_again_so_every_key_survives_the_reverse_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.file_encrypt_service import ORIGINAL_KEYS, FileEncryptService
    from services.task.checks.machine_spec_scrape import _deobfuscate

    # Arrange — the first two draws collide, every later draw is fresh
    fresh = (
        "_Fresh" + "".join(pair) for pair in itertools.product(string.ascii_lowercase, repeat=2)
    )
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
