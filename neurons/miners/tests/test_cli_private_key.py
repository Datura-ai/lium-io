"""A CLI command that takes an Ethereum private key refuses a malformed one without echoing it."""

import pytest
from click.testing import CliRunner

MALFORMED_KEYS = ["0x" + "ab" * 8, "not-a-hex-private-key", "zq" * 32]


@pytest.mark.parametrize("bad_key", MALFORMED_KEYS)
def test_associate_eth_with_a_malformed_key_exits_1_without_the_key(monkeypatch, caplog, bad_key):
    import cli as cli_module

    built = []
    monkeypatch.setattr(cli_module, "CliService", lambda **kwargs: built.append(kwargs))

    result = CliRunner().invoke(cli_module.cli, ["associate-eth", "--private-key", bad_key])

    assert result.exit_code == 1, result.output
    assert built == []
    assert "private key is malformed" in caplog.text
    assert bad_key not in caplog.text + result.output
