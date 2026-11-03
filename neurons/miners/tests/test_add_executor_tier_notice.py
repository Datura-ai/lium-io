"""add-executor tells the provider the new node starts on Spot."""

from click.testing import CliRunner


def _fake_cli_service(monkeypatch, ok):
    import cli as cli_module

    class FakeCliService:
        def __init__(self, private_key=None, with_executor_db=False, version=None):
            pass

        async def add_executor(self, address, port, price, validator):
            return ok

    monkeypatch.setattr(cli_module, "CliService", FakeCliService)


def test_add_executor_prints_spot_tier_notice(monkeypatch):
    from cli import cli

    _fake_cli_service(monkeypatch, True)
    result = CliRunner().invoke(cli, ["add-executor", "--address", "10.0.0.1", "--port", "8001", "--price", "1.5"])
    assert result.exit_code == 0, result.output
    assert "starts on the Spot tier" in result.output
    assert "Change Tier" in result.output


def test_failed_add_prints_no_tier_notice(monkeypatch):
    from cli import cli

    _fake_cli_service(monkeypatch, False)
    result = CliRunner().invoke(cli, ["add-executor", "--address", "10.0.0.1", "--port", "8001", "--price", "1.5"])
    assert "Spot tier" not in result.output
