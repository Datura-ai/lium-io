"""The validator trust anchor is resolved once, visibly, and a broken override
module fails the boot instead of silently leaving the built-in (prod) anchor in
place."""

import importlib.machinery
import sys
import types

import pytest

import core.config as config


def _install_override(monkeypatch: pytest.MonkeyPatch, **attrs: str) -> None:
    module = types.ModuleType("core.config_override")
    module.__spec__ = importlib.machinery.ModuleSpec("core.config_override", None)
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "core.config_override", module)


def test_builtin_anchor_when_no_override_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "core.config_override", raising=False)
    monkeypatch.setattr(config.importlib.util, "find_spec", lambda name: None)

    assert config._resolve_validator_hotkey() == config._BUILTIN_VALIDATOR_HOTKEY_SS58


def test_override_module_replaces_the_anchor(monkeypatch):
    override = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    _install_override(monkeypatch, _VALIDATOR_HOTKEY_SS58=override)

    assert config._resolve_validator_hotkey() == override


def test_override_with_invalid_address_fails_loudly(monkeypatch):
    _install_override(monkeypatch, _VALIDATOR_HOTKEY_SS58="5F7X5UpKSr26KU3jKfpLmT8kuKtBNyHhEnfS8xtxPCqCb13X")

    with pytest.raises(RuntimeError, match="not an ss58 address"):
        config._resolve_validator_hotkey()


def test_override_missing_the_name_is_not_swallowed(monkeypatch):
    _install_override(monkeypatch)  # module exists, name absent (typo at build time)

    with pytest.raises(ImportError):
        config._resolve_validator_hotkey()
