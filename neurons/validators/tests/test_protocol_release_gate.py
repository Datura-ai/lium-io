import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "protocol_release_gate.py"
_spec = importlib.util.spec_from_file_location("protocol_release_gate", _PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_a_newer_minor_than_the_prod_backend_waits():
    assert "release the backend first" in gate.refusal("2.20.0", "2.16.0\n")


def test_same_minor_or_older_releases():
    assert gate.refusal("2.16.0", "2.16.0\n") is None
    assert gate.refusal("2.16.3", "2.16.0") is None
    assert gate.refusal("2.15.0", "2.16.0") is None


def test_the_pin_and_protocol_files_parse():
    assert gate.main() in (0, 1)
