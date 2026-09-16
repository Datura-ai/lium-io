from types import SimpleNamespace

from services.miner_service import _ssh_key_not_accepted_text

_EXTRA = {"executor_id": "e-1", "miner_hotkey": "m-1"}


def test_an_empty_executor_list_says_the_key_was_not_accepted():
    message = _ssh_key_not_accepted_text([], _EXTRA)

    assert str(message) == "Error: no executor accepted the SSH key"
    assert message.extra["executors_returned"] == 0


def test_a_different_executor_says_the_id_does_not_match():
    returned = SimpleNamespace(uuid="e-2")

    message = _ssh_key_not_accepted_text([returned], _EXTRA)

    assert str(message) == "Error: the miner returned a different executor id"
    assert message.extra["executors_returned"] == 1
    assert message.extra["returned_executor_id"] == "e-2"


def test_a_missing_executor_list_is_read_as_no_executor():
    message = _ssh_key_not_accepted_text(None, _EXTRA)

    assert str(message) == "Error: no executor accepted the SSH key"
