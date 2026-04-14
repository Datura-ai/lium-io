import pytest

from neurons.validators.src.services.verifyx_validation_service import (
    MIN_CIPHER_LEN,
    STDERR_TAIL_BYTES,
    VerifyXFailureClass,
    _classify_failure,
    _tail_stderr,
)


def test_classify_ssh_transport_when_transport_error_set():
    result = _classify_failure(
        exit_status=None,
        stdout=None,
        stderr=None,
        transport_error="ConnectionLost: timeout",
    )
    assert result == VerifyXFailureClass.SSH_TRANSPORT


def test_classify_executor_crash_on_nonzero_exit_and_empty_stdout():
    result = _classify_failure(
        exit_status=1,
        stdout="",
        stderr="Traceback (most recent call last): ...",
        transport_error=None,
    )
    assert result == VerifyXFailureClass.EXECUTOR_CRASH


def test_classify_empty_response_on_zero_exit_with_short_stdout():
    result = _classify_failure(
        exit_status=0,
        stdout="short",
        stderr=None,
        transport_error=None,
    )
    assert result == VerifyXFailureClass.EMPTY_RESPONSE


def test_classify_cipher_rejected_on_valid_length_stdout():
    long_stdout = "a" * (MIN_CIPHER_LEN + 10)
    result = _classify_failure(
        exit_status=0,
        stdout=long_stdout,
        stderr=None,
        transport_error=None,
    )
    assert result == VerifyXFailureClass.CIPHER_REJECTED


def test_classify_unknown_fallback():
    # No transport error, no exit_status, no stdout — edge case
    result = _classify_failure(
        exit_status=None,
        stdout=None,
        stderr=None,
        transport_error=None,
    )
    assert result == VerifyXFailureClass.UNKNOWN


def test_classify_executor_crash_takes_priority_over_empty_response():
    # Non-zero exit with empty stdout -> EXECUTOR_CRASH, not EMPTY_RESPONSE
    result = _classify_failure(
        exit_status=137,
        stdout="",
        stderr="Killed",
        transport_error=None,
    )
    assert result == VerifyXFailureClass.EXECUTOR_CRASH


def test_classify_stderr_noise_does_not_beat_valid_stdout():
    # A warning on stderr with a valid-looking cipher on stdout: CIPHER_REJECTED
    long_stdout = "b" * (MIN_CIPHER_LEN + 1)
    result = _classify_failure(
        exit_status=0,
        stdout=long_stdout,
        stderr="warning: something",
        transport_error=None,
    )
    assert result == VerifyXFailureClass.CIPHER_REJECTED


def test_tail_stderr_truncates_to_last_2kb():
    long_stderr = "x" * (STDERR_TAIL_BYTES * 2)
    result = _tail_stderr(long_stderr)
    assert result is not None
    assert len(result.encode("utf-8")) == STDERR_TAIL_BYTES


def test_tail_stderr_returns_none_for_none():
    assert _tail_stderr(None) is None


def test_tail_stderr_returns_as_is_when_under_limit():
    short = "only a few bytes"
    assert _tail_stderr(short) == short
