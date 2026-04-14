import pytest

from neurons.validators.src.services.verifyx_validation_service import (
    MIN_CIPHER_LEN,
    STDERR_TAIL_BYTES,
    VerifyXFailureClass,
    _classify_failure,
    _tail_stderr,
)


LONG_STDOUT = "a" * (MIN_CIPHER_LEN + 10)


@pytest.mark.parametrize(
    "exit_status,stdout,transport_error,expected",
    [
        # transport error wins over everything else
        (None, None, "ConnectionLost: timeout", VerifyXFailureClass.SSH_TRANSPORT),
        # non-zero exit wins over stdout shape (even valid-length stdout => CRASH, not CIPHER_REJECTED)
        (1, "", None, VerifyXFailureClass.EXECUTOR_CRASH),
        (137, "", None, VerifyXFailureClass.EXECUTOR_CRASH),
        (1, LONG_STDOUT, None, VerifyXFailureClass.EXECUTOR_CRASH),
        # zero exit + short stdout => EMPTY_RESPONSE
        (0, "short", None, VerifyXFailureClass.EMPTY_RESPONSE),
        # zero exit + valid-length stdout => CIPHER_REJECTED
        (0, LONG_STDOUT, None, VerifyXFailureClass.CIPHER_REJECTED),
        # no signals at all => UNKNOWN fallback
        (None, None, None, VerifyXFailureClass.UNKNOWN),
    ],
)
def test_classify_failure(exit_status, stdout, transport_error, expected):
    # Arrange / Act
    result = _classify_failure(
        exit_status=exit_status,
        stdout=stdout,
        stderr=None,
        transport_error=transport_error,
    )

    # Assert — classification follows the priority: transport > exit_status > stdout shape
    assert result == expected


def test_tail_stderr_returns_none_for_none():
    # None input is passed through unchanged — no stderr available means nothing to tail.
    assert _tail_stderr(None) is None


def test_tail_stderr_returns_as_is_when_under_limit():
    # Arrange
    short = "only a few bytes"

    # Act / Assert — short payloads are not truncated
    assert _tail_stderr(short) == short


def test_tail_stderr_truncates_long_payload_and_keeps_valid_utf8_end():
    # Arrange — payload longer than STDERR_TAIL_BYTES, ending with a multi-byte char
    # split by the cut. Ω is 2 bytes in UTF-8 so the slice may land mid-character.
    prefix = "x" * (STDERR_TAIL_BYTES + 10)
    payload = prefix + "Ωabc"

    # Act
    result = _tail_stderr(payload)

    # Assert — tail is capped at STDERR_TAIL_BYTES and remains valid UTF-8 (no leading U+FFFD)
    assert result is not None
    assert len(result.encode("utf-8")) <= STDERR_TAIL_BYTES
    assert result[0] != "\ufffd"
    assert result.endswith("Ωabc")
