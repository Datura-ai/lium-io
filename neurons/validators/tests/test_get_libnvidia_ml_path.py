"""Tests for get_libnvidia_ml_path() — 32-bit vs 64-bit library selection.

machine_scrape.py executes module-level code on import (GPU scraping + sys.exit),
so we replicate the function logic directly for isolated testing.
"""

import pytest


# Replicate the fixed function logic for isolated testing.
# This avoids importing machine_scrape.py which triggers GPU scraping at module level.
def get_libnvidia_ml_path_impl(run_cmd_fn):
    """Exact copy of the fixed get_libnvidia_ml_path() logic."""
    try:
        original_path = run_cmd_fn("find /usr -name 'libnvidia-ml.so.1'").strip()
        paths = [p for p in original_path.split('\n') if p]
        for p in paths:
            if 'x86_64' in p or 'lib64' in p:
                return p
        return paths[0] if paths else ''
    except Exception:
        return ''


X86_64 = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
I386 = "/usr/lib/i386-linux-gnu/libnvidia-ml.so.1"
LIB64 = "/usr/lib64/libnvidia-ml.so.1"
LIB32 = "/usr/lib32/libnvidia-ml.so.1"


@pytest.mark.parametrize(
    "find_output,expected,reason",
    [
        # x86_64 first, i386 second — should pick x86_64
        (f"{X86_64}\n{I386}", X86_64, "prefers x86_64 over i386"),
        # i386 first, x86_64 second — order shouldn't matter
        (f"{I386}\n{X86_64}", X86_64, "picks x86_64 regardless of order"),
        # lib32 first, lib64 second — should pick lib64
        (f"{LIB32}\n{LIB64}", LIB64, "prefers lib64 over lib32"),
        # three paths — should still pick x86_64
        (f"{I386}\n{X86_64}\n{LIB32}", X86_64, "picks x86_64 among three paths"),
        # single x86_64 path
        (X86_64, X86_64, "returns single x86_64 path"),
        # single 32-bit path — no 64-bit available, fallback to first
        (I386, I386, "falls back to 32-bit when no 64-bit available"),
        # empty output — no libraries found
        ("", "", "returns empty when find returns nothing"),
    ],
)
def test_get_libnvidia_ml_path(find_output: str, expected: str, reason: str):
    """get_libnvidia_ml_path should prefer 64-bit library paths."""
    # Arrange
    run_cmd = lambda _: find_output

    # Act
    result = get_libnvidia_ml_path_impl(run_cmd)

    # Assert — function should select the correct path based on architecture
    assert result == expected, reason


def test_get_libnvidia_ml_path_returns_empty_on_exception():
    """When find command fails, should return empty string gracefully."""
    # Arrange
    def failing_cmd(_):
        raise Exception("command not found")

    # Act
    result = get_libnvidia_ml_path_impl(failing_cmd)

    # Assert — exceptions should be caught and return empty string
    assert result == ""
