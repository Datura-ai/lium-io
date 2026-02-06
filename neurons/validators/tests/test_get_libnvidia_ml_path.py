"""Tests for get_libnvidia_ml_path() — 32-bit vs 64-bit library selection.

machine_scrape.py executes module-level code on import (GPU scraping + sys.exit),
so we extract and test the function logic directly without importing the module.
"""

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


def test_prefers_x86_64_when_both_present():
    """When find returns both 32-bit and 64-bit paths, should pick 64-bit."""
    find_output = (
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1\n"
        "/usr/lib/i386-linux-gnu/libnvidia-ml.so.1"
    )
    assert get_libnvidia_ml_path_impl(lambda _: find_output) == "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"


def test_prefers_x86_64_reversed_order():
    """Should pick x86_64 regardless of find output order."""
    find_output = (
        "/usr/lib/i386-linux-gnu/libnvidia-ml.so.1\n"
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    )
    assert get_libnvidia_ml_path_impl(lambda _: find_output) == "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"


def test_prefers_lib64_path():
    """Should pick lib64 path over lib32."""
    find_output = (
        "/usr/lib32/libnvidia-ml.so.1\n"
        "/usr/lib64/libnvidia-ml.so.1"
    )
    assert get_libnvidia_ml_path_impl(lambda _: find_output) == "/usr/lib64/libnvidia-ml.so.1"


def test_single_path_returned_as_is():
    """When only one path exists, should return it."""
    find_output = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    assert get_libnvidia_ml_path_impl(lambda _: find_output) == "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"


def test_single_32bit_path_returned_as_fallback():
    """When only 32-bit path exists, should return it as fallback."""
    find_output = "/usr/lib/i386-linux-gnu/libnvidia-ml.so.1"
    assert get_libnvidia_ml_path_impl(lambda _: find_output) == "/usr/lib/i386-linux-gnu/libnvidia-ml.so.1"


def test_empty_find_output_returns_empty():
    """When find returns nothing, should return empty string."""
    assert get_libnvidia_ml_path_impl(lambda _: "") == ""


def test_exception_returns_empty():
    """When find command fails, should return empty string."""
    def raise_error(_):
        raise Exception("command not found")
    assert get_libnvidia_ml_path_impl(raise_error) == ""


def test_three_paths_picks_x86_64():
    """With host-mounted libs showing 3 paths, should pick x86_64."""
    find_output = (
        "/usr/lib/i386-linux-gnu/libnvidia-ml.so.1\n"
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1\n"
        "/usr/lib32/libnvidia-ml.so.1"
    )
    assert get_libnvidia_ml_path_impl(lambda _: find_output) == "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
