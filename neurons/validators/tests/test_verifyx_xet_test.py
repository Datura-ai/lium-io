"""Branch coverage for _verify_xet_test in verifyx_validation_service."""

from __future__ import annotations

import pytest

from neurons.validators.src.services import verifyx_validation_service as vvs

MIN_SPEED = 100.0


@pytest.fixture(autouse=True)
def min_speed(monkeypatch):
    monkeypatch.setattr(
        "neurons.validators.src.services.verifyx_validation_service.settings.verifyx.NETWORK_MIN_DOWNLOAD_SPEED_MBPS",
        MIN_SPEED,
    )


def _challenge(pkg="org/model", size=1000, hash_="abc"):
    return {
        "xet_challenge": {
            "download": {
                "pkg": pkg,
                "url": "https://huggingface.co/org/model/resolve/main/file.safetensors",
                "size": size,
                "hash": hash_,
            },
            "timeout_seconds": 120,
        }
    }


def _execution(pkg="org/model", size=1000, hash_="abc", speed=200.0):
    return {
        "xet_execution": {
            "pkg": pkg,
            "hash": hash_,
            "status": "ok",
            "success": True,
            "bytes_downloaded": size,
            "speed_mbps": speed,
            "elapsed_ms": 500,
            "token_fetch_ms": 50,
            "reconstruction_ms": 5,
            "error": None,
        }
    }


def test_skipped_when_no_xet_challenge():
    stats, errors = vvs._verify_xet_test({}, {})

    assert stats["status"] == "skipped"
    assert stats["success"] is True
    assert stats["bytes_downloaded"] == 0
    assert errors == []


def test_happy_path():
    stats, errors = vvs._verify_xet_test(_challenge(), _execution())

    assert stats["status"] == "ok"
    assert stats["success"] is True
    assert stats["bytes_downloaded"] == 1000
    assert stats["speed_mbps"] == 200.0
    assert stats["elapsed_ms"] == 500
    assert stats["token_fetch_ms"] == 50
    assert stats["hash"] == "abc"
    assert errors == []


def test_execution_failed():
    execution = _execution()
    execution["xet_execution"]["status"] = "failed"
    execution["xet_execution"]["success"] = False
    execution["xet_execution"]["error"] = "boom"

    stats, errors = vvs._verify_xet_test(_challenge(), execution)

    assert stats["success"] is False
    assert errors == ["Xet execution failed: boom"]


def test_execution_failed_short_circuits_speed_check():
    execution = _execution(speed=1.0)
    execution["xet_execution"]["status"] = "failed"
    execution["xet_execution"]["success"] = False
    execution["xet_execution"]["error"] = "boom"

    _, errors = vvs._verify_xet_test(_challenge(), execution)

    assert not any("speed inadequate" in error for error in errors)


def test_speed_below_minimum():
    stats, errors = vvs._verify_xet_test(_challenge(), _execution(speed=MIN_SPEED - 1))

    assert stats["success"] is False
    assert len(errors) == 1
    assert "speed inadequate" in errors[0]
    assert f"{MIN_SPEED:.0f} Mbps required" in errors[0]


def test_pkg_mismatch():
    stats, errors = vvs._verify_xet_test(_challenge(), _execution(pkg="org/other"))

    assert stats["success"] is False
    assert len(errors) == 1
    assert "Resource validation failed: org/other" in errors


def test_size_mismatch():
    stats, errors = vvs._verify_xet_test(
        _challenge(), _execution(size=999, hash_="abc")
    )

    assert stats["success"] is False
    assert len(errors) == 1
    assert "Size validation failed for org/model" in errors


def test_hash_mismatch():
    stats, errors = vvs._verify_xet_test(_challenge(), _execution(hash_="def"))

    assert stats["success"] is False
    assert len(errors) == 1
    assert "Integrity check failed for org/model" in errors


def test_all_failures_reported_together():
    execution = _execution(pkg="org/other", size=0, hash_="def", speed=1.0)

    stats, errors = vvs._verify_xet_test(_challenge(), execution)

    assert stats["success"] is False
    assert len(errors) == 4
    assert any("speed inadequate" in error for error in errors)
    assert any("Resource validation failed" in error for error in errors)
    assert any("Size validation failed" in error for error in errors)
    assert any("Integrity check failed" in error for error in errors)
