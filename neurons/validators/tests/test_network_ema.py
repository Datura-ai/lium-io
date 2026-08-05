"""Tests for the display-speed resolution published to every UI (DAH-2572)."""

from __future__ import annotations

from neurons.validators.src.services.task.checks.network_ema import resolve_display_speeds


def test_verifyx_ema_wins_over_every_other_measurement() -> None:
    network = {
        "upload_speed": 100.0,
        "download_speed": 200.0,
        "ema_upload_speed": 110.0,
        "ema_download_speed": 210.0,
        "ema_verifyx_upload_speed": 900.0,
        "ema_verifyx_download_speed": 9100.0,
    }

    assert resolve_display_speeds(network) == (900.0, 9100.0)


def test_zeroed_ema_does_not_mask_a_real_measurement() -> None:
    network = {
        "upload_speed": 100.0,
        "download_speed": 200.0,
        "ema_verifyx_upload_speed": 0.0,
        "ema_verifyx_download_speed": 0.0,
    }

    assert resolve_display_speeds(network) == (100.0, 200.0)


def test_a_node_with_no_measurement_at_all_resolves_to_none() -> None:
    assert resolve_display_speeds({}) == (None, None)
