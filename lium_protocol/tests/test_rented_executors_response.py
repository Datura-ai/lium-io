"""RentedExecutorsResponse since protocol 1.5.0: `filler_revenue_by_gpu_config` and `provider_spot_executor_ids`
ride the reply; a reply from an older backend, which sends neither, reads as empty for both, and unknown keys are
ignored, so a 1.4.0 reader takes a 1.5.0 reply."""

from lium_protocol import PROTOCOL_VERSION
from lium_protocol.http import RentedExecutorsResponse

OLDER_BACKEND_REPLY = {"executors": {}, "spot_executor_ids": ["6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c"]}
NEW_FIELDS = {
    "provider_spot_executor_ids": ["6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c"],
    "filler_revenue_by_gpu_config": [
        {"base_model": "NVIDIA B200", "gpu_count": 8, "usd_per_gpu_hour": 2.5, "gpu_hours": 120.0}
    ],
}


def test_the_fields_arrived_with_a_minor_bump() -> None:
    major, minor, _patch = (int(part) for part in PROTOCOL_VERSION.split("."))
    assert (major, minor) >= (1, 5)
    assert set(NEW_FIELDS) <= set(RentedExecutorsResponse.model_fields)


def test_an_older_backend_reply_reads_as_empty() -> None:
    reply = RentedExecutorsResponse.model_validate(OLDER_BACKEND_REPLY)
    assert reply.provider_spot_executor_ids == []
    assert reply.filler_revenue_by_gpu_config == []
    assert reply.spot_executor_ids == OLDER_BACKEND_REPLY["spot_executor_ids"]


def test_unknown_keys_are_ignored_so_an_older_reader_takes_a_newer_reply() -> None:
    assert RentedExecutorsResponse.model_config.get("extra", "ignore") == "ignore"
    reply = RentedExecutorsResponse.model_validate({**OLDER_BACKEND_REPLY, **NEW_FIELDS, "a_later_field": [1]})
    assert "a_later_field" not in reply.model_dump()
    assert reply.provider_spot_executor_ids == NEW_FIELDS["provider_spot_executor_ids"]
    assert reply.filler_revenue_by_gpu_config[0].usd_per_gpu_hour == 2.5


def test_executor_owner_hotkey_arrived_in_1_6_and_is_empty_from_an_older_backend() -> None:
    major, minor, _patch = (int(part) for part in PROTOCOL_VERSION.split("."))
    assert (major, minor) >= (1, 6)
    assert RentedExecutorsResponse.model_validate(OLDER_BACKEND_REPLY).executor_owner_hotkeys == {}
    owners = {"6f1d2c3b-4a5e-4f60-9b7c-8d9e0f1a2b3c": "5Owner"}
    reply = RentedExecutorsResponse.model_validate(
        {**OLDER_BACKEND_REPLY, "executor_owner_hotkeys": owners}
    )
    assert reply.executor_owner_hotkeys == owners


def test_maintenance_windows_arrived_in_1_16_and_are_empty_from_an_older_backend() -> None:
    major, minor, _patch = (int(part) for part in PROTOCOL_VERSION.split("."))
    assert (major, minor) >= (1, 16)
    assert RentedExecutorsResponse.model_validate(OLDER_BACKEND_REPLY).maintenance_windows_by_executor == {}
    executor = "3e2f1a0b-9c8d-4e7f-8a6b-5c4d3e2f1a0b"
    reply = RentedExecutorsResponse.model_validate(
        {
            **OLDER_BACKEND_REPLY,
            "maintenance_executor_ids": [executor],
            "maintenance_windows_by_executor": {
                executor: {"starting_at": "2026-10-23T10:00:00Z", "expiring_at": "2026-10-23T16:00:00Z"}
            },
        }
    )
    window = reply.maintenance_windows_by_executor[executor]
    assert window.starting_at.utcoffset().total_seconds() == 0
    assert (window.expiring_at - window.starting_at).total_seconds() == 6 * 3600
