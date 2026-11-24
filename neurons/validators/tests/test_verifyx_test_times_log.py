from unittest.mock import patch

from neurons.validators.src.services.verifyx_validation_service import _log_verifyx_network_speeds

SERVICE = "neurons.validators.src.services.verifyx_validation_service"


def _logged_extra(response_data: dict) -> dict:
    with patch(f"{SERVICE}.logger") as logger:
        _log_verifyx_network_speeds({"success": True}, response_data, {"executor_uuid": "exec-1"})
    return logger.info.call_args.args[0].extra


def test_network_speeds_line_carries_the_time_of_each_verifyx_test():
    # Arrange
    response_data = {
        "memory_execution": {"execution_time_ms": 41_000},
        "storage_execution": {"execution_time_ms": 23_000},
        "network_execution": {"execution_time_ms": 88_000, "download": {"time_ms": 34_000}},
    }

    # Act
    extra = _logged_extra(response_data)

    # Assert
    assert (extra["memory_ms"], extra["storage_ms"], extra["network_ms"], extra["package_ms"]) == (
        41_000,
        23_000,
        88_000,
        34_000,
    )


def test_a_time_verifyx_did_not_report_is_logged_as_none():
    # Arrange
    response_data = {"memory_execution": {"execution_time_ms": 41_000}, "network_execution": {}}

    # Act
    extra = _logged_extra(response_data)

    # Assert
    assert (extra["memory_ms"], extra["storage_ms"], extra["network_ms"], extra["package_ms"]) == (
        41_000,
        None,
        None,
        None,
    )
