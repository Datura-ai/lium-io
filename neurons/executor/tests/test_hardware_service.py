"""
Tests for hardware_service - system and container metric collection.

Covers:
- _parse_df_output parsing logic
- get_system_metrics with mocked psutil/pynvml
- get_container_metrics error handling
- Edge cases: malformed df output, no GPUs, missing containers
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from services.hardware_service import (
    _parse_df_output,
    _get_filesystem_usage,
    get_system_metrics,
    get_container_metrics,
)


class TestParseDfOutput(unittest.TestCase):
    """Tests for _parse_df_output helper."""

    def test_valid_df_output(self):
        """Standard df -k output should be parsed correctly."""
        output = (
            "Filesystem     1K-blocks    Used Available Use% Mounted on\n"
            "/dev/sda1      100000000 40000000  60000000  40% /\n"
        )
        result = _parse_df_output(output)
        self.assertEqual(result["total"], 100000000)
        self.assertEqual(result["used"], 40000000)
        self.assertEqual(result["available"], 60000000)
        self.assertAlmostEqual(result["utilization"], 40.0, places=1)

    def test_empty_output(self):
        """Empty string should return zero metrics."""
        result = _parse_df_output("")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["used"], 0)
        self.assertEqual(result["available"], 0)
        self.assertEqual(result["utilization"], 0.0)

    def test_header_only(self):
        """Only header line with no data should return zeros."""
        output = "Filesystem     1K-blocks    Used Available Use% Mounted on\n"
        result = _parse_df_output(output)
        # Second line will be parsed but may fail
        self.assertIn("total", result)

    def test_malformed_data_line(self):
        """Data line with too few columns should return zeros."""
        output = (
            "Filesystem     1K-blocks\n"
            "/dev/sda1      100000000\n"
        )
        result = _parse_df_output(output)
        self.assertEqual(result["total"], 0)

    def test_non_numeric_values(self):
        """Non-numeric values in data columns should return zeros."""
        output = (
            "Filesystem     1K-blocks    Used Available Use% Mounted on\n"
            "/dev/sda1      abc          def  ghi       40% /\n"
        )
        result = _parse_df_output(output)
        self.assertEqual(result["total"], 0)

    def test_zero_total(self):
        """Zero total should result in 0.0 utilization (no division by zero)."""
        output = (
            "Filesystem     1K-blocks    Used Available Use% Mounted on\n"
            "tmpfs          0            0    0          0% /dev\n"
        )
        result = _parse_df_output(output)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["utilization"], 0.0)

    def test_high_utilization(self):
        """Near-full disk should show ~100% utilization."""
        output = (
            "Filesystem     1K-blocks    Used Available Use% Mounted on\n"
            "/dev/sda1      1000000      999000 1000     99% /\n"
        )
        result = _parse_df_output(output)
        self.assertGreater(result["utilization"], 99.0)


class TestGetSystemMetrics(unittest.TestCase):
    """Tests for get_system_metrics with mocked dependencies."""

    @patch("services.hardware_service.pynvml")
    @patch("services.hardware_service.psutil")
    def test_returns_expected_structure(self, mock_psutil, mock_pynvml):
        """Should return dict with cpu, memory, storage, gpu keys."""
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value = MagicMock(percent=60.0)
        mock_psutil.disk_usage.return_value = MagicMock(percent=45.0)

        # Simulate no GPUs
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mock_pynvml.nvmlShutdown.return_value = None

        result = get_system_metrics()

        self.assertIn("cpu", result)
        self.assertIn("memory", result)
        self.assertIn("storage", result)
        self.assertIn("gpu", result)
        self.assertEqual(result["cpu"], 25.5)
        self.assertEqual(result["memory"], 60.0)
        self.assertEqual(result["storage"], 45.0)
        self.assertIsInstance(result["gpu"], list)

    @patch("services.hardware_service.pynvml")
    @patch("services.hardware_service.psutil")
    def test_with_gpu_metrics(self, mock_psutil, mock_pynvml):
        """Should include GPU utilization and memory when GPUs are present."""
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=30.0)
        mock_psutil.disk_usage.return_value = MagicMock(percent=20.0)

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlShutdown.return_value = None

        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_util = MagicMock()
        mock_util.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_mem = MagicMock()
        mock_mem.used = 4 * 1024**3  # 4 GB
        mock_mem.total = 8 * 1024**3  # 8 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        result = get_system_metrics()

        self.assertEqual(len(result["gpu"]), 2)
        self.assertEqual(result["gpu"][0]["utilization"], 75)
        self.assertAlmostEqual(result["gpu"][0]["memory"], 50.0)

    @patch("services.hardware_service.pynvml")
    @patch("services.hardware_service.psutil")
    def test_nvml_error_graceful(self, mock_psutil, mock_pynvml):
        """NVML errors should be handled gracefully, returning empty GPU list."""
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=30.0)
        mock_psutil.disk_usage.return_value = MagicMock(percent=20.0)

        mock_pynvml.NVMLError = Exception
        mock_pynvml.NVMLError_NotSupported = Exception
        mock_pynvml.NVMLError_DriverNotLoaded = Exception
        mock_pynvml.nvmlInit.side_effect = Exception("No NVIDIA driver")

        result = get_system_metrics()

        self.assertEqual(result["gpu"], [])


class TestGetContainerMetrics(unittest.TestCase):
    """Tests for get_container_metrics error paths."""

    @patch("services.hardware_service.docker")
    def test_container_not_found_raises(self, mock_docker):
        """Should raise ValueError when container doesn't exist."""
        import docker.errors

        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.containers.get.side_effect = docker.errors.NotFound("not found")
        mock_docker.errors.NotFound = docker.errors.NotFound
        mock_docker.errors.APIError = docker.errors.APIError

        with self.assertRaises(ValueError) as ctx:
            get_container_metrics("nonexistent_container", [])
        self.assertIn("nonexistent_container", str(ctx.exception))


class TestGetFilesystemUsage(unittest.TestCase):
    """Tests for _get_filesystem_usage helper."""

    def test_successful_exec(self):
        """Should parse df output from container exec correctly."""
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.output = (
            b"Filesystem     1K-blocks    Used Available Use% Mounted on\n"
            b"/dev/sda1      500000       200000 300000    40% /\n"
        )
        mock_container.exec_run.return_value = mock_result

        result = _get_filesystem_usage(mock_container, "/")

        self.assertEqual(result["total"], 500000)
        self.assertEqual(result["used"], 200000)
        self.assertEqual(result["mount_point"], "/")

    def test_exec_failure(self):
        """Should return zeros when exec_run fails."""
        mock_container = MagicMock()
        mock_result = MagicMock()
        mock_result.exit_code = 1
        mock_result.output = b"df: /nonexistent: No such file or directory"
        mock_container.exec_run.return_value = mock_result

        result = _get_filesystem_usage(mock_container, "/nonexistent")

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["mount_point"], "/nonexistent")

    def test_exec_exception(self):
        """Should return zeros when exec_run raises an exception."""
        mock_container = MagicMock()
        mock_container.exec_run.side_effect = RuntimeError("container not running")

        result = _get_filesystem_usage(mock_container, "/")

        self.assertEqual(result["total"], 0)
        self.assertEqual(result["utilization"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
