"""
Tests for Health Check Service

This module tests the health check functionality for the executor,
including comprehensive health checks, liveness probes, and readiness probes.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from services.health_service import (
    HealthStatus,
    ComponentHealth,
    _check_docker_health,
    _check_gpu_health,
    _check_system_resources,
    get_health_status,
    get_liveness_status,
    get_readiness_status,
)


class TestHealthStatus(unittest.TestCase):
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Verify all health status values exist."""
        self.assertEqual(HealthStatus.HEALTHY.value, "healthy")
        self.assertEqual(HealthStatus.DEGRADED.value, "degraded")
        self.assertEqual(HealthStatus.UNHEALTHY.value, "unhealthy")


class TestComponentHealth(unittest.TestCase):
    """Tests for ComponentHealth dataclass."""

    def test_component_health_creation(self):
        """Test creating a ComponentHealth instance."""
        health = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"key": "value"}
        )
        self.assertEqual(health.status, HealthStatus.HEALTHY)
        self.assertEqual(health.message, "All good")
        self.assertEqual(health.details, {"key": "value"})

    def test_component_health_without_details(self):
        """Test ComponentHealth with no details."""
        health = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message="Something wrong"
        )
        self.assertIsNone(health.details)


class TestDockerHealthCheck(unittest.TestCase):
    """Tests for Docker health check function."""

    @patch('services.health_service.docker')
    def test_docker_healthy(self, mock_docker):
        """Test Docker health check when daemon is responsive."""
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        
        # Mock containers
        mock_container1 = MagicMock()
        mock_container1.status = "running"
        mock_container2 = MagicMock()
        mock_container2.status = "exited"
        mock_client.containers.list.return_value = [mock_container1, mock_container2]
        
        mock_client.version.return_value = {"Version": "24.0.0"}
        
        result = _check_docker_health()
        
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertIn("responsive", result.message)
        self.assertEqual(result.details["containers_running"], 1)
        self.assertEqual(result.details["containers_total"], 2)
        self.assertEqual(result.details["docker_version"], "24.0.0")

    @patch('services.health_service.docker')
    def test_docker_unhealthy_exception(self, mock_docker):
        """Test Docker health check when daemon throws exception."""
        mock_docker.from_env.side_effect = Exception("Connection refused")
        mock_docker.errors.DockerException = Exception
        
        result = _check_docker_health()
        
        self.assertEqual(result.status, HealthStatus.UNHEALTHY)
        self.assertIn("error", result.message.lower())


class TestGPUHealthCheck(unittest.TestCase):
    """Tests for GPU health check function."""

    @patch('services.health_service.pynvml')
    def test_gpu_healthy(self, mock_pynvml):
        """Test GPU health check with available GPUs."""
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlShutdown.return_value = None
        
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 4090"
        
        mock_mem = MagicMock()
        mock_mem.total = 24 * 1024 * 1024 * 1024  # 24GB
        mock_mem.free = 20 * 1024 * 1024 * 1024   # 20GB free
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem
        
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 45
        mock_pynvml.NVML_TEMPERATURE_GPU = 0
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "535.183.01"
        
        result = _check_gpu_health()
        
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertIn("2", result.message)
        self.assertEqual(result.details["gpu_count"], 2)

    @patch('services.health_service.pynvml')
    def test_gpu_no_gpus(self, mock_pynvml):
        """Test GPU health check with no GPUs detected."""
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mock_pynvml.nvmlShutdown.return_value = None
        
        result = _check_gpu_health()
        
        self.assertEqual(result.status, HealthStatus.DEGRADED)
        self.assertIn("No GPUs", result.message)

    def test_gpu_no_driver(self):
        """Test GPU health check when NVIDIA driver not available."""
        # This test runs without mocking to test real behavior when no GPU
        # The function should handle this gracefully
        result = _check_gpu_health()
        
        # Should be degraded, not unhealthy (no GPU is acceptable)
        self.assertIn(result.status, [HealthStatus.HEALTHY, HealthStatus.DEGRADED])


class TestSystemResourcesCheck(unittest.TestCase):
    """Tests for system resources health check function."""

    @patch('services.health_service.psutil')
    def test_system_healthy(self, mock_psutil):
        """Test system resources check with healthy values."""
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024 ** 3  # 8GB
        mock_memory.total = 16 * 1024 ** 3     # 16GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = MagicMock()
        mock_disk.percent = 60.0
        mock_disk.free = 100 * 1024 ** 3   # 100GB
        mock_disk.total = 250 * 1024 ** 3  # 250GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        result = _check_system_resources()
        
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertIn("OK", result.message)
        self.assertEqual(result.details["cpu"]["usage_percent"], 25.0)
        self.assertEqual(result.details["memory"]["usage_percent"], 50.0)
        self.assertEqual(result.details["disk"]["usage_percent"], 60.0)

    @patch('services.health_service.psutil')
    def test_system_high_memory(self, mock_psutil):
        """Test system resources check with high memory usage."""
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = MagicMock()
        mock_memory.percent = 95.0  # High memory
        mock_memory.available = 1 * 1024 ** 3
        mock_memory.total = 16 * 1024 ** 3
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = MagicMock()
        mock_disk.percent = 60.0
        mock_disk.free = 100 * 1024 ** 3
        mock_disk.total = 250 * 1024 ** 3
        mock_psutil.disk_usage.return_value = mock_disk
        
        result = _check_system_resources()
        
        self.assertEqual(result.status, HealthStatus.DEGRADED)
        self.assertIn("memory", result.message.lower())

    @patch('services.health_service.psutil')
    def test_system_high_disk(self, mock_psutil):
        """Test system resources check with high disk usage."""
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_memory.available = 8 * 1024 ** 3
        mock_memory.total = 16 * 1024 ** 3
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = MagicMock()
        mock_disk.percent = 95.0  # High disk
        mock_disk.free = 10 * 1024 ** 3
        mock_disk.total = 250 * 1024 ** 3
        mock_psutil.disk_usage.return_value = mock_disk
        
        result = _check_system_resources()
        
        self.assertEqual(result.status, HealthStatus.DEGRADED)
        self.assertIn("disk", result.message.lower())


class TestGetHealthStatus(unittest.TestCase):
    """Tests for comprehensive health status function."""

    @patch('services.health_service._check_docker_health')
    @patch('services.health_service._check_gpu_health')
    @patch('services.health_service._check_system_resources')
    def test_all_healthy(self, mock_system, mock_gpu, mock_docker):
        """Test overall health when all components are healthy."""
        mock_docker.return_value = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Docker OK",
            details={}
        )
        mock_gpu.return_value = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="GPU OK",
            details={}
        )
        mock_system.return_value = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="System OK",
            details={}
        )
        
        result = get_health_status()
        
        self.assertEqual(result["status"], "healthy")
        self.assertIn("timestamp", result)
        self.assertIn("uptime_seconds", result)
        self.assertIn("components", result)

    @patch('services.health_service._check_docker_health')
    @patch('services.health_service._check_gpu_health')
    @patch('services.health_service._check_system_resources')
    def test_docker_unhealthy(self, mock_system, mock_gpu, mock_docker):
        """Test overall health when Docker is unhealthy."""
        mock_docker.return_value = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message="Docker down",
            details=None
        )
        mock_gpu.return_value = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="GPU OK",
            details={}
        )
        mock_system.return_value = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="System OK",
            details={}
        )
        
        result = get_health_status()
        
        self.assertEqual(result["status"], "unhealthy")

    @patch('services.health_service._check_docker_health')
    @patch('services.health_service._check_gpu_health')
    @patch('services.health_service._check_system_resources')
    def test_gpu_degraded_overall_healthy(self, mock_system, mock_gpu, mock_docker):
        """Test that GPU being degraded doesn't make overall status unhealthy."""
        mock_docker.return_value = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Docker OK",
            details={}
        )
        mock_gpu.return_value = ComponentHealth(
            status=HealthStatus.DEGRADED,
            message="No GPU",
            details={}
        )
        mock_system.return_value = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="System OK",
            details={}
        )
        
        result = get_health_status()
        
        # GPU degraded alone should result in degraded, not unhealthy
        self.assertIn(result["status"], ["healthy", "degraded"])


class TestLivenessStatus(unittest.TestCase):
    """Tests for liveness probe function."""

    def test_liveness_returns_alive(self):
        """Test that liveness probe returns alive status."""
        result = get_liveness_status()
        
        self.assertTrue(result["alive"])
        self.assertIn("timestamp", result)


class TestReadinessStatus(unittest.TestCase):
    """Tests for readiness probe function."""

    @patch('services.health_service._check_docker_health')
    def test_readiness_docker_healthy(self, mock_docker):
        """Test readiness when Docker is healthy."""
        mock_docker.return_value = ComponentHealth(
            status=HealthStatus.HEALTHY,
            message="Docker OK",
            details={}
        )
        
        result = get_readiness_status()
        
        self.assertTrue(result["ready"])
        self.assertTrue(result["checks"]["docker"])

    @patch('services.health_service._check_docker_health')
    def test_readiness_docker_unhealthy(self, mock_docker):
        """Test readiness when Docker is unhealthy."""
        mock_docker.return_value = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            message="Docker down",
            details=None
        )
        
        result = get_readiness_status()
        
        self.assertFalse(result["ready"])
        self.assertFalse(result["checks"]["docker"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
