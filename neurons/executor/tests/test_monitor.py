"""
Tests for monitor.py - Docker container event monitoring.

Covers:
- classify_stop exit code classification
- handle_event filtering and PodLog creation
- Edge cases: unknown exit codes, non-matching container names
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Mock settings before importing monitor
mock_settings = MagicMock()
mock_settings.MINER_HOTKEY_SS58_ADDRESS = "5FakeHotkey"

with patch.dict("sys.modules", {
    "core.config": MagicMock(settings=mock_settings),
    "core.db": MagicMock(),
    "daos.pod_log": MagicMock(),
    "pynvml": MagicMock(),
    "docker": MagicMock(),
    "psutil": MagicMock(),
}):
    # Need to set up PodLog mock before import
    from unittest.mock import MagicMock as MM
    pod_log_module = sys.modules["daos.pod_log"]
    
    class FakePodLog:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return self.__dict__
    
    pod_log_module.PodLog = FakePodLog
    pod_log_module.PodLogDao = MagicMock

    from monitor import classify_stop, handle_event


class TestClassifyStop(unittest.TestCase):
    """Tests for classify_stop exit code classification."""

    def test_exit_code_0_purposely_stopped(self):
        """Exit code 0 indicates intentional stop."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(0), "purposely_stopped")

    def test_exit_code_1_application_error(self):
        """Exit code 1 indicates application error."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(1), "application_error")

    def test_exit_code_125_container_failed(self):
        """Exit code 125 indicates container failed to run."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(125), "container_failed_to_run")

    def test_exit_code_126_command_invoke_error(self):
        """Exit code 126 indicates command invocation error."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(126), "command_invoke_error")

    def test_exit_code_127_file_not_found(self):
        """Exit code 127 indicates file or directory not found."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(127), "file_or_directory_not_found")

    def test_exit_code_128_invalid_argument(self):
        """Exit code 128 indicates invalid argument on exit."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(128), "invalid_argument_on_exit")

    def test_exit_code_134_sigabrt(self):
        """Exit code 134 (SIGABRT) indicates abnormal termination."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(134), "abnormal_termination")

    def test_exit_code_137_sigkill(self):
        """Exit code 137 (SIGKILL) indicates immediate termination."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(137), "immediate_termination")

    def test_exit_code_139_sigsegv(self):
        """Exit code 139 (SIGSEGV) indicates segmentation fault."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(139), "segmentation_fault")

    def test_exit_code_143_sigterm(self):
        """Exit code 143 (SIGTERM) indicates graceful termination."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(143), "graceful_termination")

    def test_exit_code_255_out_of_range(self):
        """Exit code 255 indicates exit status out of range."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(255), "exit_status_out_of_range")

    def test_unknown_exit_code(self):
        """Unrecognized exit codes should return 'unknown'."""
        with patch("subprocess.check_output", return_value=""):
            self.assertEqual(classify_stop(42), "unknown")

    def test_gpu_error_from_dmesg(self):
        """GPU errors in dmesg should override exit code classification."""
        dmesg_with_xid = "some log\nNVRM: Xid (PCI:0000:01:00): 79\nmore log"
        with patch("subprocess.check_output", return_value=dmesg_with_xid):
            self.assertEqual(classify_stop(1), "gpu_error")

    def test_gpu_fallen_off_bus(self):
        """GPU fallen off bus in dmesg should classify as gpu_error."""
        dmesg_fallen = "GPU has fallen off the bus"
        with patch("subprocess.check_output", return_value=dmesg_fallen):
            self.assertEqual(classify_stop(0), "gpu_error")

    def test_dmesg_failure_falls_through(self):
        """When dmesg fails, should still classify by exit code."""
        with patch("subprocess.check_output", side_effect=OSError("not available")):
            self.assertEqual(classify_stop(137), "immediate_termination")


class TestHandleEvent(unittest.TestCase):
    """Tests for handle_event filtering logic."""

    @patch("monitor.log_event")
    def test_ignores_non_prefixed_containers(self, mock_log):
        """Events from containers not matching prefix should be ignored."""
        event = {
            "Action": "start",
            "Actor": {"ID": "abc123", "Attributes": {"name": "random_container"}},
            "id": "abc123",
        }
        handle_event(event)
        mock_log.assert_not_called()

    @patch("monitor.log_event")
    def test_ignores_miner_hotkey_container(self, mock_log):
        """Events from the miner's own hotkey container should be ignored."""
        event = {
            "Action": "start",
            "Actor": {
                "ID": "abc123",
                "Attributes": {"name": "container_5FakeHotkey"},
            },
            "id": "abc123",
        }
        handle_event(event)
        mock_log.assert_not_called()

    @patch("monitor.log_event")
    def test_logs_start_event(self, mock_log):
        """Start events for valid containers should be logged."""
        event = {
            "Action": "start",
            "Actor": {
                "ID": "abc123",
                "Attributes": {"name": "container_user_pod"},
            },
            "id": "abc123",
        }
        handle_event(event)
        mock_log.assert_called_once()

    @patch("monitor.log_event")
    def test_logs_stop_event(self, mock_log):
        """Stop events for valid containers should be logged."""
        event = {
            "Action": "stop",
            "Actor": {
                "ID": "def456",
                "Attributes": {"name": "container_test"},
            },
            "id": "def456",
        }
        handle_event(event)
        mock_log.assert_called_once()

    @patch("monitor.log_event")
    def test_logs_oom_event(self, mock_log):
        """OOM events should be logged."""
        event = {
            "Action": "oom",
            "Actor": {
                "ID": "oom123",
                "Attributes": {"name": "container_test"},
            },
            "id": "oom123",
        }
        handle_event(event)
        mock_log.assert_called_once()

    @patch("monitor.classify_stop", return_value="application_error")
    @patch("monitor.log_event")
    def test_die_event_classifies_exit_code(self, mock_log, mock_classify):
        """Die events should classify the exit code and log with reason."""
        event = {
            "Action": "die",
            "Actor": {
                "ID": "die789",
                "Attributes": {"name": "container_app", "exitCode": "1"},
            },
            "id": "die789",
        }
        handle_event(event)
        mock_classify.assert_called_once_with(1)
        mock_log.assert_called_once()

    @patch("monitor.log_event")
    def test_ignores_untracked_actions(self, mock_log):
        """Actions like 'create', 'attach' should not be logged."""
        for action in ["create", "attach", "detach", "resize", "exec_create"]:
            mock_log.reset_mock()
            event = {
                "Action": action,
                "Actor": {
                    "ID": "xyz",
                    "Attributes": {"name": "container_test"},
                },
                "id": "xyz",
            }
            handle_event(event)
            mock_log.assert_not_called(), f"Action '{action}' should not be logged"

    @patch("monitor.log_event")
    def test_no_name_attribute_ignored(self, mock_log):
        """Events without a container name should be ignored."""
        event = {
            "Action": "start",
            "Actor": {"ID": "abc123", "Attributes": {}},
            "id": "abc123",
        }
        handle_event(event)
        mock_log.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
