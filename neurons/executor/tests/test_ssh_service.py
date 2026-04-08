"""
Tests for SSHService - SSH key management operations.

Covers:
- Adding public keys to authorized_keys
- Removing public keys from authorized_keys
- Edge cases: duplicate keys, missing files, empty keys
- OS user detection
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from services.ssh_service import SSHService


class TestSSHServiceAddKey(unittest.TestCase):
    """Tests for SSHService.add_pubkey_to_host."""

    def setUp(self):
        self.service = SSHService()
        self.tmpdir = tempfile.mkdtemp()
        self.ssh_dir = os.path.join(self.tmpdir, ".ssh")
        os.makedirs(self.ssh_dir, exist_ok=True)
        self.auth_keys_path = os.path.join(self.ssh_dir, "authorized_keys")

    def tearDown(self):
        if os.path.exists(self.auth_keys_path):
            os.remove(self.auth_keys_path)
        os.rmdir(self.ssh_dir)
        os.rmdir(self.tmpdir)

    @patch("os.path.expanduser")
    def test_add_key_creates_entry(self, mock_expand):
        """Adding a key should append it to authorized_keys with a newline."""
        mock_expand.return_value = self.auth_keys_path
        # Create the file first
        with open(self.auth_keys_path, "w") as f:
            f.write("")

        self.service.add_pubkey_to_host("ssh-rsa AAAA testkey")

        with open(self.auth_keys_path, "r") as f:
            content = f.read()
        self.assertIn("ssh-rsa AAAA testkey", content)
        self.assertTrue(content.endswith("\n"))

    @patch("os.path.expanduser")
    def test_add_multiple_keys(self, mock_expand):
        """Adding multiple keys should result in multiple entries."""
        mock_expand.return_value = self.auth_keys_path
        with open(self.auth_keys_path, "w") as f:
            f.write("")

        self.service.add_pubkey_to_host("ssh-rsa KEY1 user1")
        self.service.add_pubkey_to_host("ssh-ed25519 KEY2 user2")

        with open(self.auth_keys_path, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("ssh-rsa KEY1 user1\n", lines)
        self.assertIn("ssh-ed25519 KEY2 user2\n", lines)

    @patch("os.path.expanduser")
    def test_add_duplicate_key_appends(self, mock_expand):
        """Adding the same key twice results in duplicate entries (no dedup)."""
        mock_expand.return_value = self.auth_keys_path
        with open(self.auth_keys_path, "w") as f:
            f.write("")

        self.service.add_pubkey_to_host("ssh-rsa DUPLICATE key")
        self.service.add_pubkey_to_host("ssh-rsa DUPLICATE key")

        with open(self.auth_keys_path, "r") as f:
            lines = f.readlines()
        matching = [l for l in lines if "DUPLICATE" in l]
        self.assertEqual(len(matching), 2)


class TestSSHServiceRemoveKey(unittest.TestCase):
    """Tests for SSHService.remove_pubkey_from_host."""

    def setUp(self):
        self.service = SSHService()
        self.tmpdir = tempfile.mkdtemp()
        self.ssh_dir = os.path.join(self.tmpdir, ".ssh")
        os.makedirs(self.ssh_dir, exist_ok=True)
        self.auth_keys_path = os.path.join(self.ssh_dir, "authorized_keys")

    def tearDown(self):
        if os.path.exists(self.auth_keys_path):
            os.remove(self.auth_keys_path)
        os.rmdir(self.ssh_dir)
        os.rmdir(self.tmpdir)

    @patch("os.path.expanduser")
    def test_remove_existing_key(self, mock_expand):
        """Removing an existing key should leave only other keys."""
        mock_expand.return_value = self.auth_keys_path
        with open(self.auth_keys_path, "w") as f:
            f.write("ssh-rsa KEEP_ME user1\n")
            f.write("ssh-rsa REMOVE_ME user2\n")
            f.write("ssh-ed25519 ALSO_KEEP user3\n")

        self.service.remove_pubkey_from_host("ssh-rsa REMOVE_ME user2")

        with open(self.auth_keys_path, "r") as f:
            content = f.read()
        self.assertNotIn("REMOVE_ME", content)
        self.assertIn("KEEP_ME", content)
        self.assertIn("ALSO_KEEP", content)

    @patch("os.path.expanduser")
    def test_remove_nonexistent_key(self, mock_expand):
        """Removing a key that doesn't exist should not change the file."""
        mock_expand.return_value = self.auth_keys_path
        original = "ssh-rsa EXISTING key\n"
        with open(self.auth_keys_path, "w") as f:
            f.write(original)

        self.service.remove_pubkey_from_host("ssh-rsa NONEXISTENT key")

        with open(self.auth_keys_path, "r") as f:
            content = f.read()
        self.assertEqual(content, original)

    @patch("os.path.expanduser")
    def test_remove_from_empty_file(self, mock_expand):
        """Removing from an empty authorized_keys should succeed silently."""
        mock_expand.return_value = self.auth_keys_path
        with open(self.auth_keys_path, "w") as f:
            f.write("")

        self.service.remove_pubkey_from_host("ssh-rsa NONEXISTENT key")

        with open(self.auth_keys_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "")


class TestSSHServiceGetUser(unittest.TestCase):
    """Tests for SSHService.get_current_os_user."""

    def test_returns_string(self):
        """get_current_os_user should return a non-empty string."""
        service = SSHService()
        user = service.get_current_os_user()
        self.assertIsInstance(user, str)
        self.assertTrue(len(user) > 0)

    @patch("getpass.getuser", return_value="testuser")
    def test_returns_mocked_user(self, mock_getuser):
        """Should return the value from getpass.getuser."""
        service = SSHService()
        self.assertEqual(service.get_current_os_user(), "testuser")


if __name__ == "__main__":
    unittest.main(verbosity=2)
