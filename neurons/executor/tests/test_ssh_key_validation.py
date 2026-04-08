"""
Tests for SSH public key validation and sanitization.

Covers:
- Valid key types accepted
- Empty keys rejected
- Newline injection prevention
- Invalid key type rejection
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from services.ssh_service import validate_ssh_public_key, InvalidSSHKeyError


class TestValidateSSHPublicKey(unittest.TestCase):
    """Tests for validate_ssh_public_key."""

    def test_valid_rsa_key(self):
        """Standard RSA key should pass validation."""
        key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAB user@host"
        result = validate_ssh_public_key(key)
        self.assertEqual(result, key)

    def test_valid_ed25519_key(self):
        """Ed25519 key should pass validation."""
        key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH user@host"
        result = validate_ssh_public_key(key)
        self.assertEqual(result, key)

    def test_valid_ecdsa_key(self):
        """ECDSA key should pass validation."""
        key = "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlz user@host"
        result = validate_ssh_public_key(key)
        self.assertEqual(result, key)

    def test_valid_dss_key(self):
        """DSS key should pass validation."""
        key = "ssh-dss AAAAB3NzaC1kc3MAAACBAM user@host"
        result = validate_ssh_public_key(key)
        self.assertEqual(result, key)

    def test_valid_sk_ed25519_key(self):
        """Security key Ed25519 should pass validation."""
        key = "sk-ssh-ed25519 AAAAG user@host"
        result = validate_ssh_public_key(key)
        self.assertEqual(result, key)

    def test_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        key = "  ssh-rsa AAAAB3 user@host  \n"
        result = validate_ssh_public_key(key)
        self.assertEqual(result, "ssh-rsa AAAAB3 user@host")

    def test_empty_key_rejected(self):
        """Empty string should raise InvalidSSHKeyError."""
        with self.assertRaises(InvalidSSHKeyError) as ctx:
            validate_ssh_public_key("")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_whitespace_only_rejected(self):
        """Whitespace-only string should raise InvalidSSHKeyError."""
        with self.assertRaises(InvalidSSHKeyError):
            validate_ssh_public_key("   \n\t  ")

    def test_newline_injection_rejected(self):
        """Key with embedded newline should be rejected (prevents authorized_keys injection)."""
        malicious = "ssh-rsa AAAA user@host\ncommand=\"evil\" ssh-rsa BBBB attacker@evil"
        with self.assertRaises(InvalidSSHKeyError) as ctx:
            validate_ssh_public_key(malicious)
        self.assertIn("newline", str(ctx.exception).lower())

    def test_carriage_return_rejected(self):
        """Key with embedded carriage return should be rejected."""
        malicious = "ssh-rsa AAAA user@host\rcommand=\"evil\" ssh-rsa BBBB"
        with self.assertRaises(InvalidSSHKeyError) as ctx:
            validate_ssh_public_key(malicious)
        self.assertIn("newline", str(ctx.exception).lower())

    def test_invalid_key_type_rejected(self):
        """Key with unrecognized type prefix should be rejected."""
        with self.assertRaises(InvalidSSHKeyError) as ctx:
            validate_ssh_public_key("not-a-key AAAA user@host")
        self.assertIn("valid key type", str(ctx.exception).lower())

    def test_random_string_rejected(self):
        """Random string that's not an SSH key should be rejected."""
        with self.assertRaises(InvalidSSHKeyError):
            validate_ssh_public_key("hello world this is not a key")

    def test_is_value_error_subclass(self):
        """InvalidSSHKeyError should be a ValueError subclass for API compatibility."""
        self.assertTrue(issubclass(InvalidSSHKeyError, ValueError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
