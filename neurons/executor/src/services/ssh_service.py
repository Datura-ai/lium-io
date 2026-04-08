import getpass
import logging
import os
import re
import stat

logger = logging.getLogger(__name__)

# Accepted SSH key type prefixes per OpenSSH specification
_VALID_KEY_PREFIXES = (
    "ssh-rsa",
    "ssh-ed25519",
    "ssh-dss",
    "ecdsa-sha2-",
    "sk-ssh-ed25519",
    "sk-ecdsa-sha2-",
)


class InvalidSSHKeyError(ValueError):
    """Raised when a public key fails basic format validation."""


def validate_ssh_public_key(pub_key: str) -> str:
    """
    Validate and sanitize an SSH public key string.

    Checks that the key:
    - Is non-empty after stripping whitespace
    - Starts with a recognised key-type prefix
    - Contains no embedded newlines (prevents authorized_keys injection)

    Args:
        pub_key: The raw public key string to validate.

    Returns:
        The stripped, validated public key.

    Raises:
        InvalidSSHKeyError: If the key fails any validation check.
    """
    stripped = pub_key.strip()
    if not stripped:
        raise InvalidSSHKeyError("SSH public key must not be empty")

    # Guard against newline injection into authorized_keys
    if "\n" in stripped or "\r" in stripped:
        raise InvalidSSHKeyError(
            "SSH public key contains embedded newlines"
        )

    if not any(stripped.startswith(prefix) for prefix in _VALID_KEY_PREFIXES):
        raise InvalidSSHKeyError(
            f"SSH public key must start with a valid key type "
            f"(e.g. ssh-rsa, ssh-ed25519), got: {stripped[:30]!r}..."
        )

    return stripped


def _ensure_ssh_directory(authorized_keys_path: str) -> None:
    """
    Ensure the parent .ssh directory exists with correct permissions.

    Creates ``~/.ssh`` with mode 0700 and the ``authorized_keys`` file with
    mode 0600 if they do not already exist.
    """
    ssh_dir = os.path.dirname(authorized_keys_path)
    if not os.path.isdir(ssh_dir):
        os.makedirs(ssh_dir, mode=0o700, exist_ok=True)
        logger.info("Created SSH directory: %s", ssh_dir)

    if not os.path.isfile(authorized_keys_path):
        with open(authorized_keys_path, "w") as f:
            pass  # create empty file
        os.chmod(authorized_keys_path, stat.S_IRUSR | stat.S_IWUSR)
        logger.info("Created authorized_keys file: %s", authorized_keys_path)

from core.config import settings

logger = logging.getLogger(__name__)


class SSHService:
    """Service for managing SSH authorized keys on the host."""

    def _get_authorized_keys_path(self) -> str:
        """Return the resolved path to the authorized_keys file."""
        return os.path.expanduser("~/.ssh/authorized_keys")

    def add_pubkey_to_host(self, pub_key: str) -> None:
        """
        Append a validated public key to the authorized_keys file.

        Args:
            pub_key: SSH public key string to add.

        Raises:
            InvalidSSHKeyError: If the key fails validation.
            OSError: If the file cannot be written.
        """
        validated_key = validate_ssh_public_key(pub_key)
        auth_path = self._get_authorized_keys_path()
        _ensure_ssh_directory(auth_path)

        with open(auth_path, "a") as file:
            file.write(validated_key + "\n")

        logger.info(
            "Added SSH key (type=%s) to %s",
            validated_key.split()[0],
            auth_path,
        )

    def remove_pubkey_from_host(self, pub_key: str) -> bool:
        """
        Remove a public key from the authorized_keys file.

        Args:
            pub_key: SSH public key string to remove.

        Returns:
            True if the key was found and removed, False otherwise.

        Raises:
            OSError: If the file cannot be read or written.
        """
        authorized_keys_path = self._get_authorized_keys_path()
        target = pub_key.strip()

        if not os.path.isfile(authorized_keys_path):
            logger.warning(
                "authorized_keys file not found at %s", authorized_keys_path
            )
            return False

        with open(authorized_keys_path, "r") as file:
            lines = file.readlines()

        original_count = len(lines)
        filtered = [line for line in lines if line.strip() != target]

        with open(authorized_keys_path, "w") as file:
            file.writelines(filtered)

        removed_count = original_count - len(filtered)
        if removed_count > 0:
            logger.info(
                "Removed %d SSH key entry/entries from %s",
                removed_count,
                authorized_keys_path,
            )
            return True

        logger.debug("SSH key not found in %s", authorized_keys_path)
        return False

    def get_current_os_user(self) -> str:
        """Return the current operating system username."""
        return getpass.getuser()

    def get_host_public_key(self) -> str | None:
        host_key_path = settings.SSH_HOST_KEY_PATH
        if not host_key_path:
            return None

        path = os.path.expanduser(host_key_path)
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    candidate = line.strip()
                    if candidate:
                        return candidate
        except FileNotFoundError:
            logger.warning("SSH host key file not found at %s", path)
        except OSError as exc:
            logger.warning("Failed to read SSH host key from %s: %s", path, exc)

        return None
