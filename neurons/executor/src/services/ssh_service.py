import getpass
import logging
import os

from core.config import settings

logger = logging.getLogger(__name__)


class SSHService:
    def add_pubkey_to_host(self, pub_key: str):
        with open(os.path.expanduser("~/.ssh/authorized_keys"), "a") as file:
            file.write(pub_key + "\n")

    def remove_pubkey_from_host(self, pub_key: str):
        authorized_keys_path = os.path.expanduser("~/.ssh/authorized_keys")

        with open(authorized_keys_path, "r") as file:
            lines = file.readlines()

        with open(authorized_keys_path, "w") as file:
            for line in lines:
                if line.strip() != pub_key:
                    file.write(line)

    def get_current_os_user(self) -> str:
        return getpass.getuser()

    def get_host_public_key(self) -> str | None:
        """Return the executor's SSH host public key if available."""
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
