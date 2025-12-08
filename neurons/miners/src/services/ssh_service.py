import getpass
import hashlib
import os
import random
import string
from base64 import b64encode

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


class MinerSSHService:
    def add_pubkey_to_host(self, pub_key: bytes):
        with open(os.path.expanduser("~/.ssh/authorized_keys"), "a") as file:
            file.write(pub_key.decode() + "\n")
            
    def remove_pubkey_from_host(self, pub_key: bytes):
        pub_key_str = pub_key.decode().strip()
        authorized_keys_path = os.path.expanduser("~/.ssh/authorized_keys")

        with open(authorized_keys_path, "r") as file:
            lines = file.readlines()

        with open(authorized_keys_path, "w") as file:
            for line in lines:
                if line.strip() != pub_key_str:
                    file.write(line)

    def get_current_os_user(self) -> str:
        return getpass.getuser()

    def generate_random_string(self, length=30, string_only=False):
        """Generate a random string for encryption keys.

        Args:
            length (int): Length of the random string
            string_only (bool): If True, only use letters; otherwise include digits and special chars

        Returns:
            str: Random string
        """
        if string_only:
            characters = string.ascii_letters
        else:
            characters = string.ascii_letters + string.digits + "/ +_"
        random_string = ''.join(random.choices(characters, k=length))
        return random_string

    def _hash(self, s: bytes) -> bytes:
        """Hash a byte string using SHA256 and base64 encode it.

        Args:
            s (bytes): Input bytes to hash

        Returns:
            bytes: Base64 encoded hash
        """
        return b64encode(hashlib.sha256(s).digest(), altchars=b"-_")

    def _encrypt(self, key: str, payload: str) -> str:
        """Encrypt a payload using Fernet symmetric encryption.

        Args:
            key (str): Encryption key
            payload (str): String to encrypt

        Returns:
            str: Encrypted payload
        """
        key_bytes = self._hash(key.encode("utf-8"))
        return Fernet(key_bytes).encrypt(payload.encode("utf-8")).decode("utf-8")

    def decrypt_payload(self, key: str, encrypted_payload: str) -> str:
        """Decrypt a payload encrypted with _encrypt.

        Args:
            key (str): Decryption key
            encrypted_payload (str): Encrypted string

        Returns:
            str: Decrypted payload
        """
        key_bytes = self._hash(key.encode("utf-8"))
        return Fernet(key_bytes).decrypt(encrypted_payload.encode("utf-8")).decode("utf-8")

    def generate_ssh_key(self, encryption_key: str) -> tuple[bytes, bytes]:
        """Generate SSH key pair using Ed25519 algorithm.

        Args:
            encryption_key (str): Key to encrypt the private key

        Returns:
            tuple[bytes, bytes]: (encrypted private key bytes, public key bytes)
        """
        # Generate a new private-public key pair
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Serialize private key in OpenSSH format without encryption
        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Serialize public key in OpenSSH format
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )

        # Encrypt the private key using the provided encryption key
        encrypted_private_key = self._encrypt(
            encryption_key,
            private_key_bytes.decode("utf-8")
        ).encode("utf-8")

        return encrypted_private_key, public_key_bytes
