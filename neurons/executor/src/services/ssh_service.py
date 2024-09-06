import getpass
import os


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
