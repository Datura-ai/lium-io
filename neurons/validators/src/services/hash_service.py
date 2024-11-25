import enum
import hashlib
import string
import random
import secrets
from dataclasses import dataclass
from base64 import b64encode
import json
from typing import Self
import subprocess


class Algorithm(enum.Enum):
    SHA256 = "SHA256"
    SHA384 = "SHA384"
    SHA512 = "SHA512"

    @property
    def params(self):
        return {
            Algorithm.SHA256: {
                "hash_function": hashlib.sha256,
                "hash_type": "1410",
            },
            Algorithm.SHA384: {
                "hash_function": hashlib.sha384,
                "hash_type": "10810",
            },
            Algorithm.SHA512: {
                "hash_function": hashlib.sha512,
                "hash_type": "1710",
            },
        }

    def hash(self, *args, **kwargs):
        return self.params[self]["hash_function"](*args, **kwargs)

    @property
    def type(self):
        return self.params[self]["hash_type"]


@dataclass
class JobParam:
    algorithm: Algorithm
    num_letters: int
    num_digits: int
    num_hashes: int

    @classmethod
    def generate(
        cls,
        num_hashes: int = 100
    ) -> Self:
        algorithm = random.choice(list(Algorithm))

        return cls(
            algorithm=algorithm,
            num_letters=random.randint(5, 6),
            num_digits=random.randint(1, 2),
            num_hashes=num_hashes,
        )

    @property
    def password_length(self) -> int:
        return self.num_letters + self.num_digits

    def __str__(self) -> str:
        return (
            f"algorithm={self.algorithm} "
            f"algorithm_type={self.algorithm.type} "
            f"num_letters={self.num_letters} "
            f"num_digits={self.num_digits} "
            f"num_hashes={self.num_hashes}"
        )


@dataclass
class HashService:
    job_params_count: int
    passwords: list[list[str]]
    salts: list[bytes]
    job_params: list[JobParam]

    @classmethod
    def random_string(self, num_letters: int, num_digits: int) -> str:
        return ''.join(random.choices(string.ascii_letters, k=num_letters)) + ''.join(random.choices(string.digits, k=num_digits))

    @classmethod
    def generate(
        cls, job_params_count: int = 3, num_hashes: int = 100, salt_length_bytes: int = 8
    ) -> Self:
        # generate distinct passwords for each algorithm
        job_params = [JobParam.generate(num_hashes=num_hashes) for _ in range(job_params_count)]

        passwords = []
        for _params in job_params:
            _passwords = set()
            while len(_passwords) < _params.num_hashes:
                _passwords.add(
                    cls.random_string(
                        num_letters=_params.num_letters, num_digits=_params.num_digits
                    )
                )
            passwords.append(sorted(list(_passwords)))

        return cls(
            job_params_count=job_params_count,
            job_params=job_params,
            passwords=passwords,
            salts=[secrets.token_bytes(salt_length_bytes) for _ in range(job_params_count)],
        )

    def hash_masks(self) -> list[str]:
        return ["?1" * param.num_letters + "?d" * param.num_digits for param in self.job_params]

    def hash_hexes(self, i) -> list[str]:
        return [
            self.job_params[i].algorithm.hash(password.encode("ascii") + self.salts[i]).hexdigest()
            for password in self.passwords[i]
        ]

    def _hash(self, s: bytes) -> bytes:
        return b64encode(hashlib.sha256(s).digest(), altchars=b"-_")

    def _payload(self, i) -> str:
        return "\n".join([f"{hash_hex}:{self.salts[i].hex()}" for hash_hex in self.hash_hexes(i)])

    def _payloads(self) -> list[str]:
        payloads = []
        for i in range(self.job_params_count):
            payloads.append(self._payload(i))
        return payloads

    @property
    def payload(self) -> str | bytes:
        """Convert this instance to a hashcat argument format."""

        data = {
            "n": self.job_params_count,
            "payloads": self._payloads(),
            "masks": self.hash_masks(),
            "algorithms": [param.algorithm.type for param in self.job_params],
            "num_letters": [param.num_letters for param in self.job_params],
            "num_digits": [param.num_digits for param in self.job_params],
        }
        return json.dumps(data)

    @property
    def answer(self) -> str:
        return self._hash(
            "".join(["".join(passwords) for passwords in self.passwords]).encode("utf-8")
        ).decode("utf-8")

    def __str__(self) -> str:
        return f"JobService {self.job_params}"


if __name__ == "__main__":
    import time

    hash_service = HashService.generate(num_hashes=1)
    print(hash_service.payload)
    print(hash_service.answer)

    start_time = time.time()

    cmd = f"python dist/score.py '{hash_service.payload}'"
    result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    end_time = time.time()
    print('result ===>', result)
    print(end_time - start_time)
    # # print(job.algorithms, job.passwords, job.salts, job.params)
    # print(job.payload)
    # # print(job.raw_script())
    # # print(f"Payload: {job.payload}")
    # print(f"Answer: {job.answer}") fSCQBEvrW6wTYMbh1vO66l1K_Jqx5efRU8ofjAcA2pQ=
