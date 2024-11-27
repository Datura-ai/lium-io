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
        num_letters: int,
        num_digits: int,
        num_hashes: int,
    ) -> Self:
        algorithm = random.choice(list(Algorithm))

        return cls(
            algorithm=algorithm,
            num_letters=num_letters,
            num_digits=num_digits,
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
class HashcatJob:
    passwords: list[list[str]]
    salts: list[bytes]
    job_params: list[JobParam]


@dataclass
class HashService:
    gpu_count: int
    num_job_params: int
    jobs: list[HashcatJob]
    timeout: int

    @classmethod
    def random_string(self, num_letters: int, num_digits: int) -> str:
        return ''.join(random.choices(string.ascii_letters, k=num_letters)) + ''.join(random.choices(string.digits, k=num_digits))

    @classmethod
    def generate(
        cls,
        gpu_count: int = 1,
        timeout: int = 60,
        num_job_params: int = 1,
        num_letters: int = 0,
        num_digits: int = 11,
        num_hashes: int = 10,
        salt_length_bytes: int = 8
    ) -> Self:
        jobs = []
        for _ in range(gpu_count):
            job_params = [
                JobParam.generate(
                    num_letters=num_letters,
                    num_digits=num_digits,
                    num_hashes=num_hashes,
                )
                for _ in range(num_job_params)
            ]

            passwords = [
                sorted(
                    {
                        cls.random_string(
                            num_letters=_params.num_letters, num_digits=_params.num_digits
                        )
                        for _ in range(_params.num_hashes)
                    }
                )
                for _params in job_params
            ]

            salts = [secrets.token_bytes(salt_length_bytes) for _ in range(num_job_params)]

            jobs.append(HashcatJob(
                job_params=job_params,
                passwords=passwords,
                salts=salts,
            ))

        return cls(
            gpu_count=gpu_count,
            num_job_params=num_job_params,
            jobs=jobs,
            timeout=timeout,
        )

    def hash_masks(self, job: HashcatJob) -> list[str]:
        return ["?1" * param.num_letters + "?d" * param.num_digits for param in job.job_params]

    def hash_hexes(self, algorithm: Algorithm, passwords: list[str], salt: str) -> list[str]:
        return [
            algorithm.hash(password.encode("ascii") + salt).hexdigest()
            for password in passwords
        ]

    def _hash(self, s: bytes) -> bytes:
        return b64encode(hashlib.sha256(s).digest(), altchars=b"-_")

    # def _payload(self, i) -> str:
    #     return "\n".join([f"{hash_hex}:{self.salts[i].hex()}" for hash_hex in self.hash_hexes(i)])

    def _payloads(self, job: HashcatJob) -> list[str]:
        payloads = [
            "\n".join([
                f"{hash_hex}:{job.salts[i].hex()}"
                for hash_hex
                in self.hash_hexes(job.job_params[i].algorithm, job.passwords[i], job.salts[i])
            ])
            for i in range(self.num_job_params)
        ]
        return payloads

    @property
    def payload(self) -> str | bytes:
        """Convert this instance to a hashcat argument format."""

        data = {
            "gpu_count": self.gpu_count,
            "num_job_params": self.num_job_params,
            "jobs": [
                {
                    "payloads": self._payloads(job),
                    "masks": self.hash_masks(job),
                    "algorithms": [param.algorithm.type for param in job.job_params],
                }
                for job in self.jobs
            ],
            "timeout": self.timeout,
        }
        return json.dumps(data)

    @property
    def answer(self) -> str:
        return self._hash(
            "".join(["".join(["".join(passwords) for passwords in job.passwords]) for job in self.jobs]).encode("utf-8")
        ).decode("utf-8")

    def __str__(self) -> str:
        return f"JobService {self.jobs}"


if __name__ == "__main__":
    import time

    hash_service = HashService.generate(gpu_count=1, timeout=50)
    # print(hash_service.payload)
    print('answer ====>', hash_service.answer)

    start_time = time.time()

    cmd = f"python src/miner_jobs/score.py '{hash_service.payload}'"
    result = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    end_time = time.time()
    print('result ===>', result)
    print(end_time - start_time)
