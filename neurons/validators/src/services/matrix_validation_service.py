import time
import random
from dataclasses import dataclass
from typing import Self

@dataclass
class VerifierParams:
    def __init__(self, dim_n: int = 1000, dim_k: int = 3400, seed: int = 4000):
        self.dim_n = dim_n
        self.dim_k = dim_k
        self.seed = seed
        self.result_path = "/root/validate_result.txt"

    @classmethod
    def generate(cls) -> Self:
        # You can modify the range for more randomness or based on specific needs
        dim_n = random.randint(1000, 2000)  # Random dim_n between 500 and 2000
        dim_k = random.randint(100000, 200000)  # Random dim_k between 2000 and 4000
        seed = int(time.time())

        return cls(
            dim_n=dim_n,
            dim_k=dim_k,
            seed=seed
        )

    def __str__(self) -> str:
        return f"--dim_n {self.dim_n} --dim_k {self.dim_k} --seed {self.seed} --result_path {self.result_path}"

class H100Prover:
    def __init__(self, dim_n, dim_k, seed):
        """Initialize the prover with matrix dimensions and seed."""
        self.dim_n = dim_n
        self.dim_k = dim_k
        self.seed = seed
        self.A = 6364136223846793005
        self.C = 1442695040888963407
        self.M = 9223372036854775807
        self.bandwidth = 0
        self.matrix = []

    def lcg_rand_host(self, seed):
        """Linear congruential generator to simulate random values."""
        return ((self.A * seed + self.C) % self.M) / self.M

    def matrix_mul(self, row, col):
        """Matrix multiplication optimized by reducing redundant calculations."""
        value = 0.0
        division_factor = 1 if self.dim_n <= 100 and self.dim_k <= 100 else 10

        for i in range(self.dim_k):
            seed_a = self.seed + row * self.dim_k + i
            rand_num_a = self.lcg_rand_host(seed_a) / division_factor

            seed_b = self.seed + i * self.dim_n + col + self.dim_k * self.dim_n
            rand_num_b = self.lcg_rand_host(seed_b) / division_factor
            
            value += rand_num_a * rand_num_b
        
        index_factor = (row * self.dim_n + col + value) / (self.dim_n * self.dim_n)
        index_increase = index_factor * self.dim_n * 1.5

        return value + index_increase

    def validate_verification_result(self):
        """Validate the verification result."""
        print("H100Prover: Verifying result...")

        start_time = time.time()

        # Select a random row/column
        min_dim = min(self.dim_n, self.dim_k)
        row = random.randint(0, min_dim - 1)
        col = random.randint(0, min_dim - 1)
        sum_val = round(self.matrix_mul(row, col), 2)

        print(f"Multiplication value on validator side = {sum_val}")
        print(f"Multiplication value from verifier = {self.matrix[row * self.dim_n + col]}")

        end_time = time.time()
        print("Matrix calc Duration = ", (end_time - start_time))

        if abs(self.matrix[row * self.dim_n + col] - sum_val) < 0.001 * self.dim_n:
            print("Verification Results match.")
        else:
            print("ERROR: Verification Failed. Results do not match.")
            return False

        BANDWIDTH_MIN = 1800.0
        BANDWIDTH_MAX = 2004.0
        print(f"Bandwidth = {self.bandwidth}")
        if not (BANDWIDTH_MIN <= self.bandwidth <= BANDWIDTH_MAX):
            print(f"ERROR: Verification Failed for Nvidia H100 GPU: "
                  f"Memory bandwidth {self.bandwidth} GB/s outside expected range ({BANDWIDTH_MIN}-{BANDWIDTH_MAX} GB/s)")
            return False

        print("SUCCESS: Validate Passed for Nvidia H100 GPU")
        return True


    def parse_validate_result_content(self, contents):
        """Read and parse the validation result file."""
        try:
            # Read the dimensions DIM_N and DIM_K
            dim_line = contents[0].strip()
            dimensions_str = dim_line.split("Dimension N: ")[1]
            DIM_N_str, DIM_K_str = dimensions_str.split(", K: ")
            self.dim_n = int(DIM_N_str)
            self.dim_k = int(DIM_K_str)

            # Read the matrix values
            self.matrix = [float(value) for line in contents[2:2 + self.dim_n] for value in line.split()]

            # Read the bandwidth
            bandwidth_line = contents[2 + self.dim_n].strip()
            bandwidth = float(bandwidth_line.split(": ")[1])

            self.bandwidth = bandwidth
        except Exception as e:
            print(f"Error reading the file: {e}")
    