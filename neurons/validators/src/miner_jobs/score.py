import sys
import os
import subprocess
import tempfile
import json
import hashlib
from base64 import b64encode
import asyncio


def gen_hash(s: bytes) -> bytes:
    return b64encode(hashlib.sha256(s).digest(), altchars=b"-_")


payload = sys.argv[1]
data = json.loads(payload)

gpu_count = data["gpu_count"]
num_job_params = data["num_job_params"]
jobs = data["jobs"]
timeout = data["timeout"]


def run_hashcat(device_id: int, job: dict) -> list[str]:
    answers = []
    for i in range(num_job_params):
        payload = job["payloads"][i]
        mask = job["masks"][i]
        algorithm = job["algorithms"][i]

        with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as payload_file:
            payload_file.write(payload.encode('utf-8'))
            payload_file.flush()
            os.fsync(payload_file.fileno())

            if not os.path.exists(f"/usr/bin/hashcat{device_id}"):
                # Use list form instead of shell=True to prevent command injection
                subprocess.check_output(["cp", "/usr/bin/hashcat", f"/usr/bin/hashcat{device_id}"])

            # Use list form instead of shell=True to prevent command injection
            # Note: mask may contain special characters, so we pass it as a separate argument
            cmd = [
                f'hashcat{device_id}',
                f'--session=hashcat{device_id}',
                '--potfile-disable',
                '--restore-disable',
                '--attack-mode', '3',
                '-d', str(device_id),
                '--workload-profile', '3',
                '--optimized-kernel-enable',
                '--hash-type', str(algorithm),
                '--hex-salt',
                '-1', '?l?d?u',
                '--outfile-format', '2',
                '--quiet',
                '--hwmon-disable',
                payload_file.name,
                mask
            ]
            stdout = subprocess.check_output(cmd, text=True)
            if stdout:
                passwords = [p for p in sorted(stdout.split("\n")) if p != ""]
                answers.append(passwords)

    return answers


async def run_jobs():
    tasks = [
        asyncio.to_thread(
            run_hashcat,
            i+1,
            jobs[i]
        )
        for i in range(gpu_count)
    ]

    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    result = {
        "answer": gen_hash("".join([
            "".join([
                "".join(passwords)
                for passwords in answers
            ])
            for answers in results
        ]).encode("utf-8")).decode("utf-8")
    }

    print(json.dumps(result))

if __name__ == "__main__":
    asyncio.run(run_jobs())
