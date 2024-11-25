import sys
import os
import subprocess
import tempfile
import json
import hashlib
from base64 import b64encode


def hash(s: bytes) -> bytes:
    return b64encode(hashlib.sha256(s).digest(), altchars=b"-_")


payload = sys.argv[1]
data = json.loads(payload)

answers = []
for i in range(int(data["n"])):
    payload = data["payloads"][i]
    mask = data["masks"][i]
    algorithm = data["algorithms"][i]

    with tempfile.NamedTemporaryFile(delete=True, suffix='.txt') as payload_file:
        payload_file.write(payload.encode('utf-8'))
        payload_file.flush()
        os.fsync(payload_file.fileno())

        cmd = f'hashcat --potfile-disable --restore-disable --attack-mode 3 --workload-profile 3 --optimized-kernel-enable --hash-type {algorithm} --hex-salt -1 "?l?d?u" --outfile-format 2 --quiet {payload_file.name} "{mask}"'
        passwords = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
        passwords = [p for p in sorted(passwords.split("\n")) if p != ""]
        answers.append(passwords)


result = {
    "answer": hash("".join(["".join(passwords) for passwords in answers]).encode("utf-8")).decode("utf-8")
}

print(json.dumps(result))
