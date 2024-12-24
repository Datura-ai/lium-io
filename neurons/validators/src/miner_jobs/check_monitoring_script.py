from base64 import b64encode
import hashlib
import json
from cryptography.fernet import Fernet
import psutil
import subprocess
import time

def is_script_running(script_name):
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            # Check if the process command line contains the script name
            if script_name in proc.info['cmdline']:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def inject_signature(script_name, signature):

    if not is_script_running(script_name):
        try:
            subprocess.Popen(['python', script_name], 
                           start_new_session=True,  
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Failed to start script: {e}")
            return False

    time.sleep(2)  # Give it 2 seconds to start

    # inject the signature
    try:
        with open(script_name, 'r') as file:
            script_content = file.read()
        script_content = script_content.replace('signature', signature)
        with open(script_name, 'w') as file:
            file.write(script_content)
        return True
    except Exception as e:
        print(f"Failed to inject signature: {e}")
        return False
    
    
def _encrypt(key: str, payload: str) -> str:
    key_bytes = b64encode(hashlib.sha256(key.encode('utf-8')).digest(), altchars=b"-_")
    return Fernet(key_bytes).encrypt(payload.encode("utf-8")).decode("utf-8")

key = 'encrypt_key'
message = 'message'
encoded_signature = _encrypt(key, message)
inject_signature('check_gpu_utility.py',encoded_signature)