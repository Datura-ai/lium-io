import subprocess
import asyncio

# Approach 1: Using a list of arguments (recommended)
def run_monitoring_script(program_id: str, signature: str, executor_id: str, validator_hotkey: str, public_key: str):
    cmd = [
        "python",
        "src/cli.py",
        "--program_id", program_id,
        "--signature", signature,
        "--executor_id", executor_id,
        "--validator_hotkey", validator_hotkey,
        "--public_key", public_key,
    ]
    
    try:
        result = subprocess.run(cmd, 
                              capture_output=True,  # Capture stdout and stderr
                              text=True,           # Return strings instead of bytes
                              check=True)          # Raise exception on failure
        print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return False

if __name__ == "__main__":
    program_id = "program_id"
    signature = "signature"
    executor_id = "executor_id"
    validator_hotkey = "validator_hotkey"
    public_key = "public_key"
    asyncio.run(run_monitoring_script(program_id, signature, executor_id, validator_hotkey, public_key))
