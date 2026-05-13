import os
import subprocess
import logging
import json
import tempfile
import re

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backup_storage")

plugin_name = "s3fs-backup"

# AWS CLI only needs --expected-size for very large stdin uploads. Passing a
# large guessed value for small gzip streams can make aws-cli use multipart
# behavior and return without a usable object, so keep small backups simple.
AWS_CLI_EXPECTED_SIZE_THRESHOLD_BYTES = 50 * 1024 * 1024 * 1024
MAX_ERROR_DETAIL_CHARS = 1200
# These scripts are copied to executor hosts and report status through the public API.
# Some diagnostic path strings can be rejected before they reach the backend, so
# normalize stream paths before sending status updates or writing command logs.
STREAM_PATH_REPLACEMENTS = {
    "/dev/stdout": "stdout",
    "/dev/stderr": "stderr",
    "/dev/stdin": "stdin",
}
GENERIC_BACKUP_FAILURE_MESSAGE = "Backup failed. Detailed error could not be reported; check executor logs."


def redact_sensitive_text(text: str | None) -> str:
    if not text:
        return ""
    # Use one sanitizer for backend payloads and local executor logs so credentials
    # cannot leak and diagnostic text does not block FAILED status updates.
    text = re.sub(r"(AWS_SECRET_ACCESS_KEY|AWSSECRETACCESSKEY)=\S+", r"\1=<redacted>", text)
    for unsafe_path, replacement in STREAM_PATH_REPLACEMENTS.items():
        text = text.replace(unsafe_path, replacement)
    return text


def command_for_log(command: str | list[str]) -> str:
    if isinstance(command, list):
        return redact_sensitive_text(" ".join(command))
    return redact_sensitive_text(command)


def compact_output(label: str, output: str | None, limit: int = 500) -> str:
    output = redact_sensitive_text((output or "").strip())
    if not output:
        return ""
    output = " | ".join(line.strip() for line in output.splitlines() if line.strip())
    if len(output) > limit:
        output = f"{output[:limit]}...<truncated>"
    return f"{label}: {output}"


def upload_failure_message(
    tar_status: int,
    aws_status: int,
    tar_stderr: str | None,
    aws_stdout: str | None,
    aws_stderr: str | None,
) -> str:
    parts = [f"Backup upload failed: tar_status={tar_status}, aws_status={aws_status}"]
    parts.extend(
        detail
        for detail in [
            compact_output("tar stderr", tar_stderr),
            compact_output("aws stdout", aws_stdout),
            compact_output("aws stderr", aws_stderr),
        ]
        if detail
    )
    message = "; ".join(parts)
    if len(message) > MAX_ERROR_DETAIL_CHARS:
        message = f"{message[:MAX_ERROR_DETAIL_CHARS]}...<truncated>"
    return message


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {command_for_log(command)}")
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {command_for_log(command)}\n"
            f"{compact_output('stderr', result.stderr)}"
        )
    else:
        logger.info(f"Command succeeded: {command_for_log(command)}")
    return result


def run_command_args(command: list[str], input_stream=None, stdout=None):
    result = subprocess.run(command, stdin=input_stream, stdout=stdout, capture_output=stdout is None, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {command_for_log(command)}")
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {command_for_log(command)}\n"
            f"{compact_output('stderr', result.stderr)}"
        )
    logger.info(f"Command succeeded: {command_for_log(command)}")
    return result


def create_backup_container(args):
    # install docker volume plugin
    command = f"/usr/bin/docker plugin install mochoa/s3fs-volume-plugin --alias {plugin_name} --grant-all-permissions --disable"
    run_command(command)

    # disable volume plugin and set credential
    command = f"/usr/bin/docker plugin disable {plugin_name} -f"
    run_command(command)
    command = f"/usr/bin/docker plugin set {plugin_name} AWSACCESSKEYID={args.backup_volume_iam_user_access_key} AWSSECRETACCESSKEY={args.backup_volume_iam_user_secret_key}"
    run_command(command)
    command = f'/usr/bin/docker plugin set s3fs-backup DEFAULT_S3FSOPTS="url=https://s3-accelerate.amazonaws.com"'
    run_command(command)
    command = f"/usr/bin/docker plugin enable {plugin_name}"
    run_command(command)

    # clean up all existing volumes and containers for s3 backup 
    # Find and remove all containers whose names start with s3fs-backup
    list_cmd = "/usr/bin/docker ps -a --filter 'name=^/s3fs-backup' --format '{{.Names}}'"
    result = run_command(list_cmd)
    container_names = [name for name in result.stdout.strip().split('\n') if name]
    if container_names:
        names_str = " ".join(container_names)
        rm_cmd = f"/usr/bin/docker rm -f {names_str}"
        run_command(rm_cmd)

    # Find and remove all Docker volumes whose names start with "celium-backup-volume"
    list_volumes_cmd = "/usr/bin/docker volume ls --format '{{.Name}}'"
    result = run_command(list_volumes_cmd)
    volume_names = [name for name in result.stdout.strip().split('\n') if name.startswith("celium-backup-volume")]
    if volume_names:
        names_str = " ".join(volume_names)
        rm_volumes_cmd = f"/usr/bin/docker volume rm {names_str}"
        run_command(rm_volumes_cmd)
    
    # create volume
    volume_name = args.backup_volume_name
    command = " ".join([
        "/usr/bin/docker", "volume", "create", "-d", plugin_name, volume_name,
    ])
    run_command(command)

    # create s3fs backup container
    container_name = f"s3fs-backup-{args.backup_log_id}"
    command = (
        "/usr/bin/docker run -d "
        f"--name {container_name} "
        f"-v {volume_name}:/mnt "
        f"-v {args.source_volume}:{args.source_volume_path} "
        f"--entrypoint bash "
        f'ubuntu -c "tail -f /dev/null"'
    )
    run_command(command)
    return container_name


def start_backup(args, container_name):
    # Run cp command inside the container to copy from source_volume_path to /mnt
    run_command(f"/usr/bin/docker exec {container_name} sh -lc 'mkdir -p /mnt/{args.backup_target_path}'")
    command = f"/usr/bin/docker exec {container_name} sh -lc 'cp -a {args.backup_path} /mnt/{args.backup_target_path}'"
    run_command(command)


def clean_backup_container(container_name):
    command = f"/usr/bin/docker rm -f {container_name}"
    run_command(command)

def clean_backup_volume(volume_name):
    command = f"/usr/bin/docker volume rm {volume_name}"
    run_command(command)

def disable_backup_volume_plugin():
    command = f"/usr/bin/docker plugin disable {plugin_name} -f"
    run_command(command)


def update_backup_log(
        api_url: str,
        backup_log_id: str,
        status: str,
        logs: list[str],
        error_message: str,
        progress: float,
        auth_token: str,
        s3_metadata: dict | None = None,
        estimated_backup_size_bytes: int | None = None,
    ):
    import requests

    payload = {
        "status": status,
        "logs": [redact_sensitive_text(log) for log in logs],
        "error_message": redact_sensitive_text(error_message),
        "progress": progress,
    }
    # Size estimate is sent before upload so API/UI can show expected backup size while the job is running.
    if estimated_backup_size_bytes is not None:
        payload["estimated_backup_size_bytes"] = estimated_backup_size_bytes
    if s3_metadata:
        payload.update(
            {
                "s3_content_length": s3_metadata.get("ContentLength"),
                "s3_last_modified": s3_metadata.get("LastModified"),
                "s3_etag": s3_metadata.get("ETag"),
            }
        )

    url = f"{api_url}/backup-logs/{backup_log_id}/progress"
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = requests.put(
        url, json=payload,
        headers=headers,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        if status != "FAILED":
            raise
        # If a future diagnostic string is still rejected by the API edge, preserve
        # state correctness by marking the job failed with a known-safe message.
        logger.warning("Detailed backup failure update was rejected; retrying with a generic error message")
        fallback_payload = dict(payload)
        fallback_payload["error_message"] = GENERIC_BACKUP_FAILURE_MESSAGE
        fallback_response = requests.put(url, json=fallback_payload, headers=headers)
        fallback_response.raise_for_status()


def pull_aws_cli():
    run_command_args(["/usr/bin/docker", "pull", "daturaai/aws-cli"])


def docker_base_command(args, volumes=None, entrypoint=None, interactive=False):
    command = ["/usr/bin/docker", "run", "--rm"]
    if interactive:
        command.append("-i")
    for volume in volumes or []:
        command.extend(["-v", volume])
    if entrypoint:
        command.extend(["--entrypoint", entrypoint])
    command.extend(
        [
            "-e", f"AWS_ACCESS_KEY_ID={args.backup_volume_iam_user_access_key}",
            "-e", f"AWS_SECRET_ACCESS_KEY={args.backup_volume_iam_user_secret_key}",
            "-e", "AWS_DEFAULT_REGION=us-east-1",
            "daturaai/aws-cli",
        ]
    )
    return command


def estimate_backup_sizes(args, backup_path) -> tuple[int, int]:
    source_volume = f"{args.source_volume}:{args.source_volume_path}"
    # Run du inside Docker with the workload volume mounted; the miner host may not have this path.
    du_command = docker_base_command(args, volumes=[source_volume], entrypoint="du") + ["-sb", backup_path]
    try:
        result = run_command_args(du_command)
        source_bytes = int(result.stdout.split()[0])
    except Exception:
        result = run_command_args(docker_base_command(args, volumes=[source_volume], entrypoint="du") + ["-sk", backup_path])
        source_bytes = int(result.stdout.split()[0]) * 1024

    find_command = docker_base_command(args, volumes=[source_volume], entrypoint="find") + [backup_path, "-print"]
    entry_count = count_command_output_lines(find_command)
    entry_count = max(entry_count, 1)

    # source_bytes is the user-facing estimate. expected_upload_size is only for AWS CLI's
    # streamed multipart upload planner and intentionally has a safety cushion.
    cushion = max(source_bytes // 20, 100 * 1024 * 1024)
    expected_upload_size = source_bytes + cushion + entry_count * 4096
    return source_bytes, expected_upload_size


def estimate_expected_size(args, backup_path):
    _, expected_upload_size = estimate_backup_sizes(args, backup_path)
    return expected_upload_size


def count_command_output_lines(command: list[str]) -> int:
    # Stream line counting instead of materializing potentially huge find output in memory.
    with tempfile.TemporaryFile(mode="w+") as stderr_file:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file, text=True)
        count = 0
        try:
            for _ in proc.stdout:
                count += 1
            status = proc.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read()
        except Exception:
            proc.kill()
            raise

    if status != 0:
        logger.error(f"Command failed: {command_for_log(command)}")
        raise RuntimeError(
            f"Command failed with exit code {status}: {command_for_log(command)}\n"
            f"{compact_output('stderr', stderr)}"
        )
    logger.info(f"Command succeeded: {command_for_log(command)}")
    return count


def aws_cp(args, expected_size: int | None = None):
    backup_path = os.path.expanduser((args.backup_path or '').rstrip('/'))
    if not backup_path:
        raise ValueError("Backup path is required")

    backup_path_parent = os.path.dirname(backup_path) or "."
    backup_path_current = os.path.basename(backup_path)
    source_volume = f"{args.source_volume}:{args.source_volume_path}"

    # tar runs inside Docker with the source volume mounted and writes the archive to stdout.
    tar_command = docker_base_command(args, volumes=[source_volume], entrypoint="tar") + [
        "--xattrs",
        "--acls",
        "-C",
        backup_path_parent,
        "-czf",
        "-",
        backup_path_current,
    ]
    # AWS CLI reads stdin and uploads directly to S3; no local archive is created on miner disk.
    aws_command = docker_base_command(args, entrypoint="aws", interactive=True) + [
        "s3",
        "cp",
        "-",
        f"s3://{args.backup_volume_name}/{args.backup_target_path}",
        "--sse",
        "AES256",
    ]
    if expected_size is not None:
        aws_command.extend(["--expected-size", str(expected_size)])

    logger.info("Starting tar-to-S3 streaming backup")
    aws_proc = None
    with tempfile.TemporaryFile() as tar_stderr_file:
        # Keep tar stderr in a temp file so verbose warnings cannot fill a pipe and deadlock the stream.
        tar_proc = subprocess.Popen(tar_command, stdout=subprocess.PIPE, stderr=tar_stderr_file)
        try:
            # AWS reads tar stdout directly; no archive is written to local disk.
            aws_proc = subprocess.Popen(
                aws_command,
                stdin=tar_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            tar_proc.stdout.close()
            aws_stdout, aws_stderr = aws_proc.communicate()
            tar_status = tar_proc.wait()
            tar_stderr_file.seek(0)
            tar_stderr = tar_stderr_file.read().decode(errors="replace")
        except Exception:
            tar_proc.kill()
            if aws_proc:
                aws_proc.kill()
            raise

    if tar_status != 0 or aws_proc.returncode != 0:
        raise RuntimeError(upload_failure_message(tar_status, aws_proc.returncode, tar_stderr, aws_stdout, aws_stderr))

    logger.info("Tar-to-S3 streaming backup completed")


def aws_head_object(args):
    # A successful upload alone is not enough; completion requires S3 to acknowledge the object exists.
    command = docker_base_command(args, entrypoint="aws") + [
        "s3api",
        "head-object",
        "--bucket",
        args.backup_volume_name,
        "--key",
        args.backup_target_path,
        "--output",
        "json",
    ]
    result = run_command_args(command)
    return json.loads(result.stdout)


def backup_storage(args):
    progress = 0
    try:
        logger.info("=" * 70)
        logger.info("Environment variables:")
        logger.info("=" * 70)

        # logger.info(f"SOURCE_VOLUME: {args.source_volume}")
        # logger.info(f"BACKUP_VOLUME_NAME: {args.backup_volume_name}")
        # logger.info(f"BACKUP_VOLUME_IAM_USER_ACCESS_KEY: {args.backup_volume_iam_user_access_key}")
        # logger.info(f"BACKUP_VOLUME_IAM_USER_SECRET_KEY: {args.backup_volume_iam_user_secret_key}")
        # logger.info(f"BACKUP_PATH: {args.backup_path}")
        # logger.info(f"AUTH_TOKEN: {args.auth_token}")
        # logger.info(f"BACKUP_LOG_ID: {args.backup_log_id}")

        # logger.info("Step 1: Creating backup container...")
        # container_name = create_backup_container(args)
        # logger.info(f"Backup container created: {container_name}")
        # progress += 10 # 10
        # update_backup_log(args.api_url, "IN_PROGRESS", ["Info: Backup container created"], "", progress, args.auth_token)

        # logger.info("Step 2: Starting backup...")
        # start_backup(args, container_name)
        # logger.info("Backup started")
        # progress += 20 # 30
        # update_backup_log(args.api_url, "IN_PROGRESS", ["Info: Backup started"], "", progress, args.auth_token)

        # logger.info("Step 3: Cleanup backup container...")
        # clean_backup_container(container_name)
        # logger.info("Backup container cleaned")
        # progress += 50 # 80
        # update_backup_log(args.api_url, "IN_PROGRESS", ["Info: Backup container cleaned"], "", progress, args.auth_token)

        # logger.info("Step 4: Cleanup backup volume...")
        # clean_backup_volume(args.backup_volume_name)
        # logger.info("Backup volume cleaned")
        # progress += 10 # 90
        # update_backup_log(args.api_url, "IN_PROGRESS", ["Info: Backup volume cleaned"], "", progress, args.auth_token)

        # logger.info("Step 5: Disable backup volume plugin...")
        # disable_backup_volume_plugin()
        # logger.info("Backup volume plugin disabled")
        # progress += 10 # 100
        # update_backup_log(args.api_url, "COMPLETED", [], "", progress, args.auth_token)

        logger.info("Step 1: Pulling aws cli...")
        pull_aws_cli()
        logger.info("Aws cli pulled")
        progress = 10
        update_backup_log(args.api_url, args.backup_log_id, "IN_PROGRESS", ["Info: Aws cli pulled"], "", progress, args.auth_token)

        backup_path = os.path.expanduser((args.backup_path or '').rstrip('/'))
        if not backup_path:
            raise ValueError("Backup path is required")
        logger.info("Step 2: Estimating backup size...")
        estimated_backup_size_bytes, expected_upload_size = estimate_backup_sizes(args, backup_path)
        progress = 20
        update_backup_log(
            args.api_url,
            args.backup_log_id,
            "IN_PROGRESS",
            ["Info: Backup size estimated"],
            "",
            progress,
            args.auth_token,
            estimated_backup_size_bytes=estimated_backup_size_bytes,
        )

        logger.info("Step 3: Copying to aws s3...")
        expected_size_arg = (
            expected_upload_size
            if estimated_backup_size_bytes >= AWS_CLI_EXPECTED_SIZE_THRESHOLD_BYTES
            else None
        )
        aws_cp(args, expected_size=expected_size_arg)
        logger.info("Copying to aws s3 completed")
        progress = 90
        update_backup_log(
            args.api_url,
            args.backup_log_id,
            "IN_PROGRESS",
            ["Info: Copying to aws s3 completed"],
            "",
            progress,
            args.auth_token,
        )

        logger.info("Step 4: Verifying aws s3 object...")
        s3_metadata = aws_head_object(args)
        logger.info("Aws s3 object verified")

        progress = 100
        update_backup_log(
            args.api_url,
            args.backup_log_id,
            "COMPLETED",
            ["Info: Aws s3 object verified"],
            "",
            progress,
            args.auth_token,
            s3_metadata=s3_metadata,
        )
    except Exception as e:
        logger.error(f"Backup failed: {e}", exc_info=True)
        try:
            update_backup_log(args.api_url, args.backup_log_id, "FAILED", ["Error: Backup failed"], str(e), progress, args.auth_token)
        except Exception:
            logger.error("Failed to update backup log after backup failure", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    logger.info("Backup storage script started")

    parser = argparse.ArgumentParser(description="Backup storage script")
    parser.add_argument('--source-volume', type=str, help='Source volume to backup')
    parser.add_argument('--backup-path', type=str, help='Backup path')
    parser.add_argument('--api-url', type=str, help='API URL')
    parser.add_argument('--auth-token', type=str, help='Authentication token')
    parser.add_argument('--source-volume-path', type=str, help='Source volume path')
    parser.add_argument('--backup-target-path', type=str, help='Backup target path')
    parser.add_argument('--backup-log-id', type=str, help='Backup log ID')
    parser.add_argument('--backup-volume-name', type=str, help='Backup volume name')
    parser.add_argument('--backup-volume-iam_user_access_key', type=str, help='Backup volume IAM user access key')
    parser.add_argument('--backup-volume-iam_user_secret_key', type=str, help='Backup volume IAM user secret key')

    args = parser.parse_args()
    backup_storage(args)
    logger.info("Backup storage script completed")
