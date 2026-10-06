import os
import subprocess
import logging
import json
import tempfile
import re

from workspace_mount import VolumeAccess, detect_volume_access

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backup_storage")

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
    text = re.sub(
        r"(AWS_SECRET_ACCESS_KEY|AWSSECRETACCESSKEY)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s]+)",
        r"\1=<redacted>",
        text,
    )
    for unsafe_path, replacement in STREAM_PATH_REPLACEMENTS.items():
        text = text.replace(unsafe_path, replacement)
    return text


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


def run_command(command, command_label: str = "command"):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {command_label}")
        raise RuntimeError(
            f"{command_label} failed with exit code {result.returncode}\n"
            f"{compact_output('stderr', result.stderr)}"
        )
    else:
        logger.info(f"Command succeeded: {command_label}")
    return result


def run_command_args(command: list[str], input_stream=None, stdout=None, input_data=None, command_label: str = "command"):
    result = subprocess.run(
        command,
        stdin=input_stream,
        stdout=stdout,
        input=input_data,
        capture_output=stdout is None,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Command failed: {command_label}")
        raise RuntimeError(
            f"{command_label} failed with exit code {result.returncode}\n"
            f"{compact_output('stderr', result.stderr)}"
        )
    logger.info(f"Command succeeded: {command_label}")
    return result


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
    run_command_args(["/usr/bin/docker", "pull", "daturaai/aws-cli"], command_label="docker pull daturaai/aws-cli")


def docker_base_command(args, volumes=None, volume_args=None, entrypoint=None, interactive=False):
    command = ["/usr/bin/docker", "run", "--rm"]
    if interactive:
        command.append("-i")
    for volume in volumes or []:
        command.extend(["-v", volume])
    command.extend(volume_args or [])
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


def workspace_command(args, volume_access: VolumeAccess, entrypoint: str, interactive: bool = False) -> list[str]:
    if volume_access.encrypted:
        return volume_access.docker_exec_args(entrypoint, interactive=interactive)
    return docker_base_command(
        args,
        volume_args=volume_access.docker_run_args(),
        entrypoint=entrypoint,
        interactive=interactive,
    )


def estimate_backup_sizes(args, volume_access: VolumeAccess, backup_path: str) -> tuple[int, int]:
    # Run du inside Docker with the workload volume mounted; the miner host may not have this path.
    du_command = workspace_command(args, volume_access, "du") + ["-sb", backup_path]
    try:
        result = run_command_args(du_command, command_label="docker run du -sb <backup_path>")
        source_bytes = int(result.stdout.split()[0])
    except Exception:
        result = run_command_args(
            workspace_command(args, volume_access, "du") + ["-sk", backup_path],
            command_label="docker run du -sk <backup_path>",
        )
        source_bytes = int(result.stdout.split()[0]) * 1024

    find_command = workspace_command(args, volume_access, "find") + [backup_path, "-print"]
    entry_count = count_command_output_lines(find_command, command_label="docker run find <backup_path> -print")
    entry_count = max(entry_count, 1)

    # source_bytes is the user-facing estimate. expected_upload_size is only for AWS CLI's
    # streamed multipart upload planner and intentionally has a safety cushion.
    cushion = max(source_bytes // 20, 100 * 1024 * 1024)
    expected_upload_size = source_bytes + cushion + entry_count * 4096
    return source_bytes, expected_upload_size


def estimate_expected_size(args, volume_access: VolumeAccess, backup_path: str):
    _, expected_upload_size = estimate_backup_sizes(args, volume_access, backup_path)
    return expected_upload_size


def count_command_output_lines(command: list[str], command_label: str = "command") -> int:
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
        logger.error(f"Command failed: {command_label}")
        raise RuntimeError(
            f"{command_label} failed with exit code {status}\n"
            f"{compact_output('stderr', stderr)}"
        )
    logger.info(f"Command succeeded: {command_label}")
    return count


def aws_cp(args, volume_access: VolumeAccess, backup_path: str, expected_size: int | None = None):
    if not backup_path:
        raise ValueError("Backup path is required")

    backup_path_parent = os.path.dirname(backup_path) or "."
    backup_path_current = os.path.basename(backup_path)

    # tar runs inside Docker with the source volume mounted and writes the archive to stdout.
    tar_command = workspace_command(args, volume_access, "tar") + [
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
    result = run_command_args(command, command_label="docker run aws s3api head-object")
    return json.loads(result.stdout)


def backup_storage(args):
    progress = 0
    try:
        volume_access = detect_volume_access(args.source_volume, args.source_volume_path)

        logger.info("=" * 70)
        logger.info("Environment variables:")
        logger.info("=" * 70)

        logger.info("Step 1: Preparing workspace and pulling aws cli...")
        pull_aws_cli()
        logger.info("Aws cli pulled")
        progress = 10
        update_backup_log(args.api_url, args.backup_log_id, "IN_PROGRESS", ["Info: Aws cli pulled"], "", progress, args.auth_token)

        backup_path = volume_access.normalized_path(args.backup_path)
        if not backup_path:
            raise ValueError("Backup path is required")
        logger.info("Step 2: Estimating backup size...")
        estimated_backup_size_bytes, expected_upload_size = estimate_backup_sizes(args, volume_access, backup_path)
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
        progress = 30
        update_backup_log(
            args.api_url,
            args.backup_log_id,
            "IN_PROGRESS",
            ["Info: Backup upload started"],
            "",
            progress,
            args.auth_token,
        )
        aws_cp(args, volume_access, backup_path, expected_size=expected_size_arg)
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
