import logging
import os
import subprocess
import tempfile
import re

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("restore_storage")

plugin_name = "s3fs-restore"
MAX_ERROR_DETAIL_CHARS = 1200
# These scripts are copied to executor hosts and report status through the public API.
# Some diagnostic path strings can be rejected before they reach the backend, so
# normalize stream paths before sending status updates or writing command logs.
STREAM_PATH_REPLACEMENTS = {
    "/dev/stdout": "stdout",
    "/dev/stderr": "stderr",
    "/dev/stdin": "stdin",
}
GENERIC_RESTORE_FAILURE_MESSAGE = "Restore failed. Detailed error could not be reported; check executor logs."


def sanitize_report_text(text: str | None) -> str:
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
    output = sanitize_report_text((output or "").strip())
    if not output:
        return ""
    output = " | ".join(line.strip() for line in output.splitlines() if line.strip())
    if len(output) > limit:
        output = f"{output[:limit]}...<truncated>"
    return f"{label}: {output}"


def restore_failure_message(
    aws_status: int,
    tar_status: int,
    aws_stderr: str | None,
    tar_stdout: str | None,
    tar_stderr: str | None,
) -> str:
    parts = [f"Restore failed: aws_status={aws_status}, tar_status={tar_status}"]
    parts.extend(
        detail
        for detail in [
            compact_output("aws stderr", aws_stderr),
            compact_output("tar stdout", tar_stdout),
            compact_output("tar stderr", tar_stderr),
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


def run_command_args(command: list[str], command_label: str = "command"):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {command_label}")
        raise RuntimeError(
            f"{command_label} failed with exit code {result.returncode}\n"
            f"{compact_output('stderr', result.stderr)}"
        )
    logger.info(f"Command succeeded: {command_label}")
    return result


def update_restore_log(
    api_url: str,
    status: str,
    logs: list[str],
    error_message: str,
    progress: float,
    auth_token: str,
    restore_log_id: str,
):
    import requests

    url = f"{api_url}/restore-logs/{restore_log_id}/progress"
    payload = {
        "status": status,
        "logs": [sanitize_report_text(log) for log in logs],
        "error_message": sanitize_report_text(error_message),
        "progress": progress,
    }
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = requests.put(
        url,
        json=payload,
        headers=headers,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        if status != "FAILED":
            raise
        # If a future diagnostic string is still rejected by the API edge, preserve
        # state correctness by marking the job failed with a known-safe message.
        logger.warning("Detailed restore failure update was rejected; retrying with a generic error message")
        fallback_payload = dict(payload)
        fallback_payload["error_message"] = GENERIC_RESTORE_FAILURE_MESSAGE
        fallback_response = requests.put(url, json=fallback_payload, headers=headers)
        fallback_response.raise_for_status()


def pull_aws_cli():
    run_command_args(["/usr/bin/docker", "pull", "daturaai/aws-cli"], command_label="docker pull daturaai/aws-cli")


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


def aws_head_object(args):
    command = docker_base_command(args, entrypoint="aws") + [
        "s3api",
        "head-object",
        "--bucket",
        args.backup_volume_name,
        "--key",
        args.backup_source_path,
        "--output",
        "json",
    ]
    run_command_args(command, command_label="docker run aws s3api head-object")


def aws_restore(args):
    # aws s3 cp s3://$BUCKET_NAME/backups/my-folder-2025-09-02.tar.gz - \
    # | tar -xzpf - -C $RESTORE_PATH
    restore_path = os.path.expanduser(args.restore_path)
    aws_command = docker_base_command(args, entrypoint="aws") + [
        "s3",
        "cp",
        f"s3://{args.backup_volume_name}/{args.backup_source_path}",
        "-",
    ]
    tar_command = docker_base_command(
        args,
        volumes=[f"{args.target_volume}:{args.target_volume_path}"],
        entrypoint="tar",
        interactive=True,
    ) + [
        "--xattrs",
        "--acls",
        "-xzpf",
        "-",
        "-C",
        restore_path,
        "--strip-components=1",
    ]

    logger.info("Starting S3-to-tar streaming restore")
    tar_proc = None
    with tempfile.TemporaryFile() as aws_stderr_file:
        aws_proc = subprocess.Popen(aws_command, stdout=subprocess.PIPE, stderr=aws_stderr_file)
        try:
            # tar reads the S3 object stream directly; no local archive is written.
            tar_proc = subprocess.Popen(
                tar_command,
                stdin=aws_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            aws_proc.stdout.close()
            tar_stdout, tar_stderr = tar_proc.communicate()
            aws_status = aws_proc.wait()
            aws_stderr_file.seek(0)
            aws_stderr = aws_stderr_file.read().decode(errors="replace")
        except Exception:
            aws_proc.kill()
            if tar_proc:
                tar_proc.kill()
            raise

    if aws_status != 0 or tar_proc.returncode != 0:
        raise RuntimeError(restore_failure_message(aws_status, tar_proc.returncode, aws_stderr, tar_stdout, tar_stderr))

    logger.info("S3-to-tar streaming restore completed")


def restore_storage(args):
    progress = 0
    try:
        logger.info("=" * 70)
        logger.info("Restore operation started")
        logger.info("=" * 70)

        logger.info("Step 1: Pulling aws cli...")
        pull_aws_cli()
        logger.info("Aws cli pulled")
        progress = 10
        update_restore_log(
            args.api_url,
            "IN_PROGRESS",
            ["Info: Aws cli pulled"],
            "",
            progress,
            args.auth_token,
            args.restore_log_id,
        )

        logger.info("Step 2: Verifying aws s3 object...")
        aws_head_object(args)
        logger.info("Aws s3 object verified")
        progress = 20
        update_restore_log(
            args.api_url,
            "IN_PROGRESS",
            ["Info: Restore source object verified"],
            "",
            progress,
            args.auth_token,
            args.restore_log_id,
        )

        logger.info("Step 3: Restoring from aws s3...")
        progress = 30
        update_restore_log(
            args.api_url,
            "IN_PROGRESS",
            ["Info: Restore stream started"],
            "",
            progress,
            args.auth_token,
            args.restore_log_id,
        )
        aws_restore(args)
        logger.info("Restore from aws s3 completed")
        progress = 90
        update_restore_log(
            args.api_url,
            "IN_PROGRESS",
            ["Info: Restore from aws s3 completed"],
            "",
            progress,
            args.auth_token,
            args.restore_log_id,
        )

        progress = 100
        update_restore_log(
            args.api_url,
            "COMPLETED",
            ["Info: Restore completed"],
            "",
            progress,
            args.auth_token,
            args.restore_log_id,
        )
    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        try:
            update_restore_log(
                args.api_url,
                "FAILED",
                ["Error: Restore failed"],
                str(e),
                progress,
                args.auth_token,
                args.restore_log_id,
            )
        except Exception:
            logger.error("Failed to update restore log after restore failure", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse

    logger.info("Restore storage script started")

    parser = argparse.ArgumentParser(description="Restore storage script")
    parser.add_argument("--api-url", type=str, help="API URL")
    parser.add_argument("--auth-token", type=str, help="Authentication token")
    parser.add_argument("--backup-volume-name", type=str, help="Backup volume name")
    parser.add_argument(
        "--backup-volume-iam_user_access_key", type=str, help="Backup volume IAM user access key"
    )
    parser.add_argument(
        "--backup-volume-iam_user_secret_key", type=str, help="Backup volume IAM user secret key"
    )
    parser.add_argument("--target-volume", type=str, help="Target volume for restore")
    parser.add_argument("--backup-source-path", type=str, help="Backup source path in S3")
    parser.add_argument("--restore-path", type=str, help="Restore path")
    parser.add_argument("--target-volume-path", type=str, help="Target volume mounted path")
    parser.add_argument("--restore-log-id", type=str, help="Restore log ID")

    args = parser.parse_args()
    restore_storage(args)
    logger.info("Restore storage script completed")
