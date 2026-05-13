import logging
import os
import subprocess
import tempfile

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("restore_storage")

plugin_name = "s3fs-restore"


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {command}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}\nstderr: {result.stderr}")
    else:
        logger.info(f"Command succeeded: {command}")
    return result


def run_command_args(command: list[str]):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}\nstderr: {result.stderr}")
    logger.info(f"Command succeeded: {' '.join(command)}")
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
    response = requests.put(
        url,
        json={"status": status, "logs": logs, "error_message": error_message, "progress": progress},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    response.raise_for_status()


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
    run_command_args(command)


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
        logger.error(f"aws stderr: {aws_stderr}")
        logger.error(f"tar stdout: {tar_stdout}")
        logger.error(f"tar stderr: {tar_stderr}")
        raise RuntimeError(
            f"Restore failed: aws_status={aws_status}, tar_status={tar_proc.returncode}"
        )

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
        progress += 30  # 30
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

        logger.info("Step 3: Restoring from aws s3...")
        aws_restore(args)
        logger.info("Restore from aws s3 completed")
        progress += 70  # 100
        update_restore_log(
            args.api_url,
            "COMPLETED",
            ["Info: Restore from aws s3 completed"],
            "",
            progress,
            args.auth_token,
            args.restore_log_id,
        )
    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        update_restore_log(
            args.api_url,
            "FAILED",
            ["Error: Restore failed"],
            str(e),
            progress,
            args.auth_token,
            args.restore_log_id,
        )
        raise e


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
