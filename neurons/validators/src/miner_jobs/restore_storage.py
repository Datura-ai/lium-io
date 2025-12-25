import logging
import subprocess
import shlex

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("restore_storage")

plugin_name = "s3fs-restore"


def run_command(command):
    # Use shlex.split to safely parse command string and prevent command injection
    if isinstance(command, str):
        cmd_list = shlex.split(command)
    else:
        cmd_list = command
    result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {command}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
    else:
        logger.info(f"Command succeeded: {command}")
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
    run_command("/usr/bin/docker pull daturaai/aws-cli")


def aws_restore(args):
    # aws s3 cp s3://$BUCKET_NAME/backups/my-folder-2025-09-02.tar.gz - \
    # | tar -xzpf - -C $RESTORE_PATH
    # Use shlex.quote to safely escape all user inputs to prevent command injection
    target_volume_quoted = shlex.quote(args.target_volume)
    target_volume_path_quoted = shlex.quote(args.target_volume_path)
    access_key_quoted = shlex.quote(args.backup_volume_iam_user_access_key)
    secret_key_quoted = shlex.quote(args.backup_volume_iam_user_secret_key)
    backup_volume_name_quoted = shlex.quote(args.backup_volume_name)
    backup_source_path_quoted = shlex.quote(args.backup_source_path)
    restore_path_quoted = shlex.quote(args.restore_path)
    
    # Build command as a list to avoid shell injection
    command = [
        "docker", "run", "--rm",
        "-v", f"{target_volume_quoted}:{target_volume_path_quoted}",
        "-e", f"AWS_ACCESS_KEY_ID={access_key_quoted}",
        "-e", f"AWS_SECRET_ACCESS_KEY={secret_key_quoted}",
        "-e", "AWS_DEFAULT_REGION=us-east-1",
        "--entrypoint", "sh",
        "daturaai/aws-cli", "-lc",
        f"aws s3 cp s3://{backup_volume_name_quoted}/{backup_source_path_quoted} - | tar --xattrs --acls -xzpf - -C {restore_path_quoted} --strip-components=1"
    ]
    run_command(command)


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

        logger.info("Step 2: Restoring from aws s3...")
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
