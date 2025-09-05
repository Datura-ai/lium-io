import os
import subprocess
import logging

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
    else:
        logger.info(f"Command succeeded: {command}")
    return result

def update_restore_log(
        api_url: str, status: str, logs: list[str], error_message: str, progress: float, auth_token: str, restore_log_id: str
    ):
    import requests
    url = f"{api_url}/restore-logs/{restore_log_id}/progress"
    response = requests.put(
        url, json={"status": status, "logs": logs, "error_message": error_message, "progress": progress},
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    response.raise_for_status()


def pull_aws_cli():
    run_command("/usr/bin/docker pull daturaai/aws-cli")


def aws_restore(args):
    backup_path = (args.backup_path or '').rstrip('/')
    backup_path_parent = os.path.dirname(backup_path)

    command = (
        "docker run --rm "
        f"-v {args.target_volume}:{args.target_volume_path} "
        f"-e AWS_ACCESS_KEY_ID={args.backup_volume_iam_user_access_key} "
        f"-e AWS_SECRET_ACCESS_KEY={args.backup_volume_iam_user_secret_key} "
        "--entrypoint sh "
        "daturaai/aws-cli  -lc "
        f'"aws s3 cp s3://{args.backup_volume_name}/{args.backup_source_path} - '
        f"| tar --xattrs --acls -C {backup_path_parent} -xzf -\""
    )
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
        progress += 30 # 30
        update_restore_log(args.api_url, "IN_PROGRESS", ["Info: Aws cli pulled"], "", progress, args.auth_token, args.restore_log_id)

        logger.info("Step 2: Restoring from aws s3...")
        aws_restore(args)
        logger.info("Restore from aws s3 completed")
        progress += 70 # 100
        update_restore_log(args.api_url, "COMPLETED", ["Info: Restore from aws s3 completed"], "", progress, args.auth_token, args.restore_log_id)
    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        update_restore_log(args.api_url, "FAILED", ["Error: Restore failed"], str(e), progress, args.auth_token, args.restore_log_id)
        raise e


if __name__ == "__main__":
    import argparse
    logger.info("Restore storage script started")

    parser = argparse.ArgumentParser(description="Restore storage script")
    parser.add_argument('--api-url', type=str, help='API URL')
    parser.add_argument('--auth-token', type=str, help='Authentication token')
    parser.add_argument('--backup-volume-name', type=str, help='Backup volume name')
    parser.add_argument('--backup-volume-iam_user_access_key', type=str, help='Backup volume IAM user access key')
    parser.add_argument('--backup-volume-iam_user_secret_key', type=str, help='Backup volume IAM user secret key')
    parser.add_argument('--target-volume', type=str, help='Target volume for restore')
    parser.add_argument('--backup-path', type=str, help='Backup path')
    parser.add_argument('--backup-source-path', type=str, help='Backup source path in S3')
    parser.add_argument('--target-volume-path', type=str, help='Target volume mounted path')
    parser.add_argument('--restore-log-id', type=str, help='Restore log ID')

    args = parser.parse_args()
    restore_storage(args)
    logger.info("Restore storage script completed")