import os
import subprocess
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backup_storage")

plugin_name = "s3fs-backup"


def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {command}")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
    else:
        logger.info(f"Command succeeded: {command}")
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
        api_url: str, status: str, logs: list[str], error_message: str, progress: float, auth_token: str
    ):
    import requests
    url = f"{api_url}/backup-logs/{args.backup_log_id}/progress"
    response = requests.put(
        url, json={"status": status, "logs": logs, "error_message": error_message, "progress": progress},
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    response.raise_for_status()


def pull_aws_cli():
    run_command("/usr/bin/docker pull amazon/aws-cli")


def get_total_size(args):
    try:
        comand = (
            f"docker run --rm -v {args.source_volume}:{args.source_volume_path} "
            f' ubuntu bash -lc "du -sb {args.source_volume_path}'
             " | awk '{print \$1}' \" " 
        )
        result = run_command(comand)
        logger.info(f"Total Size: {result.stdout.strip()}")
        return int(result.stdout.strip())
    except Exception as e:
        logger.error(f"Failed to get total size: {e}", exc_info=True)
        return None



def aws_cp(args, size: int | None = None):
    expected_size_flag = f' --expected-size {size} ' if size else ''
    command = (
        "docker run --rm "
        f"-v {args.source_volume}:{args.source_volume_path} "
        f"-e AWS_ACCESS_KEY_ID={args.backup_volume_iam_user_access_key} "
        f"-e AWS_SECRET_ACCESS_KEY={args.backup_volume_iam_user_secret_key} "
        "--entrypoint sh "
        "amazon/aws-cli -lc "
        f'"aws s3 cp {args.backup_path} s3://{args.backup_volume_name}/{args.backup_target_path} '
        f' --recursive --only-show-errors {expected_size_flag} --endpoint-url https://s3-accelerate.amazonaws.com"'
    )
    run_command(command)


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
        progress += 30 # 30
        update_backup_log(args.api_url, "IN_PROGRESS", ["Info: Aws cli pulled"], "", progress, args.auth_token)

        logger.info("Step 2: Getting total size...")
        total_size = get_total_size(args)
        logger.info(f"Total size: {total_size}")
        progress += 30 # 60
        update_backup_log(args.api_url, "IN_PROGRESS", ["Info: Got total size"], "", progress, args.auth_token)

        logger.info("Step 3: Copying to aws s3...")
        aws_cp(args, total_size)
        logger.info("Copying to aws s3 completed")
        progress += 40 # 100
        update_backup_log(args.api_url, "COMPLETED", ["Info: Copying to aws s3 completed"], "", progress, args.auth_token)
    except Exception as e:
        logger.error(f"Backup failed: {e}", exc_info=True)
        update_backup_log(args.api_url, "FAILED", ["Error: Backup failed"], str(e), progress, args.auth_token)
        raise e


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