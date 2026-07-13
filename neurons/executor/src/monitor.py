import docker
import logging
import json
import time
from datetime import datetime
from core.db import get_session
from daos.pod_log import PodLog, PodLogDao

# Set up logging to a file in JSON format (one JSON object per line)
logging.basicConfig(filename="container_monitor.log", level=logging.INFO,
                    format='%(message)s')  # we will log pre-formatted JSON strings

# Container-name prefixes the validator creates for real workloads: rental pods
# (pod_*) and idle-node default jobs (filler_*). The legacy "container_" prefix
# matches only the validator's throwaway port-check probes today, so watching it
# recorded pure probe noise while every pod/filler death went unlogged (DAH-2395).
MONITORED_CONTAINER_PREFIXES: tuple[str, ...] = ("pod_", "filler_")


# Helper: Determine stop reason classification


def classify_stop(exit_code):
    """Classify the stop reason for the given container."""
    # Default classification
    reason = "unknown"

    # Check for recent GPU errors in dmesg
    try:
        # Read kernel messages (dmesg) for GPU errors
        import subprocess
        dmesg_out = subprocess.check_output(["dmesg", "--ctime", "--kernel", "--nopager"], universal_newlines=True)
    except Exception:
        dmesg_out = ""

    if "NVRM: Xid" in dmesg_out or "GPU has fallen off the bus" in dmesg_out:
        reason = "gpu_error"
    elif exit_code == 0:
        reason = "purposely_stopped"
    elif exit_code == 1:
        reason = "application_error"
    elif exit_code == 125:
        reason = "container_failed_to_run"
    elif exit_code == 126:
        reason = "command_invoke_error"
    elif exit_code == 127:
        reason = "file_or_directory_not_found"
    elif exit_code == 128:
        reason = "invalid_argument_on_exit"
    elif exit_code == 134:  # SIGABRT
        reason = "abnormal_termination"
    elif exit_code == 137:  # SIGKILL: from docker rm command
        reason = "immediate_termination"
    elif exit_code == 139:  # SIGSEGV
        reason = "segmentation_fault"
    elif exit_code == 143:  # SIGTERM
        reason = "graceful_termination"
    elif exit_code == 255:
        reason = "exit_status_out_of_range"
    return reason


def log_event(log: PodLog):
    try:
        with get_session() as session:
            pod_log_dao = PodLogDao()
            pod_log_dao.save(session, log)
            logging.info(json.dumps(log.model_dump(), default=str))
    except:
        pass


def handle_event(event):
    """Process a Docker event."""
    action = event.get("Action")
    attrs = event.get("Actor", {}).get("Attributes", {})
    container_id = event.get("id") or event.get("Actor", {}).get("ID")
    name = attrs.get("name")

    if not name or not name.startswith(MONITORED_CONTAINER_PREFIXES):
        return

    pod_log = PodLog(
        container_name=name,
        container_id=container_id,
        event=action,
        created_at=datetime.utcnow()
    )

    if action in {"start", "stop", "restart", "kill", "destroy", "oom"}:
        log_event(pod_log)
    elif action == "die":
        exit_code = int(attrs.get("exitCode", 0))
        reason = classify_stop(exit_code)
        pod_log.exit_code = exit_code
        pod_log.reason = reason
        log_event(pod_log)


def main():
    """Main loop to monitor Docker events."""
    client = docker.from_env()

    while True:
        try:
            for event in client.events(decode=True):
                if event.get("Type") != "container":
                    continue

                try:
                    handle_event(event)
                except Exception as ex:
                    pod_log = PodLog(
                        event=event.get('Action'),
                        error=f"Exception processing event {event.get('Action')}: {ex}",
                        created_at=datetime.utcnow()
                    )
                    log_event(pod_log)
        except docker.errors.APIError as api_error:
            pod_log = PodLog(
                error=f"Docker API error: {api_error}",
                created_at=datetime.utcnow()
            )
            log_event(pod_log)

            time.sleep(5)
        except Exception as ex:
            pod_log = PodLog(
                error=f"Unexpected error: {ex}",
                created_at=datetime.utcnow()
            )
            log_event(pod_log)

            time.sleep(5)


if __name__ == "__main__":
    main()
