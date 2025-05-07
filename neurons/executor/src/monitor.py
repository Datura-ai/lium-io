import docker
import psutil
import pynvml
import logging
import json
import threading
import time
from datetime import datetime
from core.config import settings
from core.db import get_session
from daos.pod_log import PodLog, PodLogDao

# Set up logging to a file in JSON format (one JSON object per line)
logging.basicConfig(filename="container_monitor.log", level=logging.INFO,
                    format='%(message)s')  # we will log pre-formatted JSON strings

# Initialize NVML for GPU monitoring
try:
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception as e:
    NVML_AVAILABLE = False
    logging.error(json.dumps({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "error": f"Failed to initialize NVML: {e}"
    }))

# Data structures for tracking
monitored = {}  # dict container_id -> {"name": ..., "pid": ..., "last_stats": {...}}
monitor_prefix = "container_"  # only monitor containers with this prefix

# # Helper: get current GPU usage for a container (if NVML is available)
# def get_container_gpu_stats(container_pid):
#     """Return GPU utilization% and memory usage (MB) for processes in the container."""
#     stats = {"gpu_util": 0, "gpu_mem_mb": 0}
#     if not NVML_AVAILABLE:
#         return stats
#     try:
#         device_count = pynvml.nvmlDeviceGetCount()
#     except pynvml.NVMLError as e:
#         # NVML error, log and return zeros
#         logging.error(json.dumps({
#             "timestamp": datetime.utcnow().isoformat() + "Z",
#             "error": f"NVML error in get_count: {str(e)}"
#         }))
#         return stats
#     total_util = 0
#     total_mem = 0
#     for i in range(device_count):
#         try:
#             handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#             util = pynvml.nvmlDeviceGetUtilizationRates(handle)
#             procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
#         except pynvml.NVMLError:
#             continue  # skip this GPU if any issue
#         # Sum usage for processes belonging to this container (match by PID namespace via container_pid, or cgroup check)
#         for p in procs:
#             pid = p.pid
#             # Check if this process is in the same container by comparing its ancestral PID namespace root or cgroup.
#             # Simplest approach: check if the container's init PID matches this or is an ancestor.
#             try:
#                 proc = psutil.Process(pid)
#                 ancestors = proc.parents()
#             except psutil.NoSuchProcess:
#                 continue
#             # If container_pid is this pid or any parent pid, assume it's part of the container
#             if any(parent.pid == container_pid for parent in ancestors) or pid == container_pid:
#                 total_util += util.gpu   # add GPU core utilization (this is per-device, so if multiple containers share, it will sum >100 possibly)
#                 # Add memory used by this process on this GPU (pynvml gives bytes)
#                 try:
#                     total_mem += p.usedGpuMemory // (1024 * 1024)  # convert to MB
#                 except Exception:
#                     # p.usedGpuMemory might be None or not available for some drivers
#                     pass
#     if device_count > 0:
#         # Clip util to 100 if one container per GPU assumption, or leave as sum if multi-GPU usage
#         stats["gpu_util"] = min(total_util, 100)
#         stats["gpu_mem_mb"] = total_mem
#     return stats

# # Thread: periodically sample GPU stats for each monitored container
# def metrics_sampler():
#     while True:
#         for cid, info in list(monitored.items()):
#             pid = info.get("pid")
#             if not pid:
#                 continue
#             stats = get_container_gpu_stats(pid)
#             # Store the latest stats in our dict (so event thread can use it)
#             monitored[cid]["last_stats"] = stats
#         time.sleep(5)  # sample interval (seconds)

# # Start metrics sampling thread
# thread = threading.Thread(target=metrics_sampler, daemon=True)
# thread.start()

# Helper: Determine stop reason classification


def classify_stop(exit_code):
    """Classify the stop reason for the given container."""
    # Default classification
    reason = "unknown"

    # If NVML is available, check for recent GPU errors in dmesg
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


while True:
    try:
        # Initialize Docker client
        client = docker.from_env()

        for event in client.events(decode=True):
            try:
                # Only care about container events
                if event.get("Type") != "container":
                    continue

                action = event.get("Action")
                attrs = event.get("Actor", {}).get("Attributes", {})
                container_id = event.get("id") or event.get("Actor", {}).get("ID")
                name = attrs.get("name")

                try:
                    container = client.containers.get(container_id)
                except Exception:
                    continue

                # skip containers not matching our prefix filter
                if not name or not name.startswith(monitor_prefix) or name == f"{monitor_prefix}{settings.MINER_HOTKEY_SS58_ADDRESS}":
                    continue

                with get_session() as session:
                    pod_log_dao = PodLogDao()
                    pod_log = PodLog(
                        container_name=name,
                        container_id=container_id,
                        event=action,
                        created_at=datetime.utcnow()
                    )
                    default_log = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "event": action,
                        "container_id": container_id[:12],
                        "container_name": name,
                    }

                    if action == "start":
                        pod_log_dao.save(session, pod_log)
                        logging.info(json.dumps(default_log))
                    elif action == "stop":
                        pod_log_dao.save(session, pod_log)
                        logging.info(json.dumps(default_log))
                    elif action == "restart":
                        pod_log_dao.save(session, pod_log)
                        logging.info(json.dumps(default_log))
                    elif action == "kill":
                        pod_log_dao.save(session, pod_log)
                        logging.info(json.dumps(default_log))
                    elif action == "destroy":
                        pod_log_dao.save(session, pod_log)
                        logging.info(json.dumps(default_log))
                    # container out of memory error
                    elif action == "oom":
                        pod_log_dao.save(session, pod_log)
                        logging.info(json.dumps(default_log))
                    # container is exited
                    elif action == "die":
                        exit_code = int(attrs.get("exitCode", 0))
                        reason = classify_stop(exit_code)
                        pod_log.exit_code = exit_code
                        pod_log.reason = reason
                        pod_log_dao.save(session, pod_log)
                        logging.info(json.dumps({
                            **default_log,
                            "exit_code": exit_code,
                            "reason": reason,
                        }))

            except Exception as ex:
                # Log any exception in event handling, but continue loop
                logging.error(json.dumps({
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "error": f"Exception processing event {event.get('Action')}: {ex}"
                }))
    except docker.errors.APIError as api_error:
        logging.error(json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": f"Docker API error: {api_error}"
        }))
        time.sleep(5)
    except Exception as ex:
        logging.error(json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": f"Unexpected error: {ex}"
        }))
        time.sleep(5)
