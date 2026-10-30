#!/bin/sh
set -eux -o pipefail

# disk reserve: validator-cycle writes must survive a 100%-full host disk
bash setup_disk_reserve.sh || echo "[disk-reserve] setup failed; continuing without reserve"
bash setup_disk_reserve.sh janitor &
# executor-v1.151 capped vm.dirty_bytes on every host; put the kernel defaults back (DAH-4978 revert)
bash restore_dirty_defaults.sh || echo "[dirty-cap] restore failed; continuing with the host's current setting"
# re-enter cwd through the overlay mount (the old dentry would bypass it)
cd /root/app

# ensure docker group matches host socket
SOCK=/var/run/docker.sock
if [ -S "$SOCK" ]; then
  SGID=$(stat -c %g "$SOCK")
  GNAME=$(getent group "$SGID" | cut -d: -f1 || true)
  if [ -z "$GNAME" ]; then
    groupadd -o -g "$SGID" dockersock
    GNAME=dockersock
  fi
  usermod -aG "$GNAME" liumuser || true
fi

# liumd's host settings, before sshd lets the validator in (the executor itself never runs liumd)
pdm run python src/liumd_host_files.py host || echo "[liumd] host files not written; continuing"

# start ssh service
ssh-keygen -A
# sshd logs to syslog (LogLevel VERBOSE from sshd_setup.sh names the key of each login); this container
# runs no syslog, so sshd_log_receiver.py is /dev/log and keeps the login lines for the validator.
python3 sshd_log_receiver.py &
for _ in $(seq 50); do [ -S /dev/log ] && break; sleep 0.1; done
service ssh start

# run fastapi app
pdm run python src/executor.py
