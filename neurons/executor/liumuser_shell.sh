#!/bin/bash
# The login shell and forced command of the `liumuser` SSH account (Dockerfile,
# sshd_liumuser.conf, DAH-3522).
#
# The platform's web terminal logs in as `liumuser` and names the renter's pod in the
# CONTAINER_NAME environment variable (AcceptEnv). Before DAH-3522 the forced command ran
# `docker exec` on whatever non-empty name arrived: the executor container itself (privileged,
# host Docker socket) or any other renter's pod (bounty ticket-0329). This script accepts
# only a rental pod name, `pod_<uuid>` (the validator's `POD_CONTAINER_PREFIX` + the pod id),
# so the account cannot reach anything else through `docker exec`. Which rental pod a session
# opens is still the caller's choice: the credential is the platform's, not a renter's. The name
# is checked here, before any docker call, and passed as one argument; nothing from the client
# is ever interpreted by a shell.
#
# It is also the account's login shell, so the account has no general-purpose shell at all:
# sshd runs a forced command as `<shell> -c <command>`, `su`/`docker exec -u liumuser` run
# the shell directly, and every one of those paths lands here. Arguments are ignored on
# purpose: the only thing this program ever does is open a shell in the named pod.
#
# Installed at /usr/local/bin/lium-pod-shell (Dockerfile). The functions run in a plain
# bash so the tests can source the file without a container.
set -u
export LC_ALL=C  # [0-9a-f] must mean ASCII hex whatever locale sshd hands the session

POD_NAME_PATTERN='^pod_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

lium_pod_shell_target() {
    # Print the validated container name, or refuse (exit 1) with one line on stderr.
    local name="${CONTAINER_NAME:-}"
    if [ -z "$name" ]; then
        echo "CONTAINER_NAME not set" >&2
        return 1
    fi
    if ! [[ "$name" =~ $POD_NAME_PATTERN ]]; then
        echo "CONTAINER_NAME is not a rental pod name (pod_<uuid>)" >&2
        return 1
    fi
    printf '%s\n' "$name"
}

lium_pod_shell_main() {
    # "$@" (for example sshd's `-c /usr/local/bin/lium-pod-shell`) is deliberately unused.
    local name
    name="$(lium_pod_shell_target)" || exit 1
    exec docker exec -it "$name" bash
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    lium_pod_shell_main "$@"
fi
