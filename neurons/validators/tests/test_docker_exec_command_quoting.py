"""`DockerCommand.exec_command` runs as root on the executor host over SSH.

Both the container name and the inner command must be shell-quoted so a value
that ever carried a metacharacter would stay a single docker argument instead of
starting a second host command.
"""

import shlex

from core.docker_utils import DockerCommand


def test_live_call_shape_unchanged():
    # The one caller today: a UUID-derived name and a constant command.
    command = DockerCommand.exec_command("pod_ab12cd34-0000", "cat /root/.ssh/authorized_keys")

    assert shlex.split(command) == [
        "/usr/bin/docker", "exec", "-u", "0", "-i", "pod_ab12cd34-0000",
        "sh", "-c", "cat /root/.ssh/authorized_keys",
    ]


def test_metacharacters_in_container_name_stay_one_argument():
    command = DockerCommand.exec_command("pod; touch /tmp/pwned", "true")

    argv = shlex.split(command)
    assert argv[5] == "pod; touch /tmp/pwned"
    assert argv[6:] == ["sh", "-c", "true"]


def test_single_quote_in_command_cannot_close_the_quoting():
    inner = "echo 'a'; touch /tmp/pwned"

    command = DockerCommand.exec_command("pod_x", inner)

    # The host shell hands the whole inner command to the container's `sh -c`
    # as one argument; nothing after the quote becomes a host command.
    assert shlex.split(command)[-1] == inner
