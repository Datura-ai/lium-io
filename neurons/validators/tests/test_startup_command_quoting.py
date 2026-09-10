"""Renter `startup_commands` become the container's argv, never a host command.

`build_container_command_argv` is what `DockerService._container_run_spec` hands
to the Docker SDK as the container command (no host shell is involved), so the
only way a renter value could reach the executor host is if it were ever
rendered back into a shell line. These tests pin the argv contract for the
payloads that matter: the real prod breakout attempt and the real top template.
"""

from services.rental_docker_sdk import build_container_command_argv


def test_simple_command_preserved():
    assert build_container_command_argv("python /app/main.py --epochs 5") == (
        "python", "/app/main.py", "--epochs", "5",
    )


def test_legit_quoted_metacharacters_preserved():
    # The real top template (CUDA 13.0.2) — `&&` lives inside a quoted bash -c arg
    # and must reach the container intact as one argument, not be dropped.
    original = '/bin/bash -c "service ssh start && tail -f /dev/null"'

    assert build_container_command_argv(original) == (
        "/bin/bash", "-c", "service ssh start && tail -f /dev/null",
    )


def test_leading_newline_breakout_neutralized():
    # The real prod attack payload.
    argv = build_container_command_argv("\n\nsh /tmp/evil.sh")

    # No raw newline survives into any token, so nothing can terminate a host line.
    assert all("\n" not in token for token in argv)
    # The remainder is reduced to harmless container arguments.
    assert argv == ("sh", "/tmp/evil.sh")


def test_semicolon_chaining_neutralized():
    # `;` is part of a single token (`hi;`) — a container argument, never a
    # command separator anywhere.
    assert build_container_command_argv("echo hi; rm -rf /") == (
        "echo", "hi;", "rm", "-rf", "/",
    )


def test_env_expansion_kept_inside_container_shell():
    # `bash -c '... $MODEL_NAME'` expands against the CONTAINER env, not the host.
    assert build_container_command_argv("bash -c 'vllm serve $MODEL_NAME'") == (
        "bash", "-c", "vllm serve $MODEL_NAME",
    )


def test_empty_and_none_fall_back_to_default():
    assert build_container_command_argv(None) == ()
    assert build_container_command_argv("") == ()
    assert build_container_command_argv("   ") == ()
    assert build_container_command_argv("\n\n") == ()


def test_unbalanced_quotes_fall_back_to_default():
    # shlex.split raises ValueError → drop to the image default rather than guess.
    assert build_container_command_argv('bash -c "unterminated') == ()
