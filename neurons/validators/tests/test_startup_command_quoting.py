import shlex

from services.docker_service import build_startup_command_args


def test_simple_command_preserved():
    result = build_startup_command_args("python /app/main.py --epochs 5")

    assert shlex.split(result) == ["python", "/app/main.py", "--epochs", "5"]


def test_legit_quoted_metacharacters_preserved():
    # The real top template (CUDA 13.0.2) — `&&` lives inside a quoted bash -c arg
    # and must reach the container intact, not be dropped.
    original = '/bin/bash -c "service ssh start && tail -f /dev/null"'

    result = build_startup_command_args(original)

    # Host shell re-tokenizes the quoted fragment back into the original argv,
    # so the `&&` runs inside the container, not on the host.
    assert shlex.split(result) == shlex.split(original)
    assert shlex.split(result) == ["/bin/bash", "-c", "service ssh start && tail -f /dev/null"]


def test_leading_newline_breakout_neutralized():
    # The real prod attack payload.
    result = build_startup_command_args("\n\nsh /tmp/evil.sh")

    # No raw newline survives, so the host `docker run` line cannot be terminated.
    assert "\n" not in result
    # The remainder is reduced to harmless container arguments.
    assert shlex.split(result) == ["sh", "/tmp/evil.sh"]


def test_semicolon_chaining_neutralized():
    result = build_startup_command_args("echo hi; rm -rf /")

    # The `;` is quoted (result is `echo 'hi;' rm -rf /`), so the host shell sees
    # a single literal token rather than a command separator — it becomes a
    # container argument and cannot start a new host command.
    assert shlex.split(result) == ["echo", "hi;", "rm", "-rf", "/"]


def test_env_expansion_kept_inside_container_shell():
    # `bash -c '... $MODEL_NAME'` expands against the CONTAINER env, not the host.
    original = "bash -c 'vllm serve $MODEL_NAME'"

    result = build_startup_command_args(original)

    assert shlex.split(result) == ["bash", "-c", "vllm serve $MODEL_NAME"]


def test_empty_and_none_fall_back_to_default():
    assert build_startup_command_args(None) == ""
    assert build_startup_command_args("") == ""
    assert build_startup_command_args("   ") == ""
    assert build_startup_command_args("\n\n") == ""


def test_unbalanced_quotes_fall_back_to_default():
    # shlex.split raises ValueError → drop to image default rather than risk a
    # malformed host command.
    assert build_startup_command_args('bash -c "unterminated') == ""


def test_quoting_round_trips_for_arbitrary_payloads():
    # General safety invariant: the quoted fragment, when interpreted by the host
    # shell, yields exactly the same argv as the original — semantics preserved,
    # host interpretation neutralized.
    for payload in [
        "python app.py",
        '/bin/bash -c "a && b | c > /tmp/x"',
        "jupyter lab --ip 0.0.0.0",
        "sh -c 'while true; do sleep 1; done'",
    ]:
        result = build_startup_command_args(payload)
        assert shlex.split(result) == shlex.split(payload), payload
