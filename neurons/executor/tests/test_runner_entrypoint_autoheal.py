"""DAH-7125: the update removes this stack's old autoheal container (compose keeps one it no longer knows) and
confirms the executor API answers, now that --wait has no healthcheck to wait for."""

import test_runner_entrypoint_updater as updater

EXTRA_CASES = """\
    "compose up "*) : > "$STUB_CALLS.up" ;;
    "compose ps --quiet executor") if [ -e "$STUB_CALLS.up" ]; then echo exec1; fi ;;
    *com.docker.compose.project\\"}}\\ exec1) echo executor ;;
    "ps --all --quiet --filter label=com.docker.compose.project=executor --filter label=com.docker.compose.service=autoheal")
        echo "${STUB_AUTOHEAL-ah1}" ;;
    "compose exec -T executor "*) exit "${STUB_API_EXIT:-0}" ;;
"""


def run(tmp_path, monkeypatch, **env):
    monkeypatch.setattr(updater, "DOCKER_STUB", updater.DOCKER_STUB.replace("esac", EXTRA_CASES + "esac"))
    return updater.run(tmp_path, updater.EXECUTOR, STUB_UPDATER_IMAGE="daturaai/lium-watchtower:1.2.4", **env)


def test_the_update_removes_only_this_stacks_old_autoheal_and_confirms_the_api(tmp_path, monkeypatch):
    result, calls = run(tmp_path, monkeypatch)

    up = calls.index("compose up --pull always --detach --wait --force-recreate")
    ready = next(i for i, c in enumerate(calls) if c.startswith("compose exec -T executor "))
    assert up < ready < calls.index("rm --force ah1")
    assert "Executor API ready." in result.stdout
    assert "Removed the old autoheal container ah1" in result.stdout
    assert [c for c in calls if c.startswith("rm ")] == ["rm --force ah1"]


def test_a_stack_without_an_old_autoheal_removes_nothing(tmp_path, monkeypatch):
    _, calls = run(tmp_path, monkeypatch, STUB_AUTOHEAL="")

    assert not [c for c in calls if c.startswith("rm ")]


def test_an_api_not_answering_yet_is_logged_and_not_recreated(tmp_path, monkeypatch):
    result, calls = run(tmp_path, monkeypatch, STUB_API_EXIT="7", RUNNER_READY_MAX_TRIES="1")

    assert "The executor API did not answer yet" in result.stdout
    assert calls.count("compose up --pull always --detach --wait --force-recreate") == 1
    assert "rm --force ah1" in calls
