"""DAH-7125: the executor has no Docker healthcheck, so `compose up --wait` only waits for the start; the runner logs
whether the executor API answers and never recreates it again for that. (This stack's old autoheal container is
removed by the monitor, tests/test_api_watcher.py.)"""

import test_runner_entrypoint_updater as updater

EXTRA_CASES = """\
    "compose exec -T executor "*) exit "${STUB_API_EXIT:-0}" ;;
"""


def run(tmp_path, monkeypatch, **env):
    monkeypatch.setattr(updater, "DOCKER_STUB", updater.DOCKER_STUB.replace("esac", EXTRA_CASES + "esac"))
    return updater.run(tmp_path, updater.EXECUTOR, STUB_UPDATER_IMAGE="daturaai/lium-watchtower:1.2.4", **env)


def test_the_update_confirms_the_executor_api_after_the_start(tmp_path, monkeypatch):
    result, calls = run(tmp_path, monkeypatch)

    up = calls.index("compose up --pull always --detach --wait --force-recreate")
    ready = next(i for i, c in enumerate(calls) if c.startswith("compose exec -T executor "))
    assert up < ready
    assert "Executor API ready." in result.stdout
    assert not [c for c in calls if c.startswith("rm ")]


def test_an_api_not_answering_yet_is_logged_and_not_recreated(tmp_path, monkeypatch):
    result, calls = run(tmp_path, monkeypatch, STUB_API_EXIT="7", RUNNER_READY_MAX_TRIES="1")

    assert "The executor API did not answer yet" in result.stdout
    assert calls.count("compose up --pull always --detach --wait --force-recreate") == 1
