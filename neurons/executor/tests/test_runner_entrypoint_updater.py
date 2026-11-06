"""Fish, 6 Nov 2026: no provider should ever have to update by hand. The runner is the one container every
updater recreates, so its entrypoint replaces a stack's out-of-date updater container (executor: anything but
the pinned daturaai/lium-watchtower; miner: containrrr/watchtower, which stops on Docker Engine 29).

Each entrypoint runs under sh with a stub `docker` that records every call.
"""

import os
import re
import shutil
import subprocess
import textwrap

import pytest

HERE = os.path.dirname(__file__)
EXECUTOR = os.path.abspath(os.path.join(HERE, "..", "entrypoint.sh"))
MINER = os.path.abspath(os.path.join(HERE, "..", "..", "miners", "entrypoint.sh"))

DOCKER_STUB = textwrap.dedent(
    """\
    #!/bin/sh
    echo "$*" >> "$STUB_CALLS"
    case "$*" in
        *com.docker.compose.project\\"}}\\ runnerhost) echo "${STUB_PROJECT-executor}" ;;
        "ps --all --quiet --filter label=com.docker.compose.project="*) echo "${STUB_UPDATER-wt1}" ;;
        "inspect --format {{.Config.Image}} wt1") echo "$STUB_UPDATER_IMAGE" ;;
        "inspect --format {{.Name}} wt1") echo "/executor-watchtower-1" ;;
        "pull "*) exit "${STUB_PULL_EXIT:-0}" ;;
        "run --detach "*) exit "${STUB_RUN_EXIT:-0}" ;;
    esac
    exit 0
    """
)


def run(tmp_path, script, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(DOCKER_STUB)
    (bin_dir / "docker").chmod(0o755)
    # the log-follow loop never ends; a `sleep` that fails ends the script after one round
    (bin_dir / "sleep").write_text("#!/bin/sh\nexit 7\n")
    (bin_dir / "sleep").chmod(0o755)
    for tool in ("sh", "tail", "timeout"):
        os.symlink(shutil.which(tool), bin_dir / tool)
    calls = tmp_path / "calls"
    calls.touch()
    result = subprocess.run(
        ["sh", script],
        env={"PATH": str(bin_dir), "STUB_CALLS": str(calls), "HOSTNAME": "runnerhost", **env},
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, calls.read_text().splitlines()


def updater_run(calls):
    return [c for c in calls if c.startswith("run --detach")]


@pytest.mark.parametrize(
    "script, old, new, args",
    [
        (EXECUTOR, "nickfedor/watchtower", "daturaai/lium-watchtower:1.2.0", ""),
        (EXECUTOR, "containrrr/watchtower:latest", "daturaai/lium-watchtower:1.2.0", ""),
        (MINER, "containrrr/watchtower:latest", "nickfedor/watchtower", " --interval 60 --cleanup --label-enable"),
    ],
)
def test_an_old_updater_is_replaced_before_the_stack_starts(tmp_path, script, old, new, args):
    result, calls = run(tmp_path, script, STUB_UPDATER_IMAGE=old)

    assert updater_run(calls) == [
        "run --detach --name executor-watchtower-1 --restart unless-stopped -v /var/run/docker.sock:/var/run/docker.sock"
        f" --label com.docker.compose.project=executor --label com.docker.compose.service=watchtower {new}{args}"
    ]
    assert calls.index(f"pull --quiet {new}") < calls.index("rename wt1 executor-watchtower-1-replaced")
    assert "rm --force wt1" in calls
    assert calls.index(updater_run(calls)[0]) < calls.index("compose up --pull always --detach --wait --force-recreate")
    assert f"Updater replaced: executor-watchtower-1 now runs {new}" in result.stdout


@pytest.mark.parametrize("script, current", [(EXECUTOR, "daturaai/lium-watchtower:1.2.0"), (MINER, "nickfedor/watchtower")])
def test_a_current_updater_is_left_alone(tmp_path, script, current):
    _, calls = run(tmp_path, script, STUB_UPDATER_IMAGE=current)

    assert not updater_run(calls)
    assert not any(c.startswith(("pull", "rename", "stop")) for c in calls)


@pytest.mark.parametrize("env", [{"STUB_UPDATER": ""}, {"STUB_UPDATER": "wt1 wt2"}, {"STUB_PROJECT": ""}, {"HOSTNAME": ""}])
def test_no_single_updater_in_the_runners_project_means_nothing_is_created(tmp_path, env):
    _, calls = run(tmp_path, EXECUTOR, STUB_UPDATER_IMAGE="nickfedor/watchtower", **env)

    assert not updater_run(calls)


def test_an_updater_that_cannot_be_pulled_is_kept(tmp_path):
    result, calls = run(tmp_path, EXECUTOR, STUB_UPDATER_IMAGE="nickfedor/watchtower", STUB_PULL_EXIT="1")

    assert not any(c.startswith(("rename", "stop", "rm")) for c in calls)
    assert "Updater kept" in result.stdout
    assert recreated_stack(calls)


def test_an_updater_that_does_not_start_puts_the_old_one_back(tmp_path):
    result, calls = run(tmp_path, EXECUTOR, STUB_UPDATER_IMAGE="nickfedor/watchtower", STUB_RUN_EXIT="125")

    assert calls.index("rename wt1 executor-watchtower-1") > calls.index(updater_run(calls)[0])
    assert "start wt1" in calls
    assert "rm --force wt1" not in calls
    assert "the old updater runs again" in result.stdout
    assert recreated_stack(calls)


def recreated_stack(calls):
    return "compose up --pull always --detach --wait --force-recreate" in calls


def test_the_executor_entrypoint_installs_the_updater_the_compose_file_pins():
    compose = open(os.path.join(HERE, "..", "docker-compose.yml")).read()
    entrypoint = open(EXECUTOR).read()
    pinned = re.search(r"image: (daturaai/lium-watchtower:\S+)", compose).group(1)
    assert f'UPDATER_IMAGE="{pinned}"' in entrypoint
