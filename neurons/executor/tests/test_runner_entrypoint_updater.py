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
NICKFEDOR = "nickfedor/watchtower:1.24.0@sha256:e96976bcbeaf0a9adeddc9016a539c27a2245ef6bb7d3ca97ed106783350d387"
OWN_ID = "f" * 64
MOUNTINFO = (
    "1 0 0:1 / / rw - overlay overlay rw\n"
    f"2 1 259:1 /var/lib/docker/containers/{OWN_ID}/resolv.conf /etc/resolv.conf rw - ext4 /dev/root rw\n"
    f"3 1 259:1 /var/lib/docker/containers/{OWN_ID}/hostname /etc/hostname rw - ext4 /dev/root rw\n"
)
COMPOSE_LABELS = (
    " --label com.docker.compose.config-hash=abc --label com.docker.compose.oneoff=False"
    " --label com.docker.compose.project=executor --label com.docker.compose.service=watchtower"
)

DOCKER_STUB = textwrap.dedent(
    """\
    #!/bin/sh
    echo "$*" >> "$STUB_CALLS"
    case "$*" in
        *com.docker.compose.project\\"}}\\ "$OWN_ID") echo "${STUB_PROJECT-executor}" ;;
        "ps --all --filter label=com.docker.compose.project="*) echo "${STUB_UPDATERS-wt1 executor-watchtower-1}" ;;
        "inspect --format {{.Config.Image}} "*) echo "$STUB_UPDATER_IMAGE" ;;
        "inspect --format {{.State.Running}} "*) echo "${STUB_RUNNING-true}" ;;
        "inspect --format {{.Name}} wt1") echo "/executor-watchtower-1" ;;
        "inspect --format {{range "*" wt1") printf '%s\\n' com.docker.compose.config-hash=abc \\
            com.docker.compose.oneoff=False com.docker.compose.project=executor \\
            com.docker.compose.service=watchtower org.opencontainers.image.title=watchtower ;;
        "pull "*) exit "${STUB_PULL_EXIT:-0}" ;;
        "image inspect --format {{.Os}}/{{.Architecture}} "*) echo "${STUB_IMAGE_PLATFORM-linux/amd64}" ;;
        "version --format {{.Server.Os}}/{{.Server.Arch}}") echo "linux/amd64" ;;
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
    for tool in ("sh", "tail", "timeout", "sed", "head"):
        os.symlink(shutil.which(tool), bin_dir / tool)
    calls = tmp_path / "calls"
    calls.touch()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(env.pop("MOUNTINFO", MOUNTINFO))
    result = subprocess.run(
        ["sh", script],
        # a recreated runner keeps the hostname of the container it replaced, which names no container
        env={
            "PATH": str(bin_dir),
            "STUB_CALLS": str(calls),
            "HOSTNAME": "0123456789ab",
            "RUNNER_MOUNTINFO": str(mountinfo),
            "OWN_ID": OWN_ID,
            **env,
        },
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
        (EXECUTOR, "nickfedor/watchtower", "daturaai/lium-watchtower:1.2.2", ""),
        (EXECUTOR, "containrrr/watchtower:latest", "daturaai/lium-watchtower:1.2.2", ""),
        (EXECUTOR, "daturaai/lium-watchtower:1.1.0-staging", "daturaai/lium-watchtower:1.2.2-staging", ""),
        (MINER, "containrrr/watchtower:latest", NICKFEDOR, " --interval 60 --cleanup --label-enable"),
        (MINER, "nickfedor/watchtower", NICKFEDOR, " --interval 60 --cleanup --label-enable"),
    ],
)
def test_an_old_updater_is_replaced_before_the_stack_starts(tmp_path, script, old, new, args):
    result, calls = run(tmp_path, script, STUB_UPDATER_IMAGE=old)

    assert updater_run(calls) == [
        "run --detach --name executor-watchtower-1 --restart unless-stopped -v /var/run/docker.sock:/var/run/docker.sock"
        f"{COMPOSE_LABELS} {new}{args}"
    ]
    assert calls.index(f"pull --quiet {new}") < calls.index("rename wt1 executor-watchtower-1-replaced")
    assert "rm --force wt1" in calls
    assert calls.index(updater_run(calls)[0]) < calls.index("compose up --pull always --detach --wait --force-recreate")
    assert f"Updater replaced: executor-watchtower-1 now runs {new}" in result.stdout


@pytest.mark.parametrize(
    "staging, current, want",
    [
        ("1", "nickfedor/watchtower", "daturaai/lium-watchtower:1.2.2-staging"),
        ("1", "daturaai/lium-watchtower:1.2.2", "daturaai/lium-watchtower:1.2.2-staging"),
        ("", "nickfedor/watchtower", "daturaai/lium-watchtower:1.2.2"),
    ],
)
def test_a_dev_runner_installs_the_staging_updater_whatever_its_stack_runs(tmp_path, staging, current, want):
    _, calls = run(tmp_path, EXECUTOR, STUB_UPDATER_IMAGE=current, UPDATER_STAGING=staging)

    assert updater_run(calls)[0].endswith(f" {want}")


@pytest.mark.parametrize(
    "script, current",
    [(EXECUTOR, "daturaai/lium-watchtower:1.2.2"), (EXECUTOR, "daturaai/lium-watchtower:1.2.2-staging"), (MINER, NICKFEDOR)],
)
def test_a_current_updater_is_left_alone(tmp_path, script, current):
    _, calls = run(tmp_path, script, STUB_UPDATER_IMAGE=current)

    assert not updater_run(calls)
    assert not any(c.startswith(("pull", "rename", "stop")) for c in calls)


@pytest.mark.parametrize(
    "env",
    [
        {"STUB_UPDATERS": ""},
        {"STUB_UPDATERS": "wt1 executor-watchtower-1 wt2 other-watchtower-1"},
        # started by `docker run`: the runner's own container has no compose project
        {"STUB_PROJECT": ""},
        {"MOUNTINFO": "1 0 0:1 / / rw - overlay overlay rw\n"},
    ],
)
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

    # the container `docker run` created but could not start is removed before the old one gets its name back
    assert calls.index(updater_run(calls)[0]) < calls.index("rm --force executor-watchtower-1") < calls.index(
        "rename wt1 executor-watchtower-1"
    )
    assert "start wt1" in calls
    assert "rm --force wt1" not in calls
    assert "the old updater runs again" in result.stdout
    assert recreated_stack(calls)


def recreated_stack(calls):
    return "compose up --pull always --detach --wait --force-recreate" in calls


@pytest.mark.parametrize(
    "script, compose",
    [
        (EXECUTOR, os.path.join(HERE, "..", "docker-compose.yml")),
        (EXECUTOR, os.path.join(HERE, "..", "docker-compose.dev.yml")),
        (MINER, os.path.join(HERE, "..", "..", "miners", "docker-compose.yml")),
        (MINER, os.path.join(HERE, "..", "..", "miners", "docker-compose.dev.yml")),
    ],
)
def test_the_entrypoint_installs_the_updater_the_compose_file_pins(script, compose):
    pinned = re.search(r"image: (\S*watchtower\S*)", open(compose).read()).group(1)
    entrypoint_image = re.search(r'UPDATER_IMAGE="([^"]+)"', open(script).read()).group(1)
    assert pinned in (entrypoint_image, f"{entrypoint_image}-staging")


def test_only_the_dev_runner_build_sets_the_staging_updater():
    build = open(os.path.join(HERE, "..", "docker_runner_build.sh")).read()
    dockerfile = open(os.path.join(HERE, "..", "Dockerfile.runner")).read()
    assert build.count('--build-arg "updaterStaging=$([[ "$TAG" == dev ]] && echo 1 || true)"') == 2
    assert "ENV UPDATER_STAGING=${updaterStaging}" in dockerfile


@pytest.mark.parametrize("script", [EXECUTOR, MINER])
def test_only_the_runners_own_stack_is_touched(tmp_path, script):
    _, calls = run(tmp_path, script, STUB_UPDATER_IMAGE="containrrr/watchtower:latest")

    # the project comes from the runner's own container, never from a label lookup that can find another stack
    assert calls[0] == f'inspect --format {{{{index .Config.Labels "com.docker.compose.project"}}}} {OWN_ID}'
    assert not any(c.startswith("ps --filter label=com.docker.compose.service=") for c in calls)
    assert not any("0123456789ab" in c for c in calls)
    assert updater_run(calls)


@pytest.mark.parametrize("script", [EXECUTOR, MINER])
@pytest.mark.parametrize(
    "found, running, expect, absent",
    [
        # cut after the rename, before `docker run`: only the stopped old updater is left
        ("wt1 executor-watchtower-1-replaced", "true", ["rename wt1 executor-watchtower-1", "start wt1"], []),
        # cut after `docker run` created the new updater (started or not): it starts and the old one goes
        (
            "wt1 executor-watchtower-1-replaced wt2 executor-watchtower-1",
            "true",
            ["start wt2", "rm --force wt1"],
            ["rm --force wt2", "rename wt1 executor-watchtower-1"],
        ),
        # the new updater does not run: it goes and the old one comes back
        (
            "wt2 executor-watchtower-1 wt1 executor-watchtower-1-replaced",
            "false",
            ["start wt2", "rm --force wt2", "rename wt1 executor-watchtower-1", "start wt1"],
            ["rm --force wt1"],
        ),
    ],
)
def test_a_swap_cut_short_is_finished_at_the_next_start(tmp_path, script, found, running, expect, absent):
    current = re.search(r'UPDATER_IMAGE="([^"]+)"', open(script).read()).group(1)
    result, calls = run(tmp_path, script, STUB_UPDATERS=found, STUB_RUNNING=running, STUB_UPDATER_IMAGE=current)

    positions = [calls.index(c) for c in expect]
    assert positions == sorted(positions)
    assert not any(c in calls for c in absent)
    assert "Updater swap finished" in result.stdout
    assert recreated_stack(calls)


@pytest.mark.parametrize("script", [EXECUTOR, MINER])
@pytest.mark.parametrize(
    "found",
    [
        "wt1 executor-watchtower-1-replaced wt3 executor-watchtower-run-0f3a",
        "wt3 0123abcd_executor-watchtower-1 wt1 executor-watchtower-1-replaced",
    ],
)
def test_a_replaced_updater_next_to_an_unrelated_container_is_left_alone(tmp_path, script, found):
    _, calls = run(tmp_path, script, STUB_UPDATERS=found, STUB_UPDATER_IMAGE="containrrr/watchtower:latest")

    assert not any(c.startswith(("start", "rm", "rename", "stop", "pull", "run --detach")) for c in calls)


@pytest.mark.parametrize("script", [EXECUTOR, MINER])
@pytest.mark.parametrize("platform", ["linux/arm64", ""])
def test_an_updater_image_for_another_platform_is_not_installed(tmp_path, script, platform):
    # an amd64-only updater on an arm64 host (or an image whose platform cannot be read) never starts there
    result, calls = run(tmp_path, script, STUB_UPDATER_IMAGE="containrrr/watchtower:latest", STUB_IMAGE_PLATFORM=platform)

    assert any(c.startswith("pull ") for c in calls)
    assert not any(c.startswith(("rename ", "stop ", "run --detach ")) for c in calls)
    assert "not built for this host's platform" in result.stdout
