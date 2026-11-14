"""nvidia_docker_sysbox_setup.sh: needrestart never restarts sysbox, Docker, containerd or NVIDIA fabric manager under pods."""

import os
import re
import shutil
import subprocess

import pytest

SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nvidia_docker_sysbox_setup.sh"))


def _function(name: str) -> str:
    text = open(SCRIPT).read()
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.S | re.M)
    assert match, name
    return match.group(0)


@pytest.mark.skipif(not shutil.which("perl"), reason="needrestart reads its config with perl")
def test_the_needrestart_override_keeps_the_container_runtime_and_is_valid_needrestart_config(tmp_path):
    script = (
        'host_path() { echo "${SYSBOX_SETUP_HOST_ROOT:-}$1"; }\n'
        + _function("keep_needrestart_off_the_container_runtime")
        + "keep_needrestart_off_the_container_runtime\n"
    )
    subprocess.run(["bash", "-c", script], check=True, env={**os.environ, "SYSBOX_SETUP_HOST_ROOT": str(tmp_path)})
    conf = tmp_path / "etc/needrestart/conf.d/lium-container-runtime.conf"
    # needrestart `eval`s each conf.d file with %nrconf in scope; a service matching an override_rc rule set to 0
    # is never restarted
    out = subprocess.run(
        ["perl", "-e", 'our %nrconf; my $r = do $ARGV[0]; die $@ if $@;'
         ' for my $s (@ARGV[1..$#ARGV]) { my @m = grep { $s =~ $_ && !$nrconf{override_rc}{$_} }'
         ' keys %{$nrconf{override_rc}}; print "$s:", (@m ? "kept" : "restarted"), "\\n" }',
         str(conf), "sysbox.service", "sysbox-mgr.service", "sysbox-fs.service", "docker.service",
         "containerd.service", "nvidia-fabricmanager.service", "nvidia-persistenced.service", "cron.service"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.split() == [
        "sysbox.service:kept", "sysbox-mgr.service:kept", "sysbox-fs.service:kept", "docker.service:kept",
        "containerd.service:kept", "nvidia-fabricmanager.service:kept", "nvidia-persistenced.service:kept",
        "cron.service:restarted",
    ]


def test_the_guard_is_written_before_the_already_working_exit():
    # a host whose sysbox already works exits at "Already working?"; it runs pods, so it needs the guard most
    text = open(SCRIPT).read()
    call = text.index("\nkeep_needrestart_off_the_container_runtime &&")
    assert text.count("\nkeep_needrestart_off_the_container_runtime &&") == 1
    assert call < text.index("# ── 2. Already working?")
