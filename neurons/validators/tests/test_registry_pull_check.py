"""REGISTRY_PULL_FAILED: an idle node whose Docker daemon cannot pull a Docker Hub image through its own registry path.

ticket-0361 root cause (reported 23 Sep 18:46Z): a node failed 16 rents in 24 h, every one of
them a template image the node did not have cached. Its dockerd pulls through the registry mirror
docker.m.daocloud.io, whose DNS lookup times out; cached templates started fine. No check pulled anything, so the
node passed every one.
"""

from __future__ import annotations



from services.task.checks import registry_pull as module

# Docker 29.1.3 (containerd image store) with that mirror configured and its DNS answers dropped: the 2.4 KB
# pull took 51 s on one run and over 60 s on another, and the 30 s bound ended it on both runs after; with the resolver refusing, the pull failed at once
# Docker 29.8.1, 23 Sep 2026, a REJECT rule in OUTPUT for every address of registry-1.docker.io (or the daemon's
# proxy), the committed hello-world digest pulled; the text is the daemon's own, verbatim


def test_defaults_bounded():
    """Regression: the check is on with no enforcement switch, pulls every 6 h and retries a failure after 30 min."""
    fields = type(module.settings).model_fields
    assert fields["REGISTRY_PULL_CHECK_ENABLED"].default is True
    assert "REGISTRY_PULL_ENFORCEMENT_ENABLED" not in fields
    assert fields["REGISTRY_PULL_PROBE_INTERVAL_HOURS"].default == 6.0
    assert fields["REGISTRY_PULL_PROBE_RETRY_MINUTES"].default == 30.0
    assert not [name for name in fields if name.startswith("REGISTRY_PULL_FLEET")]


# ------------------------------------------------------------------ the script under sh, docker stubbed


# ------------------------------------------------------------------ the check


# ------------------------------------------------------------------ the Docker Hub control


# ------------------------------------------------------------------ fleets over many cycles


