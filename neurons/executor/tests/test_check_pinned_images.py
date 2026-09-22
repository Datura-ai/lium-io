"""scripts/check_pinned_images.py finds every pinned daturaai/ image in the tree and fails on one
Docker Hub does not have. The tests run offline: the registry call is replaced."""

import importlib.util
import sys
import urllib.error
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "check_pinned_images.py"
WORKFLOW = REPO / ".github" / "workflows" / "pinned_images.yml"

_spec = importlib.util.spec_from_file_location("check_pinned_images", SCRIPT)
cpi = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cpi  # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(cpi)

DIGEST = "sha256:" + "ab" * 32


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "    image: daturaai/lium-watchtower:1.2.0",
            [("daturaai/lium-watchtower", "1.2.0", False)],
        ),
        (
            "    image: daturaai/lium-watchtower:1.2.0-staging",
            [("daturaai/lium-watchtower", "1.2.0-staging", False)],
        ),
        (
            f"    image: daturaai/compute-subnet-executor@{DIGEST}",
            [("daturaai/compute-subnet-executor", DIGEST, False)],
        ),
        ("    image: docker.io/daturaai/redis:7.4.2", [("daturaai/redis", "7.4.2", False)]),
        ("FROM daturaai/ubuntu:24.04-py3.11", [("daturaai/ubuntu", "24.04-py3.11", False)]),
        ("docker pull daturaai/dind", [("daturaai/dind", "latest", False)]),
        (
            "    image: daturaai/compute-subnet-executor@${EXECUTOR_IMAGE_SHA256}",
            [("daturaai/compute-subnet-executor", None, True)],
        ),
        (
            "docker build -t daturaai/compute-subnet-miner:$TAG .",
            [("daturaai/compute-subnet-miner", None, True)],
        ),
        ('IMAGE="daturaai/compute-subnet-$ROLE:1.0"', [("daturaai/compute-subnet", None, True)]),
        ("# image: daturaai/lium-watchtower:9.9.9", []),
        (
            "    image: daturaai/lium-watchtower:1.1.1  # was 1.1.0 (daturaai/lium-watchtower:1.1.0)",
            [("daturaai/lium-watchtower", "1.1.1", False)],
        ),
        ("see https://hub.docker.com/r/daturaai/lium-watchtower/tags", []),
        ("pull daturaai/lium-watchtower:<version> by hand", []),
    ],
)
def test_refs_in_text(line, expected):
    assert [(repo, ref, var) for _, repo, ref, var in cpi.refs_in_text(line)] == expected


def _tree(tmp_path, files):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def test_collect_dedupes_and_ignores_unscanned_files(tmp_path):
    root = _tree(
        tmp_path,
        {
            "neurons/executor/docker-compose.yml": "services:\n  w:\n    image: daturaai/lium-watchtower:1.2.0\n",
            "neurons/executor/docker-compose.dev.yml": "services:\n  w:\n    image: daturaai/lium-watchtower:1.2.0\n",
            "neurons/executor/README.md": "image: daturaai/lium-watchtower:0.0.0\n",
            "neurons/validators/src/core/config.py": 'IMG = "daturaai/dind:0.0.3"\n',
        },
    )
    refs = cpi.collect(root)
    assert [(r.display, [(loc.path, loc.line) for loc in r.locations]) for r in refs] == [
        (
            "daturaai/lium-watchtower:1.2.0",
            [
                ("neurons/executor/docker-compose.dev.yml", 3),
                ("neurons/executor/docker-compose.yml", 3),
            ],
        )
    ]


def test_main_fails_on_a_missing_tag_and_skips_variables(tmp_path, monkeypatch, capsys):
    root = _tree(
        tmp_path,
        {
            "neurons/executor/docker-compose.yml": (
                "services:\n"
                "  w:\n    image: daturaai/lium-watchtower:1.2.0\n"
                "  r:\n    image: daturaai/compute-subnet-executor-runner:latest\n"
                "  e:\n    image: daturaai/compute-subnet-executor@${EXECUTOR_IMAGE_SHA256}\n"
            ),
        },
    )
    asked = []

    def fake_exists(repository, reference, tokens):
        asked.append((repository, reference))
        return reference != "1.2.0"

    monkeypatch.setattr(cpi, "manifest_exists", fake_exists)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert cpi.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert sorted(asked) == [
        ("daturaai/compute-subnet-executor-runner", "latest"),
        ("daturaai/lium-watchtower", "1.2.0"),
    ]
    assert "MISSING  daturaai/lium-watchtower:1.2.0" in out
    assert "::error file=neurons/executor/docker-compose.yml,line=3::" in out
    assert "SKIP     daturaai/compute-subnet-executor (variable tag or digest)" in out


def test_main_passes_when_every_pin_exists(tmp_path, monkeypatch):
    root = _tree(
        tmp_path,
        {"neurons/miners/docker-compose.yml": "services:\n  m:\n    image: daturaai/redis:7.4.2\n"},
    )
    monkeypatch.setattr(cpi, "manifest_exists", lambda *a: True)
    assert cpi.main(["--root", str(root)]) == 0


def test_main_reports_an_unreadable_registry(tmp_path, monkeypatch):
    root = _tree(
        tmp_path,
        {"neurons/miners/docker-compose.yml": "services:\n  m:\n    image: daturaai/redis:7.4.2\n"},
    )

    def down(*a):
        raise cpi.RegistryUnavailable("HEAD daturaai/redis@7.4.2: HTTP 503")

    monkeypatch.setattr(cpi, "manifest_exists", down)
    assert cpi.main(["--root", str(root)]) == 2


def test_workflow_paths_cover_every_scanned_glob():
    on = yaml.safe_load(WORKFLOW.read_text())[True]
    assert on["pull_request"]["paths"] == [
        *cpi.SCAN_GLOBS,
        "scripts/check_pinned_images.py",
        ".github/workflows/pinned_images.yml",
        "neurons/executor/tests/test_check_pinned_images.py",
    ]


def test_workflow_runs_the_offline_tests():
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["pinned-images"]["steps"]
    runs = "\n".join(step.get("run") or "" for step in steps)
    assert "test_check_pinned_images.py" in runs


def test_request_sleeps_only_when_a_retry_remains(monkeypatch):
    sleeps = []
    monkeypatch.setattr(cpi.time, "sleep", sleeps.append)

    def down(*_a, **_k):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(cpi.urllib.request, "urlopen", down)
    with pytest.raises(cpi.RegistryUnavailable):
        cpi._request("https://example.invalid/")
    assert sleeps == [2 ** (i + 1) for i in range(cpi.ATTEMPTS - 1)]


def test_request_returns_a_non_retryable_status_without_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(cpi.time, "sleep", sleeps.append)

    def missing(*_a, **_k):
        raise urllib.error.HTTPError("https://example.invalid/", 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr(cpi.urllib.request, "urlopen", missing)
    status, body = cpi._request("https://example.invalid/")
    assert status == 404
    assert body == b""
    assert sleeps == []
