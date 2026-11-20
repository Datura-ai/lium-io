"""`--check-base`: a changed schema takes a version above the base branch's, and the version never goes down."""

import json

from lium_protocol import PROTOCOL_VERSION
from lium_protocol.schema import base_problem, build_schema, main, render


def _base(version: str, *, changed: bool = False) -> str:
    doc = json.loads(render(build_schema()))
    doc["protocol_version"] = version
    if changed:
        doc["$defs"].pop(sorted(doc["$defs"])[0])
    return json.dumps(doc)


def _bump(version: str, by: int) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    return f"{major}.{minor + by}.{patch}"


def test_a_changed_schema_above_the_base_version_passes():
    assert base_problem(_base(_bump(PROTOCOL_VERSION, -1), changed=True)) is None


def test_the_same_schema_at_the_same_version_passes():
    assert base_problem(_base(PROTOCOL_VERSION)) is None


def test_a_changed_schema_at_the_base_version_is_refused():
    assert "one version would name two schemas" in base_problem(_base(PROTOCOL_VERSION, changed=True))


def test_a_version_below_the_base_is_refused_even_with_the_same_schema():
    # another protocol PR merged first with a higher number: this one rebases and takes the next one
    assert "below the base branch's" in base_problem(_base(_bump(PROTOCOL_VERSION, 3)))


def test_an_unreadable_base_snapshot_is_refused():
    assert "cannot be read" in base_problem("not json")
    assert "cannot be read" in base_problem("{}")


def test_the_cli_exits_1_on_a_problem(tmp_path, capsys):
    path = tmp_path / "base.json"
    path.write_text(_base(_bump(PROTOCOL_VERSION, 3)))
    assert main(["--check-base", str(path)]) == 1
    assert "protocol version:" in capsys.readouterr().out
    path.write_text(_base(PROTOCOL_VERSION))
    assert main(["--check-base", str(path)]) == 0


def _ci(monkeypatch, tmp_path, event: dict, base_snapshot: str | None):
    from types import SimpleNamespace

    from lium_protocol import schema

    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(path))
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "show":
            ok = base_snapshot is not None
            return SimpleNamespace(returncode=0 if ok else 128, stdout=base_snapshot or "", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(schema.subprocess, "run", run)
    return calls


def test_check_in_ci_refuses_a_version_below_the_merge_queue_base(monkeypatch, tmp_path, capsys):
    # the finding's case: another protocol PR merged first with a higher number
    group = {"merge_group": {"base_ref": "refs/heads/main"}}
    calls = _ci(monkeypatch, tmp_path, group, _base(_bump(PROTOCOL_VERSION, 3)))
    assert main(["--check"]) == 1
    assert "protocol version against main" in capsys.readouterr().out
    assert calls[0][-1] == "main"
    assert calls[1][-1] == "FETCH_HEAD:lium_protocol/lium_protocol/snapshots/lium_protocol.v2.json"


def test_check_in_ci_passes_a_pr_above_its_base(monkeypatch, tmp_path, capsys):
    pr = {"pull_request": {"base": {"ref": "main"}}}
    _ci(monkeypatch, tmp_path, pr, _base(_bump(PROTOCOL_VERSION, -1), changed=True))
    assert main(["--check"]) == 0
    assert f"protocol version ok: {PROTOCOL_VERSION} follows main" in capsys.readouterr().out


def test_check_on_a_push_or_outside_ci_compares_with_no_base(monkeypatch, tmp_path):
    calls = _ci(monkeypatch, tmp_path, {"ref": "refs/heads/main"}, None)
    assert main(["--check"]) == 0 and calls == []
    monkeypatch.delenv("GITHUB_EVENT_PATH")
    assert main(["--check"]) == 0 and calls == []
