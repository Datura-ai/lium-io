"""DAH: hosts set up before 16 Nov lack the needrestart guard; the scrape reports it, the check warns."""

from __future__ import annotations

import difflib
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from neurons.validators.src.services.task.checks.needrestart_guard import (
    NEEDRESTART_GUARD_TEXT,
    NeedrestartGuardCheck,
    needrestart_guard_inner,
    needrestart_guard_script,
)
from neurons.validators.src.services.task.messages import NeedrestartGuardMessages as Msg
from neurons.validators.tests.helpers import build_scrape_namespace

from tests.helpers import build_state
from tests.test_sysbox_required_check import _rented_data

SRC = Path(__file__).resolve().parents[1] / "src"
SETUP_SCRIPT = Path(__file__).resolve().parents[2] / "executor" / "nvidia_docker_sysbox_setup.sh"
GUARD = "/etc/needrestart/conf.d/lium-container-runtime.conf"
# Ubuntu 24.04's needrestart 3.6 config, verbatim: data, then the block that evals conf.d/*.conf
STOCK_MAIN = (
    Path(__file__).resolve().parent / "fixtures" / "needrestart_ubuntu2404.conf"
).read_text()
LOADER = STOCK_MAIN[STOCK_MAIN.index("if(-d q(/etc/needrestart/conf.d))") :]
ALL = ["sysbox", "docker", "containerd", "nvidia-fabricmanager", "nvidia-persistenced"]


def _read(host_root: Path):
    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {
            "get_needrestart_guard",
            "needrestart_data_line",
            "NEEDRESTART_GUARD_FILE",
            "NEEDRESTART_GUARD_SERVICES",
            "NEEDRESTART_GUARD_UNITS",
        },
        {"difflib": difflib, "os": os, "re": re},
    )
    return ns["get_needrestart_guard"](str(host_root))[:2]


def _reason(host_root: Path):
    ns = build_scrape_namespace(
        SRC / "miner_jobs" / "machine_scrape.py",
        {
            "get_needrestart_guard",
            "needrestart_data_line",
            "NEEDRESTART_GUARD_FILE",
            "NEEDRESTART_GUARD_SERVICES",
            "NEEDRESTART_GUARD_UNITS",
        },
        {"difflib": difflib, "os": os, "re": re},
    )
    return ns["get_needrestart_guard"](str(host_root))[2]


def _host(tmp_path: Path, guard_text: str | None, needrestart: bool = True) -> Path:
    if needrestart:
        (tmp_path / "etc/needrestart/conf.d").mkdir(parents=True)
        (tmp_path / "etc/needrestart/needrestart.conf").write_text(STOCK_MAIN)
    if guard_text is not None:
        (tmp_path / GUARD.lstrip("/")).write_text(guard_text)
    return tmp_path


def _setup_script_guard() -> str:
    # the exact file the setup script writes, so the scrape and the script cannot drift apart
    script = SETUP_SCRIPT.read_text()
    return script.split("<<'NRCONF'\n", 1)[1].split("\nNRCONF\n", 1)[0]


def test_the_file_the_setup_script_writes_reads_as_present(tmp_path):
    assert _read(_host(tmp_path, _setup_script_guard())) == ("present", [])


@pytest.mark.parametrize(
    "release",
    # each distro's stock needrestart.conf from its needrestart package, verbatim
    ["ubuntu2004", "ubuntu2204", "ubuntu2404", "debian11", "debian12"],
)
def test_the_guard_reads_as_present_under_each_stock_main_config(tmp_path, release):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/needrestart.conf").write_text(
        (Path(__file__).resolve().parent / "fixtures" / f"needrestart_{release}.conf").read_text()
    )

    assert _read(host) == ("present", [])


def test_a_host_set_up_before_the_guard_reads_as_missing(tmp_path):
    status, unguarded = _read(_host(tmp_path, None))

    assert status == "missing"
    assert "sysbox" in unguarded and "nvidia-fabricmanager" in unguarded


def test_a_guard_without_one_service_reads_as_partial(tmp_path):
    text = "\n".join(
        line for line in _setup_script_guard().splitlines() if "containerd" not in line
    )

    assert _read(_host(tmp_path, text)) == ("partial", ["containerd"])


@pytest.mark.parametrize("value", ["0", "'0'", '"0"'])
def test_a_later_zero_in_any_quoting_keeps_the_guard(tmp_path, value):
    text = _setup_script_guard() + f"\n$nrconf{{override_rc}}->{{qr(^containerd)}} = {value};\n"

    assert _read(_host(tmp_path, text)) == ("present", [])


def test_a_commented_out_rule_does_not_count(tmp_path):
    text = _setup_script_guard().replace(
        "$nrconf{override_rc}->{qr(^containerd)}", "# $nrconf{override_rc}->{qr(^containerd)}"
    )

    assert _read(_host(tmp_path, text)) == ("partial", ["containerd"])


@pytest.mark.parametrize("value", ["1", "'1'", '"1"', "\"'0'\"", "'\"0\"'"])
def test_a_later_assignment_that_re_enables_a_restart_wins(tmp_path, value):
    text = _setup_script_guard() + f"\n$nrconf{{override_rc}}->{{qr(^containerd)}} = {value};\n"

    assert _read(_host(tmp_path, text)) == ("partial", ["containerd"])


@pytest.mark.parametrize(
    "extra",
    [
        "$nrconf{override_rc}->{qr/^containerd/} = 1;",  # another delimiter, same hash key in Perl
        "$nrconf{override_rc}->{qr(^sysbox)} = 0; $nrconf{override_rc}->{qr(^containerd)} = 1;",  # same line
        "$nrconf{override_rc} = {};",  # whole-map reset
        "$nrconf{override_rc}{qr(^docker)} = 1;",  # arrow omitted
        "%nrconf = ();",
    ],
)
def test_live_perl_the_parser_does_not_recognise_guards_nothing(tmp_path, extra):
    text = _setup_script_guard() + "\n" + extra + "\n"

    assert _read(_host(tmp_path, text)) == (
        "partial",
        list(("sysbox", "docker", "containerd", "nvidia-fabricmanager", "nvidia-persistenced")),
    )


def test_blank_lines_comments_and_trailing_comments_keep_the_guard(tmp_path):
    text = "# lium\n\n" + _setup_script_guard().replace(" = 0;", " = 0;  # keep pods up") + "\n\n"

    assert _read(_host(tmp_path, text)) == ("present", [])


@pytest.mark.parametrize(
    "later",
    [
        "$nrconf{override_rc}->{qr(^containerd)} = 1;",
        "%nrconf = ();",
        "undef %nrconf;",
        "delete $nrconf{override_rc};",
        "$nrconf{override_rc} = {};",
        "$nrconf{restart} = do { %nrconf = (); 'a' };",
        "$nrconf{restart} = 'a'; %nrconf = ();",
        "$nrconf{restart} = clear_guard;",  # a sub an earlier file defines
        "$nrconf{restart} = reset 'n';",
        "$nrconf{restart} = do '/tmp/x.pl';",
        "$nrconf{restart} = eval 'undef ' . chr 37 . 'nrconf';",
        '$nrconf{restart} = "@{[ %nrconf = () ]}";',
    ],
)
def test_perl_that_may_reach_the_guard_in_a_later_conf_file_makes_it_partial(tmp_path, later):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/conf.d/zz-local.conf").write_text(later + "\n")

    assert _read(host) == (
        "partial",
        ["sysbox", "docker", "containerd", "nvidia-fabricmanager", "nvidia-persistenced"],
    )


def test_a_later_conf_file_with_only_comments_or_plain_settings_of_other_keys_keeps_it(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    # the usual non-interactive setup: restart automatically, kernel hints off
    (host / "etc/needrestart/conf.d/zz-local.conf").write_text(
        "# $nrconf{override_rc} = {};\n\n$nrconf{restart} = 'a';\n$nrconf{kernelhints} = -1;  # quiet\n"
        '$nrconf{ui} = "NeedRestart::UI::stdio";\n$nrconf{verbosity} = 0;\n'
    )

    assert _read(host) == ("present", [])


@pytest.mark.parametrize("value", ["1", "'1'"])
def test_an_overlapping_restart_rule_in_the_guard_file_unguards_the_unit(tmp_path, value):
    text = _setup_script_guard() + f"\n$nrconf{{override_rc}}->{{qr(^container)}} = {value};\n"

    assert _read(_host(tmp_path, text)) == ("partial", ["containerd"])


def test_an_overlapping_restart_rule_in_an_earlier_file_unguards_the_unit(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/needrestart.conf").write_text(
        "$nrconf{override_rc} = {\n    qr(^dbus) => 0,\n    qr(^nvidia) => 1,\n    qr(^containerd) => 1,\n};\n"
        + LOADER
    )

    # qr(^containerd) is the guard file's own key, which it overrides; qr(^nvidia) is not
    assert _read(host) == ("partial", ["nvidia-fabricmanager", "nvidia-persistenced"])


def test_a_stock_main_config_with_only_zero_rules_keeps_the_guard(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/needrestart.conf").write_text(
        "$nrconf{override_rc} = {\n    qr(^dbus) => 0,\n    qr(^docker) => 0,\n    qr(^systemd-) => 0,\n};\n"
        "$nrconf{restart} = 'i';\n$nrconf{override_cont} = {};\n" + LOADER
    )

    assert _read(host) == ("present", [])


def test_the_remediation_carries_every_line_the_setup_script_writes():
    for line in _setup_script_guard().splitlines():
        if line.startswith("$nrconf"):
            assert line in Msg.MISSING.remediation


def test_a_host_without_needrestart_needs_no_guard(tmp_path):
    assert _read(_host(tmp_path, None, needrestart=False)) == ("no_needrestart", [])


def test_the_setup_script_guard_on_a_host_without_needrestart_needs_no_guard(tmp_path):
    # the setup script creates conf.d and writes the guard whether or not needrestart is installed
    (tmp_path / "etc/needrestart/conf.d").mkdir(parents=True)
    assert _read(_host(tmp_path, _setup_script_guard(), needrestart=False)) == ("no_needrestart", [])


def test_no_host_view_raises_so_the_scrape_reports_unknown():
    with pytest.raises(RuntimeError):
        _read(Path("/"))


def _ssh(stdout: str):
    return SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(exit_status=0, stdout=stdout, stderr="")))


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["missing", "partial"])
@pytest.mark.parametrize("rented", [False, True])
async def test_an_unguarded_host_gets_the_guard_written_and_keeps_its_score(context_factory, guard, rented):
    specs = {"needrestart_guard": guard, "needrestart_guard_unguarded": ["sysbox"]}
    state = build_state(specs=specs, rented_data=_rented_data() if rented else None)
    ssh = _ssh("needrestart_guard_set host\n")

    result = await NeedrestartGuardCheck().run(context_factory(state=state, ssh=ssh))

    assert result.passed is True
    assert result.event.reason_code == Msg.WRITTEN.reason
    assert not result.updates
    assert ssh.run.await_args.args[0] == needrestart_guard_script()


@pytest.mark.asyncio
@pytest.mark.parametrize("stdout", ["needrestart_guard_failed host write_denied\n", "needrestart_guard_skipped no_nsenter\n"])
async def test_a_host_the_write_does_not_reach_warns_and_keeps_its_score(context_factory, stdout):
    state = build_state(specs={"needrestart_guard": "missing"})

    result = await NeedrestartGuardCheck().run(context_factory(state=state, ssh=_ssh(stdout)))

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason
    assert result.event.severity == "warning"
    assert GUARD in Msg.MISSING.remediation


@pytest.mark.asyncio
async def test_an_ssh_error_in_the_write_never_fails_the_check(context_factory):
    ssh = SimpleNamespace(run=AsyncMock(side_effect=OSError("connection lost")))

    result = await NeedrestartGuardCheck().run(context_factory(state=build_state(specs={"needrestart_guard": "missing"}), ssh=ssh))

    assert result.passed is True
    assert result.event.reason_code == Msg.MISSING.reason


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(context_factory):
    ssh = _ssh("")

    result = await NeedrestartGuardCheck(write=False).run(
        context_factory(state=build_state(specs={"needrestart_guard": "missing"}), ssh=ssh)
    )

    assert result.event.reason_code == Msg.MISSING.reason
    ssh.run.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "specs,reason",
    [
        ({"needrestart_guard": "present"}, Msg.OK.reason),
        ({"needrestart_guard": "no_needrestart"}, Msg.OK.reason),
        ({}, Msg.UNKNOWN.reason),  # old scrape or no host view, and the write did not run: unknown, never a warning
    ],
)
async def test_guarded_or_unknown_hosts_pass_quietly(context_factory, specs, reason):
    ssh = _ssh("needrestart_guard_skipped no_needrestart\n")

    result = await NeedrestartGuardCheck().run(context_factory(state=build_state(specs=specs), ssh=ssh))

    assert result.passed is True
    assert result.event.reason_code == reason
    assert ssh.run.await_count == (0 if specs else 1)


@pytest.mark.parametrize("pipeline", ["build_checks", "build_fast_path_checks", "build_dry_run_checks"])
def test_the_guard_runs_right_after_the_scrape_before_any_fatal_check(pipeline):
    from neurons.validators.src.services.task.pipeline_factory import PipelineFactory

    names = [type(c).__name__ for c in getattr(PipelineFactory, pipeline)()]

    assert names[names.index("MachineSpecScrapeCheck") + 1] == "NeedrestartGuardCheck"


@pytest.mark.parametrize(
    "code",
    [
        "tie %nrconf, 'T';",  # a STORE can then drop the guard when a later file writes any key
        "package T;",
        "sub STORE { }",
        "BEGIN { %nrconf = () }",
        "$nrconf{override_rc} = {\n    qr(^dbus) => 0, tie(%nrconf, 'T'),\n};",
        "$nrconf{override_rc} = {\n    qr(^a(?{ %nrconf = () })) => 0,\n};",
        "$nrconf{override_rc} = {\n    qr(^${\\ undef}) => 0,\n};",
        "$nrconf{blacklist} = [\n    qr(^@{[ %nrconf = () ]}),\n];",
    ],
)
@pytest.mark.parametrize("where", ["main", "earlier conf.d"])
def test_code_in_config_loaded_before_the_guard_makes_it_partial(tmp_path, code, where):
    host = _host(tmp_path, _setup_script_guard())
    if where == "main":
        (host / "etc/needrestart/needrestart.conf").write_text(code + "\n" + STOCK_MAIN)
    else:
        (host / "etc/needrestart/conf.d/00-local.conf").write_text(code + "\n")

    assert _read(host) == ("partial", ALL)
    # DAH-6566 follow-up: the event names the file and the line the scrape could not prove
    path = "needrestart.conf" if where == "main" else "conf.d/00-local.conf"
    assert re.fullmatch(rf"/etc/needrestart/{re.escape(path)} line [12], loaded before the guard, is not plain data", _reason(host))


@pytest.mark.parametrize(
    "main",
    [
        None,  # needrestart has no config to load conf.d from
        "$nrconf{restart} = 'a';\n",  # no loader: conf.d, and so the guard, is never read
        STOCK_MAIN.replace(
            "sort </etc/needrestart/conf.d/*.conf>", "sort </etc/needrestart/other/*.conf>"
        ),
        STOCK_MAIN + "%nrconf = ();\n",  # runs after the guard file
    ],
)
def test_a_main_config_that_does_not_load_the_guard_last_makes_it_partial(tmp_path, main):
    host = _host(tmp_path, _setup_script_guard())
    path = host / "etc/needrestart/needrestart.conf"
    path.unlink()
    if main is not None:
        path.write_text(main)
    else:  # needrestart is installed, its config is not
        (host / "usr/sbin").mkdir(parents=True)
        (host / "usr/sbin/needrestart").write_text("")

    assert _read(host) == ("partial", ALL)
    assert "needrestart.conf" in _reason(host)


@pytest.mark.parametrize(
    "entry",
    [
        "qr<^container> => 1,",
        "qr!^container! => 1,",
        "qr(^container) => 1,",
        "qr{^container} => '1',",
        "'^container' => 1,",  # needrestart matches each key as a regex, a plain string too
        "qr(^container),",  # no value: not a zero
    ],
)
def test_an_overlapping_rule_in_any_delimiter_in_an_earlier_block_unguards_the_unit(
    tmp_path, entry
):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/needrestart.conf").write_text(
        "$nrconf{override_rc} = {\n    qr(^dbus) => 0,\n    " + entry + "\n};\n" + LOADER
    )

    assert _read(host) == ("partial", ["containerd"])


@pytest.mark.parametrize(
    ("entry", "unguarded"),
    [
        # a string key is its own Perl hash key: the guard's qr(^containerd) does not overwrite it
        ("'^containerd' => 1,", ["containerd"]),
        ('"^docker" => 1,', ["docker"]),
        # rules that match the sysbox package's real units, not the name sysbox.service
        ("qr(^sysbox-) => 1,", ["sysbox"]),
        ("qr(mgr) => 1,", ["sysbox"]),
        ("qr(-fs\\.service$) => 1,", ["sysbox"]),
    ],
)
def test_an_earlier_rule_the_guard_does_not_overwrite_unguards_its_unit(tmp_path, entry, unguarded):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/needrestart.conf").write_text("$nrconf{override_rc} = {\n    " + entry + "\n};\n" + LOADER)

    assert _read(host) == ("partial", unguarded)


@pytest.mark.parametrize("entry", ["qr(^containerd) => 1,", "qr<^containerd> => 1,", "qr{^sysbox} => 1,"])
def test_an_earlier_qr_key_the_guard_sets_again_is_overwritten(tmp_path, entry):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/needrestart.conf").write_text("$nrconf{override_rc} = {\n    " + entry + "\n};\n" + LOADER)

    assert _read(host) == ("present", [])


def test_an_overlapping_entry_outside_override_rc_is_not_a_restart_rule(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/needrestart.conf").write_text(
        "$nrconf{override_cont} = {\n    qr<^container> => 1,\n};\n" + LOADER
    )

    assert _read(host) == ("present", [])


def test_a_host_without_the_guard_reads_present_once_the_validator_writes_it(tmp_path):
    host = _host(tmp_path, None)
    assert _read(host)[0] == "missing"

    subprocess.run(["sh", "-c", needrestart_guard_inner(str(host))], check=True)

    assert _read(host) == ("present", [])


def test_the_written_guard_is_exactly_the_setup_scripts(tmp_path):
    assert NEEDRESTART_GUARD_TEXT == _setup_script_guard()
    host = _host(tmp_path, None)

    first = subprocess.run(["sh", "-c", needrestart_guard_inner(str(host))], capture_output=True, text=True, check=True)
    second = subprocess.run(["sh", "-c", needrestart_guard_inner(str(host))], capture_output=True, text=True, check=True)

    assert (first.stdout, second.stdout) == ("needrestart_guard_set host\n", "")
    assert (host / GUARD.lstrip("/")).read_text() == _setup_script_guard() + "\n"


def test_no_guard_is_written_without_needrestart(tmp_path):
    out = subprocess.run(["sh", "-c", needrestart_guard_inner(str(tmp_path))], capture_output=True, text=True, check=True)

    assert out.stdout == "needrestart_guard_skipped no_needrestart\n"
    assert not (tmp_path / "etc").exists()


def test_a_readable_config_gives_no_reason_and_a_guard_line_it_cannot_read_is_named(tmp_path):
    assert _reason(_host(tmp_path, _setup_script_guard())) is None

    host = _host(tmp_path / "odd", _setup_script_guard() + "\n$nrconf{override_rc}{qr(^x)} = 0;\n")

    assert _read(host) == ("partial", ALL)
    assert _reason(host) == f"{GUARD} line {len(_setup_script_guard().splitlines()) + 1} is not a guard line"


@pytest.mark.asyncio
async def test_the_partial_reason_reaches_the_event(context_factory):
    specs = {"needrestart_guard": "partial", "needrestart_guard_unguarded": ALL, "needrestart_guard_reason": "why"}
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(exit_status=0, stdout="needrestart_guard_unchanged\n", stderr="")

    result = await NeedrestartGuardCheck().run(context_factory(state=build_state(specs=specs), ssh=ssh))

    assert result.event.what_we_saw["reason"] == "why"


SECRET = "SAMPLE-SECRET-MARKER"  # a fake, recognisable value; it must not reach any output


@pytest.mark.parametrize(
    ("file", "text"),
    [
        (GUARD, None),  # appended to the guard file
        ("/etc/needrestart/needrestart.conf", f"my $webhook_token = '{SECRET}';\n" + STOCK_MAIN),
        ("/etc/needrestart/needrestart.conf", STOCK_MAIN + f"my $webhook_token = '{SECRET}';\n"),
        ("/etc/needrestart/conf.d/00-local.conf", f"my $webhook_token = '{SECRET}';\n"),
        ("/etc/needrestart/conf.d/zz-local.conf", f"my $webhook_token = '{SECRET}';\n"),
    ],
)
@pytest.mark.asyncio
async def test_a_secret_on_a_config_line_reaches_neither_the_specs_nor_the_event(tmp_path, context_factory, file, text):
    # lio#2333 review: a reason names file, line and category, never the line's text
    host = _host(tmp_path, _setup_script_guard() + (f"\nmy $webhook_token = '{SECRET}';\n" if text is None else ""))
    if text is not None:
        (host / file.lstrip("/")).write_text(text)
    guard, unguarded = _read(host)
    reason = _reason(host)
    assert guard == "partial" and file in reason and " line " in reason
    specs = {"needrestart_guard": guard, "needrestart_guard_unguarded": unguarded, "needrestart_guard_reason": reason}
    assert SECRET not in repr(specs)
    ssh = AsyncMock()
    ssh.run.return_value = SimpleNamespace(exit_status=0, stdout="needrestart_guard_unchanged\n", stderr="")

    result = await NeedrestartGuardCheck().run(context_factory(state=build_state(specs=specs), ssh=ssh))

    assert SECRET not in repr(result.event)
    assert result.event.what_we_saw["reason"] == reason


def _rules(host: Path, *entries: str) -> None:
    (host / "etc/needrestart/needrestart.conf").write_text(
        "$nrconf{override_rc} = {\n" + "".join(f"    {e}\n" for e in entries) + "};\n" + LOADER
    )


def test_a_rule_pattern_python_cannot_read_is_named_with_its_line(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    _rules(host, "qr{^sshd\\K} => 1,")

    assert _read(host) == ("partial", ALL)
    assert _reason(host) == "/etc/needrestart/needrestart.conf line 2 has a rule pattern the scrape cannot evaluate"


def test_a_computed_guard_value_is_named_with_its_line(tmp_path):
    guard = _setup_script_guard().replace("qr(^docker)} = 0;", "qr(^docker)} = (1 - 1);")
    host = _host(tmp_path, guard)
    line = next(n for n, ln in enumerate(guard.splitlines(), 1) if "(1 - 1)" in ln)

    assert _read(host) == ("partial", ["docker"])
    assert _reason(host) == f"{GUARD} line {line} sets docker to a computed value the scrape cannot evaluate"


def test_a_proven_overlap_needs_no_reason_and_an_unproven_rule_beside_it_keeps_its_own(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    _rules(host, "qr(^container) => 1,")
    assert (_read(host), _reason(host)) == (("partial", ["containerd"]), None)

    guard = _setup_script_guard().replace("qr(^docker)} = 0;", "qr(^docker)} = (1 - 1);")
    (host / GUARD.lstrip("/")).write_text(guard)
    assert _read(host) == ("partial", ["docker", "containerd"])
    assert "sets docker to a computed value" in _reason(host)

    _rules(host, "qr(^container) => 1,", "qr{^sshd\\K} => 1,")
    (host / GUARD.lstrip("/")).write_text(_setup_script_guard())
    assert _read(host) == ("partial", ALL)
    assert _reason(host) == "/etc/needrestart/needrestart.conf line 3 has a rule pattern the scrape cannot evaluate"


def test_a_changed_loader_line_is_named_and_a_missing_loader_names_the_file(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    main = host / "etc/needrestart/needrestart.conf"
    glob_line = next(n for n, ln in enumerate(STOCK_MAIN.splitlines(), 1) if "conf.d/*.conf>" in ln)
    main.write_text(STOCK_MAIN.replace("conf.d/*.conf>", "other/*.conf>"))
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {glob_line} differs from the stock conf.d loader"

    # the opener itself edited: the rest of the loader still anchors it, so the opener's line is named
    opener = next(n for n, ln in enumerate(STOCK_MAIN.splitlines(), 1) if "q(/etc/needrestart/conf.d))" in ln)
    main.write_text(STOCK_MAIN.replace("q(/etc/needrestart/conf.d))", "'/etc/needrestart/conf.d')"))
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener} differs from the stock conf.d loader"

    # the opener rewritten to an unrelated condition (an environment flag): it still opens the loader's block, so it
    # is named, not the unchanged glob below it
    stock_opener = STOCK_MAIN.splitlines()[opener - 1]
    main.write_text(STOCK_MAIN.replace(stock_opener, "if (!$ENV{NR_SKIP_CONFD}) {"))
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener} differs from the stock conf.d loader"

    # the edited opener with a trailing comment still opens the block, so it is named, not the unchanged glob
    main.write_text(STOCK_MAIN.replace(stock_opener, "if (-d '/etc/needrestart/conf.d') { # local"))
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener} differs from the stock conf.d loader"

    # the opener's quoting and the glob both changed: the stock log line below still anchors the loader
    main.write_text(
        STOCK_MAIN.replace("q(/etc/needrestart/conf.d))", "'/etc/needrestart/conf.d')").replace(
            "conf.d/*.conf>", "other/*.conf>"
        )
    )
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener} differs from the stock conf.d loader"

    # a verbatim copy of the stock loop for another directory above the conf.d loader: neither its opener nor its log
    # line anchors the conf.d loader, so the changed conf.d line is named, with or without the opener also changed
    site_loop = LOADER.replace("/etc/needrestart/conf.d", "/etc/needrestart/site.d")
    shift = len(site_loop.splitlines())
    main.write_text(site_loop + STOCK_MAIN.replace("conf.d/*.conf>", "other/*.conf>"))
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {glob_line + shift} differs from the stock conf.d loader"
    quoted = STOCK_MAIN.replace("q(/etc/needrestart/conf.d))", "'/etc/needrestart/conf.d')")
    main.write_text(site_loop + quoted.replace("conf.d/*.conf>", "other/*.conf>"))
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener + shift} differs from the stock conf.d loader"

    # the opener edited and a block with an exact stock opener appended below: that block matches one loader line,
    # the real loader's glob and the lines after it match more, so the edited opener is named, not the appended block
    main.write_text(quoted + stock_opener + "\n$nrconf{verbosity} = 0;\n}\n")
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener} differs from the stock conf.d loader"

    lines = STOCK_MAIN.splitlines()
    # a line inserted inside the loader, after the glob or after the opener: the inserted line is named, not the
    # unchanged loader line above it
    glob_text = lines[glob_line - 1]
    main.write_text(STOCK_MAIN.replace(glob_text, glob_text + "\nnext unless -r $fn;"))
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {glob_line + 1} differs from the stock conf.d loader"
    main.write_text(STOCK_MAIN.replace(stock_opener, stock_opener + "\n$nrconf{verbosity} = 0;"))
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener + 1} differs from the stock conf.d loader"

    # the opener and the glob swapped: the glob, now first, is named, not the unchanged log line below them
    main.write_text(STOCK_MAIN.replace(stock_opener + "\n" + glob_text, glob_text + "\n" + stock_opener))
    assert stock_opener + "\n" + glob_text in STOCK_MAIN
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener} differs from the stock conf.d loader"

    # a trailing comment on each of the opener, glob and log line: the loader is still found, and its first
    # commented line is named
    log_text = lines[glob_line]
    assert "print STDERR" in log_text
    commented = STOCK_MAIN
    for text in (stock_opener, glob_text, log_text):
        commented = commented.replace(text, text + " # local")
    main.write_text(commented)
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener} differs from the stock conf.d loader"

    # the optional log line deleted: the line now in its place (the eval) is named, after the intact opener and glob
    lines = STOCK_MAIN.splitlines()
    log_line = next(n for n, ln in enumerate(lines, 1) if "print STDERR" in ln and "eval $fn" in ln)
    main.write_text("\n".join(ln for n, ln in enumerate(lines, 1) if n != log_line) + "\n")
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {log_line} differs from the stock conf.d loader"

    # the directory check and its brace deleted: the glob, where the loader starts, is named, not a setting above it
    last_brace = max(n for n, ln in enumerate(lines, 1) if ln.strip() == "}")
    main.write_text("\n".join(ln for n, ln in enumerate(lines, 1) if n not in (opener, last_brace)) + "\n")
    # the glob moved up into the deleted opener's line number
    assert _reason(host) == f"/etc/needrestart/needrestart.conf line {opener} differs from the stock conf.d loader"
    assert "conf.d/*.conf>" in (host / "etc/needrestart/needrestart.conf").read_text().splitlines()[opener - 1]

    # a site loader's eval line above a stock loader whose glob changed: the glob is named, not the site loader
    site = "my $fn = '/etc/needrestart/site.conf';\nif(-f $fn) {\neval do { local(@ARGV, $/) = $fn; <>};\n}\n"
    main.write_text(site + STOCK_MAIN.replace("conf.d/*.conf>", "other/*.conf>"))
    assert _reason(host) == (
        f"/etc/needrestart/needrestart.conf line {glob_line + 4} differs from the stock conf.d loader"
    )

    # only a commented opener: it anchors the loader once its comment is stripped, but no line aligns as written,
    # so it is named (the stand's crash at 409768f)
    main.write_text("if(-d q(/etc/needrestart/conf.d)) { # c\n")
    assert _read(host) == ("partial", ALL)
    assert _reason(host) == "/etc/needrestart/needrestart.conf line 1 differs from the stock conf.d loader"
    main.write_text("$nrconf{a} = 1;\nforeach my $fn (sort </etc/needrestart/conf.d/*.conf>) { # c\n")
    assert _reason(host) == "/etc/needrestart/needrestart.conf line 2 differs from the stock conf.d loader"

    main.write_text("$nrconf{restart} = 'a';\nif (1) {\n}\n}\n")  # stray braces are not a loader
    assert _reason(host) == "/etc/needrestart/needrestart.conf has no stock conf.d loader"


def test_only_the_last_assignment_to_a_guard_file_rule_counts(tmp_path):
    # Perl keeps the last value of a key: a superseded 1 does not prove docker unguarded, so the computed docker
    # value keeps its reason
    guard = _setup_script_guard().replace("qr(^docker)} = 0;", "qr(^docker)} = (1 - 1);")
    guard += "\n$nrconf{override_rc}->{qr(^dock)} = 1;\n$nrconf{override_rc}->{qr(^dock)} = 0;\n"
    host = _host(tmp_path, guard)
    line = next(n for n, ln in enumerate(guard.splitlines(), 1) if "(1 - 1)" in ln)
    assert _read(host) == ("partial", ["docker"])
    assert _reason(host) == f"{GUARD} line {line} sets docker to a computed value the scrape cannot evaluate"


@pytest.mark.parametrize(("value", "perl_true"), [("0.0", False), ("''", False), ("-0", False), ('"0.0"', True)])
def test_a_broader_rule_proves_the_overlap_only_when_perl_reads_its_value_as_true(tmp_path, value, perl_true):
    guard = _setup_script_guard().replace("qr(^docker)} = 0;", "qr(^docker)} = (1 - 1);")
    host = _host(tmp_path, guard)
    _rules(host, f"qr(^dock) => {value},")
    assert _read(host) == ("partial", ["docker"])
    if perl_true:
        assert _reason(host) is None
    else:
        assert "sets docker to a computed value" in _reason(host)


def test_a_dotfile_in_conf_d_is_not_loaded_and_never_named(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    (host / "etc/needrestart/conf.d/.backup.conf").write_text("system('true');\n")
    assert _read(host) == ("present", [])
    (host / "etc/needrestart/conf.d/00-local.conf").write_text("system('true');\n")
    assert _reason(host) == "/etc/needrestart/conf.d/00-local.conf line 1, loaded before the guard, is not plain data"


def test_a_guard_value_that_runs_code_is_named_with_its_line(tmp_path):
    # a chained assignment sets another rule as it runs: a later line resetting the first key does not undo that
    guard = _setup_script_guard() + (
        "\n$nrconf{override_rc}->{qr(^dock)} = $nrconf{override_rc}->{qr(^docker)} = 1;\n"
        "$nrconf{override_rc}->{qr(^dock)} = 0;\n"
    )
    host = _host(tmp_path, guard)
    line = next(n for n, ln in enumerate(guard.splitlines(), 1) if "= $nrconf" in ln)
    assert _read(host) == ("partial", ALL)
    assert _reason(host) == f"{GUARD} line {line} runs code in its value"



@pytest.mark.parametrize("value", ["!1", "!0", "~0", "1 ? 1 : 0", "2 * (3 - 1) % 2"])
def test_an_operator_only_rule_for_another_service_leaves_the_guard_present(tmp_path, value):
    host = _host(tmp_path, _setup_script_guard() + f"\n$nrconf{{override_rc}}->{{qr(^cron)}} = {value};\n")
    assert _read(host) == ("present", [])


def test_an_operator_only_value_on_a_guarded_service_is_not_proven_zero(tmp_path):
    guard = _setup_script_guard().replace("qr(^docker)} = 0;", "qr(^docker)} = !1;")
    host = _host(tmp_path, guard)
    line = next(n for n, ln in enumerate(guard.splitlines(), 1) if "qr(^docker)} = !1;" in ln)
    assert _read(host) == ("partial", ["docker"])
    assert _reason(host) == f"{GUARD} line {line} sets docker to a computed value the scrape cannot evaluate"

def test_a_heredoc_value_that_swallows_the_next_guard_lines_runs_code(tmp_path):
    # Perl reads the lines after <<'' up to a blank line as the string, so the docker reset below never runs
    guard = _setup_script_guard() + (
        "\n$nrconf{override_rc}->{qr(^cron)} = <<'';\n"
        "$nrconf{override_rc}->{qr(^docker)} = 0;\n\n"
    )
    host = _host(tmp_path, guard)
    line = next(n for n, ln in enumerate(guard.splitlines(), 1) if "<<''" in ln)
    assert _read(host) == ("partial", ALL)
    assert _reason(host) == f"{GUARD} line {line} runs code in its value"


@pytest.mark.parametrize("value", ["%-", "*+", "/0/", "1 - %+", "1 ** 2", "<STDIN>", "1 x 2"])
def test_a_sigil_or_match_where_a_term_belongs_runs_code(tmp_path, value):
    host = _host(tmp_path, _setup_script_guard() + f"\n$nrconf{{override_rc}}->{{qr(^cron)}} = {value};\n")
    assert _read(host)[0] == "partial"


@pytest.mark.parametrize("value", ["0.0", "-0", "''"])
def test_only_the_guards_own_zeros_keep_a_service_guarded(tmp_path, value):
    guard = _setup_script_guard().replace("qr(^docker)} = 0;", f"qr(^docker)}} = {value};")
    host = _host(tmp_path, guard)
    line = next(n for n, ln in enumerate(guard.splitlines(), 1) if f"qr(^docker)}} = {value};" in ln)
    assert _read(host) == ("partial", ["docker"])
    assert _reason(host) == f"{GUARD} line {line} sets docker to a zero other than the guard's 0"

    host = _host(tmp_path / "broader", _setup_script_guard())
    _rules(host, f"qr(^dock) => {value},")
    assert _read(host) == ("partial", ["docker"])
    assert _reason(host) == "/etc/needrestart/needrestart.conf line 2 sets a zero other than the guard's 0"


def test_a_quoted_key_keeps_its_inner_quotes(tmp_path):
    host = _host(tmp_path, _setup_script_guard())
    _rules(host, '"^docker" => 1,', "\"'^docker'\" => 0,")
    assert _read(host) == ("partial", ["docker"])
    assert _reason(host) is None
