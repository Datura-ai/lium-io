"""The obfuscator renames every name in a function body that is not on its allowlist, but it never
touches an `import` statement — an alias is not a Name node. So a module the scrape imports and the
allowlist does not carry keeps its import intact while every usage becomes a random name nothing
assigns, and the whole call raises NameError at runtime.

That is how DAH-2514 shipped: `glob` and `socket` arrived with the docker disk breakdown, were not
on the allowlist, and `hard_disk.images/containers/volumes` came back null on all 347 prod
executors. Nothing failed loudly — the scrape swallowed it into an error key.

The check is on the obfuscated output, not on the allowlist: it stays true whatever the next import
happens to be called.
"""

import ast
import builtins
import importlib
from pathlib import Path

import pytest
from miner_jobs.obfuscator import obfuscate_code

SRC = Path(__file__).resolve().parents[1] / "src"


def _names_bound_anywhere(tree: ast.Module) -> set[str]:
    """Every name the module itself makes readable, plus builtins and star-import contents.

    Scope is deliberately ignored — a name bound in one function and read in another is not what
    this test is looking for. Renaming a usage while leaving its binding alone leaves the name
    bound nowhere at all, and that is what is being caught.
    """
    bound = set(dir(builtins))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            bound |= set(dir(importlib.import_module(node.module)))
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            bound.add(node.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            bound |= set(node.names)

    return bound


@pytest.fixture
def obfuscated() -> ast.Module:
    return ast.parse(obfuscate_code((SRC / "miner_jobs" / "machine_scrape.py").read_text()))


def test_obfuscated_scrape_reads_no_name_it_never_binds(obfuscated: ast.Module) -> None:
    # Arrange
    bound = _names_bound_anywhere(obfuscated)

    # Act
    unbound = {
        node.id
        for node in ast.walk(obfuscated)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound
    }

    # Assert
    assert not unbound, (
        f"obfuscated machine_scrape reads names nothing binds: {sorted(unbound)}. "
        "A module imported by the scrape has to be listed in obfuscator.python_keywords, "
        "otherwise its usages are renamed and its import is not."
    )
