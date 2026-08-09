"""Import `dstack.py` from the dstacktee checkout and hand back the module.

dstack.py is a CLI script, not a package: it is not on cvmd's `sys.path`, it is not pip-installed,
and it does `import host_api` — a sibling import that only resolves when its own directory is
importable. So the loader adds that directory to `sys.path` and loads the file by path.

Nothing here adapts, patches, or wraps the module. The value of importing rather than
re-implementing is that the bytes cvmd measures are produced by the same code the shell path
produces them with; any adaptation layer would be exactly the fork this task forbids.
"""

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

MODULE_NAME = "dstack"
SIBLING_MODULE = "host_api"

# The functions cvmd calls. Checked at import so a dstack.py that has moved on fails here, with
# a legible message, instead of raising AttributeError partway through a launch.
REQUIRED_ATTRIBUTES = (
    "DStackManager",
    "shutdown_instance",
    "start_server",
    "get_qemu_version_string",
)

_lock = threading.Lock()
_cache: dict[Path, ModuleType] = {}


class DStackUnavailable(Exception):
    """dstack.py could not be imported, or is not the shape cvmd calls."""


def _add_to_path(scripts_dir: str) -> None:
    """Make `import host_api` resolve.

    Appended, not prepended: this directory is not ours, and putting a third-party script
    directory ahead of the stdlib would let a file dropped in beside dstack.py shadow a module
    the daemon depends on.
    """
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)


def load_dstack(scripts_dir: Path) -> ModuleType:
    """Import dstack.py from `scripts_dir` and return it. Cached per directory.

    Raises DStackUnavailable with the reason — the caller turns that into a refusal, because a
    host that cannot import the launcher cannot launch anything.
    """
    scripts_dir = Path(scripts_dir).expanduser().resolve()
    cached = _cache.get(scripts_dir)
    if cached is not None:
        return cached

    source = scripts_dir / f"{MODULE_NAME}.py"
    if not source.is_file():
        raise DStackUnavailable(f"no {MODULE_NAME}.py at {source}")
    if not (scripts_dir / f"{SIBLING_MODULE}.py").is_file():
        raise DStackUnavailable(
            f"{source} needs {SIBLING_MODULE}.py beside it and {scripts_dir} has none"
        )

    with _lock:
        # Re-check inside the lock: two concurrent launches would otherwise both exec the
        # module, and the second would replace a module object the first is already using.
        cached = _cache.get(scripts_dir)
        if cached is not None:
            return cached

        _add_to_path(str(scripts_dir))
        spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
        if spec is None or spec.loader is None:
            raise DStackUnavailable(f"cannot build an import spec for {source}")

        module = importlib.util.module_from_spec(spec)
        # Registered before exec so the module behaves exactly like a normal import — a module
        # that is mid-exec and absent from sys.modules re-executes if anything imports it.
        sys.modules[MODULE_NAME] = module
        try:
            # dstack.py calls logging.basicConfig() at import. Under the daemon that is a
            # no-op, because uvicorn has already given the root logger a handler.
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - any import failure is the same refusal
            sys.modules.pop(MODULE_NAME, None)
            raise DStackUnavailable(f"importing {source} failed: {exc}") from exc

        absent = [name for name in REQUIRED_ATTRIBUTES if not hasattr(module, name)]
        if absent:
            sys.modules.pop(MODULE_NAME, None)
            raise DStackUnavailable(
                f"{source} has no {', '.join(absent)} — cvmd calls it as a library and this "
                f"build does not offer the entry points it calls"
            )

        _cache[scripts_dir] = module
        return module
