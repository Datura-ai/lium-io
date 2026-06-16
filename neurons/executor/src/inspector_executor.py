import argparse
import ctypes
import json
import sys
from typing import Any

LIBINSPECTOR_PATH = "/usr/lib/libinspector.so"


class InspectorExecutor:
    def __init__(self) -> None:
        lib = ctypes.CDLL(LIBINSPECTOR_PATH)
        lib.session_new.restype = ctypes.c_void_p
        lib.session_handshake_reply.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.session_handshake_reply.restype = ctypes.POINTER(ctypes.c_char)
        lib.inspector_collect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.inspector_collect.restype = ctypes.POINTER(ctypes.c_char)
        lib.collector_start.restype = ctypes.c_int
        lib.collector_stop.restype = ctypes.c_int
        lib.session_del.argtypes = [ctypes.c_void_p]
        lib.str_del.argtypes = [ctypes.POINTER(ctypes.c_char)]
        self.lib = lib
        self.session = lib.session_new()

    def __del__(self) -> None:
        if getattr(self, "session", None):
            self.lib.session_del(self.session)

    def _take(self, ptr) -> str:
        if not ptr:
            raise RuntimeError("libinspector call failed")
        try:
            return ctypes.string_at(ptr).decode("utf-8")
        finally:
            self.lib.str_del(ptr)

    def handshake_reply(self, open_json: str) -> str:
        return self._take(self.lib.session_handshake_reply(self.session, open_json.encode()))

    def execute(self, request_cipher: str) -> str:
        return self._take(self.lib.inspector_collect(self.session, request_cipher.encode()))

    def start_collector(self) -> None:
        if self.lib.collector_start() != 0:
            raise RuntimeError("collector_start failed")

    def stop_collector(self) -> None:
        if self.lib.collector_stop() != 0:
            raise RuntimeError("collector_stop failed")


def _emit(ok: bool, *, result: str = "", error: str = "") -> None:
    payload: dict[str, Any] = {"ok": ok}
    if ok:
        payload["result"] = result
    else:
        payload["error"] = error
    print(json.dumps(payload), flush=True)


def _dispatch_inspect(executor: InspectorExecutor, msg: dict[str, Any]) -> str:
    cmd = msg.get("cmd")
    if cmd == "handshake-reply":
        open_json = msg.get("open_json")
        if not open_json:
            raise ValueError("handshake-reply requires open_json")
        return executor.handshake_reply(open_json)
    if cmd == "execute":
        request_cipher = msg.get("request_cipher")
        if not request_cipher:
            raise ValueError("execute requires request_cipher")
        return executor.execute(request_cipher)
    if cmd == "start-collector":
        executor.start_collector()
        return ""
    if cmd == "quit":
        return ""
    raise ValueError(f"unknown cmd: {cmd!r}")


def run_interactive(executor: InspectorExecutor) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit(False, error=f"invalid json: {exc}")
            continue
        if msg.get("cmd") == "quit":
            _emit(True, result="")
            break
        try:
            result = _dispatch_inspect(executor, msg)
            _emit(True, result=result)
        except Exception as exc:
            _emit(False, error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="InspectorExecutor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--interactive", action="store_true", help="Inspect session REPL")
    group.add_argument(
        "--start-collector",
        action="store_true",
        help="Start collector in background and exit",
    )
    group.add_argument(
        "--stop-collector",
        action="store_true",
        help="Stop background collector and exit",
    )

    args = parser.parse_args()
    executor = InspectorExecutor()

    if args.start_collector:
        executor.start_collector()
        print("collector running", flush=True)
        return

    if args.stop_collector:
        executor.stop_collector()
        print("collector stopped", flush=True)
        return

    run_interactive(executor)


if __name__ == "__main__":
    main()
