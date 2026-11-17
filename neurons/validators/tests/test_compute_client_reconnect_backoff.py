import itertools
import re
from pathlib import Path

from websockets.client import backoff

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def _image_backoff_env() -> dict[str, float]:
    found = re.findall(r"WEBSOCKETS_BACKOFF_(\w+)=([\d.]+)", DOCKERFILE.read_text())
    return {name.lower(): float(value) for name, value in found}


def test_image_caps_reconnect_waits_under_the_backends_rent_wait() -> None:
    env = _image_backoff_env()
    assert env["max_delay"] < 10  # the backend's rent wait for a validator connection
    delays = list(itertools.islice(backoff(**env), 50))
    assert all(0 <= d <= env["max_delay"] for d in delays)
    assert delays[0] <= 1 and delays[1] <= 1  # first retries are quick
    assert delays[-1] == env["max_delay"]
