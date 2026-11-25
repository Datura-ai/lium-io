"""Every outbound request of this app is made here. It dials only addresses from its own settings (the backend,
the chain), so it has one kind of helper: a trusted one. An address someone outside Lium supplies needs a helper that
applies datura's public-address rule; ruff (TID251, root pyproject.toml) refuses the raw constructors anywhere else, so
a new call site picks one."""

import aiohttp
import requests


def trusted_session(**kwargs) -> aiohttp.ClientSession:
    """An aiohttp session for an address from this app's own settings."""
    return aiohttp.ClientSession(**kwargs)


def trusted_get(url: str, **kwargs) -> requests.Response:
    """``requests.get`` for an address from this app's own settings."""
    return requests.get(url, **kwargs)
