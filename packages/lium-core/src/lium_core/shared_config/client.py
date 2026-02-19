import logging
import threading
import time

import requests

from lium_core.shared_config.defaults import DEFAULT_SHARED_CONFIG
from lium_core.shared_config.model import SharedConfig
from lium_core.shared_config.utils import dict_diff

logger = logging.getLogger(__name__)


class SharedConfigClient:
    def __init__(self, api_url: str, refresh_interval: int = 60):
        self._api_url = api_url
        self._refresh_interval = refresh_interval
        self._lock = threading.Lock()
        self._running = True

        fetched = self._fetch()
        if fetched:
            logger.info("Shared config fetched from API: %s", self._api_url)
            self._config = fetched
        else:
            logger.warning("Using default shared config (API unavailable: %s)", self._api_url)
            self._config = DEFAULT_SHARED_CONFIG

        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    @property
    def config(self) -> SharedConfig:
        with self._lock:
            return self._config

    def _fetch(self) -> SharedConfig | None:
        """Fetch config from API, return None on failure."""
        try:
            response = requests.get(self._api_url, timeout=10)
            response.raise_for_status()
            return SharedConfig.model_validate(response.json())
        except Exception:
            logger.warning("Failed to fetch shared config from %s", self._api_url, exc_info=True)
            return None

    def _refresh_loop(self) -> None:
        """Background daemon thread: sleep + fetch."""
        logger.info("Started shared config refresh loop")
        while self._running:
            time.sleep(self._refresh_interval)
            try:
                new_config = self._fetch()
                with self._lock:
                    if new_config and self._config != new_config:
                        for change in dict_diff(self._config.model_dump(), new_config.model_dump()):
                            logger.info("Config changed %s", change)
                        self._config = new_config
                    else:
                        logger.debug("Shared config unchanged")
            except Exception:
                logger.exception("Unexpected error in shared config refresh loop")

    def stop(self) -> None:
        """Stop background refresh."""
        self._running = False
