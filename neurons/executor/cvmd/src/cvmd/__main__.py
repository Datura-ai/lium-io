"""Entry point: `cvmd` (console script) or `python -m cvmd`.

A config that cannot be loaded exits non-zero rather than starting a daemon that serves on a
half-understood config. That is a permanent failure by design — the systemd unit uses
Restart=on-failure, so pair any change here with the unit's restart policy.
"""

import logging
import sys

import uvicorn

from cvmd.app import create_app
from cvmd.auth.clients import AuthorizedClientsError
from cvmd.config import ConfigError, load_config


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        config = load_config()
        app = create_app(config)
    except (ConfigError, AuthorizedClientsError) as exc:
        logging.getLogger("cvmd").error("refusing to start: %s", exc)
        return 1

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        ssl_certfile=str(config.tls_cert) if config.tls_enabled else None,
        ssl_keyfile=str(config.tls_key) if config.tls_enabled else None,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
