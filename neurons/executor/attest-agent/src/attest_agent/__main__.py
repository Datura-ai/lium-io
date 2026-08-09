"""Entry point. TLS is terminated by the agent itself, with the key the quote binds to."""

import logging
import sys

import uvicorn

from attest_agent.app import create_app
from attest_agent.config import load_config


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        config = load_config()
    except ValueError as exc:
        logging.error("attest-agent will not start: %s", exc)
        return 2

    app = create_app(config)
    # `ssl_keyfile`/`ssl_certfile` rather than a reverse proxy: the key hashed into
    # `report_data` has to be the key that terminates the connection, or the binding
    # proves nothing about the channel the verifier is actually talking over.
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        ssl_keyfile=str(config.key_path),
        ssl_certfile=str(config.cert_path),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
