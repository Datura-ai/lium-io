from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

# Configuration
HOST = "0.0.0.0"
API_PORT = int(os.environ.get("API_PORT", "19999"))
MAX_CONCURRENT = 100
CONNECTION_TIMEOUT = 1.0
HANDLER_TIMEOUT = 0.2


class BatchPortVerifier:
    """Handles port connectivity verification."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def get_response(self, probe_host: str, port: int, nonce: str) -> str:
        """Send nonce and get response from port."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(probe_host, port), timeout=CONNECTION_TIMEOUT
            )
            try:
                writer.write((nonce + "\n").encode())
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=CONNECTION_TIMEOUT)
                return line.decode(errors="ignore").strip()
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
        except TimeoutError:
            raise

    async def check_ports(self, ports: list[tuple[int, int]], external_ip: str) -> dict[int, bool]:
        """
        Spin up lightweight TCP listeners on given ports, probe each locally, then tear down.

        Returns:
            Mapping {port: True/False} where True means "listener bound and probe succeeded".
        """
        print(f"Checking {len(ports)} ports on {external_ip}...")
        results: dict[int, bool] = {}
        nonce = secrets.token_hex(8)

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            """Single-connection handler that replies with OK + nonce."""
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=HANDLER_TIMEOUT)
                received_nonce = line.decode(errors="ignore").strip()
                if received_nonce == nonce:
                    writer.write(f"OK {nonce}\n".encode())
                    await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

        async def process_port(port_pair: tuple[int, int]) -> tuple[int, bool]:
            """Start server on internal port, probe external port, then close server."""
            internal_port, external_port = port_pair
            async with self.semaphore:
                try:
                    server = await asyncio.start_server(handler, host=HOST, port=internal_port)
                except OSError:
                    return internal_port, False

                try:
                    text = await self.get_response(external_ip, external_port, nonce)
                    ok = text == f"OK {nonce}"
                except Exception:
                    ok = False
                finally:
                    server.close()
                    with contextlib.suppress(Exception):
                        await server.wait_closed()

                return internal_port, ok

        tasks = [process_port(port_pair) for port_pair in ports]
        responses = await asyncio.gather(*tasks)

        for port, ok in responses:
            results[port] = ok

        return results


class BatchPortVerifierHandler(BaseHTTPRequestHandler):
    """HTTP request handler for port checking API."""

    def __init__(self, *args, **kwargs):
        self.port_verifier = BatchPortVerifier()
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json_response({"status": "ok"}, 200)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/check-ports":
            try:
                config = self._read_json_body()
                result = self._check_ports(config)
                self._send_json_response(result, 200)
            except Exception as e:
                self._send_json_response({"error": str(e)}, 400)
        else:
            self.send_error(404)

    def _read_json_body(self) -> dict:
        """Read and parse JSON request body."""
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        return json.loads(post_data.decode("utf-8"))

    def _check_ports(self, config: dict) -> dict:
        """Process port checking request."""
        external_ip = config.get("external_ip", "127.0.0.1")
        ports = config.get("ports", [])

        start_port = config.get("start_port")
        end_port = config.get("end_port")
        if start_port and end_port:
            extra_ports = [(port, port) for port in range(start_port, end_port + 1)]
            ports.extend(extra_ports)

        if not ports:
            raise ValueError("No ports provided")

        # Run async check_ports in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            t1 = time.time()
            result = loop.run_until_complete(self.port_verifier.check_ports(ports, external_ip))
            duration = time.time() - t1
        finally:
            loop.close()

        return {
            "duration": duration,
            "results": result,
            "success_count": sum(result.values()),
        }

    def _send_json_response(self, data: dict, status_code: int) -> None:
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging, just print essentials."""
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    """Start the port checker HTTP server."""
    print(f"Starting Batch Port Verifier HTTP Server on {HOST}:{API_PORT}")
    print("Endpoints:")
    print("  GET  /health - Health check")
    print("  POST /check-ports - Check ports (JSON body)")

    server = HTTPServer((HOST, API_PORT), BatchPortVerifierHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
