"""HTTP API using only the Python standard library.

Endpoints
---------
GET  /healthz   liveness probe            -> {"status": "ok"}
GET  /readyz    readiness probe           -> {"status": "ready"}
POST /api/v1/invert   exact mass inversion

The listening host/port come from environment variables (``API_HOST``,
``API_PORT``); the container port mapping is configured separately in
docker-compose (``API_HOST_PORT``).
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .errors import RequestError
from .service import invert

MAX_BODY_BYTES = 1 << 20  # 1 MiB is ample for <= 16 components


class Handler(BaseHTTPRequestHandler):
    server_version = "OligomerInverter/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        # Structured single-line access log to stdout.
        self.server.log.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        encoded = data.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/healthz", "/health"):
            self._send_json(200, {"status": "ok", "service": "oligomer-inverter"})
            return
        if path in ("/readyz", "/ready"):
            self._send_json(200, {"status": "ready"})
            return
        self._send_json(404, {"error": {"code": "not_found",
                                        "message": f"unknown path {path!r}"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/v1/invert":
            self._send_json(404, {"error": {"code": "not_found",
                                            "message": f"unknown path {path!r}"}})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": {"code": "bad_request",
                                            "message": "invalid Content-Length"}})
            return
        if length <= 0:
            self._send_json(400, {"error": {"code": "bad_request",
                                            "message": "empty request body"}})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": {"code": "payload_too_large",
                                            "message": "request body exceeds 1 MiB"}})
            return

        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": {
                "code": "invalid_json",
                "message": f"request body is not valid JSON: {exc}",
            }})
            return

        try:
            status, payload = invert(body)
        except RequestError as exc:
            self._send_json(400, {
                "status": "invalid_request",
                **exc.to_dict(),
            })
            return
        self._send_json(status, payload)


class _StdoutLogger:
    def info(self, fmt: str, *args: object) -> None:
        print(fmt % args, flush=True)


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.log = _StdoutLogger()  # type: ignore[attr-defined]
    return server


def main() -> None:
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8080"))
    server = build_server(host, port)
    actual = server.server_address[1]
    print(f"oligomer-inverter listening on http://{host}:{actual} "
          f"(endpoints: POST /api/v1/invert, GET /healthz)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
