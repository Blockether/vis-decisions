"""Loopback HTTP fixtures; never connect a test to a user's gateway."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from blockether.vis._contracts import definition

_HANDSHAKE = definition("gateway", "handshake")["properties"]


@contextmanager
def endpoint(respond):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle_request(self):
            if self.headers.get("Transfer-Encoding") == "chunked":
                chunks = []
                while size := int(self.rfile.readline(), 16):
                    chunks.append(self.rfile.read(size))
                    assert self.rfile.read(2) == b"\r\n"
                assert self.rfile.readline() == b"\r\n"
                body = b"".join(chunks)
            else:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            calls.append((self.command, self.path, dict(self.headers), body))
            result = respond(self.command, self.path, body)
            status, value = result[:2]
            kind = result[2] if len(result) > 2 else "application/json"
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = handle_request

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def compatible(method, path, body):
    if path == "/v1/capabilities":
        return 200, {
            "protocol": {
                "protocol": _HANDSHAKE["protocol"]["const"],
                "min_client": _HANDSHAKE["min_client"]["const"],
            }
        }
    if path == "/v1/clients":
        return 201, {"client_id": "sdk-lease"}
    if path == "/v1/clients/sdk-lease":
        return 200, {"is_released": True}
    return None
