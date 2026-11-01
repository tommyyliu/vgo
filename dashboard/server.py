"""Localhost server for the performance and research dashboards. Standard library only."""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .collect import collect

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
PAGE = HOME / "static" / "index.html"
IDEAS = HOME / "ideas.json"
TTL = 300
IDEA_FIELDS = {"title", "bucket", "status", "why", "test", "effort", "result"}
MAX_FIELD = 4000


def write_json(path: Path, value) -> None:
    """Replace atomically, so a crash can't leave half a file."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


class State:
    def __init__(self, root: Path = ROOT, ideas: Path = IDEAS):
        self.root = root
        self.ideas_path = ideas
        self.lock = threading.Lock()
        self.cached: dict | None = None

    def data(self) -> dict:
        """Collected metrics, rebuilt when stale. The project's files are only read."""
        with self.lock:
            cached = self.cached
        if cached and time.time() - cached["collected_at"] < TTL:
            return cached
        result = collect(self.root)
        result["collected_at"] = time.time()
        with self.lock:
            self.cached = result
        return result

    def ideas(self) -> list[dict]:
        return json.loads(self.ideas_path.read_text()) if self.ideas_path.exists() else []

    def save_idea(self, body: dict) -> list[dict]:
        with self.lock:
            ideas = self.ideas()
            target = body.get("id")
            if body.get("delete"):
                ideas = [i for i in ideas if i["id"] != target]
            else:
                clean = {
                    k: str(v).strip()[:MAX_FIELD]
                    for k, v in (body.get("fields") or {}).items() if k in IDEA_FIELDS
                }
                existing = next((i for i in ideas if i["id"] == target), None)
                if existing:
                    existing.update(clean)
                else:
                    ideas.append({"id": uuid.uuid4().hex[:8], "status": "Next", **clean})
            write_json(self.ideas_path, ideas)
            return ideas


def make_handler(state: State, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, body: bytes, content_type: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status=HTTPStatus.OK):
            self.send_body(json.dumps(value).encode(), "application/json", status)

        def host_ok(self) -> bool:
            # Stops another website reaching this server by DNS rebinding.
            return self.headers.get("Host") in allowed_hosts

        def do_GET(self):
            if not self.host_ok():
                return self.send_json({"error": "bad host"}, HTTPStatus.FORBIDDEN)
            if self.path == "/":
                self.send_body(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/data":
                try:
                    self.send_json({**state.data(), "ideas": state.ideas()})
                except Exception:
                    traceback.print_exc()
                    self.send_json({"error": "Collecting failed; see the server's terminal output."},
                                   HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if not self.host_ok():
                return self.send_json({"error": "bad host"}, HTTPStatus.FORBIDDEN)
            # Requiring this type forces a CORS preflight, which is never answered,
            # so another site's page can't post here.
            if self.headers.get("Content-Type") != "application/json" or self.path != "/api/ideas":
                return self.send_json({"error": "expected JSON"}, HTTPStatus.BAD_REQUEST)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                return self.send_json({"error": "expected JSON"}, HTTPStatus.BAD_REQUEST)
            self.send_json({"ideas": state.save_idea(body)})

    return Handler


def serve(port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(State(), port))
    print(f"Dashboards: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
