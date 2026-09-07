"""Isolated UI verification server. No camera, real settings or observations are changed."""
import json
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from estimate_setup import EstimateSetupStore
from test_estimate_setup import setup_api
from imu_gateway import read_imu_state


def main():
    with tempfile.TemporaryDirectory(prefix="estimate_preview_") as directory:
        store = EstimateSetupStore(Path(directory) / "setup.json")

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, data, status=200):
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                assets = {
                    "/": ("dashboard.html", "text/html; charset=utf-8"),
                    "/static/css/dashboard.css": ("static/css/dashboard.css", "text/css; charset=utf-8"),
                    "/static/js/dashboard.js": ("static/js/dashboard.js", "application/javascript; charset=utf-8"),
                    "/static/css/imu-panel.css": ("static/css/imu-panel.css", "text/css; charset=utf-8"),
                    "/static/js/imu-panel.js": ("static/js/imu-panel.js", "application/javascript; charset=utf-8"),
                }
                if self.path in assets:
                    filename, content_type = assets[self.path]
                    payload = (ROOT / filename).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path == "/api/imu":
                    self.respond(read_imu_state())
                elif self.path == "/api/estimate-setup":
                    route, _ = setup_api(store)
                    self.respond(route())
                elif self.path == "/api/inspection":
                    self.respond({"inspection": {}, "member_groups": {}, "materials": {}})
                elif self.path == "/api/latest":
                    self.respond({"camera_status": "starting", "detections": []})
                elif self.path == "/api/report":
                    self.respond({"records": [], "summary": {}})
                elif self.path == "/api/captures":
                    self.respond({"captures": []})
                else:
                    self.send_response(204)
                    self.end_headers()

            def do_POST(self):
                if self.path != "/api/estimate-setup":
                    return self.respond({}, 404)
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                route, request = setup_api(store)
                request.method = "POST"
                request.get_json = lambda **kw: payload
                result = route()
                self.respond(result[0], result[1]) if isinstance(result, tuple) else self.respond(result)

        server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
        print("Isolated estimate preview http://127.0.0.1:8765", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
