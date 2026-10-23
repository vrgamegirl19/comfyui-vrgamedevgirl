import base64
import importlib
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

builder_runner = importlib.import_module(f"{ROOT.name}.llm.builder_runner")
_generate_own_server_image = builder_runner._generate_own_server_image
_own_server_image_size = builder_runner._own_server_image_size

PNG = base64.b64encode(b"\x89PNG fake").decode("ascii")


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        _Handler.seen.append((self.path, self.headers.get("Authorization"), body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"b64_json": PNG}]}).encode("utf-8"))

    def log_message(self, *args):
        pass


class OwnServerImageTests(unittest.TestCase):
    def test_size_is_clamped_to_multiples_of_32(self):
        self.assertEqual(_own_server_image_size(1000, 5000), "992x2048")
        self.assertEqual(_own_server_image_size("", 100), "1024x256")

    def test_posts_to_images_generations(self):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            _Handler.seen.clear()
            result = _generate_own_server_image({
                "own_server_url": f"http://127.0.0.1:{server.server_port}/v1",
                "own_server_api_key": "secret",
                "prompt": "a sloth",
                "width": 1024,
                "height": 576,
                "steps": 8,
                "seed": 42,
            })
        finally:
            server.shutdown()
        path, auth, body = _Handler.seen[0]
        self.assertEqual(path, "/v1/images/generations")
        self.assertEqual(auth, "Bearer secret")
        self.assertEqual(body["size"], "1024x576")
        self.assertEqual((body["steps"], body["seed"], body["response_format"]), (8, 42, "b64_json"))
        self.assertNotIn("model", body)
        self.assertEqual(result["image_data"], f"data:image/png;base64,{PNG}")

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            _generate_own_server_image({"own_server_url": "http://127.0.0.1:1/v1", "prompt": " "})


if __name__ == "__main__":
    unittest.main()
