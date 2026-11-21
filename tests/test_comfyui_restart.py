"""The Video Builder's Restart ComfyUI button asks ComfyUI-Manager to restart ComfyUI: which route it uses, where it asks, and what an answer means."""

import ast
import http.client
import http.server
import socket
import ssl
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "core" / "system_routes.py"


def restart_functions():
    """The helpers of core/system_routes.py that restart ComfyUI (the module needs a running ComfyUI server to import)."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    wanted = {"_manager_reboot_route", "_own_base", "_call_reboot"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    body.insert(0, next(node for node in tree.body if isinstance(node, ast.Assign) and node.targets[0].id == "_MANAGER_REBOOT_PATHS"))
    namespace = {"http": http, "ssl": ssl, "urllib": urllib}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def route(path, method):
    return SimpleNamespace(resource=SimpleNamespace(canonical=path), method=method)


class Reboot(http.server.BaseHTTPRequestHandler):
    """A stand-in for ComfyUI-Manager that records what it was asked and answers with `status` (0: closes the connection without an answer).
    Like the real one it answers 400 to a form Content-Type (its guard against cross-site form posts)."""
    status = 200
    seen = []
    content_types = []

    def _answer(self):
        type(self).seen.append((self.command, self.path))
        declared = self.headers.get_content_type() if self.headers.get("Content-Type") else None
        type(self).content_types.append(declared)
        if declared in {"application/x-www-form-urlencoded", "multipart/form-data", "text/plain"}:
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if type(self).status == 0:
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        self.send_response(type(self).status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = _answer

    def log_message(self, *args):
        pass


class RebootRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.f = restart_functions()

    def test_the_v2_route_of_newer_managers_comes_first(self) -> None:
        routes = [route("/manager/reboot", "GET"), route("/v2/manager/reboot", "GET"), route("/v2/manager/reboot", "HEAD")]
        self.assertEqual(self.f["_manager_reboot_route"](routes), ("/v2/manager/reboot", "GET"))

    def test_the_older_route_is_used_when_there_is_no_v2_one(self) -> None:
        self.assertEqual(self.f["_manager_reboot_route"]([route("/manager/reboot", "GET")]), ("/manager/reboot", "GET"))

    def test_a_route_that_takes_post_is_asked_with_post(self) -> None:
        for methods in (["POST"], ["*"], ["GET", "POST"]):
            routes = [route("/manager/reboot", method) for method in methods]
            self.assertEqual(self.f["_manager_reboot_route"](routes), ("/manager/reboot", "POST"), methods)

    def test_no_manager_gives_nothing(self) -> None:
        routes = [route("/prompt", "POST"), route("/api/manager/reboot", "GET"), SimpleNamespace(method="GET")]
        self.assertIsNone(self.f["_manager_reboot_route"](routes))
        self.assertIsNone(self.f["_manager_reboot_route"]([]))


class OwnBaseTests(unittest.TestCase):
    def test_it_is_the_address_the_connection_came_in_on(self) -> None:
        own = restart_functions()["_own_base"]
        self.assertEqual(own("http", ("127.0.0.1", 8188), 1), "http://127.0.0.1:8188")
        self.assertEqual(own("https", ("192.168.1.5", 8443), 1), "https://192.168.1.5:8443")
        self.assertEqual(own("http", ("::1", 8188, 0, 0), 1), "http://[::1]:8188")

    def test_without_a_socket_address_it_is_this_machine_and_the_server_port(self) -> None:
        self.assertEqual(restart_functions()["_own_base"]("http", None, 8199), "http://127.0.0.1:8199")


class CallRebootTests(unittest.TestCase):
    def ask(self, status, method="GET"):
        Reboot.status = status
        Reboot.seen = []
        Reboot.content_types = []
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Reboot)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/v2/manager/reboot"
        return restart_functions()["_call_reboot"](url, method, False)

    def test_an_accepted_request_is_done_and_uses_the_asked_method(self) -> None:
        for method in ("GET", "POST"):
            self.assertEqual(self.ask(200, method), (True, ""))
            self.assertEqual(Reboot.seen, [(method, "/v2/manager/reboot")])

    def test_the_post_has_no_content_type_because_the_manager_refuses_a_form_one(self) -> None:
        self.assertEqual(self.ask(200, "POST"), (True, ""))
        self.assertEqual(Reboot.seen, [("POST", "/v2/manager/reboot")])
        self.assertEqual(Reboot.content_types, [None])

    def test_a_connection_closed_without_an_answer_is_a_restart_in_progress(self) -> None:
        self.assertEqual(self.ask(0), (True, ""))
        self.assertEqual(Reboot.seen, [("GET", "/v2/manager/reboot")])

    def test_a_refusal_by_the_security_level_says_so(self) -> None:
        ok, error = self.ask(403)
        self.assertFalse(ok)
        self.assertIn("security level", error)

    def test_any_other_answer_is_an_error_with_its_code(self) -> None:
        ok, error = self.ask(500)
        self.assertFalse(ok)
        self.assertIn("500", error)


if __name__ == "__main__":
    unittest.main()
