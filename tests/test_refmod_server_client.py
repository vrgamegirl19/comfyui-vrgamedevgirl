"""The pack's side of the RefMod server: settings, signing in, listing, sharing, adding to the library, reporting and removing,
against a stub server."""

import email
import hashlib
import http.server
import importlib
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load():
    for name, path in (("vrgdg_srv_test", ROOT), ("vrgdg_srv_test.minimax", ROOT / "minimax"), ("vrgdg_srv_test.core", ROOT / "core")):
        package = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None, is_package=True))
        package.__path__ = [str(path)]
        sys.modules[name] = package
    importlib.import_module("vrgdg_srv_test.minimax.refmod_library")
    return importlib.import_module("vrgdg_srv_test.minimax.refmod_server")


server = _load()
library = sys.modules["vrgdg_srv_test.minimax.refmod_library"]
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 40
TOKEN = "token-from-the-server-0123456789"


def refmod_bytes(kind="video", concept="identity", seed=1):
    frames, height, width = (3 if kind == "video" else 1), 4, 6
    length = 24 * frames * height * width * 2
    meta = {"name": "x", "kind": kind, "latent_h": height, "latent_w": width, "latent_t": frames, "concept_type": concept}
    header = json.dumps({
        "__metadata__": {"refmod_meta": json.dumps(meta)},
        "latent": {"dtype": "F16", "shape": [1, 24, frames, height, width], "data_offsets": [0, length]},
    }).encode()
    return len(header).to_bytes(8, "little") + header + bytes([seed]) + b"\0" * (length - 1)


def id_of(data):
    return hashlib.sha256(data).hexdigest()[:24]


class StubServer(http.server.ThreadingHTTPServer):
    """Answers from a table of (method, path) -> (status, content type, body) and keeps what it was asked."""

    def __init__(self):
        self.routes = {}
        self.seen = []
        super().__init__(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def _handler(self):
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                path = self.path.split("?")[0]
                stub.seen.append({"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body})
                status, content_type, payload = stub.routes.get((self.command, path), (404, "application/json", b'{"ok": false, "error": "No such route."}'))
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = do_DELETE = _answer

            def log_message(self, *args):
                pass

        return Handler

    def json_route(self, method, path, body, status=200):
        self.routes[(method, path)] = (status, "application/json", json.dumps(body).encode())

    def stop(self):
        self.shutdown()
        self.server_close()


class ClientTestCase(unittest.TestCase):
    signed_in = True

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.library_root = self.tmp / "refmods"
        self.library_root.mkdir()
        self.stub = StubServer()
        self.addCleanup(self.stub.stop)
        patches = [
            mock.patch.object(server, "_settings_path", lambda: str(self.tmp / "refmod_server.json")),
            mock.patch.object(server, "_refmod_dirs", lambda: [str(self.library_root)]),
            mock.patch.object(library, "_refmod_dirs", lambda: [str(self.library_root)]),
            mock.patch.dict("os.environ", {"H3_REFMOD_SERVER": ""}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        server._pending.clear()
        server.save_settings(url=self.stub.url)
        if self.signed_in:
            self.sign_in()
        self.stub.seen.clear()

    def sign_in(self, name="Alex"):
        """Signs in through the client the way the Viewer does, against the stub."""
        self.stub.json_route("POST", "/api/v1/auth/start", {"ok": True, "url": "https://github.com/login/oauth/authorize?state=s", "poll": "poll-secret", "interval": 2, "expires_in": 600})
        self.stub.json_route("POST", "/api/v1/auth/poll", {"ok": True, "status": "signed_in", "token": TOKEN, "name": name})
        server.signin_start("github")
        return server.signin_poll()

    def shared(self, data, **overrides):
        """What the server says about a file, and the routes that serve it."""
        refmod_id = id_of(data)
        meta = {"id": refmod_id, "type": "identity", "stem": "darrel", "name": "identity/darrel", "has_preview": True, "uploader": "Robin"}
        meta.update(overrides)
        self.stub.json_route("GET", f"/api/v1/refmods/{refmod_id}", {"ok": True, "refmod": meta})
        self.stub.routes[("GET", f"/api/v1/refmods/{refmod_id}/file")] = (200, "application/octet-stream", data)
        self.stub.routes[("GET", f"/api/v1/refmods/{refmod_id}/preview")] = (200, "image/png", PNG)
        return refmod_id

    def bearer(self, request):
        return request["headers"].get("Authorization")


class SettingsTests(ClientTestCase):
    signed_in = False

    def test_addresses_are_cleaned_and_bad_ones_refused(self):
        self.assertEqual(server.normalize_server_url(" https://aidigitalcreations.com/ "), "https://aidigitalcreations.com")
        self.assertEqual(server.normalize_server_url("http://127.0.0.1:8787/api/v1/"), "http://127.0.0.1:8787")
        for bad in ("", "ftp://x.com", "javascript:alert(1)", "not a url", "file:///etc/passwd"):
            self.assertEqual(server.normalize_server_url(bad), "", bad)
        with self.assertRaises(ValueError):
            server.save_settings(url="ftp://x.com")

    def test_settings_round_trip_and_an_empty_address_means_the_default(self):
        saved = server.save_settings(url="http://example.test:9000/")
        self.assertEqual((saved["url"], saved["signed_in"], saved["account"]), ("http://example.test:9000", False, ""))
        self.assertEqual(server.get_settings()["url"], "http://example.test:9000")
        self.assertEqual(server.save_settings(url="")["url"], server.DEFAULT_SERVER_URL)
        self.assertNotIn("uploader", server.get_settings())

    def test_hostile_names_stay_names(self):
        self.assertEqual(server.clean_type("../../Evil Type"), "______evil_type")
        self.assertEqual(server.clean_type(".."), "generic")
        self.assertEqual(server.clean_type(None), "generic")
        self.assertEqual(server.clean_stem("..\\..\\x/y:z"), "_.._x_y_z".strip(" ."))
        self.assertEqual(server.clean_stem(".."), "shared")
        self.assertNotIn("/", server.clean_stem("a/b"))


class SignInTests(ClientTestCase):
    signed_in = False

    def test_the_ways_to_sign_in_are_asked_without_a_token(self):
        self.stub.json_route("GET", "/api/v1/health", {"ok": True, "sign_in": ["github", "discord"]})
        self.assertEqual(server.sign_in_options(), {"url": self.stub.url, "providers": ["github", "discord"]})
        self.assertIsNone(self.bearer(self.stub.seen[-1]))

    def test_signing_in_saves_a_token_that_is_never_shown(self):
        self.stub.json_route("POST", "/api/v1/auth/start", {"ok": True, "url": "https://github.com/login/oauth/authorize?state=s", "poll": "poll-secret", "interval": 2, "expires_in": 600})
        started = server.signin_start("github")
        self.assertEqual(started["url"], "https://github.com/login/oauth/authorize?state=s")
        self.assertNotIn("poll-secret", json.dumps(started), "the secret that collects the token stays on this computer")
        self.assertEqual(json.loads(self.stub.seen[-1]["body"]), {"provider": "github"})

        self.stub.json_route("POST", "/api/v1/auth/poll", {"ok": True, "status": "waiting"})
        self.assertEqual(server.signin_poll(), {"status": "waiting"})
        self.assertEqual(json.loads(self.stub.seen[-1]["body"]), {"poll": "poll-secret"})
        self.assertFalse(server.get_settings()["signed_in"])

        self.stub.json_route("POST", "/api/v1/auth/poll", {"ok": True, "status": "signed_in", "token": TOKEN, "name": "alice-gh"})
        done = server.signin_poll()
        self.assertEqual((done["status"], done["signed_in"], done["account"]), ("signed_in", True, "alice-gh"))
        self.assertNotIn(TOKEN, json.dumps(server.get_settings()))
        self.assertNotIn(TOKEN, json.dumps(done))
        with self.assertRaises(ValueError):
            server.signin_poll()  # finished: nothing is pending any more

    def test_polling_before_starting_is_refused(self):
        with self.assertRaises(ValueError):
            server.signin_poll()
        self.assertEqual(self.stub.seen, [])

    def test_a_sign_in_that_failed_or_expired_says_so_and_ends(self):
        for code in ("signin_failed", "signin_expired"):
            self.stub.json_route("POST", "/api/v1/auth/start", {"ok": True, "url": "https://x.test/", "poll": "p", "interval": 2, "expires_in": 600})
            server.signin_start("discord")
            self.stub.json_route("POST", "/api/v1/auth/poll", {"ok": False, "error": "This account has been banned from the server.", "code": code}, status=401)
            with self.assertRaises(server.ServerError) as caught:
                server.signin_poll()
            self.assertEqual((caught.exception.code, caught.exception.status), (code, 401))
            self.assertIn("banned", str(caught.exception))
            with self.assertRaises(ValueError):
                server.signin_poll()
        self.assertFalse(server.get_settings()["signed_in"])

    def test_changing_the_address_midway_cancels_the_sign_in(self):
        self.stub.json_route("POST", "/api/v1/auth/start", {"ok": True, "url": "https://x.test/", "poll": "p"})
        server.signin_start("github")
        server.save_settings(url="http://other.example:9000")
        before = len(self.stub.seen)
        with self.assertRaises(ValueError):
            server.signin_poll()
        self.assertEqual(len(self.stub.seen), before, "the secret is not sent to a different server")

    def test_a_token_is_only_sent_to_the_server_that_gave_it(self):
        self.sign_in()
        self.assertTrue(server.get_settings()["signed_in"])
        server.save_settings(url="http://other.example:9000")
        self.assertEqual((server.get_settings()["signed_in"], server.get_settings()["account"]), (False, ""))
        server.save_settings(url=self.stub.url)
        self.assertFalse(server.get_settings()["signed_in"], "going back needs a new sign-in")
        self.stub.seen.clear()
        with self.assertRaises(server.ServerError) as caught:
            server.list_server_refmods()
        self.assertEqual((caught.exception.code, caught.exception.status), ("signin_required", 401))
        self.assertEqual(self.stub.seen, [], "nothing is sent without a sign-in")

    def test_a_token_for_another_default_address_is_not_sent(self):
        self.sign_in()
        path = Path(server._settings_path())
        data = json.loads(path.read_text())
        data["token_url"] = "http://127.0.0.1:1"  # signed in somewhere else
        path.write_text(json.dumps(data))
        self.stub.seen.clear()
        with self.assertRaises(server.ServerError):
            server.list_server_refmods()
        self.assertEqual(self.stub.seen, [])

    def test_signing_out_ends_the_token_on_the_server_and_here(self):
        self.sign_in()
        self.stub.json_route("POST", "/api/v1/auth/signout", {"ok": True})
        self.assertFalse(server.sign_out()["signed_in"])
        request = self.stub.seen[-1]
        self.assertEqual((request["path"], self.bearer(request)), ("/api/v1/auth/signout", "Bearer " + TOKEN))

    def test_signing_out_works_when_the_server_is_gone(self):
        Path(server._settings_path()).write_text(json.dumps({"url": "http://127.0.0.1:9", "token": TOKEN, "token_url": "http://127.0.0.1:9", "account": "Alex"}))
        self.assertTrue(server.get_settings()["signed_in"])
        self.assertFalse(server.sign_out()["signed_in"])
        self.assertEqual(json.loads(Path(server._settings_path()).read_text())["token"], "")

    def test_a_token_the_server_no_longer_knows_is_forgotten(self):
        self.sign_in()
        self.stub.json_route("GET", "/api/v1/refmods", {"ok": False, "error": "Sign in to use the server library.", "code": "signin_required"}, status=401)
        with self.assertRaises(server.ServerError) as caught:
            server.list_server_refmods()
        self.assertEqual(caught.exception.code, "signin_required")
        self.assertFalse(server.get_settings()["signed_in"])

    def test_a_banned_account_gets_the_servers_message(self):
        self.sign_in()
        self.stub.json_route("GET", "/api/v1/refmods", {"ok": False, "error": "This account has been banned from the server.", "code": "banned"}, status=403)
        with self.assertRaises(server.ServerError) as caught:
            server.list_server_refmods()
        self.assertEqual((caught.exception.code, caught.exception.status), ("banned", 403))
        self.assertTrue(server.get_settings()["signed_in"], "a ban is not a lost token")


class ListingTests(ClientTestCase):
    def test_the_library_comes_from_the_server_and_carries_the_token(self):
        self.stub.json_route("GET", "/api/v1/refmods", {"ok": True, "refmods": [{"id": "a" * 24, "mine": True}], "types": [{"type": "identity", "count": 1}]})
        result = server.list_server_refmods()
        self.assertEqual(result["refmods"][0]["mine"], True)
        self.assertEqual((result["url"], result["account"]), (self.stub.url, "Alex"))
        request = self.stub.seen[-1]
        self.assertEqual(self.bearer(request), "Bearer " + TOKEN)
        self.assertNotIn("X-Client-Key", request["headers"])
        self.assertIn("limit=", request["path"])

    def test_a_server_that_is_not_running_gives_a_message_with_its_address(self):
        path = Path(server._settings_path())
        path.write_text(json.dumps({"url": "http://127.0.0.1:9", "token": TOKEN, "token_url": "http://127.0.0.1:9", "account": "Alex"}))
        with self.assertRaises(server.ServerError) as caught:
            server.list_server_refmods()
        self.assertIn("http://127.0.0.1:9", str(caught.exception))

    def test_an_address_that_is_not_a_refmod_server_is_said_so(self):
        self.stub.routes[("GET", "/api/v1/refmods")] = (200, "text/html", b"<html>hello</html>")
        with self.assertRaises(server.ServerError) as caught:
            server.list_server_refmods()
        self.assertIn("did not answer like a RefMod server", str(caught.exception))

    def test_the_servers_own_error_text_is_passed_on(self):
        self.stub.json_route("GET", "/api/v1/refmods", {"ok": False, "error": "Too many requests."}, status=429)
        with self.assertRaises(server.ServerError) as caught:
            server.list_server_refmods()
        self.assertEqual((str(caught.exception), caught.exception.status), ("Too many requests.", 429))

    def test_previews_must_be_pngs_and_ids_must_be_ids(self):
        refmod_id = "b" * 24
        self.stub.routes[("GET", f"/api/v1/refmods/{refmod_id}/preview")] = (200, "image/png", PNG)
        self.assertEqual(server.server_preview(refmod_id), PNG)
        self.assertEqual(self.bearer(self.stub.seen[-1]), "Bearer " + TOKEN)
        self.stub.routes[("GET", f"/api/v1/refmods/{refmod_id}/preview")] = (200, "image/png", b"<script>")
        with self.assertRaises(server.ServerError):
            server.server_preview(refmod_id)
        for bad in ("", "../x", "B" * 24, "a" * 23, None):
            with self.assertRaises(ValueError):
                server.server_preview(bad)


class DownloadTests(ClientTestCase):
    def test_a_shared_refmod_is_added_with_its_preview(self):
        data = refmod_bytes()
        refmod_id = self.shared(data)
        result = server.download_refmod(refmod_id)
        self.assertEqual(result, {"name": "identity/darrel", "already": False, "renamed": False})
        self.assertEqual((self.library_root / "identity" / "darrel.safetensors").read_bytes(), data)
        self.assertEqual((self.library_root / "identity" / "darrel.preview.png").read_bytes(), PNG)
        self.assertEqual(sorted(path.name for path in (self.library_root / "identity").iterdir()), ["darrel.preview.png", "darrel.safetensors"])
        self.assertTrue(all(self.bearer(request) == "Bearer " + TOKEN for request in self.stub.seen))

    def test_the_same_file_again_is_already_there(self):
        refmod_id = self.shared(refmod_bytes())
        server.download_refmod(refmod_id)
        again = server.download_refmod(refmod_id)
        self.assertEqual(again, {"name": "identity/darrel", "already": True, "renamed": False})
        self.assertEqual(len(list((self.library_root / "identity").glob("*.safetensors"))), 1)

    def test_a_taken_name_is_never_overwritten(self):
        mine = refmod_bytes(seed=1)
        (self.library_root / "identity").mkdir()
        (self.library_root / "identity" / "darrel.safetensors").write_bytes(mine)
        refmod_id = self.shared(refmod_bytes(seed=2))
        result = server.download_refmod(refmod_id)
        self.assertEqual(result, {"name": "identity/darrel (2)", "already": False, "renamed": True})
        self.assertEqual((self.library_root / "identity" / "darrel.safetensors").read_bytes(), mine)
        self.assertTrue((self.library_root / "identity" / "darrel (2).safetensors").is_file())

    def test_names_from_the_server_cannot_leave_the_library(self):
        refmod_id = self.shared(refmod_bytes(), type="../../outside", stem="..\\..\\evil/../name")
        result = server.download_refmod(refmod_id)
        saved = Path(str(self.library_root / result["name"]) + ".safetensors")
        self.assertTrue(saved.is_file(), result)
        self.assertEqual(saved.resolve().parents[1], self.library_root.resolve())
        self.assertEqual([path.name for path in self.tmp.iterdir() if path.name != "refmods"], ["refmod_server.json"])

    def test_a_file_that_does_not_match_its_id_is_refused(self):
        data = refmod_bytes()
        refmod_id = self.shared(data)
        self.stub.routes[("GET", f"/api/v1/refmods/{refmod_id}/file")] = (200, "application/octet-stream", refmod_bytes(seed=9))
        with self.assertRaises(server.ServerError) as caught:
            server.download_refmod(refmod_id)
        self.assertIn("damaged", str(caught.exception))
        self.assertEqual(list(self.library_root.rglob("*")), [])

    def test_a_file_that_is_not_a_refmod_leaves_nothing_behind(self):
        junk = b"this is not a safetensors file at all"
        refmod_id = self.shared(junk)
        with self.assertRaises(server.ServerError):
            server.download_refmod(refmod_id)
        self.assertEqual([path for path in self.library_root.rglob("*") if path.is_file()], [])

    def test_a_missing_preview_does_not_stop_the_download(self):
        refmod_id = self.shared(refmod_bytes())
        self.stub.routes[("GET", f"/api/v1/refmods/{refmod_id}/preview")] = (404, "application/json", b'{"ok": false, "error": "No preview."}')
        result = server.download_refmod(refmod_id)
        self.assertEqual(result["name"], "identity/darrel")
        self.assertFalse((self.library_root / "identity" / "darrel.preview.png").exists())


class UploadTests(ClientTestCase):
    def test_a_refmod_is_sent_with_its_preview_and_name_and_your_token(self):
        (self.library_root / "identity").mkdir()
        data = refmod_bytes()
        path = self.library_root / "identity" / "darrel.safetensors"
        path.write_bytes(data)
        path.with_name("darrel.preview.png").write_bytes(PNG)
        self.stub.json_route("POST", "/api/v1/refmods", {"ok": True, "duplicate": False, "replaced": False, "status": "pending", "refmod": {"id": id_of(data), "mine": True}}, status=201)
        result = server.upload_refmod("identity/darrel")
        self.assertEqual(result["refmod"]["id"], id_of(data))
        self.assertEqual(result["status"], "pending")
        request = self.stub.seen[-1]
        self.assertEqual(self.bearer(request), "Bearer " + TOKEN)
        message = email.message_from_bytes(b"Content-Type: " + request["headers"]["Content-Type"].encode() + b"\r\n\r\n" + request["body"])
        parts = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True) for part in message.get_payload()}
        self.assertEqual(parts["file"], data)
        self.assertEqual(parts["preview"], PNG)
        self.assertEqual(parts["name"], b"identity/darrel")
        self.assertNotIn("uploader", parts, "the server names the uploader from the account")

    def test_a_refmod_that_is_not_in_the_library_is_refused(self):
        for name in ("identity/missing", "", "../secret"):
            with self.assertRaises(ValueError):
                server.upload_refmod(name)
        self.assertEqual(self.stub.seen, [])

    def test_removing_an_upload_sends_the_token(self):
        refmod_id = "c" * 24
        self.stub.json_route("DELETE", f"/api/v1/refmods/{refmod_id}", {"ok": True, "id": refmod_id})
        self.assertEqual(server.delete_server_refmod(refmod_id), {"id": refmod_id})
        request = self.stub.seen[-1]
        self.assertEqual((request["method"], self.bearer(request)), ("DELETE", "Bearer " + TOKEN))
        self.stub.json_route("DELETE", f"/api/v1/refmods/{refmod_id}", {"ok": False, "error": "Only the person who shared this RefMod can delete it."}, status=403)
        with self.assertRaises(server.ServerError) as caught:
            server.delete_server_refmod(refmod_id)
        self.assertEqual(caught.exception.status, 403)

    def test_a_report_sends_a_short_clean_reason(self):
        refmod_id = "d" * 24
        self.stub.json_route("POST", f"/api/v1/refmods/{refmod_id}/report", {"ok": True})
        self.assertEqual(server.report_server_refmod(refmod_id, "  not   a\nreal  person " + "x" * 400), {"id": refmod_id})
        request = self.stub.seen[-1]
        reason = json.loads(request["body"])["reason"]
        self.assertTrue(reason.startswith("not a real person x") and len(reason) == 200, reason)
        self.assertEqual(self.bearer(request), "Bearer " + TOKEN)
        with self.assertRaises(ValueError):
            server.report_server_refmod("../x", "no")


if __name__ == "__main__":
    unittest.main()
