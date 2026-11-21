"""The Hugging Face token of the Models dialog: where it is saved, what it has to look like, and how Hugging Face is asked about it."""

import http.server
import importlib
import json
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

store = importlib.import_module(f"{ROOT.name}.core.hf_token")

SECRET = "hf_secretTOKEN123"


class Whoami(http.server.BaseHTTPRequestHandler):
    """A stand-in for Hugging Face's whoami route: records the path and Authorization header it was asked with."""
    status = 200
    body = b'{"name": "alice", "type": "user"}'
    seen = []

    def do_GET(self):
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        self.send_response(type(self).status)
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, *args):
        pass


class SavedTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="vrgdg_hf_token_")
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "VRGDG_Model_Defaults" / "hf_token.json"
        patcher = patch.object(store, "_hf_token_path", lambda: str(self.path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_nothing_is_saved_at_first(self) -> None:
        self.assertEqual(store.load_token(), "")
        self.assertFalse(store.is_saved())

    def test_a_token_is_saved_trimmed_in_a_file_of_its_own_and_loaded_back(self) -> None:
        self.assertTrue(store.save_token(f"  {SECRET}  "))
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"hf_token": SECRET})
        self.assertEqual(store.load_token(), SECRET)
        self.assertTrue(store.is_saved())

    def test_an_empty_string_clears_it_and_clearing_twice_is_fine(self) -> None:
        store.save_token(SECRET)
        self.assertFalse(store.save_token(""))
        self.assertFalse(self.path.exists())
        self.assertFalse(store.is_saved())
        self.assertFalse(store.save_token("   "))

    def test_what_is_not_a_token_is_refused_and_nothing_is_written(self) -> None:
        for bad in ("not_a_token", "hf_two words", "hf_line\nbreak", "hf_" + "a" * 500):
            with self.subTest(bad=bad[:20]):
                with self.assertRaisesRegex(ValueError, "starts with hf_"):
                    store.save_token(bad)
                self.assertFalse(self.path.exists())

    def test_a_refused_token_leaves_the_saved_one_alone(self) -> None:
        store.save_token(SECRET)
        with self.assertRaises(ValueError):
            store.save_token("nope")
        self.assertEqual(store.load_token(), SECRET)

    def test_a_file_that_was_edited_into_something_else_counts_as_no_token(self) -> None:
        self.path.parent.mkdir(parents=True)
        for text in ('{"hf_token": "hf_a b"}', '{"hf_token": "plain"}', '["hf_abc"]', "not json", '{"hf_token": null}'):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                self.assertEqual(store.load_token(), "")


class VerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        Whoami.status, Whoami.body, Whoami.seen = 200, b'{"name": "alice", "type": "user"}', []
        temp = tempfile.TemporaryDirectory(prefix="vrgdg_hf_verify_")
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "hf_token.json"
        for patcher in (patch.object(store, "_hf_token_path", lambda: str(self.path)), patch.object(store, "HF_BASE", self.serve())):
            patcher.start()
            self.addCleanup(patcher.stop)

    def serve(self) -> str:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Whoami)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_hugging_face_names_the_owner_of_a_token_it_accepts(self) -> None:
        self.assertEqual(store.whoami(SECRET), "alice")
        self.assertEqual(Whoami.seen, [("/api/whoami-v2", f"Bearer {SECRET}")])

    def test_an_answer_without_a_name_is_still_a_working_token(self) -> None:
        Whoami.body = b"[]"
        self.assertEqual(store.whoami(SECRET), "")
        Whoami.body = b"not json"
        self.assertEqual(store.whoami(SECRET), "")

    def test_a_token_hugging_face_turns_down_says_so(self) -> None:
        for status in (401, 403):
            Whoami.status = status
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, "did not accept that token") as caught:
                    store.whoami(SECRET)
                self.assertNotIn(SECRET, str(caught.exception))

    def test_any_other_answer_is_an_error_with_its_code(self) -> None:
        Whoami.status = 500
        with self.assertRaisesRegex(ValueError, "answered 500"):
            store.whoami(SECRET)

    def test_hugging_face_that_cannot_be_reached_says_so_without_the_token(self) -> None:
        with socket.socket() as free:
            free.bind(("127.0.0.1", 0))
            port = free.getsockname()[1]
        with patch.object(store, "HF_BASE", f"http://127.0.0.1:{port}"):
            with self.assertRaisesRegex(ValueError, "Could not reach Hugging Face") as caught:
                store.whoami(SECRET)
        self.assertNotIn(SECRET, str(caught.exception))

    def test_without_a_token_it_says_to_paste_one_and_asks_nothing(self) -> None:
        with self.assertRaisesRegex(ValueError, "No Hugging Face token is saved"):
            store.verify()
        self.assertEqual(Whoami.seen, [])

    def test_the_saved_token_is_checked_unless_another_is_sent(self) -> None:
        store.save_token(SECRET)
        self.assertEqual(store.verify(), "alice")
        self.assertEqual(store.verify("hf_other"), "alice")
        self.assertEqual([header for _path, header in Whoami.seen], [f"Bearer {SECRET}", "Bearer hf_other"])

    def test_a_sent_token_that_is_not_shaped_like_one_is_refused_before_anything_is_asked(self) -> None:
        for bad in ("hf_a\nX-Evil: 1", "wrong", "hf_two words"):
            with self.subTest(bad=bad[:20]):
                with self.assertRaisesRegex(ValueError, "starts with hf_"):
                    store.verify(bad)
        self.assertEqual(Whoami.seen, [])


if __name__ == "__main__":
    unittest.main()
