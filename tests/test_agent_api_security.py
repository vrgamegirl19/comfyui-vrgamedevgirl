"""Security tests for the Agent API: project ids, allowed roots, links that escape a root, and token auth (Apireport Section 11, item 5)."""

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
auth = importlib.import_module(f"{pkg_name}.agent_api.auth")
paths = importlib.import_module(f"{pkg_name}.agent_api.paths")
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")


class FakeRequest:
    def __init__(self, remote="127.0.0.1", headers=None):
        self.remote = remote
        self.headers = headers or {}


def _link_directory(link, target):
    """Create a directory link without needing admin rights. Returns False when the platform refuses."""
    try:
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", link, target], capture_output=True)
            return result.returncode == 0
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        return False


class ProjectIdTests(unittest.TestCase):
    def test_traversal_and_invalid_ids_are_rejected(self):
        for bad in ("", "   ", "..", "../x", "..\\x", "a/b", "a\\b", "C:evil", "x\x00y", "a|b", "a?b", 'a"b', "a<b"):
            with self.subTest(project_id=bad):
                with self.assertRaises(errors.ValidationError):
                    paths.validate_project_id(bad)

    def test_ordinary_ids_pass(self):
        for good in ("My Song", "Higher_Ground_2", "project-1"):
            self.assertEqual(paths.validate_project_id(good), good)


class AllowedRootTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, ignore_errors=True)
        self.root = os.path.join(self.temp, "output")
        self.outside = os.path.join(self.temp, "secret")
        os.makedirs(os.path.join(self.root, "Project"))
        os.makedirs(self.outside)
        self.patch = patch.object(paths, "get_allowed_project_roots", return_value=[self.root])
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_project_inside_root_resolves(self):
        self.assertEqual(os.path.normcase(paths.resolve_project_folder("Project")), os.path.normcase(os.path.join(self.root, "Project")))

    def test_unknown_project_is_not_found(self):
        with self.assertRaises(errors.ProjectNotFoundError):
            paths.resolve_project_folder("Nope")

    def test_absolute_path_cannot_be_used_as_an_id(self):
        with self.assertRaises(errors.ValidationError):
            paths.resolve_project_folder(self.outside)

    def test_sibling_directory_with_a_shared_prefix_is_outside(self):
        sibling = self.root + "2"
        os.makedirs(sibling)
        self.assertFalse(paths.is_path_inside_root(sibling, self.root))
        self.assertTrue(paths.is_path_inside_root(os.path.join(self.root, "Project"), self.root))

    def test_link_inside_the_root_that_points_outside_is_rejected(self):
        link = os.path.join(self.root, "Escape")
        if not _link_directory(link, self.outside):
            self.skipTest("this platform cannot create directory links")
        self.assertFalse(paths.is_path_inside_root(link, self.root))
        with self.assertRaises(errors.PathOutsideRootError):
            paths.resolve_project_folder("Escape")


class TokenAuthTests(unittest.TestCase):
    CONFIG = {"enabled": True, "require_token_on_loopback": False, "token": "s3cret-token"}

    def verify(self, request, **config):
        auth.verify_auth(request, {**self.CONFIG, **config})

    def test_loopback_needs_no_token_by_default(self):
        self.verify(FakeRequest("127.0.0.1"))
        self.verify(FakeRequest("::1"))

    def test_remote_request_without_header_is_rejected(self):
        with self.assertRaises(errors.AuthError):
            self.verify(FakeRequest("192.168.1.20"))

    def test_remote_request_with_wrong_scheme_or_token_is_rejected(self):
        for header in ("Basic s3cret-token", "Bearer wrong", "Bearer", "s3cret-token", "Bearer s3cret-token extra"):
            with self.subTest(header=header):
                with self.assertRaises(errors.AuthError):
                    self.verify(FakeRequest("192.168.1.20", {"Authorization": header}))

    def test_remote_request_with_correct_token_is_accepted(self):
        self.verify(FakeRequest("192.168.1.20", {"Authorization": "Bearer s3cret-token"}))
        self.verify(FakeRequest("192.168.1.20", {"Authorization": "bearer s3cret-token"}))

    def test_loopback_token_can_be_required(self):
        with self.assertRaises(errors.AuthError):
            self.verify(FakeRequest("127.0.0.1"), require_token_on_loopback=True)
        self.verify(FakeRequest("127.0.0.1", {"Authorization": "Bearer s3cret-token"}), require_token_on_loopback=True)

    def test_disabled_api_rejects_everyone_including_loopback(self):
        with self.assertRaises(errors.AuthError) as ctx:
            self.verify(FakeRequest("127.0.0.1"), enabled=False)
        self.assertEqual(ctx.exception.code, errors.AUTH_DISABLED)

    def test_remote_request_is_rejected_when_no_token_is_configured(self):
        with self.assertRaises(errors.AuthError):
            self.verify(FakeRequest("10.0.0.5"), token="")

    def test_auth_errors_do_not_echo_the_expected_token(self):
        try:
            self.verify(FakeRequest("192.168.1.20", {"Authorization": "Bearer wrong"}))
        except errors.AuthError as exc:
            self.assertNotIn("s3cret-token", str(exc.to_dict()))


if __name__ == "__main__":
    unittest.main()
