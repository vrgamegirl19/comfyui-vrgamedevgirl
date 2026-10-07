"""PUT /story/settings honours If-Match like the other write routes (409 on a stale revision)."""

import asyncio
import importlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
router = importlib.import_module(f"{pkg_name}.agent_api.router")
story = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.storyboard_orchestrator")
from test_agent_api_references_llm import Base  # noqa: E402

API = "/vrgdg/api/v1"


class _Routes:
    """Collects the router's handlers by (METHOD, path) instead of serving them."""

    def __init__(self):
        self.handlers = {}

    def __getattr__(self, method):
        def register(path):
            def decorator(handler):
                self.handlers[(method.upper(), path)] = handler
                return handler
            return decorator
        return register


class _Request:
    def __init__(self, pid, body, if_match=None):
        self.match_info = {"pid": pid}
        self.headers = {"If-Match": str(if_match)} if if_match is not None else {}
        self.can_read_body = True
        self._body = body

    async def json(self):
        return self._body


def _story_settings_handler():
    server = MagicMock()
    server.routes = _Routes()
    with patch.object(router, "_VRGDG_AGENT_API_ROUTES_REGISTERED", False):
        router.register_agent_api_routes(server)
    return server.routes.handlers[("PUT", f"{API}/projects/{{pid}}/story/settings")]


class StorySettingsIfMatchTests(Base):
    BODY = {"defaults": {"video_style": "Noir"}, "story": {"overall_story_idea": "idea"}}

    def put(self, if_match=None):
        handler = _story_settings_handler()
        with patch.object(router, "verify_auth", lambda request: None):
            response = asyncio.run(handler(_Request("Song", json.loads(json.dumps(self.BODY)), if_match)))
        return response.status, json.loads(response.body)

    def test_a_stale_if_match_is_a_409_and_saves_nothing(self):
        before = self.read_session()
        status, body = self.put(if_match=0)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["code"], errors.REVISION_CONFLICT)
        self.assertEqual(self.read_session(), before)

    def test_the_current_revision_saves(self):
        status, body = self.put(if_match=1)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["revision"], 2)
        self.assertEqual(self.read_session()["builder_storyboard_defaults"]["video_style"], "Noir")

    def test_no_if_match_still_saves(self):
        status, _body = self.put()
        self.assertEqual(status, 200)

    def test_the_orchestrator_checks_the_revision(self):
        with self.assertRaises(errors.RevisionConflictError):
            story.set_story_settings("Song", {"story": {"overall_story_idea": "x"}}, if_match_revision=5)
        story.set_story_settings("Song", {"story": {"overall_story_idea": "x"}}, if_match_revision=1)
        self.assertEqual(self.read_session()["builder_story_layer"]["overall_story_idea"], "x")

    def test_endpoints_json_marks_story_writes_as_if_match(self):
        endpoints = json.loads((ROOT / "agent_api" / "endpoints.json").read_text(encoding="utf-8"))
        items = endpoints if isinstance(endpoints, list) else endpoints.get("endpoints", [])
        flags = {(e["method"], e["path"]): e["if_match"] for e in items}
        self.assertTrue(flags[("PUT", "/projects/{pid}/story/settings")])
        self.assertTrue(flags[("PUT", "/projects/{pid}/story")])


if __name__ == "__main__":
    unittest.main()
