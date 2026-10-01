"""The MCP tools call endpoints that exist, and every endpoint can be reached through a tool."""

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not (ROOT / "mcp_server").is_dir():  # mcp_server/ is local-only (git-ignored)
    raise unittest.SkipTest("mcp_server/ is not present")

from mcp_server import endpoint_tools, tools  # noqa: E402

ENDPOINTS = json.loads((ROOT / "agent_api" / "endpoints.json").read_text(encoding="utf-8"))["endpoints"]
ROUTES = {(e["method"], endpoint_tools._normalize(e["path"])) for e in ENDPOINTS}
ROUTE_PATHS = {path for _method, path in ROUTES}
HANDWRITTEN_SOURCE = (ROOT / "mcp_server" / "tools.py").read_text(encoding="utf-8").split("# Every Agent API endpoint without a tool above")[0]


class FakeClient:
    base_url = "http://test/vrgdg/api/v1"

    def __init__(self):
        self.calls = []

    def _record(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return {"ok": True}

    def get(self, path, params=None, **kw):
        return self._record("GET", path, params=params)

    def post(self, path, json_data=None, params=None, if_match_revision=None, **kw):
        return self._record("POST", path, json=json_data, params=params, if_match=if_match_revision)

    def put(self, path, json_data=None, params=None, if_match_revision=None, **kw):
        return self._record("PUT", path, json=json_data, params=params, if_match=if_match_revision)

    def patch(self, path, json_data=None, params=None, if_match_revision=None, **kw):
        return self._record("PATCH", path, json=json_data, params=params, if_match=if_match_revision)

    def delete(self, path, params=None, json_data=None, if_match_revision=None, **kw):
        return self._record("DELETE", path, params=params, json=json_data, if_match=if_match_revision)


class HandwrittenToolsMatchTheRouterTests(unittest.TestCase):
    def test_every_call_in_a_hand_written_tool_is_a_real_route(self):
        wrong = []
        for match in re.finditer(r'client\.(get|post|put|patch|delete)\(\s*f?"([^"]+)"', HANDWRITTEN_SOURCE):
            call = (match.group(1).upper(), endpoint_tools._normalize(match.group(2)))
            if call == ("PUT", "/projects/{}/references/{}/{}"):
                # reference_upsert chooses "subjects" or "locations"; both routes must exist
                self.assertIn(("PUT", "/projects/{}/references/subjects/{}"), ROUTES)
                self.assertIn(("PUT", "/projects/{}/references/locations/{}"), ROUTES)
                continue
            if call not in ROUTES:
                wrong.append(call)
        for match in re.finditer(r'endpoint = f?"(/[^"]+)"', HANDWRITTEN_SOURCE):
            if endpoint_tools._normalize(match.group(1)) not in ROUTE_PATHS:
                wrong.append(("endpoint", match.group(1)))
        self.assertEqual(wrong, [], "tools call routes the router does not have")

    def test_key_tools_send_what_the_routes_read(self):
        client = FakeClient()
        tools.ALL_TOOLS["audio_attach"].handler(client, {"project_id": "p", "audio_file": "C:/s.mp3"})
        tools.ALL_TOOLS["lyrics_set"].handler(client, {"project_id": "p", "lyrics": "line"})
        tools.ALL_TOOLS["project_delete"].handler(client, {"project_id": "p", "confirm": True})
        tools.ALL_TOOLS["job_cancel"].handler(client, {"job_id": "j"})
        tools.ALL_TOOLS["scene_split_merge_move_resize"].handler(client, {"project_id": "p", "scene_id": "s", "op": "split", "split_time": 2.5})
        calls = {(m, p): kw for m, p, kw in client.calls}
        self.assertEqual(calls[("PUT", "/projects/p/audio")]["json"], {"audio_path": "C:/s.mp3"})
        self.assertEqual(calls[("PUT", "/projects/p/lyrics")]["json"], {"lyrics_text": "line"})
        self.assertEqual(calls[("DELETE", "/projects/p")]["params"], {"confirm": "p"})
        self.assertIn(("POST", "/jobs/j/cancel"), calls)
        self.assertEqual(calls[("POST", "/projects/p/scenes/s/split")]["json"], {"at_time": 2.5})


class EndpointToolsTests(unittest.TestCase):
    def test_every_endpoint_is_reachable_through_a_named_tool(self):
        covered = endpoint_tools.covered_by_handwritten_tools()
        generated = {name for name in tools.ALL_TOOLS if name.startswith("api_") and name != "api_request"}
        uncovered = [e for e in ENDPOINTS if not endpoint_tools._is_covered(e, covered)]
        self.assertEqual(len(generated), len(uncovered))
        self.assertIn("api_request", tools.ALL_TOOLS)

    def test_tool_names_are_unique_and_short(self):
        names = list(tools.ALL_TOOLS)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(len(n) <= 64 and re.fullmatch(r"[a-z0-9_]+", n) for n in names), [n for n in names if len(n) > 64])

    def test_a_generated_tool_fills_the_path_and_routes_body_and_query(self):
        client = FakeClient()
        tool = tools.ALL_TOOLS["api_post_scenes_video_trim"]
        tool.handler(client, {"project_id": "p", "scene_id": "s", "body": {"start": 0.5, "duration": 4}})
        self.assertEqual(client.calls[-1][:2], ("POST", "/projects/p/scenes/s/video/trim"))
        self.assertEqual(client.calls[-1][2]["json"], {"start": 0.5, "duration": 4})
        self.assertEqual(tool.input_schema["required"], ["project_id", "scene_id"])
        tools.ALL_TOOLS["api_get_jobs_log"].handler(client, {"job_id": "j9", "query": {"since": 3}})
        self.assertEqual(client.calls[-1][:2], ("GET", "/jobs/j9/log"))
        self.assertEqual(client.calls[-1][2]["params"], {"since": 3})

    def test_a_missing_id_is_an_error_not_a_bad_request(self):
        client = FakeClient()
        res = tools.ALL_TOOLS["api_post_scenes_video_trim"].handler(client, {"project_id": "p"})
        self.assertTrue(res["isError"])
        self.assertEqual(client.calls, [])

    def test_api_request_reaches_any_endpoint_and_refuses_bad_paths(self):
        client = FakeClient()
        tools.ALL_TOOLS["api_request"].handler(client, {"method": "put", "path": "/projects/p/story", "body": {"a": 1}, "if_match_revision": 4})
        self.assertEqual(client.calls[-1][:2], ("PUT", "/projects/p/story"))
        self.assertEqual(client.calls[-1][2]["if_match"], 4)
        for bad in ("projects", "http://evil/x", "/projects/../x"):
            res = tools.ALL_TOOLS["api_request"].handler(client, {"method": "GET", "path": bad})
            self.assertTrue(res["isError"], bad)


if __name__ == "__main__":
    unittest.main()
