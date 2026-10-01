"""Unit tests for Phase A5: MCP Server (Section 8, Appendix C)."""

import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from mcp_server.client import ApiClientError, VrgdgApiClient
from mcp_server.prompts import get_prompt, list_prompts
from mcp_server.protocol import PROTOCOL_VERSION, SERVER_NAME, format_tool_result
from mcp_server.resources import list_resources, read_resource
from mcp_server.server import McpServer, run_stdio_server
from mcp_server.tools import ALL_TOOLS


class TestMcpServer(unittest.TestCase):
    def setUp(self):
        self.mock_db: dict = {
            "projects": [
                {
                    "project_name": "TestProject",
                    "audio_file": "test.wav",
                    "revision": 1,
                    "segments": [
                        {"id": "scene_001", "start": 0.0, "end": 4.0, "t2i_prompt": "Neon city", "i2v_prompt": "Pan left"},
                        {"id": "scene_002", "start": 4.0, "end": 8.0, "t2i_prompt": "Rainy street", "i2v_prompt": "Zoom in"},
                    ],
                }
            ],
            "jobs": {
                "job_001": {
                    "id": "job_001",
                    "type": "video.render",
                    "status": "completed",
                    "progress": 100.0,
                    "message": "Render completed.",
                    "error": None,
                }
            },
        }

        # Setup mock transport routing for VrgdgApiClient
        def _mock_transport(method: str, url: str, params: dict, json_data: dict):
            # Parse path from url
            path = url.split("/api/v1")[-1] if "/api/v1" in url else url

            if path == "/health":
                return 200, {"ok": True, "data": {"status": "healthy", "gpu_available": True}}

            if path == "/modes":
                return 200, {"ok": True, "data": {"image_modes": ["zimage"], "video_modes": ["i2v", "minimax_h3"]}}

            if path == "/models":
                return 200, {"ok": True, "data": {"checkpoints": ["model.safetensors"]}}

            if path == "/projects" and method == "GET":
                return 200, {"ok": True, "data": {"projects": [p["project_name"] for p in self.mock_db["projects"]]}}

            if path == "/projects" and method == "POST":
                p_name = json_data.get("project_name", "NewProj")
                new_p = {"project_name": p_name, "revision": 1, "segments": []}
                self.mock_db["projects"].append(new_p)
                return 201, {"ok": True, "data": new_p}

            if path == "/projects/TestProject" and method == "GET":
                return 200, {"ok": True, "data": self.mock_db["projects"][0]}

            if path == "/projects/TestProject/summary":
                return 200, {"ok": True, "data": {"project_name": "TestProject", "scene_count": 2}}

            if path == "/projects/TestProject/settings" and method == "PATCH":
                return 200, {"ok": True, "data": {"settings": json_data}}

            if path == "/projects/TestProject" and method == "DELETE":
                return 200, {"ok": True, "data": {"deleted": True}}

            if path == "/projects/TestProject/audio" and method == "POST":
                return 200, {"ok": True, "data": {"audio_file": json_data.get("audio_file")}}

            if path == "/projects/TestProject/audio/beats" and method == "POST":
                return 200, {"ok": True, "data": {"beats": [0.5, 1.0, 1.5, 2.0]}}

            if path == "/projects/TestProject/lyrics" and method == "PUT":
                return 200, {"ok": True, "data": {"lyrics": json_data.get("lyrics")}}

            if path == "/projects/TestProject/lyrics" and method == "GET":
                return 200, {"ok": True, "data": {"lyrics_text": "Sample lyrics line 1\nLine 2"}}

            if path == "/projects/TestProject/timeline/bulk" and method == "POST":
                return 200, {"ok": True, "data": {"scene_count": 3}}

            if path == "/projects/TestProject/scenes" and method == "GET":
                return 200, {"ok": True, "data": {"scenes": self.mock_db["projects"][0]["segments"]}}

            if path == "/projects/TestProject/scenes/scene_001" and method == "GET":
                return 200, {"ok": True, "data": self.mock_db["projects"][0]["segments"][0]}

            if path == "/projects/TestProject/scenes/scene_001" and method == "PATCH":
                return 200, {"ok": True, "data": {"scene": {**self.mock_db["projects"][0]["segments"][0], **json_data}}}

            if path == "/projects/TestProject/scenes/scene_001/split" and method == "POST":
                return 200, {"ok": True, "data": {"split": True}}

            if path == "/projects/TestProject/scenes/scene_001/image/generate" and method == "POST":
                return 202, {"ok": True, "data": {"job_id": "job_img_01", "status": "queued"}}

            if path == "/projects/TestProject/scenes/scene_001/video/render" and method == "POST":
                return 202, {"ok": True, "data": {"job_id": "job_vid_01", "status": "queued"}}

            if path == "/projects/TestProject/stitch" and method == "POST":
                return 200, {"ok": True, "data": {"final_video_path": "C:/path/to/FINAL_VIDEO.mp4"}}

            if path == "/projects/TestProject/pipelines/build-full-video" and method == "POST":
                return 202, {"ok": True, "data": {"job_id": "job_pipe_01", "status": "queued"}}

            if path == "/projects/TestProject/pipelines/plan" and method == "GET":
                return 200, {"ok": True, "data": {"can_run": True, "summary": {"total_scenes": 2}}}

            if path == "/jobs/job_001" and method == "GET":
                return 200, {"ok": True, "data": self.mock_db["jobs"]["job_001"]}

            # Default fallback
            return 200, {"ok": True, "data": {"path": path, "method": method}}

        self.client = VrgdgApiClient()
        self.client.set_transport(_mock_transport)
        self.server = McpServer(client=self.client)

    # ==========================================================================
    # 1. Lifecycle and Protocol Handling
    # ==========================================================================

    def test_initialize(self):
        """Test initialize handshake response."""
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        res = self.server.handle_request(req)
        self.assertEqual(res["id"], 1)
        result = res["result"]
        self.assertEqual(result["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(result["serverInfo"]["name"], SERVER_NAME)
        self.assertIn("tools", result["capabilities"])
        self.assertIn("resources", result["capabilities"])
        self.assertIn("prompts", result["capabilities"])

    def test_notifications_initialized(self):
        """Test notifications/initialized sends no response."""
        req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        res = self.server.handle_request(req)
        self.assertIsNone(res)

    def test_ping(self):
        """Test ping response."""
        req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
        res = self.server.handle_request(req)
        self.assertEqual(res["id"], 2)
        self.assertEqual(res["result"], {})

    def test_tools_list(self):
        """Test listing all tools (T1 to T50)."""
        req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
        res = self.server.handle_request(req)
        tools = res["result"]["tools"]
        self.assertEqual(len(tools), 50)
        tool_names = {t["name"] for t in tools}
        self.assertIn("system_health", tool_names)
        self.assertIn("project_create", tool_names)
        self.assertIn("timeline_build", tool_names)
        self.assertIn("pipeline_build_full_video", tool_names)
        self.assertIn("upload_file", tool_names)

    def test_unknown_method(self):
        """Test unknown method returns -32601 error."""
        req = {"jsonrpc": "2.0", "id": 4, "method": "non_existent_method"}
        res = self.server.handle_request(req)
        self.assertIn("error", res)
        self.assertEqual(res["error"]["code"], -32601)

    # ==========================================================================
    # 2. Tool Execution (T1 through T50)
    # ==========================================================================

    def test_tool_system_health(self):
        """Test system_health tool call (T1)."""
        req = {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "system_health", "arguments": {}}}
        res = self.server.handle_request(req)
        self.assertFalse(res["result"]["isError"])
        content_text = res["result"]["content"][0]["text"]
        self.assertIn("healthy", content_text)

    def test_tool_project_create_and_summary(self):
        """Test project_create (T5) and project_summary (T7)."""
        # Create
        req1 = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "project_create", "arguments": {"project_name": "BrandNewProject"}},
        }
        res1 = self.server.handle_request(req1)
        self.assertFalse(res1["result"]["isError"])
        self.assertIn("BrandNewProject", res1["result"]["content"][0]["text"])

        # Summary
        req2 = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {"name": "project_summary", "arguments": {"project_id": "TestProject"}},
        }
        res2 = self.server.handle_request(req2)
        self.assertFalse(res2["result"]["isError"])
        self.assertIn("scene_count", res2["result"]["content"][0]["text"])

    def test_tool_project_delete_confirm_guard(self):
        """Test project_delete (T12) requires confirm=true."""
        # Unconfirmed
        req_unconf = {
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {"name": "project_delete", "arguments": {"project_id": "TestProject", "confirm": False}},
        }
        res_unconf = self.server.handle_request(req_unconf)
        self.assertTrue(res_unconf["result"]["isError"])
        self.assertIn("confirm", res_unconf["result"]["content"][0]["text"])

        # Confirmed
        req_conf = {
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tools/call",
            "params": {"name": "project_delete", "arguments": {"project_id": "TestProject", "confirm": True}},
        }
        res_conf = self.server.handle_request(req_conf)
        self.assertFalse(res_conf["result"]["isError"])

    def test_tool_scene_split_merge_move_resize(self):
        """Test scene_split_merge_move_resize tool (T23)."""
        req = {
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": {
                "name": "scene_split_merge_move_resize",
                "arguments": {"project_id": "TestProject", "scene_id": "scene_001", "op": "split", "split_time": 2.0},
            },
        }
        res = self.server.handle_request(req)
        self.assertFalse(res["result"]["isError"])
        self.assertIn("split", res["result"]["content"][0]["text"])

    def test_tool_error_mapping_and_next_steps(self):
        """Test Rule 5: API errors map to isError with next_steps."""
        def _failing_transport(*args):
            return 409, {
                "ok": False,
                "error": {
                    "code": "PREDECESSOR_MISSING",
                    "message": "Scene 2 requires rendered video from Scene 1.",
                    "details": {"missing_predecessor": 1},
                },
            }

        client = VrgdgApiClient()
        client.set_transport(_failing_transport)
        server = McpServer(client=client)

        req = {
            "jsonrpc": "2.0",
            "id": 16,
            "method": "tools/call",
            "params": {"name": "video_render", "arguments": {"project_id": "TestProject", "scene_id": "scene_002"}},
        }
        res = server.handle_request(req)
        self.assertTrue(res["result"]["isError"])
        err_text = res["result"]["content"][0]["text"]
        self.assertIn("[PREDECESSOR_MISSING]", err_text)
        self.assertIn("Next steps:", err_text)
        self.assertIn("predecessor", err_text)

    def test_tool_job_wait(self):
        """Test job_wait tool (T45)."""
        req = {
            "jsonrpc": "2.0",
            "id": 17,
            "method": "tools/call",
            "params": {"name": "job_wait", "arguments": {"job_id": "job_001", "timeout_seconds": 5.0}},
        }
        res = self.server.handle_request(req)
        self.assertFalse(res["result"]["isError"])
        self.assertIn("completed", res["result"]["content"][0]["text"])

    # ==========================================================================
    # 3. Resources (Section 8.2)
    # ==========================================================================

    def test_resources_list(self):
        """Test listing resources."""
        req = {"jsonrpc": "2.0", "id": 20, "method": "resources/list"}
        res = self.server.handle_request(req)
        resources = res["result"]["resources"]
        self.assertTrue(len(resources) >= 4)
        uris = [r.get("uri") or r.get("uriTemplate") for r in resources]
        self.assertIn("vrgdg://projects", uris)
        self.assertIn("vrgdg://modes", uris)

    def test_resources_read(self):
        """Test reading resource contents."""
        # Read projects
        req1 = {"jsonrpc": "2.0", "id": 21, "method": "resources/read", "params": {"uri": "vrgdg://projects"}}
        res1 = self.server.handle_request(req1)
        self.assertIn("contents", res1["result"])
        self.assertIn("TestProject", res1["result"]["contents"][0]["text"])

        # Read lyrics
        req2 = {"jsonrpc": "2.0", "id": 22, "method": "resources/read", "params": {"uri": "vrgdg://project/TestProject/lyrics"}}
        res2 = self.server.handle_request(req2)
        self.assertIn("Sample lyrics", res2["result"]["contents"][0]["text"])

        # Read job log
        req3 = {"jsonrpc": "2.0", "id": 23, "method": "resources/read", "params": {"uri": "vrgdg://jobs/job_001/log"}}
        res3 = self.server.handle_request(req3)
        self.assertIn("Job ID: job_001", res3["result"]["contents"][0]["text"])

    # ==========================================================================
    # 4. Prompts (Section 8.3)
    # ==========================================================================

    def test_prompts_list_and_get(self):
        """Test listing and getting prompt templates."""
        req_list = {"jsonrpc": "2.0", "id": 30, "method": "prompts/list"}
        res_list = self.server.handle_request(req_list)
        prompts = res_list["result"]["prompts"]
        names = [p["name"] for p in prompts]
        self.assertIn("make_music_video", names)
        self.assertIn("review_scene", names)
        self.assertIn("fix_failed_render", names)
        self.assertIn("polish_timeline", names)

        # Get make_music_video prompt
        req_get = {
            "jsonrpc": "2.0",
            "id": 31,
            "method": "prompts/get",
            "params": {"name": "make_music_video", "arguments": {"project_name": "EpicSong", "audio_file": "song.mp3"}},
        }
        res_get = self.server.handle_request(req_get)
        self.assertIn("messages", res_get["result"])
        msg_text = res_get["result"]["messages"][0]["content"]["text"]
        self.assertIn("EpicSong", msg_text)
        self.assertIn("project_create", msg_text)

    # ==========================================================================
    # 5. Stdio Server Loop
    # ==========================================================================

    def test_stdio_server_loop(self):
        """Test executing commands through the stdio stream processing loop."""
        in_stream = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "system_health", "arguments": {}}}) + "\n"
        )
        out_stream = io.StringIO()

        self.server.run(in_stream=in_stream, out_stream=out_stream)
        output_lines = [line.strip() for line in out_stream.getvalue().split("\n") if line.strip()]
        self.assertEqual(len(output_lines), 2)

        resp1 = json.loads(output_lines[0])
        self.assertEqual(resp1["id"], 1)

        resp2 = json.loads(output_lines[1])
        self.assertEqual(resp2["id"], 2)
        self.assertFalse(resp2["result"]["isError"])

    # ==========================================================================
    # 6. Sample Agent Flow (Section 8.4)
    # ==========================================================================

    def test_sample_agent_flow_section_8_4(self):
        """Verify the full agent workflow from Section 8.4."""
        # 1. project_create
        r1 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "project_create", "arguments": {"project_name": "SampleFlowProject"}},
        })
        self.assertFalse(r1["result"]["isError"])

        # 2. audio_attach
        r2 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "audio_attach", "arguments": {"project_id": "TestProject", "audio_file": "track.mp3"}},
        })
        self.assertFalse(r2["result"]["isError"])

        # 3. audio_analyze
        r3 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "audio_analyze", "arguments": {"project_id": "TestProject"}},
        })
        self.assertFalse(r3["result"]["isError"])

        # 4. lyrics_set
        r4 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "lyrics_set", "arguments": {"project_id": "TestProject", "lyrics": "First line of the song"}},
        })
        self.assertFalse(r4["result"]["isError"])

        # 5. timeline_build
        r5 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "timeline_build", "arguments": {"project_id": "TestProject", "text": "4, 4, 4", "mode": "durations"}},
        })
        self.assertFalse(r5["result"]["isError"])

        # 6. prompts_generate (image)
        r6 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "prompts_generate", "arguments": {"project_id": "TestProject", "kind": "image", "scope": "all"}},
        })
        self.assertFalse(r6["result"]["isError"])

        # 7. image_generate
        r7 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "image_generate", "arguments": {"project_id": "TestProject", "scene_id": "scene_001"}},
        })
        self.assertFalse(r7["result"]["isError"])

        # 8. video_render
        r8 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {"name": "video_render", "arguments": {"project_id": "TestProject", "scene_id": "scene_001"}},
        })
        self.assertFalse(r8["result"]["isError"])

        # 9. stitch_final
        r9 = self.server.handle_request({
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": "stitch_final", "arguments": {"project_id": "TestProject"}},
        })
        self.assertFalse(r9["result"]["isError"])
        self.assertIn("FINAL_VIDEO.mp4", r9["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
