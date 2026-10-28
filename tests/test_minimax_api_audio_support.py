"""Offline API/MCP settings and native-audio parity, without loading ComfyUI."""

import ast
import asyncio
import copy
import importlib.util
import os
from pathlib import Path
import sys
import textwrap
import threading
from typing import Any, Dict, Optional
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("minimax_api_export", ROOT / "scripts/export_openapi.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
SCHEMAS = EXPORT._load_agent_api_module("schemas")
SETTINGS = sys.modules["_vrgdg_pack.minimax.settings_payload"]


class MiniMaxApiAudioSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = {"revision": 7, "minimax_h3_settings": {
            "pipeline": "standard", "video_mode": "text_to_video", "render_pass": "two_pass",
        }}
        self.namespace = {
            "Any": Any, "Dict": Dict, "Optional": Optional,
            "_BUILDER_SAVE_LOCK": threading.RLock(),
            "validate_settings_patch": SCHEMAS.validate_settings_patch,
            "extract_effective_settings": SCHEMAS.extract_effective_settings,
            "apply_minimax_h3_settings_patch": SETTINGS.apply_minimax_h3_settings_patch,
            "_get_active_session_and_folder": lambda pid: ("unused", self.session),
            "_persist_session": lambda folder, session: {"session": copy.deepcopy(session), "revision": 8},
        }
        tree = ast.parse((ROOT / "agent_api/mutations.py").read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name == "patch_project_settings"
        ) or (
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {
                    "_SETTINGS_GROUP_SESSION_KEYS", "_SETTINGS_FLAT_GROUPS"
                } for target in node.targets
            )
        )]
        exec(compile(ast.Module(nodes, []), "api_settings", "exec"), self.namespace)
        self.addCleanup(patch.stopall)
        patch.object(SCHEMAS, "global_llm_runner", return_value={}).start()

    def patch_settings(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self.namespace["patch_project_settings"]("project", body)

    def test_explicit_api_pass_selection_survives_legacy_migration(self) -> None:
        legacy = SETTINGS.minimax_h3_settings_for_scene(self.session, {})
        self.assertEqual(SETTINGS.minimax_workflow_key(legacy), "minimax_h3")
        result = self.patch_settings({"minimax_h3": {
            "render_pass": "two_pass", "audio_mode": "built_in_audio",
        }})
        settings = result["settings"]["minimax_h3"]
        self.assertEqual(result["revision"], 8)
        self.assertEqual(settings["video_mode"], "text_to_video")
        self.assertEqual(settings["pipeline"], "standard")
        self.assertEqual(SETTINGS.minimax_workflow_key(settings), "minimax_h3_2pass")
        self.assertEqual(SETTINGS.build_minimax_render_payload(settings)["audio_mode"], "built_in_audio")
        reloaded = SETTINGS.minimax_h3_settings_for_scene(self.session, {})
        self.assertEqual(SETTINGS.minimax_workflow_key(reloaded), "minimax_h3_2pass")

    def test_unrelated_api_edit_keeps_legacy_single_pass_behavior(self) -> None:
        result = self.patch_settings({"minimax_h3": {"audio_mode": "built_in_audio"}})
        self.assertEqual(result["settings"]["minimax_h3"]["render_pass"], "single")

    def test_scene_pass_patch_supports_inherited_t2v_and_i2v_modes(self) -> None:
        for mode in ("text_to_video", "image_to_video"):
            scene_settings = SETTINGS.apply_minimax_h3_settings_patch(
                {"render_pass": "single"}, {"render_pass": "two_pass", "audio_mode": "built_in_audio"}, mode,
            )
            session = {"minimax_h3_settings": {"video_mode": mode}}
            scene = {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": scene_settings}
            effective = SETTINGS.minimax_h3_settings_for_scene(session, scene)
            self.assertEqual(SETTINGS.minimax_workflow_key(effective), "minimax_h3_2pass")
            self.assertEqual(effective["audio_mode"], "built_in_audio")

    def test_api_native_audio_skips_supplied_audio_and_speaking_mix(self) -> None:
        source = (ROOT / "agent_api/orchestrator/video_orchestrator.py").read_text(encoding="utf-8")
        block = source.split("# Audio preparation", 1)[1].split("# SRT preparation", 1)[0]
        namespace = {"Dict": Dict, "Any": Any, "os": os, "asyncio": asyncio,
                     "session_audio_path": Mock(side_effect=AssertionError("must not read supplied audio"))}
        signature = "async def prepare(mode_group, payload, p, session, seg):\n"
        body = textwrap.indent(textwrap.dedent(block), "    ")
        exec(signature + body + "\n    return audio_path, mask_restore\n", namespace)
        result = asyncio.run(namespace["prepare"](
            "minimax_h3", {"audio_mode": "built_in_audio"}, {"audio_path": "unused.wav"},
            {"video_type": "speaking", "audio_clips": [{"path": "unused.wav"}]}, {},
        ))
        self.assertEqual(result, ("", {}))
        namespace["session_audio_path"].assert_not_called()

    def test_mcp_settings_and_render_tools_preserve_native_t2v_two_pass(self) -> None:
        if not (ROOT / "mcp_server").is_dir():
            self.skipTest("mcp_server is local-only and absent in this checkout")
        sys.path.insert(0, str(ROOT))
        from mcp_server.tools import ALL_TOOLS

        calls = []
        owner = self

        class Client:
            def patch(self, path: str, json_data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
                calls.append(("PATCH", path, json_data))
                return {"ok": True, "data": owner.patch_settings(json_data)}

            def post(self, path: str, json_data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
                calls.append(("POST", path, json_data))
                return {"ok": True, "data": {"job_id": "offline-placeholder"}}

        client = Client()
        body = {"minimax_h3": {"render_pass": "two_pass", "audio_mode": "built_in_audio"}}
        result = ALL_TOOLS["project_update_settings"].handler(client, {"project_id": "project", "settings": body})
        self.assertFalse(result.get("isError", False))
        ALL_TOOLS["video_render"].handler(client, {
            "project_id": "project", "scene_id": "scene", "params": {"mode": "minimax_h3"},
        })
        self.assertEqual(calls, [
            ("PATCH", "/projects/project/settings", body),
            ("POST", "/projects/project/scenes/scene/video/render", {"mode": "minimax_h3"}),
        ])
        effective = SETTINGS.minimax_h3_settings_for_scene(self.session, {})
        self.assertEqual(SETTINGS.minimax_workflow_key(effective), "minimax_h3_2pass")
        self.assertEqual(effective["audio_mode"], "built_in_audio")


if __name__ == "__main__":
    unittest.main()
