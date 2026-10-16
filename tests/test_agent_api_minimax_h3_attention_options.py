"""The Agent API reads and sets the MiniMax H3 attention render options, globally and per scene.

Session keys (``web/music_video_builder/minimax_h3.mjs`` DEFAULT_MINIMAX_H3_SETTINGS):
``sage_attention`` (MINIMAX_H3_SAGE_ATTENTION_OPTIONS), ``use_memory_efficient_sage_attention``,
``use_block_sparse_attention`` (single pass), ``two_pass_use_block_sparse_attention`` (2 Pass / 2 Pass Advanced)
and the optional per-pass ``pass1_use_block_sparse_attention`` / ``pass2_use_block_sparse_attention``.
Global: PATCH /projects/{pid}/settings {"minimax_h3": {...}}. Per scene: PATCH /projects/{pid}/scenes/{sid}
{"use_scene_minimax_h3_settings": true, "minimax_h3_settings": {...}}.
"""

import importlib
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

projects = importlib.import_module(f"{ROOT.name}.agent_api.projects")
mutations = importlib.import_module(f"{ROOT.name}.agent_api.mutations")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")
payload_mod = importlib.import_module(f"{ROOT.name}.minimax.settings_payload")
patches = importlib.import_module(f"{ROOT.name}.runner.minimax_patches")
from test_agent_api_references_llm import Base  # noqa: E402

UI_FILE = ROOT / "web" / "music_video_builder" / "minimax_h3.mjs"


def _ui_sage_modes():
    text = UI_FILE.read_text(encoding="utf-8")
    block = text.split("MINIMAX_H3_SAGE_ATTENTION_OPTIONS = [", 1)[1].split("];", 1)[0]
    return tuple(re.findall(r'value:\s*"([^"]+)"', block))


def _rev(session):
    return int(session.get("revision") or 0)


class AttentionOptionsTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        session["minimax_h3_settings"] = {"video_mode": "reference_to_video", "render_pass": "two_pass", "two_pass_use_block_sparse_attention": True,
                                          "sage_attention": "disabled", "use_memory_efficient_sage_attention": False}
        self.write_session(session)

    # --- allowed values ---------------------------------------------------------------------------------------
    def test_sage_modes_match_the_builder_and_the_render(self):
        ui = _ui_sage_modes()
        self.assertIn("auto", ui)
        self.assertEqual(tuple(payload_mod.SETTINGS_ENUMS["sage_attention"]), ui)
        self.assertEqual(set(ui) - {"disabled", "auto"}, set(patches._MINIMAX_H3_SAGE_ATTENTION_MODES) - {"disabled", "auto"})
        schema = payload_mod.minimax_h3_settings_schema()["settings"]
        self.assertEqual(schema["sage_attention"]["enum"], list(ui))

    # --- global -----------------------------------------------------------------------------------------------
    def test_global_get_shows_the_options(self):
        mm = projects.get_project_detail("Song", ["settings"])["settings"]["minimax_h3"]
        self.assertEqual(mm["sage_attention"], "disabled")
        self.assertIs(mm["two_pass_use_block_sparse_attention"], True)
        self.assertIs(mm["use_memory_efficient_sage_attention"], False)
        self.assertIn("use_block_sparse_attention", mm)

    def test_global_patch_sets_them(self):
        rev = _rev(self.read_session())
        mutations.patch_project_settings("Song", {"minimax_h3": {
            "two_pass_use_block_sparse_attention": False, "sage_attention": "auto",
            "use_memory_efficient_sage_attention": True}}, if_match_revision=rev)
        saved = self.read_session()["minimax_h3_settings"]
        self.assertEqual(saved["sage_attention"], "auto")
        self.assertIs(saved["two_pass_use_block_sparse_attention"], False)
        self.assertIs(saved["use_memory_efficient_sage_attention"], True)
        self.assertEqual(saved["render_pass"], "two_pass")  # untouched keys stay

    def test_global_bad_sage_mode_is_400(self):
        with self.assertRaises(errors.SettingsInvalidError) as ctx:
            mutations.patch_project_settings("Song", {"minimax_h3": {"sage_attention": "fast"}})
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("auto", ctx.exception.details["invalid_fields"]["minimax_h3.sage_attention"])
        self.assertEqual(self.read_session()["minimax_h3_settings"]["sage_attention"], "disabled")

    def test_global_bad_boolean_is_400(self):
        with self.assertRaises(errors.SettingsInvalidError):
            mutations.patch_project_settings("Song", {"minimax_h3": {"use_memory_efficient_sage_attention": "yes"}})

    def test_global_stale_if_match_is_409(self):
        rev = _rev(self.read_session())
        with self.assertRaises(errors.RevisionConflictError) as ctx:
            mutations.patch_project_settings("Song", {"minimax_h3": {"sage_attention": "auto"}}, if_match_revision=rev + 5)
        self.assertEqual(ctx.exception.status, 409)

    # --- per scene --------------------------------------------------------------------------------------------
    def test_scene_patch_sets_them_and_merges_into_the_locked_settings(self):
        session = self.read_session()
        session["segments"][7].update({"use_scene_minimax_h3_settings": True,
                                       "minimax_h3_settings": {"render_pass": "two_pass", "steps": 12}})
        self.write_session(session)
        rev = _rev(self.read_session())
        mutations.patch_scene("Song", "8", {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": {
            "two_pass_use_block_sparse_attention": False, "sage_attention": "auto",
            "use_memory_efficient_sage_attention": True}}, if_match_revision=rev)
        scene = self.read_session()["segments"][7]
        self.assertIs(scene["use_scene_minimax_h3_settings"], True)
        self.assertEqual(scene["minimax_h3_settings"]["sage_attention"], "auto")
        self.assertIs(scene["minimax_h3_settings"]["use_memory_efficient_sage_attention"], True)
        self.assertEqual(scene["minimax_h3_settings"]["steps"], 12)  # merge, not replace

    def test_scene_get_shows_the_resolved_options_and_their_source(self):
        mutations.patch_scene("Song", "8", {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": {
            "two_pass_use_block_sparse_attention": False, "sage_attention": "auto",
            "use_memory_efficient_sage_attention": True}})
        att = projects.get_scene_detail("Song", "8")["minimax_h3_attention"]
        self.assertEqual(att["source"], "scene")
        self.assertEqual(att["render_pass"], "two_pass")
        self.assertEqual(att["sage_attention"], "auto")
        self.assertIs(att["use_memory_efficient_sage_attention"], True)
        self.assertIs(att["block_sparse_attention"], False)
        self.assertEqual(att["block_sparse_attention_by_pass"], {"pass1": False, "pass2": False})
        other = projects.get_scene_detail("Song", "1")["minimax_h3_attention"]
        self.assertEqual(other["source"], "project")
        self.assertEqual(other["sage_attention"], "disabled")
        self.assertIs(other["block_sparse_attention"], True)

    def test_scene_bad_values_are_400_and_nothing_is_saved(self):
        before = self.read_session()["segments"][7]
        for bad in ({"sage_attention": "turbo"}, {"use_block_sparse_attention": "off"}, {"not_a_setting": 1}):
            with self.assertRaises(errors.ValidationError) as ctx:
                mutations.patch_scene("Song", "8", {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": bad})
            self.assertEqual(ctx.exception.status, 400)
            self.assertIn("minimax_h3_settings.", str(ctx.exception))
        with self.assertRaises(errors.ValidationError):
            mutations.patch_scene("Song", "8", {"use_scene_minimax_h3_settings": "yes"})
        self.assertEqual(self.read_session()["segments"][7], before)

    def test_scene_stale_if_match_is_409(self):
        rev = _rev(self.read_session())
        with self.assertRaises(errors.RevisionConflictError):
            mutations.patch_scene("Song", "8", {"use_scene_minimax_h3_settings": True,
                                                "minimax_h3_settings": {"sage_attention": "auto"}}, if_match_revision=rev + 3)

    # --- render resolution ------------------------------------------------------------------------------------
    def test_render_payload_uses_scene_over_global_only_when_locked(self):
        mutations.patch_scene("Song", "8", {"use_scene_minimax_h3_settings": True, "minimax_h3_settings": {
            "two_pass_use_block_sparse_attention": False, "sage_attention": "auto",
            "use_memory_efficient_sage_attention": True}})
        session = self.read_session()
        locked = payload_mod.build_minimax_render_payload(
            payload_mod.minimax_h3_settings_for_scene(session, session["segments"][7]))
        self.assertEqual(locked["sage_attention"], "auto")
        self.assertIs(locked["use_memory_efficient_sage_attention"], True)
        self.assertIs(locked["two_pass_use_block_sparse_attention"], False)
        self.assertIs(locked["pass1_use_block_sparse_attention"], False)
        self.assertIs(locked["pass2_use_block_sparse_attention"], False)
        project = payload_mod.build_minimax_render_payload(
            payload_mod.minimax_h3_settings_for_scene(session, session["segments"][0]))
        self.assertEqual(project["sage_attention"], "disabled")
        self.assertIs(project["two_pass_use_block_sparse_attention"], True)
        # Unlocking the scene drops back to the project values without deleting the saved scene settings.
        mutations.patch_scene("Song", "8", {"use_scene_minimax_h3_settings": False})
        session = self.read_session()
        unlocked = payload_mod.build_minimax_render_payload(
            payload_mod.minimax_h3_settings_for_scene(session, session["segments"][7]))
        self.assertEqual(unlocked["sage_attention"], "disabled")
        self.assertEqual(session["segments"][7]["minimax_h3_settings"]["sage_attention"], "auto")


if __name__ == "__main__":
    unittest.main()
