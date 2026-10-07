"""Tests for the Video Builder layout profiles and the "last selected" memory of both profile kinds."""

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
sys.path.insert(0, str(ROOT / "tests"))
from builder_source import read_builder_source  # noqa: E402

BUILDER_SOURCE = read_builder_source()
ui_profiles = importlib.import_module(f"{pkg_name}.builder.ui_profiles")
video_profiles = importlib.import_module(f"{pkg_name}.builder.video_profiles")


class UiProfileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.join(self._tmp.name, "VRGDG_UI_Profiles", "music_video_builder")
        patcher = patch.object(ui_profiles, "profile_root", return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _layout(self, **overrides):
        layout = {
            "left_collapsed": True, "right_collapsed": True, "llm_popout_open": True, "left_panel_width": 300,
            "right_panel_width": 400, "timeline_panel_height": 640, "llm_popout_width": 520,
            "llm_popout_height": 410, "llm_popout_x": 120, "llm_popout_y": 80,
        }
        layout.update(overrides)
        return layout

    def test_save_list_load_round_trip(self):
        saved = ui_profiles.save_ui_profile("Wide timeline", self._layout())
        self.assertEqual(saved["name"], "Wide timeline")
        self.assertEqual([item["name"] for item in ui_profiles.list_ui_profiles()], ["Wide timeline"])
        loaded = ui_profiles.load_ui_profile("wide TIMELINE")  # names are not case sensitive
        self.assertEqual(loaded["layout"], self._layout())

    def test_the_timeline_can_be_taller_than_the_old_520_pixel_limit(self):
        layout = ui_profiles.normalize_layout({"timeline_panel_height": 900})
        self.assertEqual(layout["timeline_panel_height"], 900)

    def test_bad_values_are_clamped_or_defaulted(self):
        layout = ui_profiles.normalize_layout({
            "left_collapsed": 1, "left_panel_width": 5, "right_panel_width": "wide", "timeline_panel_height": 10 ** 9,
        })
        self.assertIs(layout["left_collapsed"], True)
        self.assertEqual(layout["left_panel_width"], 180)
        self.assertEqual(layout["right_panel_width"], 360)
        self.assertEqual(layout["timeline_panel_height"], 4000)
        self.assertEqual(layout["llm_popout_width"], 460)
        self.assertEqual(ui_profiles.normalize_layout({"llm_popout_width": 10})["llm_popout_width"], 300)
        # the floating window is not placed until the user moves it
        self.assertIsNone(layout["llm_popout_x"])
        self.assertIsNone(layout["llm_popout_y"])
        placed = ui_profiles.normalize_layout({"llm_popout_x": "35.6", "llm_popout_y": 10 ** 9, "llm_popout_height": 1})
        self.assertEqual((placed["llm_popout_x"], placed["llm_popout_y"], placed["llm_popout_height"]), (36, 10000, 200))
        self.assertIsNone(ui_profiles.normalize_layout({"llm_popout_x": "left"})["llm_popout_x"])
        self.assertIs(layout["right_collapsed"], False)
        self.assertIs(layout["llm_popout_open"], False)
        self.assertEqual(ui_profiles.normalize_layout(None), ui_profiles.DEFAULT_LAYOUT)

    def test_saving_over_an_existing_name_needs_overwrite(self):
        ui_profiles.save_ui_profile("Mine", self._layout())
        with self.assertRaises(ui_profiles.UiProfileExistsError):
            ui_profiles.save_ui_profile("mine", self._layout(timeline_panel_height=500))
        ui_profiles.save_ui_profile("mine", self._layout(timeline_panel_height=500), overwrite=True)
        self.assertEqual(ui_profiles.load_ui_profile("Mine")["layout"]["timeline_panel_height"], 500)

    def test_updating_the_layout_keeps_the_name_and_requires_an_existing_profile(self):
        ui_profiles.save_ui_profile("Mine", self._layout())
        updated = ui_profiles.update_ui_profile_layout("MINE", self._layout(left_collapsed=False))
        self.assertEqual(updated["name"], "Mine")
        self.assertIs(ui_profiles.load_ui_profile("Mine")["layout"]["left_collapsed"], False)
        with self.assertRaises(FileNotFoundError):
            ui_profiles.update_ui_profile_layout("Missing", self._layout())

    def test_the_last_saved_or_selected_profile_is_remembered(self):
        self.assertEqual(ui_profiles.get_last_ui_profile_name(), "")
        ui_profiles.save_ui_profile("First", self._layout())
        self.assertEqual(ui_profiles.get_last_ui_profile_name(), "First")
        ui_profiles.save_ui_profile("Second", self._layout())
        self.assertEqual(ui_profiles.get_last_ui_profile_name(), "Second")
        ui_profiles.set_last_ui_profile("first")
        self.assertEqual(ui_profiles.get_last_ui_profile_name(), "First")
        # choosing no profile is remembered too
        ui_profiles.set_last_ui_profile("")
        self.assertEqual(ui_profiles.get_last_ui_profile_name(), "")
        with self.assertRaises(FileNotFoundError):
            ui_profiles.set_last_ui_profile("Nope")

    def test_the_memory_file_is_not_listed_and_a_deleted_profile_is_forgotten(self):
        ui_profiles.save_ui_profile("Gone", self._layout())
        self.assertTrue(os.path.isfile(os.path.join(self.root, ui_profiles.LAST_SELECTED_FILE)))
        self.assertEqual([item["name"] for item in ui_profiles.list_ui_profiles()], ["Gone"])
        ui_profiles.delete_ui_profile("Gone")
        self.assertEqual(ui_profiles.list_ui_profiles(), [])
        self.assertEqual(ui_profiles.get_last_ui_profile_name(), "")

    def test_unreadable_files_are_skipped(self):
        os.makedirs(self.root, exist_ok=True)
        Path(self.root, "broken.json").write_text("{not json", encoding="utf-8")
        Path(self.root, "other.json").write_text(json.dumps({"name": "x"}), encoding="utf-8")
        self.assertEqual(ui_profiles.list_ui_profiles(), [])


class LastVideoProfileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.join(self._tmp.name, "VRGDG_Video_Profiles", "minimax_h3")
        patcher = patch.object(video_profiles, "profile_root", return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_last_saved_or_selected_video_profile_is_remembered_and_not_listed(self):
        self.assertEqual(video_profiles.get_last_video_profile_name(), "")
        settings = {"video_mode": "reference_to_video", "render_pass": "two_pass"}
        video_profiles.save_video_profile("A", settings)
        video_profiles.save_video_profile("B", settings)
        self.assertEqual(video_profiles.get_last_video_profile_name(), "B")
        self.assertEqual([item["name"] for item in video_profiles.list_video_profiles()], ["A", "B"])
        video_profiles.set_last_video_profile("a")
        self.assertEqual(video_profiles.get_last_video_profile_name(), "A")
        video_profiles.set_last_video_profile("")
        self.assertEqual(video_profiles.get_last_video_profile_name(), "")

    def test_a_deleted_video_profile_is_no_longer_the_last_one(self):
        video_profiles.save_video_profile("A", {"video_mode": "text_to_video"})
        video_profiles.delete_video_profile("A")
        self.assertEqual(video_profiles.get_last_video_profile_name(), "")


class BuilderLayoutWiringTests(unittest.TestCase):
    def test_the_left_panel_has_a_tab_that_hides_it_and_the_timeline_limit_follows_the_window(self):
        self.assertIn("leftPanelToggle.onclick", BUILDER_SOURCE)
        self.assertIn("state.leftPanelCollapsed = !state.leftPanelCollapsed;", BUILDER_SOURCE)
        self.assertIn("${collapsed ? 0 : left}px ${collapsed ? 0 : 7}px minmax(0,1fr)", BUILDER_SOURCE)
        # flex, not an empty value: clearing it would drop the display:flex the panels are built with
        self.assertIn("segmentList.style.display = collapsed ? \"none\" : \"flex\";", BUILDER_SOURCE)
        self.assertIn("inspector.style.display = rightCollapsed ? \"none\" : \"flex\";", BUILDER_SOURCE)
        self.assertIn("state.rightPanelCollapsed = !state.rightPanelCollapsed;", BUILDER_SOURCE)
        self.assertIn("(shell.clientHeight || window.innerHeight) - 230", BUILDER_SOURCE)
        self.assertNotIn("Math.min(520, Number(state.timelinePanelHeight", BUILDER_SOURCE)

    def test_layout_changes_reach_the_selected_profile_and_the_project_session(self):
        self.assertIn("state.onLayoutChanged?.();", BUILDER_SOURCE)
        self.assertIn("/vrgdg/music_builder/update_ui_profile_layout", BUILDER_SOURCE)
        self.assertIn("left_panel_collapsed: Boolean(state.leftPanelCollapsed),", BUILDER_SOURCE)
        # a project's saved sizes load first, then the selected profile wins
        self.assertEqual(BUILDER_SOURCE.count("state.reapplyUiProfileLayout?.();"), 2)

    def test_the_ui_layout_selector_sits_next_to_video_type(self):
        self.assertIn("projectActions.append(menuButton, videoTypeField, uiProfileField, saveButton);", BUILDER_SOURCE)
        self.assertIn("createUiProfileActions({ controls: uiProfileControls", BUILDER_SOURCE)

    def test_the_last_video_profile_is_preselected_and_applied_to_new_projects_only(self):
        self.assertIn('renderOptions(preferredName || data?.last || "");', BUILDER_SOURCE)
        self.assertIn("state.applyVideoProfileToNewProject = applyToNewProject;", BUILDER_SOURCE)
        self.assertIn("await state.applyVideoProfileToNewProject?.();", BUILDER_SOURCE)


class LlmPopoutWiringTests(unittest.TestCase):
    def test_the_prompting_window_floats_and_leaves_the_builder_grid_alone(self):
        # five columns, the right panel in 4 and 5, exactly as before the pop-out existed
        self.assertIn("minmax(0,1fr) ${rightCollapsed ? 0 : 7}px ${rightCollapsed ? 0 : right}px`;", BUILDER_SOURCE)
        source = BUILDER_SOURCE.replace("\r\n", "\n")
        self.assertIn('rightResizeHandle.style.gridColumn = "4";\n    inspector.style.gridColumn = "5";', source)
        self.assertNotIn("popoutOn", BUILDER_SOURCE)
        self.assertNotIn('makePanelResize(llmPopout.handle', BUILDER_SOURCE)
        # a fixed-position window inside the Builder overlay, so it goes away when the Builder closes
        self.assertIn("position:fixed;z-index:100002;", BUILDER_SOURCE)
        self.assertIn("overlay.append(win);", BUILDER_SOURCE)
        self.assertIn("llmPopout.activate();", BUILDER_SOURCE)

    def test_the_window_mirrors_the_panel_fields_instead_of_moving_them(self):
        for text in (
            "original.value = mirror.value;",
            'original.dispatchEvent(new Event("input", { bubbles: true }));',
            "saveMirror.addEventListener(\"click\", () => saveButton.click());",
            "promptMirror.value = prompt.value",
            "pass2Mirror.value = pass2Prompt.value",
            "statusMirror.textContent = status.textContent",
            "saveMirror.disabled = saveButton.disabled",
        ):
            self.assertIn(text, BUILDER_SOURCE)
        # the panel's own fields are never re-parented
        self.assertNotIn("body.append(content)", BUILDER_SOURCE)
        self.assertNotIn("MutationObserver", BUILDER_SOURCE)
        self.assertIn("fields: {", BUILDER_SOURCE)
        self.assertIn("pass2Field: miniMaxPass2PromptField,", BUILDER_SOURCE)

    def test_the_window_can_be_moved_resized_and_remembered(self):
        self.assertIn("resize:both;", BUILDER_SOURCE)
        self.assertIn("new ResizeObserver(", BUILDER_SOURCE)
        self.assertIn("header.addEventListener(\"pointerdown\"", BUILDER_SOURCE)
        for key in ("llm_popout_open: Boolean(state.llmPopoutOpen),", "llm_popout_width: state.llmPopoutWidth,",
                    "llm_popout_height: state.llmPopoutHeight,", "llm_popout_x: state.llmPopoutX,", "llm_popout_y: state.llmPopoutY,",
                    "right_panel_collapsed: Boolean(state.rightPanelCollapsed),"):
            self.assertIn(key, BUILDER_SOURCE)
        self.assertIn("right_collapsed: Boolean(state.rightPanelCollapsed),", BUILDER_SOURCE)
        self.assertIn("llm_popout_x: Number.isFinite(state.llmPopoutX)", BUILDER_SOURCE)

    def test_the_checkbox_is_always_clickable_and_the_window_needs_no_other_module(self):
        self.assertNotIn("toggle.input.disabled", BUILDER_SOURCE)
        self.assertNotIn("miniMaxSubTabs.detach", BUILDER_SOURCE)
        self.assertNotIn("reasonNote", BUILDER_SOURCE)
        self.assertIn('normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"', BUILDER_SOURCE)

    def test_the_pop_out_is_created_after_the_builder_state_exists(self):
        # Creating it first threw "Cannot access 'state' before initialization" and the Builder never opened.
        self.assertGreater(BUILDER_SOURCE.index("const llmPopout = createLlmPopout({"), BUILDER_SOURCE.index("const state = {"))

    def test_the_layout_sets_the_right_panel_columns_so_a_stale_inspector_module_cannot_misplace_it(self):
        source = BUILDER_SOURCE.replace("\r\n", "\n")
        self.assertIn('rightResizeHandle.style.gridColumn = "4";\n    inspector.style.gridColumn = "5";', source)

    def test_the_side_tabs_start_with_a_label_and_a_spot_in_case_an_older_layout_module_is_cached(self):
        self.assertIn('leftPanelToggle.textContent = "\u25C0";', BUILDER_SOURCE)
        self.assertIn('leftPanelToggle.style.left = "267px";', BUILDER_SOURCE)
        self.assertIn('rightPanelToggle.textContent = "\u25B6";', BUILDER_SOURCE)
        self.assertIn('rightPanelToggle.style.right = "360px";', BUILDER_SOURCE)


if __name__ == "__main__":
    unittest.main()
