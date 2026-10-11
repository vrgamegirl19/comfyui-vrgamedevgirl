"""MiniMax H3 scene inputs (references, continuity, last frame) resolved server-side like the browser."""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

si = importlib.import_module(f"{ROOT.name}.minimax.scene_inputs")


def _builder():
    return {
        "use_subject_reference": True,
        "subject_count": 2,
        "extras_enabled": True,
        "subjects": [
            {"id": "s1", "name": "Ava", "image": {"path": "C:/refs/ava.png"}},
            {"id": "s2", "name": "Ben", "image": {"path": "C:/refs/ben.png"}},
            {"id": "s1b", "name": "Ava side", "extra_reference_for": "s1", "image": {"path": "C:/refs/ava_side.png"}},
        ],
        "subject_scene_map": {"scene-1": ["s1"], "scene-2": ["s2"]},
        "extra_subjects": [
            {"id": "x1", "title": "Crowd", "description": "dancers", "send_to_minimax": True, "image": {"path": "C:/refs/crowd.png"}},
        ],
        "extra_scene_map": {"scene-1": [{"extra_id": "x1", "interaction": "background"}]},
        "locations": [{"id": "l1", "name": "Rooftop", "image": {"path": "C:/refs/roof.png"}}],
        "scene_map": {"scene-1": "l1", "scene-2": "l1"},
        "ingredients_sheets": [{"id": "i1", "name": "Props", "image": {"path": "C:/refs/props.png"}}],
        "ingredients_scene_map": {"scene-2": "i1"},
    }


def _session():
    return {"flux_reference_builder": _builder()}


def _segment(scene_id="scene-1", **extra):
    return {"id": scene_id, **extra}


def _never_extract(*_args):
    raise AssertionError("final frame extraction should not run")


class ReferenceImageTests(unittest.TestCase):
    def test_mapped_references_follow_scene_maps_in_catalog_order(self):
        paths = si.render_reference_image_paths(_session(), _segment("scene-1"), "reference_to_video", 0)
        # Ava's extra view is expanded with her (expandSubjectReferencesForRender), then the forced extra and location.
        self.assertEqual(paths, ["C:/refs/ava.png", "C:/refs/ava_side.png", "C:/refs/crowd.png", "C:/refs/roof.png"])

    def test_second_scene_gets_its_subject_location_and_ingredients(self):
        paths = si.render_reference_image_paths(_session(), _segment("scene-2"), "reference_to_video", 1)
        self.assertEqual(paths, ["C:/refs/ben.png", "C:/refs/roof.png", "C:/refs/props.png"])

    def test_explicit_reference_keys_replace_mapping_but_keep_forced_extras(self):
        segment = _segment("scene-1", minimax_h3_reference_keys=["location:l1"])
        paths = si.render_reference_image_paths(_session(), segment, "reference_to_video", 0)
        self.assertEqual(paths, ["C:/refs/roof.png", "C:/refs/crowd.png"])

    def test_scene_start_frame_comes_first_when_enabled(self):
        segment = _segment("scene-1", minimax_h3_use_scene_image_as_start_frame=True, approved_image_path="C:/img/s1.png")
        paths = si.render_reference_image_paths(_session(), segment, "image_reference_to_video", 0)
        self.assertEqual(paths[0], "C:/img/s1.png")
        self.assertIn("C:/refs/ava.png", paths)

    def test_a_subject_added_through_the_api_is_used_without_the_flag(self):
        """The API never sets use_subject_reference; the UI derives it from a subject with an image."""
        session = {"flux_reference_builder": {"subjects": [{"id": "darrel", "name": "Darrel", "image": {"path": "C:/pics/darrel.png"}}]}}
        paths = si.render_reference_image_paths(session, _segment("scene-1"), "reference_to_video", 0)
        self.assertEqual(paths, ["C:/pics/darrel.png"])

    def test_placeholder_subjects_without_content_are_not_used(self):
        session = {"flux_reference_builder": {"subject_count": 1, "subjects": [{"id": "s", "name": "Character 1", "image": {}}]}}
        self.assertEqual(si.render_reference_image_paths(session, _segment("scene-1"), "reference_to_video", 0), [])

    def test_reference_mode_ignores_legacy_scene_start_flags_and_choices(self):
        segment = _segment("scene-1", minimax_h3_scene_image_use="exact_start_frame", approved_image_path="C:/img/s1.png")
        for legacy_flag in (False, True):
            segment["minimax_h3_use_scene_image_as_start_frame"] = legacy_flag
            paths = si.render_reference_image_paths(_session(), segment, "reference_to_video", 0)
            self.assertEqual(paths[0], "C:/refs/ava.png")
            self.assertNotIn("C:/img/s1.png", paths)

    def test_image_to_video_uses_only_the_selected_scene_image(self):
        segment = _segment("scene-1", image_history=["C:/img/a.png", "C:/img/b.png"], image_history_index=1)
        self.assertEqual(si.render_reference_image_paths(_session(), segment, "image_to_video", 0), ["C:/img/b.png"])

    def test_text_to_video_has_no_images(self):
        self.assertEqual(si.render_reference_image_paths(_session(), _segment(), "text_to_video", 0), [])

    def test_scene_number_keys_work_when_the_map_is_numeric(self):
        session = _session()
        session["flux_reference_builder"]["scene_map"] = {"1": "l1"}
        session["flux_reference_builder"]["use_subject_reference"] = False
        paths = si.render_reference_image_paths(session, {"id": "other"}, "reference_to_video", 0)
        self.assertIn("C:/refs/roof.png", paths)

    def test_caps_at_nine_and_dedupes_by_normalized_path(self):
        configured = [f"C:/x/{i}.png" for i in range(12)] + ["c:\\x\\0.png"]
        paths = si.render_reference_image_paths({}, _segment(), "reference_to_video", 0, configured)
        self.assertEqual(len(paths), 9)
        self.assertEqual(len({si.media_path_key(p) for p in paths}), 9)


class SceneInputsTests(unittest.TestCase):
    def _resolve(self, segment, mode="reference_to_video", **kwargs):
        defaults = dict(
            continuity_mode="off", previous_segment=None, project_folder="C:/proj", scene_number=2,
            extract_final_frame=_never_extract,
        )
        defaults.update(kwargs)
        return si.resolve_scene_inputs(_session(), segment, mode, 1, **defaults)

    def test_continuity_off_makes_no_extraction(self):
        result = self._resolve(_segment("scene-2"))
        self.assertEqual(result["continuity_mode"], "off")

    def test_the_retired_previous_final_frame_modes_are_off_and_never_extract_a_frame(self):
        for name in ("spatial", "spatial_reference", "exact", "exact_start_frame"):
            with self.subTest(name=name):
                result = self._resolve(
                    _segment("scene-2"), continuity_mode=name, previous_segment={"video_path": "v.mp4"},
                )
                self.assertEqual(result["continuity_mode"], "off")
                self.assertEqual(result["continuity_image_number"], 0)

    def test_masked_continuity_is_unavailable_in_image_modes(self):
        for mode in ("text_to_video", "reference_to_video", "video_to_video"):
            with self.subTest(mode=mode):
                self.assertTrue(si.continuity_allowed_for_mode("latent_continuation_masked", mode))
        self.assertFalse(si.continuity_allowed_for_mode("latent_continuation_masked", "image_reference_to_video"))
        self.assertFalse(si.continuity_allowed_for_mode("latent_continuation_masked", "image_to_video"))
        for render_pass in ("single", "two_pass"):
            result = self._resolve(
                _segment("scene-2", approved_image_path="C:/x/a.png"), mode="image_to_video", continuity_mode="latent_masked",
                previous_segment={"video_path": "v.mp4"}, configured_image_paths=["C:/x/a.png"],
                render_pass=render_pass,
            )
            self.assertEqual(result["continuity_mode"], "off")
            self.assertEqual(result["continuity_image_number"], 0)
        # 2 Pass Advanced (three_pass) is not supported, Single and 2 Pass are
        self.assertTrue(si.continuity_allowed_for_mode("latent_continuation_masked", "reference_to_video", "two_pass"))
        self.assertFalse(si.continuity_allowed_for_mode("latent_continuation_masked", "reference_to_video", "three_pass"))
        result = self._resolve(
            _segment("scene-2"), continuity_mode="latent_masked", previous_segment={"video_path": "v.mp4"}, render_pass="three_pass",
        )
        self.assertEqual(result["continuity_mode"], "off")
        for mode in ("text_to_video", "image_to_video", "image_reference_to_video", "reference_to_video", "video_to_video"):
            self.assertTrue(si.continuity_allowed_for_mode("off", mode))
        result = self._resolve(
            _segment("scene-2"), mode="text_to_video", continuity_mode="latent_masked", previous_segment={"video_path": "v.mp4"},
        )
        self.assertEqual(result["continuity_mode"], "latent_continuation_masked")
        result = self._resolve(
            _segment("scene-2"), mode="image_reference_to_video", continuity_mode="latent_masked",
            previous_segment={"video_path": "v.mp4"}, configured_image_paths=["C:/x/a.png"],
        )
        self.assertEqual(result["continuity_mode"], "off")

    def test_the_retired_latent_modes_continue_masked_without_extracting_a_frame(self):
        for name in ("latent", "latent_continuation", "latent_exact", "latent_continuation_exact_frame"):
            with self.subTest(name=name):
                result = self._resolve(_segment("scene-2"), continuity_mode=name, previous_segment={"video_path": "v.mp4"})
                self.assertEqual(result["continuity_mode"], "latent_continuation_masked")
                self.assertEqual(result["continuity_image_number"], 0)

    def test_latent_continuation_rejects_scene_one(self):
        with self.assertRaises(ValueError):
            self._resolve(_segment("scene-1"), continuity_mode="latent", previous_segment={}, scene_number=1)

    def test_video_to_video_requires_a_reference_video(self):
        with self.assertRaises(ValueError):
            self._resolve(_segment("scene-2"), mode="video_to_video")
        result = self._resolve(
            _segment("scene-2", minimax_h3_video_references=[{"path": "C:/ref.mp4"}]), mode="video_to_video",
        )
        self.assertEqual(result["video_references"], [{"path": "C:/ref.mp4"}])

    def test_reference_mode_needs_an_image_and_i2v_needs_a_scene_image(self):
        for mode in ("reference_to_video", "image_to_video"):
            with self.assertRaises(ValueError):
                si.resolve_scene_inputs(
                    {}, _segment("x"), mode, 0, continuity_mode="off", previous_segment=None,
                    project_folder="C:/p", scene_number=1, extract_final_frame=_never_extract,
                )

    def test_image_to_video_last_frame_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "first.png")
            last = os.path.join(tmp, "last.png")
            Path(first).write_bytes(b"x")
            Path(last).write_bytes(b"x")
            segment = _segment("x", approved_image_path=first, first_last_frame_end_image_path=last)
            result = si.resolve_scene_inputs(
                {}, segment, "image_to_video", 0, continuity_mode="spatial", previous_segment=None,
                project_folder=tmp, scene_number=1, extract_final_frame=_never_extract,
            )
            self.assertEqual(result["image_paths"], [first])
            self.assertEqual(result["last_frame_path"], last)
            self.assertEqual(result["continuity_mode"], "off")
            self.assertEqual(result["missing_image_paths"], [])
            segment["approved_image_path"] = os.path.join(tmp, "gone.png")
            gone = si.resolve_scene_inputs(
                {}, segment, "image_to_video", 0, continuity_mode="off", previous_segment=None,
                project_folder=tmp, scene_number=1, extract_final_frame=_never_extract,
            )
            self.assertEqual(gone["missing_image_paths"], [segment["approved_image_path"]])
            segment["approved_image_path"] = first
            segment["minimax_h3_i2v_frame_mode"] = "normal"
            normal = si.resolve_scene_inputs(
                {}, segment, "image_to_video", 0, continuity_mode="off", previous_segment=None,
                project_folder=tmp, scene_number=1, extract_final_frame=_never_extract,
            )
            self.assertNotIn("last_frame_path", normal)
            segment["minimax_h3_i2v_frame_mode"] = "flf"
            segment["first_last_frame_end_image_path"] = ""
            with self.assertRaisesRegex(ValueError, "saved last-frame"):
                si.resolve_scene_inputs(
                    {}, segment, "image_to_video", 0, continuity_mode="off", previous_segment=None,
                    project_folder=tmp, scene_number=1, extract_final_frame=_never_extract,
                )


class ContinuityModeTests(unittest.TestCase):
    def test_aliases_match_the_ui(self):
        for raw, expected in (
            ("latent", "latent_continuation_masked"), ("latent-exact", "latent_continuation_masked"),
            ("latent_masked", "latent_continuation_masked"), ("spatial", "off"), ("exact", "off"),
            ("nonsense", "off"), (None, "off"),
        ):
            self.assertEqual(si.normalize_continuity_mode(raw), expected)


if __name__ == "__main__":
    unittest.main()
