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

    def test_scene_image_use_choice_enables_the_start_frame_like_the_flag(self):
        segment = _segment("scene-1", minimax_h3_scene_image_use="exact_start_frame", approved_image_path="C:/img/s1.png")
        paths = si.render_reference_image_paths(_session(), segment, "reference_to_video", 0)
        self.assertEqual(paths[0], "C:/img/s1.png")

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
        self.assertEqual(result["latent_exact_frame_path"], "")

    def test_spatial_continuity_appends_the_previous_final_frame(self):
        calls = []

        def extract(folder, video, number):
            calls.append((folder, video, number))
            return "C:/proj/frames/prev_last.png"

        result = self._resolve(
            _segment("scene-2"), continuity_mode="spatial",
            previous_segment={"video_path": "C:/proj/rendered_scene_videos/video_0001.mp4"}, extract_final_frame=extract,
        )
        self.assertEqual(calls, [("C:/proj", "C:/proj/rendered_scene_videos/video_0001.mp4", 2)])
        self.assertEqual(result["image_paths"][-1], "C:/proj/frames/prev_last.png")
        self.assertEqual(result["continuity_image_number"], len(result["image_paths"]))
        self.assertEqual(result["continuity_mode"], "spatial_reference")

    def test_continuity_without_previous_video_renders_without_a_frame(self):
        result = self._resolve(_segment("scene-2"), continuity_mode="exact_start_frame", previous_segment={})
        self.assertEqual(result["continuity_image_number"], 0)

    def test_ninth_slot_error_when_full(self):
        configured = [f"C:/x/{i}.png" for i in range(9)]
        with self.assertRaises(ValueError):
            si.resolve_scene_inputs(
                _session(), _segment("scene-2"), "reference_to_video", 1,
                continuity_mode="spatial_reference", previous_segment={"video_path": "v.mp4"},
                project_folder="C:/proj", scene_number=2,
                extract_final_frame=lambda *_: "C:/proj/new.png", configured_image_paths=configured,
            )

    def test_exact_start_cannot_combine_with_scene_image_start_frame(self):
        with self.assertRaises(ValueError):
            self._resolve(
                _segment("scene-2", minimax_h3_use_scene_image_as_start_frame=True),
                continuity_mode="exact_start_frame", previous_segment={"video_path": "v.mp4"},
            )

    def test_latent_exact_frame_is_a_separate_field_not_an_image(self):
        result = self._resolve(
            _segment("scene-2"), continuity_mode="latent_exact",
            previous_segment={"video_path": "C:/proj/v1.mp4"}, extract_final_frame=lambda *_: "C:/proj/exact.png",
        )
        self.assertEqual(result["latent_exact_frame_path"], "C:/proj/exact.png")
        self.assertNotIn("C:/proj/exact.png", result["image_paths"])

    def test_latent_continuation_rejects_scene_one(self):
        with self.assertRaises(ValueError):
            self._resolve(_segment("scene-1"), continuity_mode="latent", previous_segment={}, scene_number=1)

    def test_latent_exact_needs_a_rendered_previous_video(self):
        with self.assertRaises(ValueError):
            self._resolve(_segment("scene-2"), continuity_mode="latent_exact", previous_segment={})

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


class ContinuityModeTests(unittest.TestCase):
    def test_aliases_match_the_ui(self):
        for raw, expected in (
            ("latent", "latent_continuation"), ("latent-exact", "latent_continuation_exact_frame"),
            ("spatial", "spatial_reference"), ("exact", "exact_start_frame"), ("nonsense", "off"), (None, "off"),
        ):
            self.assertEqual(si.normalize_continuity_mode(raw), expected)


if __name__ == "__main__":
    unittest.main()
