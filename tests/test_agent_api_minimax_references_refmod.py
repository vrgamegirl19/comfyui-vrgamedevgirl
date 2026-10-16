"""GET /scenes/{sid}/minimax-references reports what a RefMod render really sends when the project uses RefMods.

Cut and Gun S13: Ayame (character RefMod, video), a Glock 17 prop card (RefMod prop/Glock17, video) and the corridor
location (RefMod background/corridor, image). The render (agent_api/orchestrator/video_orchestrator.py ->
runner/minimax_refmod.py) orders them with refmod_items_for_scene + assign_labels, reads each RefMod's kind and token
count from the file on disk (refmod_library.find_refmod) and adds <Audio 1> for input audio.
"""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

mutations = importlib.import_module(f"{ROOT.name}.agent_api.mutations")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")
library = importlib.import_module(f"{ROOT.name}.minimax.refmod_library")
from test_agent_api_references_llm import Base  # noqa: E402

# Library entries as find_refmod returns them (tokens = frames x (h/2) x (w/2)); 1458 + 486 + 402 = 2346.
LIBRARY = {
    "identity/Ayame_Realistic": {"name": "identity/Ayame_Realistic", "kind": "video", "frames": 6, "canvas": [432, 432], "tokens": 1458},
    "prop/Glock17": {"name": "prop/Glock17", "kind": "video", "frames": 2, "canvas": [432, 432], "tokens": 486},
    "background/corridor": {"name": "background/corridor", "kind": "image", "frames": 1, "canvas": [864, 480], "tokens": 402},
}


def _card(card_id, name, reference_type, mod, kind, tokens):
    return {"id": card_id, "name": name, "reference_type": reference_type, "source": "refmod", "description": f"{name} look",
            "image": {"path": ""}, "refmod": {"name": mod, "kind": kind, "tokens": tokens, "strength": 1.0}}


class RefmodMinimaxReferencesTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        ayame = _card("ayame", "Ayame", "character", "identity/Ayame_Realistic", "video", 999)  # stale cached tokens
        glock = _card("subj_1791577825883_4299", "Glock 17", "prop", "prop/Glock17", "video", 486)
        corridor = _card("corridor", "Corridor", "environment", "background/corridor", "image", 402)
        session["flux_reference_builder"] = {
            "use_subject_reference": True, "use_location_references": True, "subject_count": 2,
            "subjects": [ayame, glock], "locations": [corridor],
            "subject_scene_map": {"seg_2": "ayame,subj_1791577825883_4299"}, "scene_map": {"seg_2": "corridor"},
        }
        session["minimax_h3_settings"] = {"pipeline": "refmod", "video_mode": "reference_to_video", "audio_mode": "input_audio"}
        self.write_session(session)
        patcher = patch.object(library, "find_refmod", lambda name: dict(LIBRARY[name]) if name in LIBRARY else None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_refmod_view_matches_the_render(self):
        view = mutations.get_scene_minimax_references("Song", "3")
        self.assertEqual(view["pipeline"], "refmod")
        got = [(i["label"], i["card_id"], i["category"], i["source"]["refmod"]) for i in view["items"]]
        self.assertEqual(got, [
            ("<Video 1>", "ayame", "character", "identity/Ayame_Realistic"),
            ("<Video 2>", "subj_1791577825883_4299", "object", "prop/Glock17"),
            ("<Picture 1>", "corridor", "background", "background/corridor"),
            ("<Audio 1>", "", "audio", ""),
        ])
        prop = view["items"][1]
        self.assertEqual(prop["reference_type"], "prop")
        self.assertEqual(prop["source"]["type"], "refmod")
        self.assertTrue(prop["source"]["found"])
        self.assertEqual([i["tokens"] for i in view["items"]], [1458, 486, 402, 0])  # library, not the card cache
        self.assertEqual(view["tokens"]["total"], 2346)
        self.assertFalse(view["tokens"]["over_limit"])
        self.assertEqual(view["labels"], ["<Video 1>", "<Video 2>", "<Picture 1>", "<Audio 1>"])
        self.assertFalse(view["editable"])

    def test_built_in_audio_has_no_audio_label(self):
        session = self.read_session()
        session["minimax_h3_settings"]["audio_mode"] = "built_in_audio"
        self.write_session(session)
        view = mutations.get_scene_minimax_references("Song", "seg_2")
        self.assertEqual(view["labels"], ["<Video 1>", "<Video 2>", "<Picture 1>"])

    def test_a_missing_refmod_file_is_flagged(self):
        with patch.object(library, "find_refmod", lambda name: None if name == "prop/Glock17" else dict(LIBRARY[name])):
            view = mutations.get_scene_minimax_references("Song", "3")
        prop = view["items"][1]
        self.assertFalse(prop["source"]["found"])
        self.assertEqual(prop["tokens"], 486)  # falls back to the card's saved count
        self.assertTrue(view["warnings"])

    def test_standard_view_is_unchanged_apart_from_pipeline(self):
        session = self.read_session()
        session["minimax_h3_settings"]["pipeline"] = "standard"
        self.write_session(session)
        view = mutations.get_scene_minimax_references("Song", "3")
        self.assertEqual(view["pipeline"], "standard")
        for key in ("available", "selected", "automatic_keys", "limits", "custom", "video_mode"):
            self.assertIn(key, view)
        self.assertNotIn("items", view)

    def test_put_is_refused_in_refmod_mode_and_nothing_is_saved(self):
        before = self.read_session()
        for kwargs in ({"keys": ["subject:ayame"]}, {"automatic": True}):
            with self.assertRaises(errors.NotSupportedForModeError) as ctx:
                mutations.set_scene_minimax_references("Song", "3", **kwargs)
            self.assertEqual(ctx.exception.details["mode"], "refmod")
            self.assertIn("scene-mapping", str(ctx.exception))
        self.assertEqual(self.read_session(), before)


if __name__ == "__main__":
    unittest.main()
