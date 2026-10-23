"""Story Mode telling: the book follows the song map and is told as a music video, spoken story or story only."""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

story_book = importlib.import_module(f"{ROOT.name}.storyboard.story_book")
story_plan = importlib.import_module(f"{ROOT.name}.storyboard.story_plan")
builder_runner = importlib.import_module(f"{ROOT.name}.llm.builder_runner")

SCENES = [
    {"id": "s1", "start": 0, "end": 6, "lyrics": "[instrumental]", "lyric_section": "Intro"},
    {"id": "s2", "start": 6, "end": 14, "lyrics": "I walk alone", "lyric_section": "Verse 1"},
    {"id": "s3", "start": 14, "end": 22, "lyrics": "Nobody waits", "lyric_section": "Verse 1"},
    {"id": "s4", "start": 22, "end": 30, "lyrics": "I'm gone by morning", "lyric_section": "Chorus"},
    {"id": "s5", "start": 30, "end": 34, "lyrics": ""},
    {"id": "s6", "start": 34, "end": 40, "lyrics": "Gone"},
]
BOOK = "Title: T\nPremise: P\n\nChapter 1: Night\nSetting: Street\n\nBrandon walks."


class Fake:
    def __init__(self):
        self.prompts = []

    def __call__(self, payload, instruction, **kwargs):
        self.prompts.append(instruction)
        if kwargs.get("label") == "Story Mode Cast Check":
            return '{"characters": []}', {}
        return BOOK, {}


def draft(style=None):
    layer = {"story_flow_version": 2}
    if style:
        layer["story_video_style"] = style
    fake = Fake()
    with patch.object(builder_runner, "_run_builder_text_llm", fake):
        story_book.draft_story_book({
            "story_layer": layer, "scenes": SCENES, "story_idea": "A player",
            "reference_builder": {"subjects": [{"id": "b", "name": "Brandon", "reference_type": "character"}]},
        })
    return fake.prompts[0]


class StoryTellingTests(unittest.TestCase):
    def test_song_map_groups_sections_in_order(self):
        self.assertEqual(story_book.song_map(SCENES).splitlines(), [
            "0:00-0:06 Intro (no vocals)",
            "0:06-0:22 Verse 1: I walk alone / Nobody waits",
            "0:22-0:30 Chorus: I'm gone by morning",
            "0:30-0:34 Instrumental (no vocals)",
            "0:34-0:40 Vocals: Gone",
        ])

    def test_music_video_is_the_default_and_the_lead_performs(self):
        prompt = draft()
        self.assertIn("This is the story of a music video. Brandon performs the song inside the story", prompt)
        self.assertIn("Follow the song map from start to finish", prompt)
        self.assertIn("0:22-0:30 Chorus: I'm gone by morning", prompt)

    def test_spoken_and_story_only(self):
        self.assertIn("The characters talk", draft("spoken"))
        self.assertIn("no one singing or speaking on camera", draft("story_only"))

    def test_scene_performance_follows_the_style(self):
        fit = story_plan.style_performance
        self.assertEqual(fit("singing", "music_video"), "singing")
        self.assertEqual(fit("singing", "spoken"), "speaking")
        self.assertEqual(fit("speaking", "story_only"), "no_lip_sync")
        self.assertEqual(fit("singing", "story_only"), "no_lip_sync")
        self.assertEqual(fit("instrumental", "story_only"), "instrumental")


if __name__ == "__main__":
    unittest.main()
