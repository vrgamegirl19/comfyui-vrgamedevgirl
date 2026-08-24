import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "VRGDG_StoryboardBuilderNodes.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")


class StoryboardSceneBeatAudioLanguageTests(unittest.TestCase):
    def test_scene_beat_prompt_is_visual_only(self):
        self.assertIn("visual narrative Scene Story Beat only", SOURCE)
        self.assertIn("never copy those assignments into the story_beat", SOURCE)
        self.assertIn("Exclude all audio, lyric, vocal, singing, lip-sync", SOURCE)

    def test_scene_beat_has_output_guard_and_repair(self):
        self.assertIn("_scene_beat_has_audio_language(text)", SOURCE)
        self.assertIn('label="Storyboard Scene Beat Audio Language Repair Gemma"', SOURCE)
        self.assertIn("_strip_scene_beat_audio_language(text)", SOURCE)

    def test_example_beat_language_is_detected(self):
        pattern = re.compile(
            r"\b(?:lip[\s-]?sync(?:ing)?|sings?|singing|sang|lyrics?|lyric|"
            r"vocals?|vocalizing|vocalizes?|rapping|raps?|music|instrumental|"
            r"performing vocals?)\b",
            re.IGNORECASE,
        )
        example = (
            "At the monumental white marble arch, Singer visibly sings through "
            "the instrumental opening as her fingertips trace gray veining."
        )
        self.assertIsNotNone(pattern.search(example))


if __name__ == "__main__":
    unittest.main()
