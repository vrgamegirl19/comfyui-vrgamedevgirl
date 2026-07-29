import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIZARD_SOURCE = ROOT / "web" / "VRGDG_MusicVideoWizardUI.js"
BUILDER_SOURCE = ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js"


def run_wizard_export(export_name, payload):
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("Node.js is required for the Wizard lyric tests.")

    with tempfile.TemporaryDirectory() as folder:
        module_path = Path(folder) / "VRGDG_MusicVideoWizardUI.mjs"
        module_path.write_text(WIZARD_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
        script = """
import { pathToFileURL } from "url";
const modulePath = process.argv[1];
const payload = JSON.parse(process.argv[2]);
const module = await import(pathToFileURL(modulePath).href);
const result = module[payload.exportName](...payload.args);
console.log(JSON.stringify(result));
"""
        completed = subprocess.run(
            [
                node,
                "--input-type=module",
                "-e",
                script,
                str(module_path),
                json.dumps({"exportName": export_name, "args": payload}),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    return json.loads(completed.stdout)


class WizardReferenceLyricSanitizerTests(unittest.TestCase):
    def sanitize(self, text):
        return run_wizard_export("sanitizeWizardReferenceLyrics", [text])

    def test_parenthesized_lyric_keeps_words_and_removes_parentheses(self):
        self.assertEqual(
            self.sanitize("[Intro]\n(Acoustic guitar riff playing)\nRemember our front yard?"),
            "[Intro]\nAcoustic guitar riff playing\nRemember our front yard?",
        )

    def test_inline_parentheses_are_removed_without_removing_lyrics(self):
        self.assertEqual(
            self.sanitize("We go home (echoes: home, home)\nStill singing"),
            "We go home echoes: home, home\nStill singing",
        )

    def test_square_bracket_section_headers_are_preserved(self):
        self.assertEqual(self.sanitize("[Verse (soft)]\nA lyric"), "[Verse (soft)]\nA lyric")


class WizardSharedScenePathTests(unittest.TestCase):
    def test_wizard_uses_shared_creator_without_timing_overrides(self):
        wizard_source = WIZARD_SOURCE.read_text(encoding="utf-8")
        builder_source = BUILDER_SOURCE.read_text(encoding="utf-8")

        self.assertIn("api.createScenesFromLyrics?.({", wizard_source)
        self.assertIn("referenceLyrics: wizardState.lyrics", wizard_source)
        self.assertIn("if (created === false) return;", wizard_source)
        self.assertIn("wizardStripLyricParentheses: true", wizard_source)
        self.assertNotIn("wizardRepairShortReferenceScenes", wizard_source)
        self.assertNotIn("wizardUseTranscriptionTimestamps", wizard_source)
        self.assertNotIn("wizardAttachUnalignedLyricsToDetectedScenes", wizard_source)
        self.assertNotIn("wizardEstimateUnalignedLyricTiming", wizard_source)
        self.assertIn("await createScenesFromTimestampedLyrics();", builder_source)
        self.assertIn("the lyric line wins over minimum/maximum duration", wizard_source)
        self.assertIn("If “Playing in the rain” is only 1.5 seconds long", builder_source)

    def test_parenthesized_reference_text_requires_explicit_confirmation(self):
        wizard_source = WIZARD_SOURCE.read_text(encoding="utf-8")
        builder_source = BUILDER_SOURCE.read_text(encoding="utf-8")

        self.assertIn("function hasParenthesizedReferenceText(value)", builder_source)
        self.assertIn("function confirmParenthesizedReferenceLyrics(value)", builder_source)
        self.assertIn("Text inside parentheses will be treated as lyrics.", builder_source)
        self.assertIn("remove any parenthesized lines that are NOT lyrics", builder_source)
        self.assertIn("Click OK only if all text inside parentheses is actual lyrics.", builder_source)
        self.assertIn("presetOptions && !confirmParenthesizedReferenceLyrics", builder_source)
        self.assertNotIn(
            "referenceLyrics: sanitizeWizardReferenceLyrics(wizardState.lyrics)",
            wizard_source,
        )

    def test_stale_inspector_values_cannot_overwrite_another_scene(self):
        builder_source = BUILDER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("vrgdgInspectorSegmentId", builder_source)
        self.assertIn("Ignored stale inspector values for a different scene", builder_source)


if __name__ == "__main__":
    unittest.main()
