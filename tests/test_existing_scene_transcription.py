import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js"


def mapping_source():
    source = BUILDER_SOURCE.read_text(encoding="utf-8")
    start = source.index("  function referenceVocalLines(")
    end = source.index("\n  function mergeTimestampedLyricText(", start)
    return source[start:end]


def run_mapping(payload, scenes, reference_lyrics, instrumental_text="[instrumental]"):
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("Node.js is required for existing-scene transcription tests.")
    helpers = r"""
function cleanTimestampedLyricText(text) {
  return String(text || "")
    .replace(/\[[^\]]{2,80}\]/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?])/g, "$1")
    .trim();
}
function normalizedLyricMatchText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}
"""
    script = helpers + mapping_source() + r"""
const args = JSON.parse(process.argv[1]);
try {
  const result = mapTimestampedReferenceLyricsToExistingScenes(
    args.payload,
    args.scenes,
    args.referenceLyrics,
    args.instrumentalText,
  );
  console.log(JSON.stringify({
    ok: true,
    mappings: result.map((item) => ({
      lyricText: item.lyricText,
      referenceLineCount: item.referenceLineCount,
    })),
  }));
} catch (error) {
  console.log(JSON.stringify({ok: false, error: String(error && error.message || error)}));
}
"""
    completed = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            script,
            json.dumps(
                {
                    "payload": payload,
                    "scenes": scenes,
                    "referenceLyrics": reference_lyrics,
                    "instrumentalText": instrumental_text,
                }
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


class ExistingSceneReferenceTranscriptionTests(unittest.TestCase):
    def test_words_are_assigned_to_the_scene_where_they_start_without_pipe_delimiters(self):
        reference = "\n".join(
            [
                "[Intro]",
                "I give up",
                "I don’t care",
                "You can win",
                "I’ll just lay here",
                "[Verse 1]",
                "Take your trophy",
                "Take your glory",
                "I know it feels good",
            ]
        )
        payload = {
            "segments": [
                {
                    "type": "vocal", "start": 1.2, "end": 2.06, "text": "I give up",
                    "words": [
                        {"start": 1.2, "end": 1.3, "text": "I"},
                        {"start": 1.3, "end": 1.62, "text": "give"},
                        {"start": 1.66, "end": 2.06, "text": "up"},
                    ],
                },
                {
                    "type": "vocal", "start": 2.06, "end": 4.32, "text": "I don’t care",
                    "words": [
                        {"start": 2.06, "end": 3.06, "text": "I"},
                        {"start": 3.84, "end": 4.08, "text": "don’t"},
                        {"start": 4.22, "end": 4.32, "text": "care"},
                    ],
                },
                {
                    "type": "vocal", "start": 4.32, "end": 7.22, "text": "You can win",
                    "words": [
                        {"start": 4.32, "end": 5.92, "text": "You"},
                        {"start": 5.94, "end": 6.18, "text": "can"},
                        {"start": 6.18, "end": 6.62, "text": "win"},
                    ],
                },
                {
                    "type": "vocal", "start": 8.4, "end": 16.16, "text": "I’ll just lay here",
                    "words": [
                        {"start": 8.4, "end": 8.52, "text": "I’ll"},
                        {"start": 8.52, "end": 9.02, "text": "just"},
                        {"start": 9.02, "end": 9.42, "text": "lay"},
                        {"start": 11.55, "end": 11.95, "text": "here"},
                    ],
                },
                {
                    "type": "vocal", "start": 16.16, "end": 18.6, "text": "Take your trophy",
                    "words": [
                        {"start": 16.16, "end": 16.3, "text": "Take"},
                        {"start": 17.76, "end": 17.86, "text": "your"},
                        {"start": 17.86, "end": 18.5, "text": "trophy"},
                    ],
                },
                {
                    "type": "vocal", "start": 18.6, "end": 20.6, "text": "Take your glory",
                    "words": [
                        {"start": 18.6, "end": 19.7, "text": "Take"},
                        {"start": 19.78, "end": 20.14, "text": "your"},
                        {"start": 20.14, "end": 20.6, "text": "glory"},
                    ],
                },
                {
                    "type": "vocal", "start": 20.6, "end": 23.98, "text": "I know it feels good",
                    "words": [
                        {"start": 20.6, "end": 21.98, "text": "I"},
                        {"start": 22.1, "end": 22.3, "text": "know"},
                        {"start": 22.3, "end": 22.66, "text": "it"},
                        {"start": 22.66, "end": 23.16, "text": "feels"},
                        {"start": 23.16, "end": 23.98, "text": "good"},
                    ],
                },
            ]
        }
        scenes = [
            {"start": 0.0, "end": 5.0},
            {"start": 5.0, "end": 10.0},
            {"start": 10.0, "end": 15.0},
            {"start": 15.0, "end": 20.0},
            {"start": 20.0, "end": 25.0},
        ]

        result = run_mapping(payload, scenes, reference)

        self.assertTrue(result["ok"])
        self.assertEqual(result["mappings"][0]["lyricText"], "I give up I don’t care")
        self.assertEqual(result["mappings"][1]["lyricText"], "You can win I’ll just lay")
        self.assertEqual(result["mappings"][2]["lyricText"], "here")
        self.assertEqual(result["mappings"][3]["lyricText"], "Take your trophy Take your")
        self.assertEqual(result["mappings"][4]["lyricText"], "glory I know it feels good")
        self.assertTrue(all("|" not in item["lyricText"] for item in result["mappings"]))

    def test_repeated_reference_lines_remain_distinct_occurrences(self):
        result = run_mapping(
            {
                "segments": [
                    {"type": "vocal", "start": 1.0, "end": 2.0, "text": "Colder than me"},
                    {"type": "vocal", "start": 3.0, "end": 4.0, "text": "Colder than me"},
                ]
            },
            [{"start": 0.0, "end": 2.5}, {"start": 2.5, "end": 5.0}],
            "[Outro]\nColder than me\nColder than me",
        )

        self.assertTrue(result["ok"])
        self.assertEqual([item["referenceLineCount"] for item in result["mappings"]], [1, 1])

    def test_sustained_final_words_follow_vocal_tail_across_scene_boundary(self):
        reference = "I’ll just lay here\nWhat a good guy what a perfect man"
        payload = {
            "segments": [
                {
                    "type": "vocal",
                    "start": 8.4,
                    "end": 10.56,
                    "text": "I’ll just lay here",
                    "words": [
                        {"start": 8.4, "end": 8.5, "text": "I’ll"},
                        {"start": 8.5, "end": 8.84, "text": "just"},
                        {"start": 8.84, "end": 9.38, "text": "lay"},
                        {"start": 9.38, "end": 9.96, "text": "here"},
                    ],
                },
                {
                    "type": "vocal",
                    "start": 34.7,
                    "end": 40.54,
                    "text": "What a good guy what a perfect man",
                    "words": [
                        {"start": 34.7, "end": 35.5, "text": "What"},
                        {"start": 35.54, "end": 35.78, "text": "a"},
                        {"start": 35.78, "end": 36.1, "text": "good"},
                        {"start": 36.1, "end": 36.78, "text": "guy"},
                        {"start": 38.38, "end": 38.48, "text": "what"},
                        {"start": 38.48, "end": 38.68, "text": "a"},
                        {"start": 38.68, "end": 39.7, "text": "perfect"},
                        {"start": 39.7, "end": 39.94, "text": "man"},
                    ],
                },
            ]
        }
        scenes = [
            {"start": 5.0, "end": 10.0},
            {"start": 10.0, "end": 15.0},
            {"start": 35.0, "end": 40.0},
            {"start": 40.0, "end": 45.0},
        ]

        result = run_mapping(payload, scenes, reference)

        self.assertTrue(result["ok"])
        self.assertEqual(
            [item["lyricText"] for item in result["mappings"]],
            ["I’ll just lay", "here", "What a good guy what a perfect", "man"],
        )

    def test_final_word_stays_before_boundary_when_vocal_tail_does_not_cross_it(self):
        result = run_mapping(
            {
                "segments": [
                    {
                        "type": "vocal",
                        "start": 8.4,
                        "end": 9.99,
                        "text": "Stay here",
                        "words": [
                            {"start": 8.4, "end": 9.2, "text": "Stay"},
                            {"start": 9.2, "end": 9.94, "text": "here"},
                        ],
                    }
                ]
            },
            [{"start": 5.0, "end": 10.0}, {"start": 10.0, "end": 15.0}],
            "Stay here",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(
            [item["lyricText"] for item in result["mappings"]],
            ["Stay here", "[instrumental]"],
        )

    def test_missing_timestamped_reference_line_aborts_mapping(self):
        result = run_mapping(
            {"segments": [{"type": "vocal", "start": 0.0, "end": 1.0, "text": "First line"}]},
            [{"start": 0.0, "end": 4.0}],
            "First line\nSecond line",
        )

        self.assertFalse(result["ok"])
        self.assertIn("existing timeline was left unchanged", result["error"].lower())

    def test_manual_and_beat_modes_share_the_new_existing_scene_path(self):
        source = BUILDER_SOURCE.read_text(encoding="utf-8")
        manual_start = source.index("  async function transcribeExistingScenesWithOptions(")
        manual_end = source.index("\n  async function transcribeLyricsForTimeline(", manual_start)
        manual_source = source[manual_start:manual_end]
        beat_start = source.index('      if (options.segmentMode === "beat_scenes") {')
        beat_end = source.index("\n      } else {", beat_start)
        beat_source = source[beat_start:beat_end]

        self.assertIn('/vrgdg/workflow_runner/build_timestamped_transcribe_prompt', manual_source)
        self.assertIn('segment_mode: "reference_scene_words"', manual_source)
        self.assertNotIn("build_transcribe_prompt", manual_source)
        self.assertIn("created = createSegmentsFromBeatSrt(text.srt);", beat_source)
        self.assertNotIn("text.whisper", beat_source)
        self.assertLess(
            source.index("      state.segments = created;", beat_start),
            source.index("        beatTranscriptionResult = await transcribeExistingScenesWithOptions({", beat_start),
        )


if __name__ == "__main__":
    unittest.main()

