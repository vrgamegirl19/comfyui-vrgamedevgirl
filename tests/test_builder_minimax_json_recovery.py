import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)
BACKEND_SOURCE = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(
    encoding="utf-8"
)


def _function_source(name, next_name):
    start = BUILDER_SOURCE.index(f"  function {name}")
    end = BUILDER_SOURCE.index(f"  function {next_name}", start)
    return BUILDER_SOURCE[start:end]


CONTEXT_SOURCE = _function_source(
    "miniMaxH3CreativePromptContextForSegment", "miniMaxH3Timecode"
)
RECOVERY_SOURCE = _function_source(
    "extractMiniMaxH3CompleteShotDescriptions",
    "parseMiniMaxH3ShotDescriptionPayload",
)
FALLBACK_SOURCE = _function_source(
    "miniMaxH3FallbackShotDescription",
    "extractMiniMaxH3CompleteShotDescriptions",
)
PARSER_SOURCE = _function_source(
    "parseMiniMaxH3ShotDescriptionPayload", "normalizeMiniMaxH3DialogueTags"
)
RUNNER_SOURCE = BUILDER_SOURCE[
    BUILDER_SOURCE.index("  async function runMiniMaxH3PromptGeneration") :
    BUILDER_SOURCE.index("  async function createMiniMaxH3PromptWithLLM")
]
BATCH_SOURCE = BUILDER_SOURCE[
    BUILDER_SOURCE.index("    const runWizardStoryboardGemmaAll = async () => {") :
    BUILDER_SOURCE.index("    const wizardSnapshot = () => {", BUILDER_SOURCE.index("    const runWizardStoryboardGemmaAll = async () => {"))
]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for JavaScript behavior tests")
class BuilderMiniMaxJsonRecoveryTests(unittest.TestCase):
    def _run_parser(self, raw_prompt, shot_count=2, allow_full_fallback=False):
        script = f"""
const notices = [];
function miniMaxH3ModeForSegment() {{ return "text_to_video"; }}
function miniMaxH3OfficialShotPlan() {{ return Array.from({{ length: {shot_count} }}, () => ({{}})); }}
function miniMaxH3FallbackShotDescription(_segment, index) {{ return `fallback-${{index + 1}}`; }}
function normalizeMiniMaxH3ShotDescription(value) {{ return String(value || "").trim(); }}
function toast(message, isError) {{ notices.push({{ message, isError }}); }}
{RECOVERY_SOURCE}
{PARSER_SOURCE}
try {{
  const result = parseMiniMaxH3ShotDescriptionPayload({json.dumps(raw_prompt)}, {{}}, null, "text_to_video", {{ allowFullFallback: {json.dumps(allow_full_fallback)} }});
  process.stdout.write(JSON.stringify({{ ok: true, result, notices }}));
}} catch (error) {{
  process.stdout.write(JSON.stringify({{ ok: false, error: String(error.message || error), notices }}));
}}
"""
        completed = subprocess.run(
            [shutil.which("node"), "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_broken_json_recovers_only_complete_descriptions_and_fills_the_rest(self):
        first = 'A close-up shows her say "hello" beside C:\\stage.'
        broken = (
            '{"shots":[{"description":'
            + json.dumps(first)
            + '},{"description":"An unfinished second shot that starts lip-syncing\n\n'
            + "Audio 1: use unchanged as the primary and only audio track.\n\n"
            + "Continuity: preserve the same character."
        )
        payload = self._run_parser(broken)

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["result"], [first, "fallback-2"])
        self.assertEqual(len(payload["notices"]), 1)
        self.assertIn("recovered 1/2", payload["notices"][0]["message"])
        self.assertIn("filled 1 missing shot", payload["notices"][0]["message"])

    def test_unrecoverable_json_still_fails_closed(self):
        broken = (
            '{"shots":[{"description":"unfinished\n\n'
            "Audio 1: use unchanged.\nContinuity: preserve identity."
        )
        payload = self._run_parser(broken)

        self.assertFalse(payload["ok"])
        self.assertIn("did not return valid JSON", payload["error"])

    def test_second_failed_attempt_discards_unfinished_text_and_uses_fallback(self):
        broken = (
            '{"shots":[{"description":"A close-up reaches precise lip-sync to\n\n'
            "Audio 1: use unchanged.\nContinuity: preserve identity."
        )
        payload = self._run_parser(broken, allow_full_fallback=True)

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["result"], ["fallback-1", "fallback-2"])
        self.assertIn("invalid JSON twice", payload["notices"][0]["message"])
        self.assertNotIn("precise lip-sync to", " ".join(payload["result"]))

    def test_unescaped_quote_is_not_mistaken_for_a_complete_shot_object(self):
        broken = '{"shots":[{"description":"She says "hello" without escaped quotes'
        payload = self._run_parser(broken)

        self.assertFalse(payload["ok"])
        self.assertIn("did not return valid JSON", payload["error"])

    def test_valid_json_keeps_strict_shot_count_validation(self):
        payload = self._run_parser(
            json.dumps({"shots": [{"description": "Only one complete shot."}]})
        )

        self.assertFalse(payload["ok"])
        self.assertIn("builder expected 2", payload["error"])

    def test_generation_retries_an_unusable_payload_once_with_stricter_settings(self):
        script = f"""
const requests = [];
const notices = [];
const miniMaxGemmaModelSelect = {{ value: "vision.gguf" }};
const miniMaxTextGemmaModelSelect = {{ value: "text.gguf" }};
const miniMaxMmprojSelect = {{ value: "mmproj.gguf" }};
const GEMMA_VIDEO_PROMPT_TIMEOUT_MS = 600000;
function miniMaxH3PromptVisionImagesForRunner() {{ return []; }}
function segmentUsesNoLipSyncPerformance() {{ return false; }}
function isInstrumentalLyricText() {{ return false; }}
function flattenLyricForPrompt(value) {{ return String(value || ""); }}
function textGemmaRunnerPayload() {{ return {{ runner: "builtin" }}; }}
function activeProjectFolderForSave() {{ return "project"; }}
function miniMaxH3InstructionKey() {{ return "minimax_h3_t2v"; }}
function miniMaxH3CreativePromptContextForSegment() {{ return "context"; }}
function miniMaxH3SceneImageIsPromptInspiration() {{ return false; }}
function effectiveVideoPerformanceModeForSegment() {{ return "singing"; }}
function miniMaxH3SettingsForSegment() {{ return {{ audio_mode: "audio_1" }}; }}
function miniMaxDialogueAssignmentsForSegment() {{ return []; }}
function miniMaxH3ModeLabel() {{ return "Text to Video"; }}
function toast(message) {{ notices.push(message); }}
async function postJson(_url, payload) {{
  requests.push(payload);
  return requests.length === 1 ? {{ prompt: "broken" }} : {{ prompt: "valid" }};
}}
function assembleMiniMaxH3PromptFromCreative(_segment, _mode, prompt) {{
  if (prompt === "broken") throw new Error("The LLM did not return valid JSON shot descriptions.");
  return `assembled:${{prompt}}`;
}}
{RUNNER_SOURCE}
(async () => {{
  try {{
    const result = await runMiniMaxH3PromptGeneration({{ id: "scene-1", lyric_text: "line", lyric_singers: [] }}, "text_to_video", {{ userNotes: "source contract" }});
    process.stdout.write(JSON.stringify({{ ok: true, result, requests, notices }}));
  }} catch (error) {{
    process.stdout.write(JSON.stringify({{ ok: false, error: String(error.message || error), requests, notices }}));
  }}
}})();
"""
        completed = subprocess.run(
            [shutil.which("node"), "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["result"]["prompt"], "assembled:valid")
        self.assertEqual(len(payload["requests"]), 2)
        self.assertEqual(payload["requests"][0]["temperature"], 0.45)
        self.assertEqual(payload["requests"][1]["temperature"], 0.15)
        self.assertTrue(payload["requests"][0]["structured_shot_descriptions"])
        self.assertIn("source contract", payload["requests"][1]["user_notes"])
        self.assertIn("JSON RETRY:", payload["requests"][1]["user_notes"])
        self.assertEqual(len(payload["notices"]), 1)


class BuilderMiniMaxJsonContextTests(unittest.TestCase):
    def test_creative_context_does_not_feed_builder_owned_fixed_sections_to_gemma(self):
        self.assertNotIn('segment?.audio_direction', CONTEXT_SOURCE)
        self.assertNotIn('segment?.continuity', CONTEXT_SOURCE)
        self.assertIn("creativeStoryboardContext", CONTEXT_SOURCE)
        self.assertIn("Exact manual audio", CONTEXT_SOURCE)
        self.assertIn("Exact manual continuity requirements", CONTEXT_SOURCE)
        self.assertIn("stop immediately after the closing JSON brace", CONTEXT_SOURCE)
        self.assertIn("Builder adds Audio and Continuity separately", CONTEXT_SOURCE)

    def test_parser_uses_strict_complete_value_recovery_and_existing_fallback(self):
        self.assertIn("extractMiniMaxH3CompleteShotDescriptions(text)", PARSER_SOURCE)
        self.assertIn("miniMaxH3FallbackShotDescription(segment, index, mode)", PARSER_SOURCE)
        self.assertIn("recoveredDescriptions.length > shotPlan.length", PARSER_SOURCE)

    def test_full_fallback_preserves_music_video_lyric_and_audio_sync(self):
        self.assertIn("flattenLyricForPrompt(segment?.lyric_text)", FALLBACK_SOURCE)
        self.assertIn('"precisely lip-sync"', FALLBACK_SOURCE)
        self.assertIn('"precisely lip-syncs"', FALLBACK_SOURCE)
        self.assertIn("<d>[English]", FALLBACK_SOURCE)

    def test_generation_retries_only_structured_output_failures_once(self):
        self.assertIn("JSON RETRY:", RUNNER_SOURCE)
        self.assertIn("requestGeneratedPrompt(true)", RUNNER_SOURCE)
        self.assertIn("options.retryInvalidJson === false", RUNNER_SOURCE)
        self.assertIn("structuredOutputFailure", RUNNER_SOURCE)
        self.assertIn("assembleGeneratedPrompt(data, true)", RUNNER_SOURCE)

    def test_backend_does_not_apply_legacy_final_prompt_formatter_to_shot_json(self):
        generator = BACKEND_SOURCE[
            BACKEND_SOURCE.index("def _generate_builder_t2v_prompt") :
            BACKEND_SOURCE.index("def _video_prompt_enhancement_instructions")
        ]
        self.assertIn('payload.get("structured_shot_descriptions")', generator)
        self.assertIn(
            "if is_minimax_h3_prompt and not structured_shot_descriptions:",
            generator,
        )

    def test_batch_saves_completed_scenes_before_reporting_a_later_failure(self):
        self.assertIn('autoSaveSessionQuiet("wizard storyboard llm partial")', BATCH_SOURCE)
        self.assertIn("Completed scenes were saved", BATCH_SOURCE)


if __name__ == "__main__":
    unittest.main()
