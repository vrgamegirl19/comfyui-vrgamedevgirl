import unittest
from pathlib import Path

from builder_source import read_builder_source, read_builder_backend_source


ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE = read_builder_source()
BACKEND_SOURCE = read_builder_backend_source()
INSTRUCTION_SOURCE = (ROOT / "llm" / "prompts" / "minimax.py").read_text(encoding="utf-8")


class MiniMaxFrameContinuityPromptTests(unittest.TestCase):
    def test_setting_is_saved_and_only_enabled_for_latent_modes(self):
        self.assertIn("continuity_prompt_from_last_frame: false", UI_SOURCE)
        self.assertIn("continuity_prompt_from_last_frame: miniMaxContinuityPromptFromLastFrame.input.checked", UI_SOURCE)
        self.assertIn("isMiniMaxH3LatentContinuationMode(settings.continuity_mode)", UI_SOURCE)
        self.assertIn('sceneSlotNumber(segment) <= 1', UI_SOURCE)

    def test_previous_final_frame_is_first_vision_input(self):
        self.assertIn('const visionImages = [{ path: framePath, frame_continuity_source: true }]', UI_SOURCE)
        self.assertIn("Attached Picture 1 is the previous rendered scene's actual final frame", UI_SOURCE)
        self.assertIn("HIGHEST PRIORITY", UI_SOURCE)
        self.assertIn("frame_continuity_prompt", BACKEND_SOURCE)

    def test_subject_absent_from_last_frame_gets_physical_entrance(self):
        self.assertIn("OPENING SUBJECT VISIBILITY — IMAGE-AWARE", UI_SOURCE)
        self.assertIn("currently mapped subject absent from Attached Picture 1 begins physically offscreen", UI_SOURCE)
        self.assertIn("an entrance through a frame edge, doorway, path, foreground layer", UI_SOURCE)
        self.assertIn("the exact entrance or camera-reveal route", UI_SOURCE)
        self.assertIn("Their first visible moment occurs through that physical entrance or reveal", UI_SOURCE)

    def test_location_change_contract_requires_physical_threshold_turn(self):
        for required in (
            "PHYSICAL THRESHOLD TURN",
            "natural full-frame occlusion",
            "clear camera arc around the subject",
            "passed fully beyond the rear camera plane",
            "final viewing direction explicitly",
            "continuous camera travel",
            "stable geometry",
        ):
            self.assertIn(required, UI_SOURCE)
        self.assertNotIn("recedes toward the edge of frame and behind the camera", UI_SOURCE)

    def test_location_transition_runs_once_then_current_location_takes_over(self):
        self.assertIn("previousKey && previousKey !== currentKey", UI_SOURCE)
        self.assertIn("ESTABLISHED CURRENT LOCATION", UI_SOURCE)
        self.assertIn("every newly revealed environmental feature", UI_SOURCE)
        self.assertIn("preserve its established spatial logic", UI_SOURCE)
        self.assertIn("End looking deeper into", UI_SOURCE)

    def test_location_transition_presets_use_global_settings_and_scene_lock(self):
        for preset in (
            'value: "normal"',
            'value: "surreal"',
            'value: "cinematic"',
            'value: "inner_world"',
            'value: "match"',
            'value: "motion"',
            'value: "creative_auto"',
            'value: "masked"',
            'value: "custom"',
        ):
            self.assertIn(preset, UI_SOURCE)
        self.assertIn('location_transition_preset: "normal"', UI_SOURCE)
        self.assertIn("location_transition_preset: miniMaxLocationTransitionPreset.value", UI_SOURCE)
        self.assertIn("location_transition_custom: miniMaxLocationTransitionCustom.value", UI_SOURCE)
        self.assertIn("if (segment?.use_scene_minimax_h3_settings)", UI_SOURCE)
        self.assertIn("state.miniMaxH3Settings = settings", UI_SOURCE)
        self.assertIn("legacyTransitionSegment", UI_SOURCE)
        self.assertIn("hasSavedTransitionPreset", UI_SOURCE)
        self.assertIn("const transitionSettings = miniMaxH3SettingsForSegment(segment)", UI_SOURCE)
        self.assertIn("Global for all unlocked scenes", UI_SOURCE)
        self.assertIn("if (previousKey && previousKey !== currentKey)", UI_SOURCE)
        self.assertIn("return `${directions[preset]} ${commonEnding}`", UI_SOURCE)

    def test_masked_preset_continues_then_makes_one_move_and_keeps_the_vocal_performance(self):
        self.assertIn("LOCATION PHASE — MASKED CONTINUATION TRANSITION", UI_SOURCE)
        # a set hold, then one smooth move, with no effects that stage a new shot
        self.assertIn("for the first ${holdSeconds} seconds simply continue the opening frame's action", UI_SOURCE)
        self.assertIn("begin ONE smooth, motivated movement", UI_SOURCE)
        self.assertIn("with no wipe, flash, portal, morph, or cut", UI_SOURCE)
        # singing / speaking carries through the move when the scene has vocals, and is skipped for b-roll
        self.assertIn("function miniMaxH3MaskedPerformanceText(segment, part)", UI_SOURCE)
        self.assertIn("if (segmentUsesNoLipSyncPerformance(segment)) return \"\";", UI_SOURCE)
        self.assertIn("singing the scene's lyrics", UI_SOURCE)
        self.assertIn("speaking their dialogue", UI_SOURCE)
        self.assertIn("${performanceLine}", UI_SOURCE)
        self.assertIn('miniMaxH3MaskedPerformanceText(segment, "opening")', UI_SOURCE)
        # picking masked continuity selects the matching preset
        self.assertIn('miniMaxLocationTransitionPreset.value = "masked"', UI_SOURCE)

    def test_continuation_direction_is_saved_per_scene_and_placed_after_a_hold_without_a_cut(self):
        # one textarea in the continuation row, saved on the scene like the scene prompt
        self.assertIn("miniMaxContinuationDirection", UI_SOURCE)
        self.assertIn("segment.minimax_h3_continuation_direction = String(miniMaxContinuationDirection.value", UI_SOURCE)
        self.assertIn("miniMaxContinuationDirection.value = String(segment?.minimax_h3_continuation_direction", UI_SOURCE)
        # the LLM is told when it starts (a third of the scene, shared with the Masked preset) and that it never cuts
        self.assertIn("function miniMaxH3ContinuationHoldSeconds(segment)", UI_SOURCE)
        # the start is the author's choice: 0.5 s by default, at most half of the scene, and the LLM is told the scene length
        self.assertIn("segment?.minimax_h3_continuation_start_seconds", UI_SOURCE)
        self.assertIn("Math.floor(Math.max(0, Number(sceneSeconds) || 0) * 0.5 * 10 + 1e-9) / 10", UI_SOURCE)
        self.assertIn("SCENE TIMING: This scene is ${sceneLength} seconds long.", UI_SOURCE)
        self.assertIn("perform its actions in quicker succession rather than leaving any out", UI_SOURCE)
        self.assertNotIn("Do not skip, soften, shorten", UI_SOURCE)
        # a slider under the direction box sets it, saves it only when moved, and the pop-out has the same slider
        self.assertIn('miniMaxContinuationStart.type = "range";', UI_SOURCE)
        self.assertIn("segment.minimax_h3_continuation_start_seconds = Number(miniMaxContinuationStart.value);", UI_SOURCE)
        self.assertIn("miniMaxContinuationStartField,\n        makeField(\"MiniMax H3 prompt\", miniMaxPrompt)", UI_SOURCE.replace("\r\n", "\n"))
        self.assertIn("const startMirror = document.createElement(\"input\");", UI_SOURCE)
        self.assertIn("AUTHOR'S DIRECTION FOR THIS SCENE — MANDATORY", UI_SOURCE)
        self.assertIn("Never cut, change shot, or restart the action to reach it.", UI_SOURCE)
        self.assertIn("miniMaxH3ContinuationDirectionText(segment)", UI_SOURCE)
        # the timing is written into the finished description, and the direction is last in the concept
        self.assertIn('"For the first ${holdSeconds} seconds, ..."', UI_SOURCE)
        self.assertIn('"At about ${holdSeconds} seconds, ..."', UI_SOURCE)
        self.assertIn("THE FINISHED DESCRIPTION MUST CONTAIN IT", UI_SOURCE)
        self.assertIn("then carry on exactly as the AUTHOR'S DIRECTION at the end of this scene concept says", UI_SOURCE)
        self.assertLess(
            UI_SOURCE.index('add(parts, "Continuity notes for staging only"'),
            UI_SOURCE.index("if (continuationDirection) parts.push(continuationDirection);"),
        )
        # vocals keep going through it
        self.assertIn('miniMaxH3MaskedPerformanceText(segment, "transition")', UI_SOURCE)

    def test_a_continued_scene_does_not_let_a_framing_preset_restage_shot_one(self):
        # a preset such as Intimate close-ups listed "side-lying close-up" as Shot 1 and the LLM obeyed it over the previous frame
        self.assertIn("function miniMaxH3PerShotFramingLines(segment, shotPlan = [], continuation = false)", UI_SOURCE)
        self.assertIn("Shot 1 framing: begin exactly as Attached Picture 1 shows it", UI_SOURCE)
        self.assertIn("miniMaxH3PerShotFramingLines(segment, shotPlan, Boolean(options.frameContinuityPrompt))", UI_SOURCE)
        # and the author's direction bridges to the body position it needs
        self.assertIn("first describe the natural movement that gets the subject there", UI_SOURCE)

    def test_decimal_points_are_not_sentence_ends(self):
        self.assertIn("VRGDGDECIMALTOKEN", UI_SOURCE)

    def test_each_location_transition_preset_resolves_to_one_instruction(self):
        for heading in (
            "PHYSICAL THRESHOLD TURN",
            "SURREAL MATERIAL TRANSFORMATION",
            "CINEMATIC CONCEAL AND REVEAL",
            "INNER WORLD PORTAL",
            "VISUAL MATCH TRANSITION",
            "MOTION-DRIVEN TRANSITION",
            "CREATIVE IMAGE-AWARE TRANSITION",
            "CUSTOM TRANSITION",
        ):
            self.assertIn(heading, UI_SOURCE)
        self.assertIn("Apply this scene's authored transition direction", UI_SOURCE)
        self.assertIn("miniMaxLocationTransitionCustomField.style.display", UI_SOURCE)

    def test_generation_retries_ten_times_and_never_accepts_blank(self):
        self.assertIn("attempt <= 10", UI_SOURCE)
        self.assertIn("returned an empty frame-to-frame continuity prompt", UI_SOURCE)
        self.assertIn("could not create a valid frame-to-frame continuity prompt after 10 attempts", UI_SOURCE)
        generation_call = UI_SOURCE.index("await createMiniMaxH3FrameContinuityPrompt")
        prompt_validation = UI_SOURCE.index("if (!prompt) throw new Error", generation_call)
        self.assertLess(generation_call, prompt_validation)

    def test_blank_saved_prompt_is_generated_before_render_validation(self):
        validation_start = UI_SOURCE.index("function validateMiniMaxSceneReadyForVideo")
        validation_end = UI_SOURCE.index("async function prepareMiniMaxH3ContinuityReference", validation_start)
        validation_source = UI_SOURCE[validation_start:validation_end]
        self.assertIn("if (!frameContinuityPromptEnabled)", validation_source)
        self.assertIn("if (!savedPrompt)", validation_source)

        render_start = UI_SOURCE.index("async function renderMiniMaxSceneVideoWithProgress")
        render_end = UI_SOURCE.index("async function createMiniMaxSceneVideo", render_start)
        render_source = UI_SOURCE[render_start:render_end]
        generation = render_source.index("await createMiniMaxH3FrameContinuityPrompt")
        saved_prompt_fallback = render_source.index("segment?.minimax_h3_prompt || segment?.i2v_prompt", generation)
        empty_prompt_rejection = render_source.index("needs a MiniMax H3 prompt", saved_prompt_fallback)
        self.assertLess(generation, saved_prompt_fallback)
        self.assertLess(saved_prompt_fallback, empty_prompt_rejection)

        create_start = UI_SOURCE.index("async function createMiniMaxH3FrameContinuityPrompt")
        create_end = UI_SOURCE.index("async function renderMiniMaxSceneVideoWithProgress", create_start)
        create_source = UI_SOURCE[create_start:create_end]
        self.assertIn("segment.minimax_h3_prompt = generatedPrompt", create_source)
        self.assertIn("await autoSaveSessionQuiet", create_source)

    def test_feature_forces_one_continuous_shot(self):
        self.assertIn("frequency: 0, cut_times_seconds: [], cue_driven: false", UI_SOURCE)
        self.assertIn("All listed cues occur inside the same uninterrupted shot", UI_SOURCE)
        self.assertIn("Continuing seamlessly from the previous shot, the camera maintains its established course as", UI_SOURCE)

    def test_generated_shot_prose_removes_negative_sentences_without_failing(self):
        self.assertIn("stripMiniMaxH3NegativePromptSentences", UI_SOURCE)
        self.assertIn("Removed negative prompt wording", UI_SOURCE)
        self.assertIn("using the positive fallback shot", UI_SOURCE)
        self.assertNotIn("Gemma used negative prompt wording", UI_SOURCE)
        self.assertIn("VRGDGDIALOGUE", UI_SOURCE)
        self.assertIn("restoreDialogue", UI_SOURCE)
        self.assertIn("The assigned performer visibly delivers", UI_SOURCE)
        self.assertNotIn("No additional environmental or physical sounds are added", UI_SOURCE)
        self.assertIn("remains the sole complete audience-facing soundtrack", UI_SOURCE)

    def test_llm_preset_contains_no_inactive_setting_branches(self):
        core = INSTRUCTION_SOURCE.split("_MINIMAX_H3_TEXT_TO_VIDEO_MODE", 1)[0]
        for inactive_branch in (
            "If a lyric/dialogue line",
            "If the scene context supplies a vocal cue map",
            "For instrumental or otherwise non-vocal intervals",
            "In multi-subject vocal scenes",
            "If the scene is visual-only",
        ):
            self.assertNotIn(inactive_branch, core)
        self.assertIn("already contains the Builder's resolved mode", core)
        self.assertNotIn("is_minimax_h3_shot_json_task and not frame_continuity_prompt", BACKEND_SOURCE)

    def test_character_motion_rule_is_resolved_for_current_performance(self):
        self.assertIn("visualOnly || segment?.no_character_present || !lyricText", UI_SOURCE)
        self.assertIn("Do not add singing, speaking, or lip sync", UI_SOURCE)
        self.assertIn('performanceMode === "speaking"', UI_SOURCE)
        self.assertIn("required dialogue lip sync", UI_SOURCE)
        self.assertIn("required singing lip sync", UI_SOURCE)

    def test_frame_prompt_finalizer_is_in_render_scope(self):
        self.assertIn("  function ensureBuilderManagedFx(prompt, scene = {})", UI_SOURCE)
        self.assertNotIn("    const ensureBuilderManagedFx =", UI_SOURCE)
        create_start = UI_SOURCE.index("async function createMiniMaxH3FrameContinuityPrompt")
        create_end = UI_SOURCE.index("async function renderMiniMaxSceneVideoWithProgress", create_start)
        self.assertIn("ensureBuilderManagedFx(prompt, segment)", UI_SOURCE[create_start:create_end])


if __name__ == "__main__":
    unittest.main()
