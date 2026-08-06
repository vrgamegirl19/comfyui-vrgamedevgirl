import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)
INSTRUCTION_SOURCE = (ROOT / "VRGDG_MiniMaxH3PromptInstructions.py").read_text(
    encoding="utf-8"
)
NODE_SOURCE = (ROOT / "VRGDG_MusicVideoBuilderNodes.py").read_text(
    encoding="utf-8"
)


class BuilderMiniMaxStartFrameCharacterInfluenceTests(unittest.TestCase):
    def test_reference_to_video_exposes_saved_character_influence_dropdown(self):
        self.assertIn(
            'label: "Face + hair only (keep the rest of the start frame)"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'label: "Full character identity (face, hair, clothing, and body)"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "segment.minimax_h3_start_frame_character_influence = normalizeMiniMaxH3StartFrameCharacterInfluence(",
            BUILDER_SOURCE,
        )

    def test_scene_image_use_and_influence_apply_to_all_eligible_scenes(self):
        self.assertIn(
            'label: "Environment inspiration only (LLM only — ignore framing)"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'label: "Environment + framing inspiration (LLM only)"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            'miniMaxH3ModeForSegment(item) === "reference_to_video"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "item.minimax_h3_scene_image_use = sceneImageUse;",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Skipped ${blocked.length} scene",
            BUILDER_SOURCE,
        )

    def test_global_start_frame_controls_never_change_locked_scenes(self):
        self.assertIn(
            "const sceneLocked = Boolean(segment.use_scene_minimax_h3_settings);",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "&& (sceneLocked || !item.use_scene_minimax_h3_settings)",
            BUILDER_SOURCE,
        )
        self.assertIn(
            '"Scene image use — this locked scene"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            '"Character reference influence — this locked scene"',
            BUILDER_SOURCE,
        )

    def test_environment_only_contract_ignores_framing_and_all_character_details(self):
        self.assertIn(
            '"PROMPT-ONLY SCENE-IMAGE INSPIRATION — MANDATORY:\\n"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Explicitly ignore camera framing and shot distance, camera angle and lens, and image composition.",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Always ignore every visible character's identity, face, hair, body, clothing, accessories, pose, placement, and activity",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "assigned character-reference images are the sole visual authority",
            INSTRUCTION_SOURCE,
        )

    def test_prompt_only_scene_image_is_vision_first_but_not_a_renderer_reference(self):
        self.assertIn(
            "return path || data ? [{ path, data, prompt_only_scene_inspiration: true }, ...rendererImages] : rendererImages;",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Attached Picture 1 is shown only to the prompt-writing vision LLM. It is NOT supplied to the MiniMax H3 renderer",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Renderer Image 1 is attached as Picture 2",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "prompt_only_scene_inspiration: miniMaxH3SceneImageIsPromptInspiration(segment)",
            BUILDER_SOURCE,
        )
        self.assertIn(
            'reference_limit = 10 if is_minimax_h3_prompt and prompt_only_scene_inspiration else 9 if is_minimax_h3_prompt else 4',
            NODE_SOURCE,
        )
        self.assertIn(
            "Attached <Picture 1> is vision input for prompt writing only.",
            NODE_SOURCE,
        )

    def test_legacy_checkbox_state_migrates_to_exact_start_frame(self):
        self.assertIn(
            "normalizeMiniMaxH3SceneImageUse(\n      segment.minimax_h3_scene_image_use,\n      Boolean(segment.minimax_h3_use_scene_image_as_start_frame)",
            BUILDER_SOURCE,
        )
        self.assertIn(
            'segment.minimax_h3_use_scene_image_as_start_frame = segment.minimax_h3_scene_image_use === "exact_start_frame";',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "!item?.use_scene_minimax_h3_settings\n        && miniMaxH3ContinuityModeForSegment(item)",
            BUILDER_SOURCE,
        )

    def test_face_hair_mode_changes_ordered_media_assignments(self):
        self.assertIn(
            '"exact start frame and authority for every visible detail except face identity and hair"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            '"face identity and hair reference only; do not copy clothing, body proportions, pose, accessories, framing, lighting, or background"',
            BUILDER_SOURCE,
        )

    def test_llm_context_receives_strict_first_frame_priority_contract(self):
        self.assertIn(
            '"START-FRAME / CHARACTER-REFERENCE PRIORITY — MANDATORY:\\n"',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Character-reference images may override Image 1 ONLY for the subject's face identity, facial features, and hair.",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "The literal first generated frame must already depict the character-reference face and hair",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "never describe or show a face swap, replacement process, morph, transition, or transformation",
            BUILDER_SOURCE,
        )

    def test_permanent_reference_to_video_instructions_enforce_selected_scope(self):
        self.assertIn(
            "obey the supplied `START-FRAME / CHARACTER-REFERENCE PRIORITY — MANDATORY` contract exactly",
            INSTRUCTION_SOURCE,
        )
        self.assertIn(
            "Character-reference images are authoritative ONLY for face identity, facial features, and hair.",
            INSTRUCTION_SOURCE,
        )
        self.assertIn(
            "never import the character reference's clothing, body, pose, accessories, framing, lighting, or background",
            INSTRUCTION_SOURCE,
        )

    def test_managed_subject_count_block_respects_face_hair_only_scope(self):
        self.assertIn(
            'use those views ONLY to learn that one person\'s face identity, facial features, and hair.',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Image 1 remains authoritative for clothing, body proportions, pose, accessories, framing, lighting, and background.",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Extract only the character properties granted by its ordered assignment",
            INSTRUCTION_SOURCE,
        )

    def test_prompt_guidance_avoids_implied_on_screen_replacement(self):
        self.assertIn(
            "Do not use temporal comparison language such as 'now featuring,' 'becomes,' 'changes into,' or 'replaced by'",
            BUILDER_SOURCE,
        )
        self.assertIn(
            "do not use temporal comparison language such as `now featuring`, `becomes`, `changes into`, or `replaced by`",
            INSTRUCTION_SOURCE,
        )


if __name__ == "__main__":
    unittest.main()
