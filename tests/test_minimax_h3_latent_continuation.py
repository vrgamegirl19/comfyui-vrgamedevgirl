import ast
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")
RUNNER_SOURCE = (ROOT / "VRGDG_WorkflowRunnerNodes.py").read_text(encoding="utf-8")

_SPEC = importlib.util.spec_from_file_location(
    "vrgdg_minimax_h3_latent_manager", ROOT / "VRGDG_MiniMaxH3LatentManager.py"
)
MANAGER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MANAGER)


def _runner_namespace():
    """The latent-continuation helpers of the workflow runner, executed in isolation."""
    module = ast.parse(RUNNER_SOURCE)
    wanted = {
        "_int_payload",
        "_first_payload_value",
        "_api_node_id_by_class",
        "_minimax_h3_latent_continuation_mode",
        "_minimax_h3_is_latent_mode",
        "_minimax_h3_latent_context_frames_setting",
        "_minimax_h3_latent_exact_plan",
        "_minimax_h3_tail_padding_frames",
        "_minimax_h3_effective_warmup_frames",
        "_patch_minimax_h3_latent_continuation",
    }
    constants = {"_MMH3_LATENT_MODE", "_MMH3_LATENT_EXACT_MODE"}
    body = [
        node for node in module.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in constants)
    ]
    namespace = {
        "json": json,
        "os": os,
        "SceneLatentManager": MANAGER.SceneLatentManager,
        "plan_latent_context": MANAGER.plan_latent_context,
        "_frames_to_tokens": MANAGER._frames_to_tokens,
        "_tokens_to_frames": MANAGER._tokens_to_frames,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "latent_continuation_runner", "exec"), namespace)
    return namespace


class LatentContextPlanTests(unittest.TestCase):
    def test_plain_plan_takes_the_tail_of_the_latent(self):
        plan = MANAGER.plan_latent_context(72, None, 22, exact_frame=False)
        self.assertEqual((plan["start_token"], plan["end_token"]), (65, 72))
        self.assertEqual(plan["warmup_frames"], 22)
        self.assertIsNone(plan["image_frame_offset"])

    def test_exact_plan_snaps_to_the_token_grid_and_lands_on_the_last_visible_frame(self):
        total = 72
        for padding in range(0, 17):
            for context_frames in (22, 39, 56):
                plan = MANAGER.plan_latent_context(total, padding, context_frames, exact_frame=True)
                visible = MANAGER._tokens_to_frames(total) - padding
                first_context_frame = sum(
                    MANAGER._FRAME_PER_TOKEN[k % 5] for k in range(plan["start_token"])
                )
                with self.subTest(padding=padding, context_frames=context_frames):
                    self.assertTrue(plan["exact"])
                    self.assertEqual(plan["start_token"] % 5, 0)
                    self.assertLessEqual(plan["end_token"], total)
                    # the exact image sits on the predecessor's real last frame
                    self.assertEqual(plan["image_frame_offset"], visible - 1 - first_context_frame)
                    # and the context never overlaps that frame
                    self.assertLess(plan["context_frames"], plan["warmup_frames"])

    def test_exact_plan_falls_back_when_the_latent_is_too_short(self):
        plan = MANAGER.plan_latent_context(3, 0, 22, exact_frame=True)
        self.assertFalse(plan["exact"])


class SceneLatentStorageTests(unittest.TestCase):
    def _save(self, folder, scene, tokens=12, **kwargs):
        latent = {"video": torch.zeros(1, 24, tokens, 4, 4), "audio": torch.zeros(1, 32, 2, 8)}
        return MANAGER.SceneLatentManager.save_latent(folder, scene, latent, **kwargs)

    def test_tail_padding_round_trips_through_the_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            self._save(folder, 3, metadata={"tail_padding_frames": 7})
            info = MANAGER.SceneLatentManager.get_latent_info(folder, 3)
            self.assertEqual(info["tail_padding_frames"], 7)
            self.assertEqual(info["token_count"], 12)

    def test_latents_saved_without_padding_report_it_as_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            self._save(folder, 3)
            self.assertIsNone(MANAGER.SceneLatentManager.get_latent_info(folder, 3)["tail_padding_frames"])

    def test_saving_a_scene_marks_an_existing_successor_dirty(self):
        with tempfile.TemporaryDirectory() as folder:
            self._save(folder, 2)
            self._save(folder, 1)
            self.assertTrue(MANAGER.SceneLatentManager.is_dirty(folder, 2))
            self._save(folder, 2)
            self.assertFalse(MANAGER.SceneLatentManager.is_dirty(folder, 2))

    def test_deleting_a_scene_latent_keeps_the_numbering_of_the_others(self):
        with tempfile.TemporaryDirectory() as folder:
            for scene in (1, 2, 3):
                self._save(folder, scene)
            MANAGER.SceneLatentManager.delete_latent(folder, 2)
            self.assertFalse(MANAGER.SceneLatentManager.latent_exists(folder, 2))
            self.assertTrue(MANAGER.SceneLatentManager.latent_exists(folder, 1))
            self.assertTrue(MANAGER.SceneLatentManager.latent_exists(folder, 3))

    def test_delete_all_removes_only_scene_latent_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for scene in (1, 2):
                self._save(folder, scene)
            keep = Path(folder) / "latents" / "notes.txt"
            keep.write_text("keep", encoding="utf-8")
            self.assertGreater(MANAGER.SceneLatentManager.delete_all_latents(folder), 0)
            self.assertEqual([p.name for p in (Path(folder) / "latents").iterdir()], ["notes.txt"])


class RunnerLatentContinuationTests(unittest.TestCase):
    def setUp(self):
        self.ns = _runner_namespace()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        MANAGER.SceneLatentManager.save_latent(
            self.folder.name,
            21,
            {"video": torch.zeros(1, 24, 72, 4, 4), "audio": torch.zeros(1, 32, 2, 8)},
            metadata={"tail_padding_frames": 5},
        )
        self.image = Path(self.folder.name) / "last.png"
        self.image.write_bytes(b"png")
        self.payload = {
            "project_folder": self.folder.name,
            "scene_number": 22,
            "latent_context_frames": 22,
            "pre_frames": 0,
        }

    @staticmethod
    def _prompt():
        return {
            "126": {"class_type": "BasicGuider", "inputs": {"conditioning": ["136", 0], "model": ["208", 0]}},
            "136": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"vae": ["119", 0]}},
            "119": {"class_type": "VAELoader", "inputs": {}},
        }

    def test_mode_names_are_normalised(self):
        mode = self.ns["_minimax_h3_latent_continuation_mode"]
        self.assertEqual(mode({"continuity_mode": "Latent Continuation"}), "latent_continuation")
        self.assertEqual(mode({"continuity_mode": "latent-exact"}), "latent_continuation_exact_frame")
        self.assertEqual(mode({"continuity_mode": "off"}), "off")

    def test_warmup_covers_the_context_only_when_latent_continuation_is_active(self):
        warmup = self.ns["_minimax_h3_effective_warmup_frames"]
        self.assertEqual(warmup({**self.payload, "continuity_mode": "off"}), 0)
        self.assertEqual(warmup({**self.payload, "continuity_mode": "latent_continuation"}), 22)
        self.assertEqual(warmup({**self.payload, "continuity_mode": "latent_continuation", "scene_number": 1}), 0)
        # the 16-frame option really covers 17 frames (5 tokens)
        self.assertEqual(
            warmup({**self.payload, "continuity_mode": "latent_continuation", "latent_context_frames": 16}), 17
        )
        # a larger user warm-up always wins
        self.assertEqual(warmup({**self.payload, "continuity_mode": "latent_continuation", "pre_frames": 40}), 40)

    def test_plain_mode_adds_a_guide_and_no_image_nodes(self):
        prompt = self._prompt()
        result = self.ns["_patch_minimax_h3_latent_continuation"](prompt, {**self.payload, "continuity_mode": "latent_continuation"})
        self.assertTrue(result["enabled"])
        self.assertEqual(prompt["126"]["inputs"]["conditioning"], [result["guide_node_id"], 0])
        self.assertIsNone(result["exact_guide_node_id"])
        classes = {node["class_type"] for node in prompt.values()}
        self.assertIn("VRGDG_MiniMaxH3LoadLatent", classes)
        self.assertNotIn("MiniMaxH3AddGuide", classes)

    def test_exact_mode_chains_the_native_add_guide_on_the_last_frame(self):
        prompt = self._prompt()
        payload = {
            **self.payload,
            "continuity_mode": "latent_continuation_exact_frame",
            "latent_exact_frame_path": str(self.image),
        }
        result = self.ns["_patch_minimax_h3_latent_continuation"](prompt, payload)
        add_guide = prompt[result["exact_guide_node_id"]]
        self.assertEqual(add_guide["class_type"], "MiniMaxH3AddGuide")
        self.assertEqual(prompt["126"]["inputs"]["conditioning"], [result["exact_guide_node_id"], 0])
        # the image is the last frame of the warm-up
        self.assertEqual(add_guide["inputs"]["frame_idx"], result["warmup_frames"] - 1)
        self.assertEqual(prompt[result["exact_image_node_id"]]["class_type"], "VRGDG_MiniMaxH3LoadExactFrame")
        self.assertTrue(prompt[result["load_node_id"]]["inputs"]["exact_frame_mode"])

    def test_exact_mode_requires_the_last_frame_image(self):
        payload = {**self.payload, "continuity_mode": "latent_continuation_exact_frame"}
        with self.assertRaises(FileNotFoundError):
            self.ns["_patch_minimax_h3_latent_continuation"](self._prompt(), payload)

    def test_missing_predecessor_latent_is_a_clear_error(self):
        payload = {**self.payload, "scene_number": 30, "continuity_mode": "latent_continuation"}
        with self.assertRaises(FileNotFoundError):
            self.ns["_patch_minimax_h3_latent_continuation"](self._prompt(), payload)

    def test_tail_padding_is_the_render_length_minus_the_visible_scene(self):
        padding = self.ns["_minimax_h3_tail_padding_frames"]
        self.assertEqual(
            padding({"actual_warmup_seconds": 0.917, "scene_duration_seconds": 7.25, "h3_frame_count": 209}), 13
        )
        self.assertIsNone(padding({}))


class BuilderLatentContinuationWiringTests(unittest.TestCase):
    def test_both_modes_are_offered_and_normalised(self):
        self.assertIn('value: "latent_continuation", label: "Latent Continuation (native H3 temporal context)"', BUILDER_SOURCE)
        self.assertIn('value: "latent_continuation_exact_frame"', BUILDER_SOURCE)
        self.assertIn("function isMiniMaxH3LatentContinuationMode(mode)", BUILDER_SOURCE)

    def test_render_payload_carries_the_latent_settings(self):
        for key in (
            "continuity_mode: continuityInput?.continuityMode",
            "latent_context_frames: latentContextFrames",
            "latent_exact_frame_path: continuityInput?.exactFramePath",
        ):
            self.assertIn(key, BUILDER_SOURCE)

    def test_exact_frame_is_not_injected_as_a_reference_image_or_prompt_block(self):
        start = BUILDER_SOURCE.index("if (isMiniMaxH3LatentContinuationMode(continuityMode)) {")
        end = BUILDER_SOURCE.index("previousSegment,\n      };", start)
        block = BUILDER_SOURCE[start:end]
        self.assertIn('framePath: ""', block)
        self.assertIn("exactFramePath", block)

    def test_deleting_a_video_removes_its_stale_latent(self):
        start = BUILDER_SOURCE.index("async function deleteSelectedMedia()")
        end = BUILDER_SOURCE.index("function sendPromptToEnhance", start)
        self.assertIn("deleteStaleSceneLatents(media.segment)", BUILDER_SOURCE[start:end])
        # a video delete must never renumber the other scenes' latents
        self.assertIn("payload.reindex = false", BUILDER_SOURCE)


if __name__ == "__main__":
    unittest.main()
