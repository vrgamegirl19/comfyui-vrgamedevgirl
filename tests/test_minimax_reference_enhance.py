"""Exercise the vision task without loading ComfyUI or calling a paid LLM."""

import ast
import copy
import json
import math
import re
import sys
import types
import unittest
from unittest.mock import Mock, patch

from test_minimax_prompt_enhance import ROOT, ORIGINAL, REFERENCES, VISION_REPLY, pe, minimax_instructions

try:
    from PIL import Image, ImageDraw, ImageOps
except ImportError:
    Image = ImageDraw = ImageOps = None


def fixture():
    tree = ast.parse((ROOT / "llm" / "minimax_reference_enhance.py").read_text(encoding="utf-8"))
    code = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
    context = {
        "json": json, "math": math, "re": re, "Image": Image, "ImageDraw": ImageDraw, "ImageOps": ImageOps,
        "merge_reference_enhanced_prompt": pe.merge_reference_enhanced_prompt,
        "_extract_json_object_from_text": json.loads,
        "MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT": minimax_instructions.MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT,
        "MINIMAX_H3_PROMPT_ENHANCE_WITH_REFERENCES_INSTRUCTIONS": minimax_instructions.MINIMAX_H3_PROMPT_ENHANCE_WITH_REFERENCES_INSTRUCTIONS,
        "_effective_builder_instruction": Mock(side_effect=lambda payload, key, default: default),
    }
    exec(compile(code, "reference_enhancement", "exec"), context)
    return context


class ReferenceEnhanceTests(unittest.TestCase):
    def test_vision_task_sends_actual_images_and_separate_instructions_then_merges(self):
        c = fixture()
        images = [object(), object()]
        c["prepare_reference_images"] = Mock(return_value=(images, "Image map"))
        c["run_reference_vision"] = Mock(return_value=(json.dumps(VISION_REPLY), {"runner": "test_vision"}))
        payload = {"current_prompt": ORIGINAL, "image_references": REFERENCES}
        result = c["enhance_with_references"](payload)
        args = c["run_reference_vision"].call_args.args
        self.assertIs(args[2], images)
        self.assertIn("character", args[1])
        self.assertIn("<Picture 2>", args[1])
        self.assertEqual(c["_effective_builder_instruction"].call_args.args[1], "minimax_h3_prompt_enhance_with_references")
        self.assertIn("red bob", result["prompt"])
        self.assertEqual(result["reference_image_count"], 2)
        self.assertTrue(result["used_image_reference"])

    def test_invalid_or_missing_images_fail_without_a_vision_request(self):
        for mode in ("empty", "duplicate", "no image", "refmod", "only frames"):
            c = fixture()
            c["prepare_reference_images"] = Mock()
            refs = copy.deepcopy(REFERENCES)
            if mode == "empty": refs = []
            elif mode == "duplicate": refs.append(refs[0])
            elif mode == "no image": refs[0]["path"] = ""
            elif mode == "refmod": refs[0]["path"] = "refmod://character"
            else:
                for ref in refs: ref["describe"] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                c["enhance_with_references"]({"current_prompt": ORIGINAL, "image_references": refs})
            c["prepare_reference_images"].assert_not_called()

    def test_remote_runner_receives_all_prepared_images(self):
        c = fixture()
        c.update(_llm_runner_from_payload=lambda p: "llm_api", _EXTERNAL_LLM_RUNNERS={"llm_api"},
                 _runner_output_token_limit=lambda p, tokens: tokens,
                 _try_run_remote_vision=Mock(return_value=("reply", {})))
        images = [object(), object(), object()]
        c["run_reference_vision"]({}, "instruction", images)
        self.assertIs(c["_try_run_remote_vision"].call_args.args[2], images)

    def test_local_vision_uses_selected_projector_and_cleans_up_on_success_or_failure(self):
        for fail in (False, True):
            c = fixture()
            llm = Mock()
            llm.MISSING_MODEL_OPTION = "missing"
            llm._resolve_dropdown_path.return_value = "model.gguf"
            llm._run_gguf_vision_pipeline.return_value = "reply"
            if fail: llm._run_gguf_vision_pipeline.side_effect = RuntimeError("vision failed")
            c.update(
                __package__="_reference_vision_test.llm",
                _llm_runner_from_payload=lambda p: "builtin", _EXTERNAL_LLM_RUNNERS={"llm_api"},
                _runner_output_token_limit=lambda p, tokens: tokens,
                _builder_local_llm=lambda p: llm,
                _builder_local_model_file=lambda p, current: current,
                _builder_local_mmproj_file=lambda p, current: current,
                _resolve_mmproj_dropdown_path=Mock(return_value="projector.gguf"),
            )
            cache = types.ModuleType("_reference_vision_test.llm.cache")
            cache._clear_vrgdg_llm_caches = Mock()
            with self.subTest(fail=fail), patch.dict(sys.modules, {cache.__name__: cache}):
                payload = {"model_file": "selected.gguf", "mmproj_file": "selected-projector.gguf"}
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "vision failed"):
                        c["run_reference_vision"](payload, "instruction", [object()])
                else:
                    text, info = c["run_reference_vision"](payload, "instruction", [object()])
                    self.assertEqual(text, "reply")
                    self.assertTrue(info["unloaded"])
                self.assertEqual(c["_resolve_mmproj_dropdown_path"].call_args.args[1], "selected-projector.gguf")
                llm._unload_gguf_model.assert_called_once()
                cache._clear_vrgdg_llm_caches.assert_called_once()

    @unittest.skipIf(Image is None, "Pillow is available in the ComfyUI Python environment")
    def test_nine_references_are_all_packed_and_labeled_in_at_most_four_images(self):
        c = fixture()
        refs = [{"path": str(index), "picture_label": f"<Picture {index + 1}>", "label": f"<Subject {index + 1}>"}
                for index in range(9)]
        c["_image_from_prompt_payload"] = lambda path, data, title: Image.new("RGB", (20, 20), (int(path) * 20, 0, 0))
        sheets, mapping = c["prepare_reference_images"](refs)
        self.assertLessEqual(len(sheets), 4)
        for index in range(9):
            sheet = sheets[index // 3]
            offset = index % 3
            self.assertEqual(sheet.getpixel(((offset % 2) * 768 + 384, (offset // 2) * 808 + 424)), (index * 20, 0, 0))
        self.assertIn("tile heading", mapping)


if __name__ == "__main__":
    unittest.main()
