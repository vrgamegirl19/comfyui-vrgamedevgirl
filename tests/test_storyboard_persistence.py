"""Storyboard settings and scene data survive a disk save and reopen."""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_persistence() -> dict:
    """Load the real disk helpers without starting ComfyUI or loading models."""
    namespace = {"os": os, "json": json}
    for relative in ("core/atomic_write.py", "storyboard/scene_helpers.py"):
        spec = importlib.util.spec_from_file_location(
            Path(relative).stem, ROOT / relative
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace.update(vars(module))
    namespace["_enforce_storyboard_video_facial_requirements"] = (
        lambda prompt, scene: prompt
    )
    source = ROOT / "storyboard/persistence.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace["datetime"] = datetime
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


class StoryboardPersistenceTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("node"), "Node.js is required for UI behavior tests"
    )
    def test_save_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                ["node", "--test", str(ROOT / "tests/storyboard_persistence.cjs")],
                env={
                    **os.environ,
                    "STORYBOARD_TEST_PYTHON": sys.executable,
                    "STORYBOARD_TEST_PROJECT": folder,
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_zero_motion_speeds_and_scene_fields_survive_disk(self) -> None:
        persistence = load_persistence()
        with tempfile.TemporaryDirectory() as folder:
            scene = {
                "id": "a", "project_video_engine": "minimax_h3",
                "video_prompt_type": "flf", "no_character_present": True,
                "lyric_singers": ["Singer"], "lyric_no_lip_sync": True,
                "lyric_instrumental": True, "flf_start_state": "A",
                "flf_transformation": "Turn", "flf_end_state": "B",
                "flf_carry_forward": "C",
                "minimax_h3_pass2_prompt": "Second pass",
                "image_name": "Frame", "image_data": "data:image/png;base64,A",
            }
            payload = {
                "project_folder": folder,
                "storyboard": {
                    "camera_motion_speed": 0, "character_motion_speed": 0,
                    "scenes": [scene],
                },
            }
            persistence["_save_storyboard"](payload)
            loaded = persistence["_load_storyboard"]({"project_folder": folder})
            self.assertEqual(loaded["camera_motion_speed"], 0)
            self.assertEqual(loaded["character_motion_speed"], 0)
            for field, value in scene.items():
                self.assertEqual(loaded["scenes"][0][field], value, field)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"save", "load"}:
        persistence = load_persistence()
        payload = json.load(sys.stdin)
        print(json.dumps(persistence["_" + sys.argv[1] + "_storyboard"](payload)))
    else:
        unittest.main()
