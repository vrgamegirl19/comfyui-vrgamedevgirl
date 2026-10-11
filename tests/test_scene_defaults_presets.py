"""Shared disk presets for Scene Defaults, independent of any project."""
import importlib
import ast
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
presets = importlib.import_module(f"{ROOT.name}.builder.scene_defaults_presets")


class SceneDefaultsPresetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = os.path.join(temporary.name, "scene_defaults")
        patcher = patch.object(presets, "preset_root", return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_roundtrip_filters_project_data_and_enforces_baseline(self):
        presets.save_scene_defaults_preset("My mix", {
            "cameraMotionSpeed": "5", "facialPerformanceCustom": "Smile", "videoStyle": "grunge",
            "temporalWorldEffect": "time_lapse", "fxPreset": "glow", "scenes": ["private"],
            "customCameraFlowSequence": [{"shot": "medium shot", "camera": "pan", "extra": "discard"}],
        })
        saved = presets.load_scene_defaults_preset("My mix")
        self.assertEqual(saved["settings"]["cameraMotionSpeed"], 5)
        self.assertEqual(saved["settings"]["facialPerformanceCustom"], "Smile")
        self.assertNotIn("scenes", saved["settings"])
        for key in presets.BASELINE:
            self.assertEqual(saved["settings"][key], "")
        self.assertEqual(saved["settings"]["customCameraFlowSequence"], [{"shot": "medium shot", "camera": "pan"}])
        self.assertEqual(presets.list_scene_defaults_presets(), [{"name": "My mix"}])

    def test_collision_and_overwrite(self):
        presets.save_scene_defaults_preset("A/B", {"cutFrequency": 1})
        with self.assertRaises(presets.SceneDefaultsPresetExistsError):
            presets.save_scene_defaults_preset("A/B", {"cutFrequency": 7})
        with self.assertRaises(ValueError):
            presets.save_scene_defaults_preset("A?B", {}, overwrite=True)
        presets.save_scene_defaults_preset("A/B", {"cutFrequency": 7}, overwrite=True)
        self.assertEqual(presets.load_scene_defaults_preset("A/B")["settings"]["cutFrequency"], 7)

    def test_names_cannot_escape_root_and_windows_devices_are_safe(self):
        for name in ("../../outside", "CON", "NUL", ".."):
            presets.save_scene_defaults_preset(name, {})
            target = os.path.abspath(presets._preset_path(name))
            self.assertEqual(os.path.commonpath([target, self.root]), self.root)
            self.assertTrue(os.path.isfile(target))
        with self.assertRaises(ValueError):
            presets.save_scene_defaults_preset(" ", {})

    def test_validation_and_corrupt_files(self):
        for value in (-1, 11, 2.5, True, "nan", "bad"):
            with self.assertRaises(ValueError):
                presets.save_scene_defaults_preset("Bad", {"cutFrequency": value})
        with self.assertRaises(ValueError):
            presets.save_scene_defaults_preset("Bad", {"cameraFlow": "custom", "customCameraFlowSequence": []})
        presets.atomic_write_json(os.path.join(self.root, "bad.json"), {"name": "Bad", "settings": []})
        self.assertEqual(presets.list_scene_defaults_presets(), [])

    def test_failed_atomic_write_keeps_previous_file(self):
        presets.save_scene_defaults_preset("Keep", {"cutFrequency": 1})
        with patch.object(presets, "atomic_write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                presets.save_scene_defaults_preset("Keep", {"cutFrequency": 7}, overwrite=True)
        self.assertEqual(presets.load_scene_defaults_preset("Keep")["settings"]["cutFrequency"], 1)


class SceneDefaultsPresetRoutesTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from aiohttp import web
        self.scope = {"asyncio": asyncio, "web": web,
                      "SceneDefaultsPresetExistsError": presets.SceneDefaultsPresetExistsError}
        tree = ast.parse((ROOT / "builder/routes.py").read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if isinstance(function, ast.AsyncFunctionDef) and function.name in {
                "vrgdg_list_scene_defaults_presets", "vrgdg_load_scene_defaults_preset", "vrgdg_save_scene_defaults_preset"
            }:
                function.decorator_list = []
                module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
                exec(compile(module, "builder/routes.py", "exec"), self.scope)

    async def test_list_load_save_envelopes_and_errors(self):
        import json

        class Request:
            async def json(self):
                return {"name": "Mix", "settings": {"cutFrequency": 4}, "overwrite": True}

        calls = []
        self.scope.update({
            "list_scene_defaults_presets": lambda: [{"name": "Mix"}],
            "load_scene_defaults_preset": lambda name: {"name": name, "settings": {}},
            "save_scene_defaults_preset": lambda name, settings, overwrite: calls.append(overwrite) or {"name": name, "settings": settings},
        })
        for verb in ("list", "load", "save"):
            response = await self.scope[f"vrgdg_{verb}_scene_defaults_preset{'s' if verb == 'list' else ''}"](Request())
            self.assertEqual(response.status, 200)
            self.assertTrue(json.loads(response.text)["ok"])
        self.assertEqual(calls, [True])

        def fail(*args):
            raise presets.SceneDefaultsPresetExistsError("exists")

        self.scope["save_scene_defaults_preset"] = fail
        response = await self.scope["vrgdg_save_scene_defaults_preset"](Request())
        self.assertEqual(response.status, 409)
        self.assertFalse(json.loads(response.text)["ok"])

        def missing(*args):
            raise FileNotFoundError()

        self.scope["load_scene_defaults_preset"] = missing
        response = await self.scope["vrgdg_load_scene_defaults_preset"](Request())
        self.assertEqual(response.status, 404)


if __name__ == "__main__":
    unittest.main()
