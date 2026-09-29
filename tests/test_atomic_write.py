import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vrgdg_atomic_write", ROOT / "core/atomic_write.py")
atomic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atomic)


class AtomicWriteTests(unittest.TestCase):
    def test_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "settings.json")
            atomic.atomic_write_json(path, {"models_root": "A"})
            atomic.atomic_write_json(path, {"models_root": "B"})
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"models_root": "B"})
            self.assertEqual(os.listdir(folder), ["settings.json"])

    def test_interrupted_write_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "storyboard.json")
            atomic.atomic_write_json(path, {"scenes": [1, 2, 3]})

            def interrupted(handle_fd):
                raise KeyboardInterrupt

            with patch.object(atomic.os, "fsync", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
                atomic.atomic_write_json(path, {"scenes": []})
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"scenes": [1, 2, 3]})
            self.assertEqual(os.listdir(folder), ["storyboard.json"])


if __name__ == "__main__":
    unittest.main()
