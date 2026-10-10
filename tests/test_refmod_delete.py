import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def _load():
    for name, path in (("vrgdg_del_test", ROOT), ("vrgdg_del_test.minimax", ROOT / "minimax")):
        package = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None, is_package=True))
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return importlib.import_module("vrgdg_del_test.minimax.refmod_library")


library = _load()


class DeleteRefModTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "refmods"
        (self.root / "identity").mkdir(parents=True)
        for name in ("brad", "darrel"):
            (self.root / "identity" / f"{name}.safetensors").write_bytes(b"x")
            (self.root / "identity" / f"{name}.preview.png").write_bytes(b"x")
        self.outside = Path(self._tmp.name) / "secret.safetensors"
        self.outside.write_bytes(b"x")

    def _entry(self, name, path, directory):
        return {"name": name, "path": path, "directory": directory}

    def _patched(self):
        directories = mock.patch.object(library, "_refmod_dirs", lambda: [str(self.root)])
        entries = mock.patch.object(library, "_entry", self._entry)
        return directories, entries

    def test_deletes_the_file_and_its_preview_only(self):
        directories, entries = self._patched()
        with directories, entries:
            result = library.delete_refmod("identity/brad")
        self.assertEqual(result, {"name": "identity/brad"})
        self.assertFalse((self.root / "identity" / "brad.safetensors").exists())
        self.assertFalse((self.root / "identity" / "brad.preview.png").exists())
        self.assertTrue((self.root / "identity" / "darrel.safetensors").exists())
        self.assertTrue((self.root / "identity" / "darrel.preview.png").exists())

    def test_deletes_a_refmod_that_has_no_preview(self):
        (self.root / "identity" / "brad.preview.png").unlink()
        directories, entries = self._patched()
        with directories, entries:
            library.delete_refmod("identity/brad")
        self.assertFalse((self.root / "identity" / "brad.safetensors").exists())

    def test_unknown_and_escaping_names_are_refused(self):
        directories, entries = self._patched()
        with directories, entries:
            for name in ("identity/missing", "", "../secret", "identity/../../secret", "/secret", "C:/secret"):
                with self.assertRaises(ValueError):
                    library.delete_refmod(name)
        self.assertTrue(self.outside.exists())

    def test_pack_mods_folder_is_protected(self):
        pack_mods = Path(self._tmp.name) / "ComfyUI-MiniMaxH3Mod" / "mods"
        (pack_mods / "identity").mkdir(parents=True)
        (pack_mods / "identity" / "stock.safetensors").write_bytes(b"x")
        with mock.patch.object(library, "_refmod_dirs", lambda: [str(pack_mods)]), mock.patch.object(library, "_entry", self._entry):
            with self.assertRaises(ValueError):
                library.delete_refmod("identity/stock")
        self.assertTrue((pack_mods / "identity" / "stock.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
