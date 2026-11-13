"""The Models dialog's allowlisted model installer: status, download, checks and cancel."""

import importlib
import io
import json
import re
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

downloads = importlib.import_module(f"{ROOT.name}.core.model_downloads")
MODELS_MJS = ROOT / "web" / "music_video_builder" / "models.mjs"


def safetensors_bytes(extra: int = 64) -> bytes:
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * extra


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, length=None, delay: float = 0.0):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if length is None else length)}
        self.delay = delay

    def read(self, size=-1):
        if self.delay:
            time.sleep(self.delay)
        return super().read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ModelDownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vrgdg_models_")
        self.addCleanup(self.temp.cleanup)
        self.loras = Path(self.temp.name) / "models" / "loras"
        patcher = patch.object(downloads, "_folder_dirs", side_effect=lambda folder: [str(self.loras)] if folder == "loras" else [])
        patcher.start()
        self.addCleanup(patcher.stop)
        with downloads._lock:
            downloads._jobs.clear()
        self.addCleanup(downloads._jobs.clear)

    def _wait(self, model_id: str, timeout: float = 10.0) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            with downloads._lock:
                job = dict(downloads._jobs[model_id])
            if job["state"] != "downloading":
                return job
            time.sleep(0.02)
        self.fail(f"{model_id} did not finish")

    def test_status_lists_every_allowlisted_file_missing_with_the_destination(self) -> None:
        status = {item["id"]: item for item in downloads.model_status()}
        self.assertEqual(set(status), {"dmad_lora", "dmad_refiner", "moss_turbo_lora", "moss_edit_lora"})
        for item in status.values():
            self.assertFalse(item["installed"])
            self.assertEqual(item["destination"], str(self.loras))
            self.assertEqual(item["job"]["state"], "idle")
        self.assertTrue(self.loras.is_dir(), "the destination folder is created when it is missing")

    def test_status_finds_a_file_in_a_subfolder(self) -> None:
        name = downloads.DOWNLOADABLE_MODELS["dmad_lora"]["filename"]
        (self.loras / "minimax").mkdir(parents=True)
        (self.loras / "minimax" / name).write_bytes(safetensors_bytes())
        item = next(entry for entry in downloads.model_status() if entry["id"] == "dmad_lora")
        self.assertTrue(item["installed"])
        self.assertEqual(Path(item["path"]).name, name)

    def test_install_downloads_checks_and_names_the_file(self) -> None:
        data = safetensors_bytes(4096)
        seen = []

        def fake_open(request, timeout):
            seen.append(request.full_url)
            return FakeResponse(data)

        with patch.object(downloads, "urlopen_verified", fake_open):
            self.assertEqual(downloads.start_install(["dmad_refiner"]), ["dmad_refiner"])
            job = self._wait("dmad_refiner")
        item = downloads.DOWNLOADABLE_MODELS["dmad_refiner"]
        self.assertEqual(job["state"], "done", job["error"])
        self.assertEqual(seen, [item["url"]])
        self.assertEqual((self.loras / item["filename"]).read_bytes(), data)
        self.assertEqual(job["downloaded"], len(data))
        self.assertEqual(list(self.loras.glob("*.part")), [])
        # Installed now, so a second request starts nothing.
        self.assertEqual(downloads.start_install(["dmad_refiner"]), [])

    def test_a_page_that_is_not_a_model_is_refused_and_leaves_nothing(self) -> None:
        with patch.object(downloads, "urlopen_verified", lambda request, timeout: FakeResponse(b"<html>login</html>")):
            downloads.start_install(["dmad_lora"])
            job = self._wait("dmad_lora")
        self.assertEqual(job["state"], "error")
        self.assertIn("not a valid .safetensors", job["error"])
        self.assertEqual(list(self.loras.iterdir()), [])

    def test_a_short_download_is_an_error(self) -> None:
        data = safetensors_bytes()
        with patch.object(downloads, "urlopen_verified", lambda request, timeout: FakeResponse(data, length=len(data) + 100)):
            downloads.start_install(["dmad_lora"])
            job = self._wait("dmad_lora")
        self.assertEqual(job["state"], "error")
        self.assertIn("stopped early", job["error"])
        self.assertEqual(list(self.loras.iterdir()), [])

    def test_not_enough_disk_space_is_reported_before_downloading(self) -> None:
        class Usage:
            free = 1024

        with patch.object(downloads, "urlopen_verified", lambda request, timeout: FakeResponse(safetensors_bytes())), \
                patch.object(downloads.shutil, "disk_usage", return_value=Usage()):
            downloads.start_install(["dmad_lora"])
            job = self._wait("dmad_lora")
        self.assertEqual(job["state"], "error")
        self.assertIn("free disk space", job["error"])

    def test_cancel_stops_the_download_and_removes_the_partial_file(self) -> None:
        data = safetensors_bytes(8 * 1024 * 1024)
        with patch.object(downloads, "urlopen_verified", lambda request, timeout: FakeResponse(data, delay=0.05)):
            downloads.start_install(["dmad_lora"])
            time.sleep(0.2)
            self.assertEqual(downloads.cancel_install(["dmad_lora"]), ["dmad_lora"])
            job = self._wait("dmad_lora")
        self.assertEqual(job["state"], "cancelled")
        self.assertEqual(list(self.loras.iterdir()), [])
        self.assertEqual(downloads.cancel_install(["dmad_lora"]), [])

    def test_only_allowlisted_ids_are_accepted(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            downloads.start_install(["../../evil", "dmad_lora"])
        self.assertEqual(downloads._jobs, {})

    def test_existing_files_are_never_replaced(self) -> None:
        item = downloads.DOWNLOADABLE_MODELS["dmad_lora"]
        self.loras.mkdir(parents=True)
        (self.loras / item["filename"]).write_bytes(b"mine")
        self.assertEqual(downloads.start_install(["dmad_lora"]), [])
        self.assertEqual((self.loras / item["filename"]).read_bytes(), b"mine")

    def test_the_edit_lora_is_saved_under_its_own_name_not_the_one_in_the_url(self) -> None:
        data = safetensors_bytes(4096)
        seen = []

        def fake_open(request, timeout):
            seen.append(request.full_url)
            return FakeResponse(data)

        with patch.object(downloads, "urlopen_verified", fake_open):
            self.assertEqual(downloads.start_install(["moss_edit_lora"]), ["moss_edit_lora"])
            job = self._wait("moss_edit_lora")
        self.assertEqual(job["state"], "done", job["error"])
        self.assertEqual(seen, ["https://huggingface.co/chinoll/Moss-Image-Edit/resolve/main/adapter_model.safetensors"])
        self.assertEqual((self.loras / "moss_image_edit_lora.safetensors").read_bytes(), data)
        self.assertFalse((self.loras / "adapter_model.safetensors").exists())
        self.assertEqual(list(self.loras.glob("*.part")), [])
        item = next(entry for entry in downloads.model_status() if entry["id"] == "moss_edit_lora")
        self.assertTrue(item["installed"], "the status finds it by the name it was saved as")
        self.assertEqual(downloads.start_install(["moss_edit_lora"]), [], "installed, so nothing starts")

    def test_the_turbo_lora_is_saved_under_its_own_name_not_lora_safetensors(self) -> None:
        data = safetensors_bytes(4096)
        seen = []

        def fake_open(request, timeout):
            seen.append(request.full_url)
            return FakeResponse(data)

        with patch.object(downloads, "urlopen_verified", fake_open):
            self.assertEqual(downloads.start_install(["moss_turbo_lora"]), ["moss_turbo_lora"])
            job = self._wait("moss_turbo_lora")
        self.assertEqual(job["state"], "done", job["error"])
        self.assertEqual(seen, ["https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/resolve/main/lora.safetensors"])
        self.assertEqual((self.loras / "moss_image_sft_8_steps_dmd_lora.safetensors").read_bytes(), data)
        self.assertFalse((self.loras / "lora.safetensors").exists())
        self.assertEqual(list(self.loras.glob("*.part")), [])
        item = next(entry for entry in downloads.model_status() if entry["id"] == "moss_turbo_lora")
        self.assertTrue(item["installed"], "the status finds it by the name it was saved as")
        # The two MOSS-Image LoRAs are different files with different names: one does not count as the other.
        edit = next(entry for entry in downloads.model_status() if entry["id"] == "moss_edit_lora")
        self.assertFalse(edit["installed"])

    def test_the_moss_lora_names_are_the_ones_the_moss_settings_look_for(self) -> None:
        settings = (ROOT / "web" / "music_video_builder" / "moss_settings.mjs").read_text(encoding="utf-8")
        for model_id, constant in (("moss_turbo_lora", "MOSS_TURBO_LORA"), ("moss_edit_lora", "MOSS_EDIT_LORA")):
            with self.subTest(model=model_id):
                wanted = re.search(rf'export const {constant} = "([^"]+)"', settings).group(1)
                self.assertEqual(downloads.DOWNLOADABLE_MODELS[model_id]["filename"], wanted)

    def test_the_urls_match_the_ones_the_dialog_shows(self) -> None:
        source = MODELS_MJS.read_text(encoding="utf-8")

        def block_of(name: str) -> str:
            block = source[source.index(f"export const {name}"):]
            return block[:block.index("];")]

        # The DMAD cards list the files with their ids. The MOSS-Image download list marks the one button that installs itself.
        blocks = {"dmad": block_of("DMAD_MODEL_FILES"), "moss": block_of("MOSS_IMAGE_MODEL_DOWNLOADS")}
        for model_id, item in downloads.DOWNLOADABLE_MODELS.items():
            with self.subTest(model=model_id):
                block = blocks[item["group"]]
                marker = f'id: "{model_id}"' if item["group"] == "dmad" else f'install: "{model_id}"'
                self.assertIn(marker, block)
                self.assertIn(item["url"], block)
                self.assertIn(item["filename"], block)
                self.assertTrue(item["url"].startswith("https://huggingface.co/") and "/resolve/" in item["url"])
        dmad_urls = re.findall(r'url: "([^"]+)"', blocks["dmad"])
        self.assertEqual(
            sorted(dmad_urls), sorted(item["url"] for item in downloads.DOWNLOADABLE_MODELS.values() if item["group"] == "dmad")
        )
        # Only the two LoRA buttons install themselves: every other MOSS-Image button still opens its page.
        self.assertEqual(re.findall(r'install: "([^"]+)"', blocks["moss"]), ["moss_turbo_lora", "moss_edit_lora"])

    def test_dmad_file_names_match_the_ui_settings(self) -> None:
        source = (ROOT / "web" / "music_video_builder" / "minimax_h3.mjs").read_text(encoding="utf-8")
        for model_id in ("dmad_lora", "dmad_refiner"):
            self.assertIn(f'"{downloads.DOWNLOADABLE_MODELS[model_id]["filename"]}"', source)


if __name__ == "__main__":
    unittest.main()
