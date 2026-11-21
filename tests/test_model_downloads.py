"""The Models dialog's allowlisted model installer: status, download, checks and cancel."""

import importlib
import http.server
import io
import json
import re
import struct
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

downloads = importlib.import_module(f"{ROOT.name}.core.model_downloads")
MODELS_MJS = ROOT / "web" / "music_video_builder" / "models.mjs"
# The real folder lookup, before a test replaces it.
REAL_FOLDER_DIRS = downloads._folder_dirs
MODEL_FOLDERS = {"loras", "diffusion_models", "text_encoders", "vae", "latent_upscale_models", "LLM"}


def gguf_bytes(extra: int = 64) -> bytes:
    return b"GGUF" + b"\x03\x00\x00\x00" + b"\0" * extra


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
        self.models = Path(self.temp.name) / "models"
        self.loras = self.models / "loras"
        patcher = patch.object(downloads, "_folder_dirs", side_effect=lambda folder: [str(self.models / folder)])
        patcher.start()
        self.addCleanup(patcher.stop)
        # No Hugging Face token is saved, whatever this computer has: a test that needs one says so.
        no_token = patch.object(downloads, "load_token", return_value="")
        no_token.start()
        self.addCleanup(no_token.stop)
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
        self.assertEqual(set(status), set(downloads.DOWNLOADABLE_MODELS))
        self.assertTrue({"dmad_lora", "dmad_refiner", "moss_turbo_lora", "moss_edit_lora"} <= set(status))
        for item in status.values():
            self.assertFalse(item["installed"])
            self.assertEqual(item["destination"], str(self.models / item["folder"]))
            self.assertEqual(item["job"]["state"], "idle")
        self.assertTrue(self.loras.is_dir(), "the destination folder is created when it is missing")

    def test_status_tells_the_dialog_where_each_file_belongs(self) -> None:
        status = {item["id"]: item for item in downloads.model_status()}
        item = status["qwen_mmproj"]
        self.assertEqual((item["category"], item["groups"], item["folder"], item["format"], item["default"]), ("llm", ["Qwen"], "LLM", "gguf", True))
        self.assertEqual(item["filename"], "qwen-mmproj-BF16.gguf")
        self.assertGreater(item["download_size"], 0)
        shared = status["qwen_3_4b"]
        self.assertEqual(shared["groups"], ["ZImage", "Flux/Klein 4B"], "a file two cards share is one entry")
        self.assertFalse(shared["default"], "nothing in the image models is part of a recommended set")
        self.assertTrue(status["flux_klein_9b_diffusion"]["gated"])

    def test_the_catalog_is_sound(self) -> None:
        seen = set()
        for model_id, item in downloads.DOWNLOADABLE_MODELS.items():
            with self.subTest(model=model_id):
                self.assertIn(item["folder"], MODEL_FOLDERS)
                self.assertIn(item["category"], {"minimax", "image", "llm", "dmad"})
                self.assertTrue(item["groups"] and item["label"] and item["page"].startswith("https://huggingface.co/"))
                self.assertTrue(item["url"].startswith("https://huggingface.co/") and "/resolve/" in item["url"])
                self.assertIn(item["format"], {"safetensors", "gguf"})
                self.assertTrue(item["filename"].endswith(".gguf" if item["format"] == "gguf" else ".safetensors"))
                self.assertEqual(item["folder"] == "LLM", item["format"] == "gguf", "GGUF files go to models/LLM and only they do")
                self.assertGreater(item["download_size"], 1_000_000)
                self.assertEqual(Path(item["filename"]).name, item["filename"], "a file name, never a path")
                # Two entries saved as the same file would make one count as the other.
                key = (item["folder"], item["filename"])
                self.assertNotIn(key, seen)
                seen.add(key)

    def test_what_the_builder_selects_by_default_is_in_the_recommended_set(self) -> None:
        defaults = json.loads((ROOT / "minimax" / "h3_settings_defaults.json").read_text(encoding="utf-8"))
        recommended = {item["filename"] for item in downloads.DOWNLOADABLE_MODELS.values() if item["default"] and item["category"] == "minimax"}
        for key in ("diffusion_model_name", "clip_name", "video_vae_name", "audio_vae_name", "two_pass_lora_name", "three_pass_lightx_lora_name",
                    "two_pass_latent_upscaler_name"):
            with self.subTest(setting=key):
                self.assertIn(defaults[key], recommended)

    def test_a_gated_file_cannot_be_started_without_a_saved_token_and_the_status_says_what_is_missing(self) -> None:
        with self.assertRaisesRegex(ValueError, "login.*box at the top"):
            downloads.start_install(["flux_klein_9b_diffusion"])
        self.assertEqual(downloads._jobs, {})
        rows = {item["id"]: item for item in downloads.model_status()}
        self.assertEqual((rows["flux_klein_9b_diffusion"]["gated"], rows["flux_klein_9b_diffusion"]["token_saved"]), (True, False))
        self.assertIn("box at the top of this tab", rows["flux_klein_9b_diffusion"]["gated_message"])
        self.assertIsNone(rows["dmad_lora"]["gated_message"], "a file that is not gated has nothing missing")
        with patch.object(downloads, "load_token", return_value="hf_secret"):
            rows = {item["id"]: item for item in downloads.model_status()}
            ready = rows["flux_klein_9b_diffusion"]
            self.assertEqual((ready["gated"], ready["token_saved"], ready["gated_message"]), (True, True, None))
            self.assertNotIn("hf_secret", json.dumps(list(rows.values())), "the status never carries the token")

    def test_a_download_sends_the_token_as_a_bearer_header_and_without_one_sends_none(self) -> None:
        headers = []

        def fake_open(request, timeout):
            headers.append(request.get_header("Authorization"))
            return FakeResponse(safetensors_bytes(2048))

        with patch.object(downloads, "load_token", return_value="hf_secret"), patch.object(downloads, "urlopen_verified", fake_open):
            self.assertEqual(downloads.start_install(["flux_klein_9b_diffusion"]), ["flux_klein_9b_diffusion"])
            job = self._wait("flux_klein_9b_diffusion")
        self.assertEqual(job["state"], "done", job["error"])
        self.assertEqual(headers, ["Bearer hf_secret"])
        self.assertTrue((self.models / "diffusion_models" / downloads.DOWNLOADABLE_MODELS["flux_klein_9b_diffusion"]["filename"]).is_file())
        with patch.object(downloads, "urlopen_verified", fake_open):
            downloads.start_install(["dmad_lora"])
            self.assertEqual(self._wait("dmad_lora")["state"], "done")
        self.assertEqual(headers[1:], [None], "no token saved, no Authorization header")

    def test_the_token_goes_to_huggingface_co_and_nowhere_else(self) -> None:
        token_for = downloads._token_for
        self.assertEqual(token_for("https://huggingface.co/a/b/resolve/main/c.safetensors", "hf_abc"), "hf_abc")
        self.assertEqual(token_for("https://huggingface.co/a/b/resolve/main/c.safetensors", ""), "", "no token saved")
        for url in ("https://example.com/huggingface.co/x", "https://huggingface.co.evil.test/x", "http://huggingface.co/x", "https://huggingface.co@evil.test/x"):
            with self.subTest(url=url):
                self.assertEqual(token_for(url, "hf_abc"), "")
        # Every address in the catalog is on huggingface.co, so a saved token goes with all of them.
        self.assertTrue(all(token_for(item["url"], "hf_abc") for item in downloads.DOWNLOADABLE_MODELS.values()))

    def test_a_gated_file_whose_token_is_refused_says_to_accept_the_licence(self) -> None:
        for code in (401, 403):
            headers = []

            def refuse(request, timeout, code=code):
                headers.append(request.get_header("Authorization"))
                raise urllib.error.HTTPError(request.full_url, code, "Refused", {}, None)

            with self.subTest(code=code), patch.object(downloads, "load_token", return_value="hf_secret"), patch.object(downloads, "urlopen_verified", refuse):
                downloads.start_install(["flux_klein_9b_diffusion"])
                job = self._wait("flux_klein_9b_diffusion")
                self.assertEqual(job["state"], "error")
                self.assertIn("refused your token", job["error"])
                self.assertIn("licence", job["error"])
                self.assertNotIn("hf_secret", job["error"], "the token is never in a message")
                self.assertEqual(headers, ["Bearer hf_secret"], "no second try for a gated file")
                self.assertEqual(list((self.models / "diffusion_models").iterdir()), [])

    def test_a_token_hugging_face_does_not_accept_does_not_stop_a_public_file(self) -> None:
        headers = []

        def open_file(request, timeout):
            headers.append(request.get_header("Authorization"))
            if request.get_header("Authorization"):
                raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)
            return FakeResponse(safetensors_bytes(1024))

        with patch.object(downloads, "load_token", return_value="hf_revoked"), patch.object(downloads, "urlopen_verified", open_file):
            downloads.start_install(["dmad_lora"])
            job = self._wait("dmad_lora")
        self.assertEqual(job["state"], "done", job["error"])
        self.assertEqual(headers, ["Bearer hf_revoked", None], "the second try is anonymous")

    def test_the_token_is_not_copied_to_the_server_a_redirect_leads_to(self) -> None:
        # Hugging Face sends the file itself to a storage server of its own. A real redirect between two local servers shows what each one gets.
        seen = {"origin": [], "storage": []}

        class Storage(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen["storage"].append(self.headers.get("Authorization"))
                self.send_response(200)
                self.send_header("Content-Length", "4")
                self.end_headers()
                self.wfile.write(b"file")

            def log_message(self, *args):
                pass

        storage = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Storage)

        class Origin(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen["origin"].append(self.headers.get("Authorization"))
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{storage.server_address[1]}/storage/file")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        origin = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Origin)
        for server in (storage, origin):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        item = {"url": f"http://127.0.0.1:{origin.server_address[1]}/org/repo/resolve/main/file.safetensors", "gated": True}
        with downloads._open_download(item, "hf_secret") as response:
            self.assertEqual(response.read(), b"file")
        self.assertEqual(seen["origin"], ["Bearer hf_secret"])
        self.assertEqual(seen["storage"], [None])

    def test_a_model_goes_to_diffusion_models_not_the_legacy_unet_folder_comfyui_lists_first(self) -> None:
        unet, current = self.models / "unet", self.models / "diffusion_models"
        with patch.object(downloads, "_folder_dirs", lambda folder: [str(unet), str(current)] if folder == "diffusion_models" else [str(self.models / folder)]):
            self.assertEqual(downloads.destination_dir("diffusion_models"), str(current))
            self.assertEqual(downloads.destination_dir("loras"), str(self.loras), "a folder with no legacy name is used as it is")
        with patch.object(downloads, "_folder_dirs", lambda folder: [str(unet)]):
            self.assertEqual(downloads.destination_dir("diffusion_models"), str(unet), "the legacy folder when it is the only one")
        self.assertEqual(downloads._preferred_dir("text_encoders", [str(self.models / "clip"), str(self.models / "text_encoders")]), str(self.models / "text_encoders"))

    def test_the_llm_folder_falls_back_to_models_llm(self) -> None:
        with patch.object(downloads.folder_paths, "get_folder_paths", side_effect=KeyError("LLM")), \
                patch.object(downloads.folder_paths, "models_dir", str(self.models), create=True), \
                patch.object(downloads, "register_custom_model_root", lambda: []):
            self.assertEqual(REAL_FOLDER_DIRS("LLM"), [str(self.models / "LLM")])
            self.assertEqual(REAL_FOLDER_DIRS("vae"), [], "only the LLM folder has a fallback")

    def test_files_for_different_folders_go_to_their_own_folder(self) -> None:
        by_url = {
            downloads.DOWNLOADABLE_MODELS["minimax_h3_audio_vae"]["url"]: safetensors_bytes(2048),
            downloads.DOWNLOADABLE_MODELS["qwen_mmproj"]["url"]: gguf_bytes(2048),
            downloads.DOWNLOADABLE_MODELS["minimax_h3_latent_upscaler_3d"]["url"]: safetensors_bytes(1024),
        }
        with patch.object(downloads, "urlopen_verified", lambda request, timeout: FakeResponse(by_url[request.full_url])):
            ids = ["minimax_h3_audio_vae", "qwen_mmproj", "minimax_h3_latent_upscaler_3d"]
            self.assertEqual(downloads.start_install(ids), ids)
            for model_id in ids:
                self.assertEqual(self._wait(model_id)["state"], "done", model_id)
        self.assertTrue((self.models / "vae" / "minimax_h3_audio_vae_fp32.safetensors").is_file())
        self.assertTrue((self.models / "latent_upscale_models" / "minimax_h3_latent_upscaler_3d_bf16.safetensors").is_file())
        # Gemma's and Qwen's projector have one name on Hugging Face. Qwen's is saved under its own, so both can be installed.
        self.assertTrue((self.models / "LLM" / "qwen-mmproj-BF16.gguf").is_file())
        self.assertFalse((self.models / "LLM" / "mmproj-BF16.gguf").exists())
        installed = {item["id"] for item in downloads.model_status() if item["installed"]}
        self.assertEqual(installed, set(ids))

    def test_a_gguf_download_is_checked_for_the_gguf_header(self) -> None:
        with patch.object(downloads, "urlopen_verified", lambda request, timeout: FakeResponse(safetensors_bytes(256))):
            downloads.start_install(["supergemma_gguf"])
            job = self._wait("supergemma_gguf")
        self.assertEqual(job["state"], "error")
        self.assertIn("not a valid .gguf", job["error"])
        self.assertEqual(list((self.models / "LLM").iterdir()), [])

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
        legacy = {model_id: item for model_id, item in downloads.DOWNLOADABLE_MODELS.items() if item["group"] in blocks}
        self.assertEqual(set(legacy), {"dmad_lora", "dmad_refiner", "moss_turbo_lora", "moss_edit_lora"})
        for model_id, item in legacy.items():
            with self.subTest(model=model_id):
                block = blocks[item["group"]]
                marker = f'id: "{model_id}"' if item["group"] == "dmad" else f'install: "{model_id}"'
                self.assertIn(marker, block)
                self.assertIn(item["url"], block)
                self.assertIn(item["filename"], block)
                self.assertTrue(item["url"].startswith("https://huggingface.co/") and "/resolve/" in item["url"])
        dmad_urls = re.findall(r'url: "([^"]+)"', blocks["dmad"])
        self.assertEqual(
            sorted(dmad_urls), sorted(item["url"] for item in legacy.values() if item["group"] == "dmad")
        )
        # Only the two LoRA buttons install themselves: every other MOSS-Image button still opens its page.
        self.assertEqual(re.findall(r'install: "([^"]+)"', blocks["moss"]), ["moss_turbo_lora", "moss_edit_lora"])

    def test_dmad_file_names_match_the_ui_settings(self) -> None:
        source = (ROOT / "web" / "music_video_builder" / "minimax_h3.mjs").read_text(encoding="utf-8")
        for model_id in ("dmad_lora", "dmad_refiner"):
            self.assertIn(f'"{downloads.DOWNLOADABLE_MODELS[model_id]["filename"]}"', source)


if __name__ == "__main__":
    unittest.main()
