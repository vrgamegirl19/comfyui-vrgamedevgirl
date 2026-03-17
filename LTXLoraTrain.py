import os
import re
import shutil
import subprocess
import time
import gc
import math
import sys
import webbrowser
from datetime import datetime

import comfy
import torch
import cv2
import numpy as np
import folder_paths
from aiohttp import web
from server import PromptServer


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

_VRGDG_TENSORBOARD_ROUTE_REGISTERED = False
_VRGDG_TENSORBOARD_RUNS = {}


def _ensure_tensorboard_route_registered():
    global _VRGDG_TENSORBOARD_ROUTE_REGISTERED
    if _VRGDG_TENSORBOARD_ROUTE_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    @server_instance.routes.post("/vrgdg/ltx/tensorboard/open")
    async def vrgdg_open_tensorboard(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        workspace_dir = os.path.normpath(str(payload.get("workspace_dir", "") or "").strip())
        port = int(payload.get("port", 6006) or 6006)
        if not workspace_dir:
            return web.json_response({"ok": False, "error": "workspace_dir is required."}, status=400)

        logs_dir = os.path.join(workspace_dir, "logs")
        if not os.path.isdir(logs_dir):
            return web.json_response(
                {"ok": False, "error": f"Logs folder does not exist: {logs_dir}"},
                status=400,
            )

        key = os.path.normcase(logs_dir)
        url = f"http://127.0.0.1:{port}"
        existing = _VRGDG_TENSORBOARD_RUNS.get(key)
        if existing:
            process = existing.get("process")
            if process is not None and process.poll() is None:
                webbrowser.open(url)
                return web.json_response({"ok": True, "url": url, "reused": True})

        candidate_commands = [
            [sys.executable, "-m", "tensorboard.main", "--logdir", logs_dir, "--host", "127.0.0.1", "--port", str(port)],
            ["tensorboard", "--logdir", logs_dir, "--host", "127.0.0.1", "--port", str(port)],
        ]

        last_error = None
        launched_process = None
        for command in candidate_commands:
            try:
                launched_process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                break
            except Exception as exc:
                last_error = exc

        if launched_process is None:
            return web.json_response(
                {
                    "ok": False,
                    "error": f"Failed to start TensorBoard: {last_error}",
                },
                status=500,
            )

        _VRGDG_TENSORBOARD_RUNS[key] = {
            "process": launched_process,
            "url": url,
            "logs_dir": logs_dir,
        }
        time.sleep(1.0)
        webbrowser.open(url)
        return web.json_response({"ok": True, "url": url, "reused": False})

    _VRGDG_TENSORBOARD_ROUTE_REGISTERED = True


class VRGDG_LTXLoraTrainChunk:
    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
    }

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "model",
        "latest_state_path",
        "log_path",
        "video_filename_prefix",
        "output_name",
        "completed_steps",
        "total_target_steps",
    )
    FUNCTION = "run"
    CATEGORY = "VRGDG/Training"
    DESCRIPTION = (
        "Runs one LTX-2 LoRA training chunk using musubi-tuner, optionally caches if needed, "
        "and can export the latest Comfy-compatible LoRA for downstream preview generation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Base model to return downstream with the latest trained LoRA optionally applied."
                }),
                "dataset_images_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Folder containing your training images, or a parent folder that will be organized into an images subfolder."
                }),
                "workspace_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Working folder for cache, logs, config files, checkpoints, and training state."
                }),
                "run_name": ("STRING", {
                    "default": "LTXChunkRun",
                    "multiline": False,
                    "tooltip": "Name prefix used for the log file."
                }),
                "output_name": ("STRING", {
                    "default": "LTXChunkRun",
                    "multiline": False,
                    "tooltip": "Name prefix used for saved LoRA files and state folders."
                }),
                "resolution_width": ("INT", {
                    "default": 1920, "min": 64, "max": 8192, "step": 1,
                    "tooltip": "Training bucket width written to the musubi dataset config. Examples: 960 for lighter tests, 1280 for medium runs, 1920 for full HD style training."
                }),
                "resolution_height": ("INT", {
                    "default": 1080, "min": 64, "max": 8192, "step": 1,
                    "tooltip": "Training bucket height written to the musubi dataset config. Examples: 540 for lighter tests, 720 for medium runs, 1080 for full HD style training."
                }),
                "steps_per_run": ("INT", {
                    "default": 250, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "How many steps to train per run, and also when to save the LoRA/state at the end of that run. Examples: 50 for quick tests, 250 for normal preview cadence, 500 for longer chunks."
                }),
                "total_target_steps": ("INT", {
                    "default": 3000, "min": 1, "max": 1000000, "step": 1,
                    "tooltip": "Training stops once the latest saved step reaches this total. Examples: 1000 for a short experiment, 3000 for a normal run, 6000+ for longer training."
                }),
                "network_dim": ("INT", {
                    "default": 64, "min": 1, "max": 2048, "step": 1,
                    "tooltip": "LoRA rank. Higher values increase capacity and VRAM usage. Examples: 16 for very small tests, 32 for lighter runs, 64 as a common default, 128 for larger higher-capacity LoRAs."
                }),
                "network_alpha": ("INT", {
                    "default": 32, "min": 1, "max": 2048, "step": 1,
                    "tooltip": "LoRA alpha scaling value. A common pairing is alpha at half the rank. Examples: rank 16 -> alpha 8, rank 32 -> alpha 16, rank 64 -> alpha 32."
                }),
                "blocks_to_swap": ("INT", {
                    "default": 4, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Higher values reduce VRAM usage but usually slow training. Use 0 to disable block swapping. Examples: 0 for max speed if VRAM is sufficient, 4 as a balanced default, 8 to 12 for lower VRAM cards."
                }),
                "clear_memory_before_gemma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Tries to unload ComfyUI models and clear VRAM/RAM before Gemma text encoder caching. Keep enabled if stage 2 tends to stall."
                }),
                "learning_rate_preset": ([
                    "Custom",
                    "1e-4",
                    "7e-5",
                    "5e-5",
                    "3e-5",
                    "1e-5",
                ], {
                    "default": "7e-5",
                    "tooltip": "Quick preset for the training learning rate. Examples: 1e-4 for aggressive training, 7e-5 as a common default, 5e-5 or 3e-5 for gentler training. Choose Custom to use the float input below."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 7e-5, "min": 1e-8, "max": 1.0, "step": 1e-6,
                    "tooltip": "Custom learning rate used only when the preset is set to Custom. Examples: 0.0001 = 1e-4, 0.00007 = 7e-5, 0.00005 = 5e-5, 0.00003 = 3e-5."
                }),
                "num_repeats": ("INT", {
                    "default": 1, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "How many times each image-caption pair is repeated in the dataset. Examples: 1 for normal use, 2 to 4 if the dataset is very small, higher only when you intentionally want more repeats."
                }),
                "cache_strategy": (["auto", "force", "skip"], {
                    "default": "auto",
                    "tooltip": "Auto builds cache only when needed, Force always rebuilds it, Skip goes straight to training."
                }),
                "copy_latest_to_comfy_loras": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Copies the latest Comfy-compatible LoRA into the ComfyUI loras folder after training."
                }),
                "keep_only_comfy_lora": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, deletes the standard .safetensors LoRA files after a matching .comfy.safetensors file exists. Resume state folders are kept."
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Strength used if the node applies the latest LoRA back onto the output model. Examples: 1.0 for normal preview, 0.7 for a lighter effect, 0.0 to effectively disable applying the LoRA to the returned model."
                }),
                "create_captions": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, missing caption txt files are created automatically using the caption text input."
                }),
                "caption_text": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Base caption text used when create_captions is enabled and an image has no caption file. Example: woman portrait, cinematic close-up, soft natural light."
                }),
                "add_trigger_word": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, the trigger text is prepended to each caption."
                }),
                "trigger_text": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Trigger word or phrase to prepend to captions when add_trigger_word is enabled. Examples: miranda, my_character, retro-future heroine."
                }),
                "musubi_root": ("STRING", {
                    "default": "A:/MUSUBI/musubi-tuner-ltx2", "multiline": False,
                    "tooltip": "Root folder of your musubi-tuner-ltx2 install."
                }),
                "ltx2_checkpoint": (
                    "STRING",
                    {
                        "default": "A:/MUSUBI/models/ltx2/ltx-2.3-22b-dev.safetensors",
                        "multiline": False,
                        "tooltip": "Path to the base LTX-2 checkpoint used for caching and training."
                    },
                ),
                "gemma_root": ("STRING", {
                    "default": "A:/MUSUBI/models/gemma3", "multiline": False,
                    "tooltip": "Folder containing the Gemma model files used for text encoder caching."
                }),
            }
        }

    @staticmethod
    def _norm(path):
        return os.path.normpath(str(path or "").strip())

    @staticmethod
    def _safe_name(value, default_value):
        raw = str(value or "").strip() or default_value
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
        return cleaned.strip("._-") or default_value

    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _quote(path):
        return path.replace("\\", "/")

    @staticmethod
    def _parse_step(name):
        match = re.search(r"step(\d+)", name or "")
        return int(match.group(1)) if match else 0

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _print_stage_banner(self, log_handle, stage_number, total_stages, title, detail_lines=None):
        lines = [
            "",
            "=" * 78,
            f"[VRGDG] STAGE {stage_number}/{total_stages}: {title}",
        ]
        if detail_lines:
            lines.extend(f"[VRGDG] {line}" for line in detail_lines)
        lines.append("=" * 78)
        lines.append("")
        banner = "\n".join(lines)
        print(banner)
        log_handle.write(banner + "\n")
        log_handle.flush()

    def _run_stage_command(self, stage_number, total_stages, title, command, cwd, log_handle, detail_lines=None):
        self._print_stage_banner(log_handle, stage_number, total_stages, title, detail_lines)
        started_at = time.time()
        self._run_command(command, cwd, log_handle)
        elapsed = self._format_duration(time.time() - started_at)
        completion_line = f"[VRGDG] Completed stage {stage_number}/{total_stages}: {title} in {elapsed}"
        print(completion_line)
        log_handle.write(completion_line + "\n")
        log_handle.flush()

    def _count_dataset_files(self, images_dir):
        image_count = 0
        caption_count = 0
        if not os.path.isdir(images_dir):
            return image_count, caption_count
        for entry in os.scandir(images_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in self.IMAGE_EXTENSIONS:
                image_count += 1
            elif ext == ".txt":
                caption_count += 1
        return image_count, caption_count

    def _count_cache_files(self, cache_dir):
        if not os.path.isdir(cache_dir):
            return 0
        file_count = 0
        for _, _, files in os.walk(cache_dir):
            file_count += len(files)
        return file_count

    def _get_dataset_label(self, dataset_images_dir):
        dataset_images_dir = self._norm(dataset_images_dir)
        base_name = os.path.basename(dataset_images_dir)
        if base_name.lower() == "images":
            parent = os.path.dirname(dataset_images_dir)
            base_name = os.path.basename(parent) or base_name
        return self._safe_name(base_name, "dataset")

    def _get_or_create_video_output_subfolder(self, config_dir, dataset_images_dir):
        subfolder_file = os.path.join(config_dir, "video_output_subfolder.txt")
        if os.path.isfile(subfolder_file):
            with open(subfolder_file, "r", encoding="utf-8") as handle:
                existing_subfolder = handle.read().strip()
            if existing_subfolder:
                return existing_subfolder

        dataset_label = self._get_dataset_label(dataset_images_dir)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        relative_subfolder = f"{dataset_label}_{stamp}"

        with open(subfolder_file, "w", encoding="utf-8") as handle:
            handle.write(relative_subfolder)

        output_root = folder_paths.get_output_directory()
        os.makedirs(os.path.join(output_root, relative_subfolder), exist_ok=True)
        return relative_subfolder

    def _build_video_filename_prefix(self, video_output_subfolder, output_name, step_number):
        return f"{video_output_subfolder}/{self._safe_name(output_name, 'LTXChunkRun')}_step_{int(step_number)}"

    def _get_or_create_video_filename_prefix(self, config_dir, dataset_images_dir, output_name, step_number):
        prefix_file = os.path.join(config_dir, "video_filename_prefix.txt")
        current_prefix = self._build_video_filename_prefix(
            self._get_or_create_video_output_subfolder(config_dir, dataset_images_dir),
            output_name,
            step_number,
        )
        with open(prefix_file, "w", encoding="utf-8") as handle:
            handle.write(current_prefix)
        return current_prefix

    def _clear_memory_before_gemma(self, log_handle):
        messages = ["[VRGDG] Clearing ComfyUI and CUDA memory before Gemma stage."]
        try:
            comfy.model_management.unload_all_models()
            messages.append("[VRGDG] unload_all_models() completed.")
        except Exception as exc:
            messages.append(f"[VRGDG] unload_all_models() skipped: {exc}")

        try:
            comfy.model_management.cleanup_models()
            messages.append("[VRGDG] cleanup_models() completed.")
        except Exception as exc:
            messages.append(f"[VRGDG] cleanup_models() skipped: {exc}")

        try:
            comfy.model_management.soft_empty_cache(force=True)
            messages.append("[VRGDG] soft_empty_cache(force=True) completed.")
        except Exception as exc:
            messages.append(f"[VRGDG] soft_empty_cache(force=True) skipped: {exc}")

        gc.collect()
        messages.append("[VRGDG] Python garbage collection completed.")

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                messages.append("[VRGDG] torch.cuda.empty_cache() completed.")
            except Exception as exc:
                messages.append(f"[VRGDG] torch.cuda.empty_cache() skipped: {exc}")
            try:
                torch.cuda.ipc_collect()
                messages.append("[VRGDG] torch.cuda.ipc_collect() completed.")
            except Exception as exc:
                messages.append(f"[VRGDG] torch.cuda.ipc_collect() skipped: {exc}")

        for message in messages:
            print(message)
            log_handle.write(message + "\n")
        log_handle.flush()

    @staticmethod
    def _resolve_learning_rate(learning_rate_preset, learning_rate):
        preset = str(learning_rate_preset or "Custom").strip()
        if preset and preset != "Custom":
            return float(preset)
        return float(learning_rate)

    def _latest_state_dir(self, output_dir, output_name):
        if not os.path.isdir(output_dir):
            return "", 0
        prefix = f"{output_name}-step"
        candidates = []
        for entry in os.scandir(output_dir):
            if not entry.is_dir():
                continue
            if not entry.name.startswith(prefix) or not entry.name.endswith("-state"):
                continue
            step = self._parse_step(entry.name)
            candidates.append((step, entry.path))
        if not candidates:
            return "", 0
        step, path = max(candidates, key=lambda item: item[0])
        return os.path.normpath(path), step

    def _latest_file(self, output_dir, output_name, suffix):
        if not os.path.isdir(output_dir):
            return "", 0
        prefix = f"{output_name}-step"
        candidates = []
        for entry in os.scandir(output_dir):
            if not entry.is_file():
                continue
            if not entry.name.startswith(prefix) or not entry.name.endswith(suffix):
                continue
            if suffix == ".safetensors" and entry.name.endswith(".comfy.safetensors"):
                continue
            step = self._parse_step(entry.name)
            candidates.append((step, entry.path))
        if not candidates:
            return "", 0
        step, path = max(candidates, key=lambda item: item[0])
        return os.path.normpath(path), step

    def _write_dataset_config(self, path, dataset_images_dir, cache_dir, width, height, num_repeats):
        content = (
            "[general]\n"
            f"resolution = [{int(width)}, {int(height)}]\n"
            'caption_extension = ".txt"\n'
            "batch_size = 1\n"
            "enable_bucket = true\n"
            "bucket_no_upscale = false\n\n"
            "[[datasets]]\n"
            f'image_directory = "{self._quote(dataset_images_dir)}"\n'
            f'cache_directory = "{self._quote(cache_dir)}"\n'
            f"num_repeats = {int(num_repeats)}\n"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _write_training_config(
        self,
        path,
        dataset_config,
        checkpoint,
        gemma_root,
        output_dir,
        log_dir,
        output_name,
        network_dim,
        network_alpha,
        blocks_to_swap,
        learning_rate,
        max_train_steps,
        steps_per_run,
        total_target_steps,
    ):
        content = (
            "# Auto-generated by VRGDG LTX chunk trainer\n"
            f"# total_target_steps_from_workflow = {int(total_target_steps)}\n"
            f"# chunk_target_steps_this_run = {int(max_train_steps)}\n"
            f"# save_interval_per_run = {int(steps_per_run)}\n"
            f'ltx2_checkpoint = "{self._quote(checkpoint)}"\n'
            f'gemma_root = "{self._quote(gemma_root)}"\n'
            f'dataset_config = "{self._quote(dataset_config)}"\n\n'
            'ltx_mode = "video"\n'
            'ltx_version = "2.3"\n'
            'ltx_version_check_mode = "error"\n'
            'lora_target_preset = "full"\n\n'
            "cache_text_encoder_outputs = true\n"
            "cache_text_encoder_outputs_to_disk = false\n\n"
            "fp8_base = true\n"
            "fp8_scaled = true\n"
            "sdpa = true\n"
            "gradient_checkpointing = true\n"
            "gradient_accumulation_steps = 1\n"
            f"blocks_to_swap = {int(blocks_to_swap)}\n\n"
            'optimizer_type = "AdamW8Bit"\n'
            f"learning_rate = {learning_rate}\n"
            'lr_scheduler = "constant_with_warmup"\n'
            "lr_warmup_steps = 100\n\n"
            'network_module = "networks.lora_ltx2"\n'
            f"network_dim = {int(network_dim)}\n"
            f"network_alpha = {int(network_alpha)}\n"
            'timestep_sampling = "shifted_logit_normal"\n'
            "ltx2_first_frame_conditioning_p = 0.5\n\n"
            f'output_dir = "{self._quote(output_dir)}"\n'
            f'output_name = "{output_name}"\n'
            'log_with = "tensorboard"\n'
            f'logging_dir = "{self._quote(log_dir)}"\n'
            "log_config = true\n"
            f"max_train_steps = {int(max_train_steps)}\n"
            f"save_every_n_steps = {int(steps_per_run)}\n"
            'save_model_as = "safetensors"\n'
            'mixed_precision = "bf16"\n'
            "save_state = true\n"
            "save_state_on_train_end = true\n"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _resolve_musubi_executables(self, musubi_root):
        checked = []
        candidate_env_names = [".venv", "venv", "env"]

        for env_name in candidate_env_names:
            scripts_dir = os.path.join(musubi_root, env_name, "Scripts")
            python_candidate = os.path.join(scripts_dir, "python.exe")
            accelerate_candidate = os.path.join(scripts_dir, "accelerate.exe")
            checked.append((python_candidate, accelerate_candidate))
            if os.path.isfile(python_candidate) and os.path.isfile(accelerate_candidate):
                return os.path.normpath(python_candidate), os.path.normpath(accelerate_candidate), env_name

        current_python = os.path.normpath(sys.executable)
        current_scripts_dir = os.path.dirname(current_python)
        for accelerate_name in ("accelerate.exe", "accelerate"):
            accelerate_candidate = os.path.join(current_scripts_dir, accelerate_name)
            checked.append((current_python, accelerate_candidate))
            if os.path.isfile(current_python) and os.path.isfile(accelerate_candidate):
                return current_python, os.path.normpath(accelerate_candidate), "current_python_env"

        path_python = shutil.which("python")
        path_accelerate = shutil.which("accelerate")
        checked.append((path_python or "(python not found on PATH)", path_accelerate or "(accelerate not found on PATH)"))
        if path_python and path_accelerate:
            return os.path.normpath(path_python), os.path.normpath(path_accelerate), "PATH"

        checked_lines = "\n".join(
            f"  python={python_path}\n  accelerate={accelerate_path}"
            for python_path, accelerate_path in checked
        )
        raise ValueError(
            "Could not resolve musubi Python/accelerate executables.\n"
            "Checked these locations:\n"
            f"{checked_lines}\n"
            "Supported layouts include .venv, venv, env, or the current/PATH environment."
        )

    def _run_command(self, command, cwd, log_handle):
        command_line = f"$ {' '.join(command)}"
        log_handle.write(command_line + "\n")
        log_handle.flush()
        print(command_line)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")

    def _should_build_cache(self, cache_strategy, cache_dir):
        if cache_strategy == "force":
            return True
        if cache_strategy == "skip":
            return False
        if not os.path.isdir(cache_dir):
            return True
        return not any(os.scandir(cache_dir))

    def _export_latest_to_comfy(self, latest_comfy_lora, output_name):
        if not latest_comfy_lora or not os.path.isfile(latest_comfy_lora):
            return ""
        lora_dirs = folder_paths.get_folder_paths("loras")
        if not lora_dirs:
            raise RuntimeError("ComfyUI loras folder could not be resolved.")
        target_dir = lora_dirs[0]
        self._ensure_dir(target_dir)
        target_path = os.path.join(target_dir, f"{output_name}_latest.comfy.safetensors")
        shutil.copy2(latest_comfy_lora, target_path)
        return os.path.normpath(target_path)

    def _delete_standard_lora_files(self, output_dir, output_name):
        if not os.path.isdir(output_dir):
            return 0

        prefix = f"{output_name}-step"
        deleted_count = 0
        for entry in os.scandir(output_dir):
            if not entry.is_file():
                continue
            if not entry.name.startswith(prefix):
                continue
            if not entry.name.endswith(".safetensors"):
                continue
            if entry.name.endswith(".comfy.safetensors"):
                continue
            os.remove(entry.path)
            deleted_count += 1

        return deleted_count

    def _apply_lora_to_model(self, model, lora_path, strength_model):
        if not lora_path or not os.path.isfile(lora_path):
            return model

        strength = float(strength_model)
        if strength == 0:
            return model

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return model_lora

    def _log_message(self, message, log_path=None):
        print(message)
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
            except Exception:
                pass

    def _compose_caption_text(self, base_caption, add_trigger_word, trigger_text):
        caption = str(base_caption or "").strip()
        trigger = str(trigger_text or "").strip()
        if not add_trigger_word or not trigger:
            return caption
        if caption:
            if caption == trigger or caption.startswith(f"{trigger},") or caption.startswith(f"{trigger} "):
                return caption
            return f"{trigger}, {caption}"
        return trigger

    def _ensure_captions(self, images_dir, create_captions, caption_text, add_trigger_word, trigger_text):
        image_entries = [
            entry for entry in os.scandir(images_dir)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in self.IMAGE_EXTENSIONS
        ]
        base_caption = str(caption_text or "").strip()

        created_count = 0
        updated_count = 0
        for entry in image_entries:
            stem = os.path.splitext(entry.name)[0]
            caption_path = os.path.join(images_dir, f"{stem}.txt")

            existing_caption = ""
            if os.path.isfile(caption_path):
                with open(caption_path, "r", encoding="utf-8") as handle:
                    existing_caption = handle.read().strip()
            elif not create_captions:
                continue

            caption_body = existing_caption if existing_caption else base_caption
            final_caption = self._compose_caption_text(caption_body, add_trigger_word, trigger_text)

            if not existing_caption and not final_caption:
                continue
            if existing_caption == final_caption:
                continue

            with open(caption_path, "w", encoding="utf-8") as handle:
                handle.write(final_caption)

            if existing_caption:
                updated_count += 1
            else:
                created_count += 1

        print(
            f"[VRGDG] Caption prep complete. created={created_count} updated={updated_count} "
            f"trigger={'on' if add_trigger_word else 'off'}"
        )

    def _prepare_dataset_directory(
        self,
        dataset_root,
        create_captions,
        caption_text,
        add_trigger_word,
        trigger_text,
    ):
        dataset_root = self._norm(dataset_root)
        if not os.path.isdir(dataset_root):
            raise ValueError(f"dataset_images_dir does not exist: {dataset_root}")

        if os.path.basename(dataset_root).lower() == "images":
            self._ensure_captions(
                dataset_root,
                create_captions,
                caption_text,
                add_trigger_word,
                trigger_text,
            )
            return dataset_root

        images_dir = os.path.join(dataset_root, "images")
        if os.path.isdir(images_dir):
            self._ensure_captions(
                images_dir,
                create_captions,
                caption_text,
                add_trigger_word,
                trigger_text,
            )
            return os.path.normpath(images_dir)

        os.makedirs(images_dir, exist_ok=True)

        root_files = [
            entry for entry in os.scandir(dataset_root)
            if entry.is_file()
        ]
        image_stems = {
            os.path.splitext(entry.name)[0]
            for entry in root_files
            if os.path.splitext(entry.name)[1].lower() in self.IMAGE_EXTENSIONS
        }

        moved_count = 0
        for entry in root_files:
            ext = os.path.splitext(entry.name)[1].lower()
            stem = os.path.splitext(entry.name)[0]
            should_move = ext in self.IMAGE_EXTENSIONS or (ext == ".txt" and stem in image_stems)
            if not should_move:
                continue

            target_path = os.path.join(images_dir, entry.name)
            if os.path.exists(target_path):
                continue
            shutil.move(entry.path, target_path)
            moved_count += 1

        self._ensure_captions(
            images_dir,
            create_captions,
            caption_text,
            add_trigger_word,
            trigger_text,
        )
        print(f"[VRGDG] Dataset prep complete. Using images folder: {images_dir} (moved {moved_count} file(s))")
        return os.path.normpath(images_dir)

    def run(
        self,
        model,
        dataset_images_dir,
        workspace_dir,
        run_name,
        output_name,
        resolution_width,
        resolution_height,
        steps_per_run,
        total_target_steps,
        network_dim,
        network_alpha,
        blocks_to_swap,
        clear_memory_before_gemma,
        learning_rate_preset,
        learning_rate,
        num_repeats,
        cache_strategy,
        copy_latest_to_comfy_loras,
        keep_only_comfy_lora,
        strength_model,
        create_captions,
        caption_text,
        add_trigger_word,
        trigger_text,
        musubi_root,
        ltx2_checkpoint,
        gemma_root,
    ):
        dataset_images_dir = self._norm(dataset_images_dir)
        workspace_dir = self._norm(workspace_dir)
        musubi_root = self._norm(musubi_root)
        ltx2_checkpoint = self._norm(ltx2_checkpoint)
        gemma_root = self._norm(gemma_root)
        run_name = self._safe_name(run_name, "LTXChunkRun")
        output_name = self._safe_name(output_name, run_name)
        effective_learning_rate = self._resolve_learning_rate(learning_rate_preset, learning_rate)

        dataset_images_dir = self._prepare_dataset_directory(
            dataset_images_dir,
            create_captions,
            caption_text,
            add_trigger_word,
            trigger_text,
        )
        workspace_dir = self._ensure_dir(workspace_dir)
        if not os.path.isdir(musubi_root):
            raise ValueError(f"musubi_root does not exist: {musubi_root}")
        if not os.path.isfile(ltx2_checkpoint):
            raise ValueError(f"ltx2_checkpoint does not exist: {ltx2_checkpoint}")
        if not os.path.isdir(gemma_root):
            raise ValueError(f"gemma_root does not exist: {gemma_root}")

        python_exe, accelerate_exe, env_source = self._resolve_musubi_executables(musubi_root)

        cache_dir = self._ensure_dir(os.path.join(workspace_dir, "cache"))
        output_dir = self._ensure_dir(os.path.join(workspace_dir, "output"))
        logs_dir = self._ensure_dir(os.path.join(workspace_dir, "logs"))
        config_dir = self._ensure_dir(os.path.join(workspace_dir, "config"))
        dataset_config = os.path.join(config_dir, "dataset-01.toml")
        training_config = os.path.join(config_dir, "training_args.toml")

        latest_state_path, completed_steps = self._latest_state_dir(output_dir, output_name)
        if completed_steps >= int(total_target_steps):
            raise RuntimeError(
                f"Training complete: reached {completed_steps}/{int(total_target_steps)} steps. Stopping workflow."
            )

        next_target_steps = min(completed_steps + int(steps_per_run), int(total_target_steps))
        video_filename_prefix = self._get_or_create_video_filename_prefix(
            config_dir,
            dataset_images_dir,
            output_name,
            next_target_steps,
        )

        self._write_dataset_config(
            dataset_config,
            dataset_images_dir,
            cache_dir,
            resolution_width,
            resolution_height,
            num_repeats,
        )
        self._write_training_config(
            training_config,
            dataset_config,
            ltx2_checkpoint,
            gemma_root,
            output_dir,
            logs_dir,
            output_name,
            network_dim,
            network_alpha,
            blocks_to_swap,
            effective_learning_rate,
            next_target_steps,
            steps_per_run,
            total_target_steps,
        )

        log_path = os.path.join(
            logs_dir,
            f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        with open(log_path, "w", encoding="utf-8") as log_handle:
            log_handle.write(f"dataset_images_dir={dataset_images_dir}\n")
            log_handle.write(f"workspace_dir={workspace_dir}\n")
            log_handle.write(f"completed_steps={completed_steps}\n")
            log_handle.write(f"next_target_steps={next_target_steps}\n\n")
            log_handle.flush()

            image_count, caption_count = self._count_dataset_files(dataset_images_dir)
            cache_file_count_before = self._count_cache_files(cache_dir)
            should_build_cache = self._should_build_cache(cache_strategy, cache_dir)
            total_stages = 3 if should_build_cache else 1

            print(f"[VRGDG] dataset_images_dir={dataset_images_dir}")
            print(f"[VRGDG] workspace_dir={workspace_dir}")
            print(f"[VRGDG] video_filename_prefix={video_filename_prefix}")
            print(f"[VRGDG] completed_steps={completed_steps}")
            print(f"[VRGDG] next_target_steps={next_target_steps}")
            print(f"[VRGDG] steps_per_run_and_save={steps_per_run}")
            print(f"[VRGDG] total_target_steps={total_target_steps}")
            print(f"[VRGDG] blocks_to_swap={int(blocks_to_swap)}")
            print(f"[VRGDG] musubi_env_source={env_source}")
            print(f"[VRGDG] musubi_python={python_exe}")
            print(f"[VRGDG] musubi_accelerate={accelerate_exe}")
            print(f"[VRGDG] clear_memory_before_gemma={clear_memory_before_gemma}")
            print(f"[VRGDG] keep_only_comfy_lora={keep_only_comfy_lora}")
            print(
                f"[VRGDG] learning_rate={effective_learning_rate} "
                f"(preset={learning_rate_preset})"
            )
            print(f"[VRGDG] dataset summary: images={image_count} captions={caption_count}")
            print(
                f"[VRGDG] cache summary: strategy={cache_strategy} build_cache={'yes' if should_build_cache else 'no'} "
                f"existing_cache_files={cache_file_count_before}"
            )
            if latest_state_path:
                print(f"[VRGDG] resume state detected: {latest_state_path}")
            else:
                print("[VRGDG] resume state detected: none")

            if should_build_cache:
                self._run_stage_command(
                    1,
                    total_stages,
                    "Cache latents",
                    [
                        python_exe,
                        "ltx2_cache_latents.py",
                        "--dataset_config",
                        dataset_config,
                        "--ltx2_checkpoint",
                        ltx2_checkpoint,
                        "--device",
                        "cuda",
                        "--vae_dtype",
                        "bf16",
                        "--ltx2_mode",
                        "video",
                    ],
                    musubi_root,
                    log_handle,
                    [
                        f"Dataset images dir: {dataset_images_dir}",
                        f"Images found: {image_count}",
                        f"Captions found: {caption_count}",
                        f"Cache dir: {cache_dir}",
                    ],
                )
                if clear_memory_before_gemma:
                    self._clear_memory_before_gemma(log_handle)
                self._run_stage_command(
                    2,
                    total_stages,
                    "Cache text encoder outputs",
                    [
                        python_exe,
                        "ltx2_cache_text_encoder_outputs.py",
                        "--dataset_config",
                        dataset_config,
                        "--ltx2_checkpoint",
                        ltx2_checkpoint,
                        "--gemma_root",
                        gemma_root,
                        "--gemma_load_in_8bit",
                        "--device",
                        "cuda",
                        "--mixed_precision",
                        "bf16",
                        "--ltx2_mode",
                        "video",
                        "--batch_size",
                        "1",
                    ],
                    musubi_root,
                    log_handle,
                    [
                        f"Gemma root: {gemma_root}",
                        "This is usually the slowest setup stage.",
                        "You should see per-item progress from the text encoder cache script.",
                    ],
                )
                print(f"[VRGDG] cache summary after build: files={self._count_cache_files(cache_dir)}")
            else:
                self._print_stage_banner(
                    log_handle,
                    1,
                    total_stages,
                    "Skip cache build",
                    [
                        f"Cache strategy: {cache_strategy}",
                        f"Existing cache files: {cache_file_count_before}",
                        "Proceeding directly to training.",
                    ],
                )

            train_command = [
                accelerate_exe,
                "launch",
                "--num_cpu_threads_per_process",
                "1",
                "--mixed_precision",
                "bf16",
                "ltx2_train_network.py",
                "--config_file",
                training_config,
                "--ltx2_checkpoint",
                ltx2_checkpoint,
            ]
            if latest_state_path:
                train_command.extend(["--resume", latest_state_path])

            self._run_stage_command(
                total_stages,
                total_stages,
                "Train LoRA",
                train_command,
                musubi_root,
                log_handle,
                [
                    f"Output dir: {output_dir}",
                    f"Target steps this run: {completed_steps} -> {next_target_steps}",
                    f"Steps per run and save interval: {steps_per_run}",
                    f"Blocks to swap: {int(blocks_to_swap)}",
                    f"Learning rate: {effective_learning_rate}",
                ],
            )

        latest_lora_path, latest_lora_step = self._latest_file(output_dir, output_name, ".safetensors")
        latest_comfy_lora_path, latest_comfy_step = self._latest_file(
            output_dir, output_name, ".comfy.safetensors"
        )
        latest_state_path, latest_state_step = self._latest_state_dir(output_dir, output_name)

        completed_steps = max(latest_lora_step, latest_comfy_step, latest_state_step)
        if completed_steps < next_target_steps:
            raise RuntimeError(
                f"Training chunk did not produce the expected checkpoint. Expected step {next_target_steps}, got {completed_steps}."
            )

        print(
            f"[VRGDG] post-run summary: state_step={latest_state_step} "
            f"lora_step={latest_lora_step} comfy_lora_step={latest_comfy_step}"
        )

        if keep_only_comfy_lora and latest_comfy_lora_path:
            deleted_count = self._delete_standard_lora_files(output_dir, output_name)
            latest_lora_path = ""
            print(f"[VRGDG] Deleted {deleted_count} standard LoRA file(s); keeping only Comfy LoRA files.")

        if copy_latest_to_comfy_loras:
            latest_comfy_lora_path = self._export_latest_to_comfy(latest_comfy_lora_path, output_name)

        applied_lora_path = latest_comfy_lora_path if latest_comfy_lora_path else latest_lora_path
        self._log_message(
            f"[VRGDG] Latest state path selected: {os.path.normpath(latest_state_path) if latest_state_path else '(none)'}",
            log_path,
        )
        self._log_message(
            f"[VRGDG] Latest standard LoRA selected: {os.path.normpath(latest_lora_path) if latest_lora_path else '(none)'}",
            log_path,
        )
        self._log_message(
            f"[VRGDG] Latest Comfy LoRA selected: {os.path.normpath(latest_comfy_lora_path) if latest_comfy_lora_path else '(none)'}",
            log_path,
        )
        self._log_message(
            f"[VRGDG] Applying LoRA to returned MODEL: {os.path.normpath(applied_lora_path) if applied_lora_path else '(none)'} "
            f"with strength_model={float(strength_model)}",
            log_path,
        )
        output_model = self._apply_lora_to_model(model, applied_lora_path, strength_model)
        if applied_lora_path and os.path.isfile(applied_lora_path) and float(strength_model) != 0:
            self._log_message("[VRGDG] LoRA applied successfully to returned MODEL.", log_path)
        else:
            self._log_message("[VRGDG] Returned MODEL is unchanged (no LoRA file selected or strength_model is 0).", log_path)

        return (
            output_model,
            os.path.normpath(latest_state_path) if latest_state_path else "",
            os.path.normpath(log_path),
            video_filename_prefix,
            output_name,
            int(completed_steps),
            int(total_target_steps),
        )


class VRGDG_SpeedCharacterLoraTraining(VRGDG_LTXLoraTrainChunk):
    DESCRIPTION = (
        "Runs the LTX trainer with a fast character-LoRA preset using dynamic IMAGE and caption inputs."
    )
    MAX_IMAGE_SLOTS = 20
    PRESET_TRAINING_STEPS = 400
    PRESET_LEARNING_RATE = 0.0002
    PRESET_LORA_RANK = 16
    PRESET_LORA_ALPHA = 16
    PRESET_NUM_REPEATS = 1
    PRESET_COPY_LATEST = False
    PRESET_KEEP_ONLY_COMFY = True

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"image{i}": ("IMAGE", {
                "forceInput": True,
            })
            for i in range(1, cls.MAX_IMAGE_SLOTS + 1)
        }
        optional_inputs.update({
            f"caption_{i}": ("STRING", {
                "default": "",
                "multiline": False,
            })
            for i in range(1, cls.MAX_IMAGE_SLOTS + 1)
        })
        return {
            "required": {
                "model": ("MODEL",),
                "workspace_dir": ("STRING", {
                    "default": "A:/MUSUBI/Training/SpeedCharacterLoraTraining",
                    "multiline": False,
                    "tooltip": "Workspace folder for cache, output, logs, config, and the managed dynamic dataset."
                }),
                "run_name": ("STRING", {
                    "default": "SpeedCharacterLoraTrainingRun",
                    "multiline": False,
                    "tooltip": "Run name used for logs."
                }),
                "output_name": ("STRING", {
                    "default": "SpeedCharacterLoraTraining",
                    "multiline": False,
                    "tooltip": "LoRA output name used for checkpoints and downstream preview naming."
                }),
                "image_count": ("INT", {
                    "default": 4, "min": 1, "max": cls.MAX_IMAGE_SLOTS, "step": 1,
                    "tooltip": "How many dynamic image inputs and caption fields to show."
                }),
                "resolution_width": ("INT", {
                    "default": 1256, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Training bucket width. Pick the resolution preset you want to train at."
                }),
                "resolution_height": ("INT", {
                    "default": 1256, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Training bucket height. Pick the resolution preset you want to train at."
                }),
                "blocks_to_swap": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "How many transformer blocks to swap to CPU. 0 is fastest if VRAM allows it."
                }),
                "clear_memory_before_gemma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clears Comfy and CUDA memory before the Gemma cache stage."
                }),
                "cache_strategy": ([
                    "auto",
                    "force",
                    "skip",
                ], {
                    "default": "auto",
                    "tooltip": "Cache behavior. auto reuses cache when present, force rebuilds, skip bypasses cache creation."
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Strength used when applying the newest trained LoRA back onto the returned MODEL."
                }),
                "musubi_root": ("STRING", {
                    "default": "A:/MUSUBI/musubi-tuner",
                    "multiline": False,
                    "tooltip": "Root folder of your musubi install."
                }),
                "ltx2_checkpoint": ("STRING", {
                    "default": "A:/MUSUBI/models/ltx2/ltx-2.3-22b-dev.safetensors",
                    "multiline": False,
                    "tooltip": "Path to the LTX-2.3 DiT checkpoint."
                }),
                "gemma_root": ("STRING", {
                    "default": "A:/MUSUBI/models/gemma3",
                    "multiline": False,
                    "tooltip": "Path to the Gemma model root used by this preset."
                }),
            },
            "optional": optional_inputs,
        }

    def _extract_single_image_tensor(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value
            if tensor.ndim == 4:
                if int(tensor.shape[0]) <= 0:
                    return None
                return tensor[0]
            if tensor.ndim == 3:
                return tensor
            return None
        if isinstance(value, dict):
            for nested_value in value.values():
                tensor = self._extract_single_image_tensor(nested_value)
                if tensor is not None:
                    return tensor
            return None
        if isinstance(value, (list, tuple, set)):
            for nested_value in value:
                tensor = self._extract_single_image_tensor(nested_value)
                if tensor is not None:
                    return tensor
            return None
        return None

    def _save_dynamic_dataset_inputs(self, workspace_dir, image_count, kwargs):
        dataset_root = self._ensure_dir(os.path.join(workspace_dir, "dynamic_dataset"))
        images_dir = self._ensure_dir(os.path.join(dataset_root, "images"))

        for entry in os.scandir(images_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in self.IMAGE_EXTENSIONS or ext == ".txt":
                os.remove(entry.path)

        saved_count = 0
        for index in range(1, int(image_count) + 1):
            image_tensor = self._extract_single_image_tensor(kwargs.get(f"image{index}"))
            if image_tensor is None:
                continue

            image_array = image_tensor.detach().cpu().numpy()
            image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            stem = f"image{index:03d}"
            image_path = os.path.join(images_dir, f"{stem}.png")
            caption_path = os.path.join(images_dir, f"{stem}.txt")
            cv2.imwrite(image_path, image_bgr)

            caption_text = str(kwargs.get(f"caption_{index}", "") or "").strip()
            with open(caption_path, "w", encoding="utf-8") as handle:
                handle.write(caption_text)

            saved_count += 1

        if saved_count <= 0:
            raise ValueError("No connected images were found. Connect at least one image input.")

        print(f"[VRGDG] Prepared dynamic dataset with {saved_count} image-caption pair(s): {images_dir}")
        return os.path.normpath(images_dir)

    def run(
        self,
        model,
        workspace_dir,
        run_name,
        output_name,
        image_count,
        resolution_width,
        resolution_height,
        blocks_to_swap,
        clear_memory_before_gemma,
        cache_strategy,
        strength_model,
        musubi_root,
        ltx2_checkpoint,
        gemma_root,
        **kwargs,
    ):
        workspace_dir = self._norm(workspace_dir)
        managed_dataset_dir = self._save_dynamic_dataset_inputs(workspace_dir, image_count, kwargs)

        return super().run(
            model=model,
            dataset_images_dir=managed_dataset_dir,
            workspace_dir=workspace_dir,
            run_name=run_name,
            output_name=output_name,
            resolution_width=resolution_width,
            resolution_height=resolution_height,
            steps_per_run=self.PRESET_TRAINING_STEPS,
            total_target_steps=self.PRESET_TRAINING_STEPS,
            network_dim=self.PRESET_LORA_RANK,
            network_alpha=self.PRESET_LORA_ALPHA,
            blocks_to_swap=blocks_to_swap,
            clear_memory_before_gemma=clear_memory_before_gemma,
            learning_rate_preset="Custom",
            learning_rate=self.PRESET_LEARNING_RATE,
            num_repeats=self.PRESET_NUM_REPEATS,
            cache_strategy=cache_strategy,
            copy_latest_to_comfy_loras=self.PRESET_COPY_LATEST,
            keep_only_comfy_lora=self.PRESET_KEEP_ONLY_COMFY,
            strength_model=strength_model,
            create_captions=False,
            caption_text="",
            add_trigger_word=False,
            trigger_text="",
            musubi_root=musubi_root,
            ltx2_checkpoint=ltx2_checkpoint,
            gemma_root=gemma_root,
        )


class VRGDG_LTXPreviewXYZPlot:
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("xyz_video_path", "created", "status")
    FUNCTION = "run"
    CATEGORY = "VRGDG/Training"
    DESCRIPTION = (
        "Creates a final checkpoint comparison video from saved preview videos once training reaches the target step."
    )

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
    LABEL_BAND_HEIGHT = 40

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vhs_filenames": (any_typ, {
                    "forceInput": True,
                    "tooltip": "Trigger input from VideoHelperSuite Combine Video so this node runs after preview videos are written."
                }),
                "preview_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Folder containing the saved preview videos, or the VHS filename_prefix from the trainer node. If a filename_prefix is provided, this node automatically uses its parent output folder."
                }),
                "output_name": ("STRING", {
                    "default": "LTXChunkRun",
                    "multiline": False,
                    "tooltip": "LoRA/output name used to filter matching preview videos and to name the final XYZ video."
                }),
                "completed_steps": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1,
                    "tooltip": "Current completed training step from the trainer node."
                }),
                "total_target_steps": ("INT", {
                    "default": 3000, "min": 1, "max": 1000000, "step": 1,
                    "tooltip": "Final training target step. The XYZ compare video is only created when completed_steps reaches this value."
                }),
                "cell_width": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Width of each tile in the final comparison grid. Use 0 to auto-detect from the first preview video. Examples: 320 for many checkpoints, 512 for a balanced layout, 768 for larger tiles."
                }),
                "cell_height": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Height of each tile in the final comparison grid. Use 0 to auto-detect from the first preview video. If labels are enabled, the label band is added automatically."
                }),
                "label_tiles": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Adds a label band above each tile using the preview filename, which should include the LoRA name and step."
                }),
                "output_fps": ("INT", {
                    "default": 24, "min": 1, "max": 120, "step": 1,
                    "tooltip": "FPS for the final comparison video. Example: 24 for standard preview playback."
                }),
                "render_backend": ([
                    "CPU (libx264)",
                    "NVIDIA GPU (h264_nvenc)",
                ], {
                    "default": "NVIDIA GPU (h264_nvenc)",
                    "tooltip": "Final XYZ export backend. CPU uses libx264. NVIDIA GPU uses NVENC for faster encoding when ffmpeg has NVENC support. Note: the grid compose filters still run in ffmpeg, so GPU mode mainly speeds up the encode stage."
                }),
            }
        }

    @staticmethod
    def _norm(path):
        return os.path.normpath(str(path or "").strip())

    @staticmethod
    def _safe_name(value, default_value):
        raw = str(value or "").strip() or default_value
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
        return cleaned.strip("._-") or default_value

    @staticmethod
    def _parse_step_from_name(name):
        text = str(name or "")
        for pattern in (r"step[_-]?(\d+)", r"[_-](\d{3,})\b", r"(\d+)"):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return -1

    @staticmethod
    def _find_ffmpeg_path():
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return "ffmpeg"
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                import imageio_ffmpeg

                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"[VRGDG] Using fallback ffmpeg from imageio: {ffmpeg_path}")
                return ffmpeg_path
            except Exception as exc:
                raise RuntimeError(f"FFmpeg was not found: {exc}") from exc

    @staticmethod
    def _resolve_preview_folder(preview_folder):
        preview_folder = str(preview_folder or "").strip()
        if not preview_folder:
            raise ValueError("preview_folder is required.")
        if os.path.isabs(preview_folder):
            resolved = preview_folder
        else:
            resolved = os.path.join(folder_paths.get_output_directory(), preview_folder)
        resolved = os.path.normpath(resolved)
        if not os.path.isdir(resolved):
            parent = os.path.dirname(resolved)
            if parent and os.path.isdir(parent):
                return parent
            raise ValueError(f"preview_folder does not exist: {resolved}")
        return resolved

    def _find_matching_videos(self, preview_folder, output_name):
        output_name = self._safe_name(output_name, "LTXChunkRun")
        matches = []
        for entry in os.scandir(preview_folder):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in self.VIDEO_EXTENSIONS:
                continue
            if "_XYZ_COMPARE_" in entry.name.upper():
                continue
            if output_name.lower() not in entry.name.lower():
                continue
            step = self._parse_step_from_name(entry.name)
            matches.append((step, entry.stat().st_mtime, entry.path))
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        return [os.path.normpath(item[2]) for item in matches]

    @staticmethod
    def _get_unique_output_path(folder, filename):
        candidate = os.path.normpath(os.path.join(folder, filename))
        if not os.path.exists(candidate):
            return candidate

        stem, ext = os.path.splitext(filename)
        version = 2
        while True:
            candidate = os.path.normpath(os.path.join(folder, f"{stem}_v{version}{ext}"))
            if not os.path.exists(candidate):
                return candidate
            version += 1

    @staticmethod
    def _get_video_resolution(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for resolution probe: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Could not determine video resolution: {video_path}")
        return width, height

    def _resolve_cell_size(self, video_paths, cell_width, cell_height, label_tiles):
        cell_width = int(cell_width)
        cell_height = int(cell_height)
        if cell_width > 0 and cell_height > 0:
            return cell_width, cell_height

        video_width, video_height = self._get_video_resolution(video_paths[0])
        resolved_width = cell_width if cell_width > 0 else video_width
        if cell_height > 0:
            resolved_height = cell_height
        else:
            resolved_height = video_height + (self.LABEL_BAND_HEIGHT if label_tiles else 0)
        return int(resolved_width), int(resolved_height)

    @staticmethod
    def _choose_columns(item_count):
        if item_count <= 0:
            return 1
        return max(1, math.ceil(math.sqrt(item_count)))

    @staticmethod
    def _find_font_path():
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return ""

    @staticmethod
    def _escape_drawtext(value):
        text = str(value or "")
        text = text.replace("\\", "/")
        text = text.replace(":", r"\:")
        text = text.replace("'", r"\'")
        text = text.replace(",", r"\,")
        text = text.replace("[", r"\[")
        text = text.replace("]", r"\]")
        return text

    @staticmethod
    def _parse_custom_labels(custom_labels):
        text = str(custom_labels or "").strip()
        if not text:
            return []
        labels = []
        for raw_line in text.replace("|", "\n").splitlines():
            label = str(raw_line).strip()
            if label:
                labels.append(label)
        return labels

    def _resolve_tile_labels(self, video_paths, custom_labels=None):
        parsed_labels = self._parse_custom_labels(custom_labels)
        resolved_labels = []
        for index, path in enumerate(video_paths):
            if index < len(parsed_labels):
                resolved_labels.append(parsed_labels[index])
            else:
                resolved_labels.append(os.path.splitext(os.path.basename(path))[0])
        return resolved_labels

    def _build_filter_complex(self, video_paths, cell_width, cell_height, columns, label_tiles, output_fps, tile_labels=None):
        font_path = self._find_font_path()
        filter_parts = []
        resolved_labels = tile_labels or self._resolve_tile_labels(video_paths)
        for index, path in enumerate(video_paths):
            label = resolved_labels[index]
            label_band_height = self.LABEL_BAND_HEIGHT if label_tiles else 0
            video_height = max(16, int(cell_height) - int(label_band_height))
            pad_y = f"{label_band_height}+(oh-{label_band_height}-ih)/2"
            chain = (
                f"[{index}:v]fps={int(output_fps)},"
                f"scale={int(cell_width)}:{int(video_height)}:force_original_aspect_ratio=decrease,"
                f"pad={int(cell_width)}:{int(cell_height)}:(ow-iw)/2:{pad_y}:black,setsar=1"
            )
            if label_tiles:
                escaped_label = self._escape_drawtext(label)
                if font_path:
                    escaped_font = font_path.replace("\\", "/").replace(":", r"\:")
                    chain += (
                        f",drawtext=fontfile='{escaped_font}':text='{escaped_label}':"
                        f"x=(w-text_w)/2:y={(label_band_height - 24) // 2}:"
                        "fontcolor=white:fontsize=22"
                    )
                else:
                    chain += (
                        f",drawtext=text='{escaped_label}':"
                        f"x=(w-text_w)/2:y={(label_band_height - 24) // 2}:"
                        "fontcolor=white:fontsize=22"
                    )
            chain += f"[v{index}]"
            filter_parts.append(chain)

        rows = math.ceil(len(video_paths) / columns)
        layout_parts = []
        for index in range(len(video_paths)):
            col = index % columns
            row = index // columns
            layout_parts.append(f"{col * int(cell_width)}_{row * int(cell_height)}")

        stack_inputs = "".join(f"[v{index}]" for index in range(len(video_paths)))
        filter_parts.append(
            f"{stack_inputs}xstack=inputs={len(video_paths)}:layout={'|'.join(layout_parts)}[outv]"
        )
        return ";".join(filter_parts), rows

    @staticmethod
    def _build_encoder_args(render_backend):
        mode = str(render_backend or "").strip()
        if "NVIDIA" in mode:
            return (
                ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "18", "-pix_fmt", "yuv420p"],
                "NVIDIA GPU (h264_nvenc)",
            )
        return (
            ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"],
            "CPU (libx264)",
        )

    def run(
        self,
        vhs_filenames,
        preview_folder,
        output_name,
        completed_steps,
        total_target_steps,
        cell_width,
        cell_height,
        label_tiles,
        output_fps,
        render_backend,
    ):
        completed_steps = int(completed_steps)
        total_target_steps = int(total_target_steps)
        if completed_steps < total_target_steps:
            return (
                "",
                False,
                f"Skipped XYZ plot creation because training is not final yet: {completed_steps}/{total_target_steps}.",
            )

        preview_folder = self._resolve_preview_folder(preview_folder)
        output_name = self._safe_name(output_name, "LTXChunkRun")
        video_paths = self._find_matching_videos(preview_folder, output_name)
        if not video_paths:
            return (
                "",
                False,
                f"No preview videos found for '{output_name}' in {preview_folder}. "
                "You can pass either the preview folder or the trainer's VHS filename_prefix.",
            )

        ffmpeg_path = self._find_ffmpeg_path()
        cell_width, cell_height = self._resolve_cell_size(
            video_paths,
            cell_width,
            cell_height,
            bool(label_tiles),
        )
        columns = self._choose_columns(len(video_paths))
        filter_complex, rows = self._build_filter_complex(
            video_paths,
            int(cell_width),
            int(cell_height),
            int(columns),
            bool(label_tiles),
            int(output_fps),
        )

        output_filename = f"{output_name}_XYZ_COMPARE_step{completed_steps}.mp4"
        output_path = self._get_unique_output_path(preview_folder, output_filename)
        encoder_args, encoder_label = self._build_encoder_args(render_backend)

        command = [ffmpeg_path, "-y"]
        for path in video_paths:
            command.extend(["-i", path])
        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[outv]",
                "-r",
                str(int(output_fps)),
            ]
        )
        command.extend(encoder_args)
        command.extend(
            [
                output_path,
            ]
        )

        print(
            f"[VRGDG] Creating XYZ compare video from {len(video_paths)} preview(s) "
            f"using a {int(columns)}x{int(rows)} grid at {int(cell_width)}x{int(cell_height)} per tile "
            f"with {encoder_label}."
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"FFmpeg failed while creating XYZ compare video: {stderr}")

        return (
            output_path,
            True,
            f"Created XYZ compare video from {len(video_paths)} preview(s): {output_path}",
        )


class VRGDG_VideoFolderGridPlot(VRGDG_LTXPreviewXYZPlot):
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "filename_prefix", "output_fps", "status")
    FUNCTION = "run"
    CATEGORY = "VRGDG/Video"
    DESCRIPTION = (
        "Creates a simple labeled grid image sequence from videos in a folder or connected inputs."
    )
    MAX_VIDEO_SLOTS = 20

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"video{i}": ("IMAGE", {
                "forceInput": True,
            })
            for i in range(1, cls.MAX_VIDEO_SLOTS + 1)
        }
        optional_inputs.update({
            f"label_{i}": ("STRING", {
                "default": "",
                "multiline": False,
            })
            for i in range(1, cls.MAX_VIDEO_SLOTS + 1)
        })
        return {
            "required": {
                "video_folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Folder containing the videos to place into the grid. Leave this as the source, or connect explicit video inputs below if you want to include only selected videos."
                }),
                "output_name": ("STRING", {
                    "default": "VideoGrid",
                    "multiline": False,
                    "tooltip": "Base filename prefix to send downstream into a video combine node."
                }),
                "filename_prefix": ("STRING", {
                    "default": "VRGDG/VideoGrid",
                    "multiline": False,
                    "tooltip": "Filename prefix to pass downstream to a video combine node. Example: VRGDG/MyGrid or tests/compare_grid."
                }),
                "video_count": ("INT", {
                    "default": 4, "min": 1, "max": cls.MAX_VIDEO_SLOTS, "step": 1,
                    "tooltip": "How many explicit video input slots and matching label fields to show. If any connected video inputs are present, those are used instead of scanning the folder."
                }),
                "cell_width": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Width of each tile. Use 0 to auto-detect from the first video."
                }),
                "cell_height": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Height of each tile. Use 0 to auto-detect from the first video. If labels are enabled, the label band is added automatically."
                }),
                "label_tiles": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Adds a label above each tile using the video filename."
                }),
                "output_fps": ("INT", {
                    "default": 24, "min": 1, "max": 120, "step": 1,
                    "tooltip": "FPS to pass downstream to a video combine node."
                }),
            },
            "optional": optional_inputs,
        }

    def _find_all_videos(self, video_folder):
        matches = []
        for entry in os.scandir(video_folder):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in self.VIDEO_EXTENSIONS:
                continue
            if "_XYZ_COMPARE_" in entry.name.upper():
                continue
            if "_VIDEOGRID_" in entry.name.upper():
                continue
            matches.append((entry.name.lower(), entry.stat().st_mtime, entry.path))
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        return [os.path.normpath(item[2]) for item in matches]

    def _extract_image_batches_from_value(self, value):
        batches = []
        if value is None:
            return batches
        if isinstance(value, torch.Tensor):
            if value.ndim == 3:
                batches.append(value.unsqueeze(0))
            elif value.ndim == 4:
                batches.append(value)
            return batches
        if isinstance(value, dict):
            for nested_value in value.values():
                batches.extend(self._extract_image_batches_from_value(nested_value))
            return batches
        if isinstance(value, (list, tuple, set)):
            for nested_value in value:
                batches.extend(self._extract_image_batches_from_value(nested_value))
            return batches
        return batches

    def _collect_selected_image_batches(self, kwargs):
        selected = []
        for i in range(1, self.MAX_VIDEO_SLOTS + 1):
            key = f"video{i}"
            selected.extend(self._extract_image_batches_from_value(kwargs.get(key)))
        return selected

    def _collect_dynamic_labels(self, kwargs):
        labels = []
        for i in range(1, self.MAX_VIDEO_SLOTS + 1):
            label = str(kwargs.get(f"label_{i}", "") or "").strip()
            if label:
                labels.append(label)
            else:
                labels.append("")
        return labels

    def _resolve_dynamic_tile_labels(self, video_paths, kwargs):
        labels = self._collect_dynamic_labels(kwargs)
        resolved = []
        for index, path in enumerate(video_paths):
            custom = labels[index] if index < len(labels) else ""
            resolved.append(custom or os.path.splitext(os.path.basename(path))[0])
        return resolved

    def _resolve_dynamic_tensor_labels(self, image_batches, kwargs):
        labels = self._collect_dynamic_labels(kwargs)
        resolved = []
        for index in range(len(image_batches)):
            custom = labels[index] if index < len(labels) else ""
            resolved.append(custom or f"video{index + 1}")
        return resolved

    @staticmethod
    def _fit_frame_to_tile(frame_bgr, cell_width, cell_height, label_tiles, label_text, label_band_height):
        canvas = np.zeros((int(cell_height), int(cell_width), 3), dtype=np.uint8)
        content_height = max(16, int(cell_height) - int(label_band_height if label_tiles else 0))

        frame_height, frame_width = frame_bgr.shape[:2]
        scale = min(float(cell_width) / max(1, frame_width), float(content_height) / max(1, frame_height))
        new_width = max(1, int(round(frame_width * scale)))
        new_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

        x_offset = max(0, (int(cell_width) - new_width) // 2)
        y_offset = int(label_band_height if label_tiles else 0) + max(0, (content_height - new_height) // 2)
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        if label_tiles:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.45, min(1.0, float(cell_width) / 420.0))
            thickness = 2
            text = str(label_text or "")
            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = max(8, (int(cell_width) - text_size[0]) // 2)
            text_y = max(text_size[1] + 6, (int(label_band_height) + text_size[1]) // 2 - baseline)
            cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return canvas

    def _build_grid_frames(self, video_paths, cell_width, cell_height, columns, label_tiles, tile_labels):
        rows = int(math.ceil(len(video_paths) / max(1, int(columns))))
        label_band_height = self.LABEL_BAND_HEIGHT if label_tiles else 0
        captures = []
        last_frames = []
        finished = []
        blank_frame = np.zeros((max(16, int(cell_height) - int(label_band_height)), int(cell_width), 3), dtype=np.uint8)

        for path in video_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for grid render: {path}")
            captures.append(cap)
            last_frames.append(None)
            finished.append(False)

        output_frames = []
        try:
            while True:
                fresh_any = False
                all_finished = True
                tile_images = []

                for index, cap in enumerate(captures):
                    frame = None
                    if not finished[index]:
                        ok, read_frame = cap.read()
                        if ok and read_frame is not None:
                            frame = read_frame
                            last_frames[index] = read_frame
                            fresh_any = True
                            all_finished = False
                        else:
                            finished[index] = True
                    if frame is None:
                        frame = last_frames[index] if last_frames[index] is not None else blank_frame
                    if not finished[index]:
                        all_finished = False
                    tile_images.append(
                        self._fit_frame_to_tile(
                            frame,
                            int(cell_width),
                            int(cell_height),
                            bool(label_tiles),
                            tile_labels[index],
                            int(label_band_height),
                        )
                    )

                if not tile_images:
                    break
                if all_finished and not fresh_any:
                    break

                grid_frame = np.zeros((rows * int(cell_height), int(columns) * int(cell_width), 3), dtype=np.uint8)
                for index, tile_image in enumerate(tile_images):
                    col = index % int(columns)
                    row = index // int(columns)
                    y0 = row * int(cell_height)
                    x0 = col * int(cell_width)
                    grid_frame[y0:y0 + int(cell_height), x0:x0 + int(cell_width)] = tile_image

                grid_rgb = cv2.cvtColor(grid_frame, cv2.COLOR_BGR2RGB)
                output_frames.append(torch.from_numpy(grid_rgb.astype(np.float32) / 255.0))

                if all(finished):
                    break
        finally:
            for cap in captures:
                cap.release()

        if not output_frames:
            raise RuntimeError("No grid frames could be created from the provided videos.")

        return torch.stack(output_frames, dim=0)

    @staticmethod
    def _get_tensor_resolution(image_batch):
        if not isinstance(image_batch, torch.Tensor):
            raise RuntimeError("Expected IMAGE tensor input.")
        tensor = image_batch
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise RuntimeError("Expected IMAGE tensor with shape [frames, height, width, channels].")
        _, height, width, _ = tensor.shape
        return int(width), int(height)

    def _resolve_cell_size_from_images(self, image_batches, cell_width, cell_height, label_tiles):
        cell_width = int(cell_width)
        cell_height = int(cell_height)
        if cell_width > 0 and cell_height > 0:
            return cell_width, cell_height

        image_width, image_height = self._get_tensor_resolution(image_batches[0])
        resolved_width = cell_width if cell_width > 0 else image_width
        if cell_height > 0:
            resolved_height = cell_height
        else:
            resolved_height = image_height + (self.LABEL_BAND_HEIGHT if label_tiles else 0)
        return int(resolved_width), int(resolved_height)

    def _build_grid_frames_from_images(self, image_batches, cell_width, cell_height, columns, label_tiles, tile_labels):
        prepared_batches = []
        max_frames = 1
        for batch in image_batches:
            tensor = batch
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            prepared_batches.append(tensor)
            max_frames = max(max_frames, int(tensor.shape[0]))

        rows = int(math.ceil(len(prepared_batches) / max(1, int(columns))))
        label_band_height = self.LABEL_BAND_HEIGHT if label_tiles else 0
        output_frames = []

        for frame_index in range(max_frames):
            tile_images = []
            for index, tensor in enumerate(prepared_batches):
                source_index = min(frame_index, int(tensor.shape[0]) - 1)
                frame_rgb = tensor[source_index].detach().cpu().numpy()
                frame_rgb = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                tile_images.append(
                    self._fit_frame_to_tile(
                        frame_bgr,
                        int(cell_width),
                        int(cell_height),
                        bool(label_tiles),
                        tile_labels[index],
                        int(label_band_height),
                    )
                )

            grid_frame = np.zeros((rows * int(cell_height), int(columns) * int(cell_width), 3), dtype=np.uint8)
            for index, tile_image in enumerate(tile_images):
                col = index % int(columns)
                row = index // int(columns)
                y0 = row * int(cell_height)
                x0 = col * int(cell_width)
                grid_frame[y0:y0 + int(cell_height), x0:x0 + int(cell_width)] = tile_image

            grid_rgb = cv2.cvtColor(grid_frame, cv2.COLOR_BGR2RGB)
            output_frames.append(torch.from_numpy(grid_rgb.astype(np.float32) / 255.0))

        if not output_frames:
            raise RuntimeError("No grid frames could be created from the provided IMAGE inputs.")

        return torch.stack(output_frames, dim=0)

    def run(
        self,
        video_folder,
        output_name,
        filename_prefix,
        video_count,
        cell_width,
        cell_height,
        label_tiles,
        output_fps,
        **kwargs,
    ):
        selected_image_batches = self._collect_selected_image_batches(kwargs)
        if selected_image_batches:
            item_count = len(selected_image_batches)
            output_name = self._safe_name(output_name, "VideoGrid")
            tile_labels = self._resolve_dynamic_tensor_labels(selected_image_batches, kwargs)
            cell_width, cell_height = self._resolve_cell_size_from_images(
                selected_image_batches,
                cell_width,
                cell_height,
                bool(label_tiles),
            )
            columns = self._choose_columns(len(selected_image_batches))
            grid_frames = self._build_grid_frames_from_images(
                selected_image_batches,
                int(cell_width),
                int(cell_height),
                int(columns),
                bool(label_tiles),
                tile_labels,
            )
            source_status = f"Created grid image sequence from {item_count} connected video/image input(s)."
        else:
            video_folder = self._resolve_preview_folder(video_folder)
            output_name = self._safe_name(output_name, self._safe_name(os.path.basename(video_folder), "VideoGrid"))
            video_paths = self._find_all_videos(video_folder)
            tile_labels = self._resolve_dynamic_tile_labels(video_paths, kwargs)
            if not video_paths:
                return (
                    torch.zeros((1, 64, 64, 3), dtype=torch.float32),
                    str(filename_prefix or output_name),
                    int(output_fps),
                    f"No video files were found in {video_folder}. Connect video inputs or point video_folder at a folder with videos.",
                )

            cell_width, cell_height = self._resolve_cell_size(
                video_paths,
                cell_width,
                cell_height,
                bool(label_tiles),
            )
            columns = self._choose_columns(len(video_paths))
            grid_frames = self._build_grid_frames(
                video_paths,
                int(cell_width),
                int(cell_height),
                int(columns),
                bool(label_tiles),
                tile_labels,
            )
            item_count = len(video_paths)
            source_status = f"Created grid image sequence from {item_count} videos."
        rows = int(math.ceil(item_count / max(1, int(columns))))
        resolved_prefix = str(filename_prefix or output_name or "VRGDG/VideoGrid").strip() or "VRGDG/VideoGrid"

        print(
            f"[VRGDG] Creating folder grid image sequence "
            f"using a {int(columns)}x{int(rows)} grid at {int(cell_width)}x{int(cell_height)} per tile."
        )

        return (
            grid_frames,
            resolved_prefix,
            int(output_fps),
            source_status,
        )


NODE_CLASS_MAPPINGS = {
    "VRGDG_LTXLoraTrainChunk": VRGDG_LTXLoraTrainChunk,
    "VRGDG_SpeedCharacterLoraTraining": VRGDG_SpeedCharacterLoraTraining,    
    "VRGDG_LTXPreviewXYZPlot": VRGDG_LTXPreviewXYZPlot,
    "VRGDG_VideoFolderGridPlot": VRGDG_VideoFolderGridPlot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LTXLoraTrainChunk": "VRGDG LTX LoRA Train Chunk",
    "VRGDG_SpeedCharacterLoraTraining": "VRGDG Speed Character Lora Training",    
    "VRGDG_LTXPreviewXYZPlot": "VRGDG LTX Preview XYZ Plot",
    "VRGDG_VideoFolderGridPlot": "VRGDG Video Folder Grid Plot",
}

try:
    _ensure_tensorboard_route_registered()
except Exception as exc:
    print(f"[VRGDG] Failed to register TensorBoard route: {exc}")
