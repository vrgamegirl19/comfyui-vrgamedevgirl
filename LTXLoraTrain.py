import os
import asyncio
import json
import re
import shutil
import subprocess
import threading
import time
import gc
import math
import sys
import tempfile
import urllib.request
import urllib.error
import zipfile
import webbrowser
from datetime import datetime

import comfy
import torch
import cv2
import numpy as np
import folder_paths
from aiohttp import web
from server import PromptServer

try:
    import torchaudio
except Exception:
    torchaudio = None


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")

_VRGDG_TENSORBOARD_ROUTE_REGISTERED = False
_VRGDG_TENSORBOARD_RUNS = {}
_VRGDG_MUSUBI_INSTALL_ROUTE_REGISTERED = False
_VRGDG_MUSUBI_INSTALL_LOCK = threading.Lock()
_VRGDG_MUSUBI_INSTALL_JOB = None

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
}


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


def _ensure_musubi_install_route_registered():
    global _VRGDG_MUSUBI_INSTALL_ROUTE_REGISTERED
    global _VRGDG_MUSUBI_INSTALL_JOB
    if _VRGDG_MUSUBI_INSTALL_ROUTE_REGISTERED:
        return

    server_instance = getattr(PromptServer, "instance", None)
    if server_instance is None:
        return

    def _emit(log_lines, message):
        message = str(message)
        print(message)
        log_lines.append(message)

    def _resolve_supported_python():
        launcher = shutil.which("py")
        if launcher:
            for minor in (12, 11, 10):
                candidate = [launcher, f"-3.{minor}"]
                try:
                    result = subprocess.run(
                        candidate + ["--version"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=True,
                    )
                except Exception:
                    continue
                version_text = (result.stdout or result.stderr or "").strip()
                if re.search(r"Python 3\.(10|11|12)\.\d+", version_text):
                    return candidate, version_text

        current_version = sys.version_info
        if current_version.major == 3 and 10 <= current_version.minor <= 12:
            return [sys.executable], f"Python {current_version.major}.{current_version.minor}.{current_version.micro}"

        raise RuntimeError(
            "Musubi-Tuner requires Python 3.10, 3.11, or 3.12. "
            "Install one of those versions or make sure the `py` launcher can resolve one of them."
        )

    def _resolve_unique_install_path(parent_root):
        base_name = "Musubi-tuner"
        primary = os.path.join(parent_root, base_name)
        if not os.path.exists(primary):
            return primary

        prefixed = os.path.join(parent_root, f"VRGDG_{base_name}")
        if not os.path.exists(prefixed):
            return prefixed

        index = 2
        while True:
            candidate = os.path.join(parent_root, f"VRGDG_{base_name}_{index}")
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def _resolve_existing_install_path(target_root):
        target_root = os.path.normpath(str(target_root or "").strip())
        if not target_root:
            return ""

        if os.path.isdir(target_root):
            venv_python = os.path.join(target_root, "venv", "Scripts", "python.exe")
            if os.path.isfile(venv_python):
                return target_root

        candidates = [
            os.path.join(target_root, "Musubi-tuner"),
            os.path.join(target_root, "VRGDG_Musubi-tuner"),
        ]
        for candidate in candidates:
            venv_python = os.path.join(candidate, "venv", "Scripts", "python.exe")
            if os.path.isfile(venv_python):
                return os.path.normpath(candidate)

        if os.path.isdir(target_root):
            with os.scandir(target_root) as entries:
                for entry in sorted(entries, key=lambda item: item.name.lower()):
                    if not entry.is_dir():
                        continue
                    venv_python = os.path.join(entry.path, "venv", "Scripts", "python.exe")
                    if os.path.isfile(venv_python):
                        return os.path.normpath(entry.path)

        return ""

    def _download_file(url, destination, log_lines):
        _emit(log_lines, f"[VRGDG] Downloading archive: {url}")
        with urllib.request.urlopen(url) as response, open(destination, "wb") as output_handle:
            total_size = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            last_reported = 0
            report_step = max(1, total_size // 10) if total_size else 5 * 1024 * 1024

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output_handle.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_reported >= report_step or (total_size and downloaded >= total_size):
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        _emit(
                            log_lines,
                            f"[VRGDG] Download progress: {downloaded}/{total_size} bytes ({percent:.1f}%)",
                        )
                    else:
                        _emit(log_lines, f"[VRGDG] Downloaded {downloaded} bytes")
                    last_reported = downloaded

        _emit(log_lines, f"[VRGDG] Download complete: {destination}")

    def _run_command(command, cwd, log_lines):
        command_line = "$ " + " ".join(command)
        _emit(log_lines, command_line)
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\n")
            _emit(log_lines, line)
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")

    def _run_python_check(venv_python, code, cwd, log_lines, label, required=True):
        _emit(log_lines, f"[VRGDG] Check: {label}")
        try:
            _run_command([venv_python, "-c", code], cwd, log_lines)
            return {"label": label, "required": required, "ok": True, "message": "PASS"}
        except Exception as exc:
            level = "ERROR" if required else "WARN"
            message = str(exc)
            _emit(log_lines, f"[VRGDG] {level}: {label}: {message}")
            return {"label": label, "required": required, "ok": False, "message": message}

    def _run_install_verification(install_path, venv_python, version_text, log_lines):
        checks = []
        checks.append(
            _run_python_check(
                venv_python,
                "import sys; print(sys.executable); print(sys.version)",
                install_path,
                log_lines,
                "Python interpreter and version",
            )
        )
        checks.append(
            _run_python_check(
                venv_python,
                "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())",
                install_path,
                log_lines,
                "torch import and CUDA availability",
            )
        )
        checks.append(
            _run_python_check(
                venv_python,
                "import accelerate, diffusers, transformers, safetensors, av, cv2, PIL, huggingface_hub, sentencepiece; print('core imports ok')",
                install_path,
                log_lines,
                "Core musubi dependencies",
            )
        )
        checks.append(
            _run_python_check(
                venv_python,
                "import musubi_tuner; print(musubi_tuner.__file__)",
                install_path,
                log_lines,
                "musubi_tuner import",
            )
        )
        checks.append(
            _run_python_check(
                venv_python,
                "import bitsandbytes; print(bitsandbytes.__file__)",
                install_path,
                log_lines,
                "bitsandbytes import",
                required=False,
            )
        )
        checks.append(
            _run_python_check(
                venv_python,
                "import pathlib; p = pathlib.Path('ltx2_train_network.py'); print(p.resolve()); print(p.exists())",
                install_path,
                log_lines,
                "musubi training script exists",
            )
        )

        pass_count = sum(1 for item in checks if item.get("ok"))
        warn_count = sum(1 for item in checks if not item.get("ok") and not item.get("required"))
        fail_count = sum(1 for item in checks if not item.get("ok") and item.get("required"))

        summary_line = (
            f"[VRGDG] Verification summary: pass={pass_count} warn={warn_count} fail={fail_count} "
            f"python={version_text}"
        )
        _emit(log_lines, summary_line)

        report_lines = [
            "VRGDG Musubi-Tuner Installation Report",
            f"Install path: {install_path}",
            f"Python: {version_text}",
            "",
            "Checks:",
        ]
        for item in checks:
            state = "PASS" if item.get("ok") else ("WARN" if not item.get("required") else "FAIL")
            report_lines.append(f"- [{state}] {item['label']}: {item['message']}")
        report_lines.append("")
        report_lines.append(summary_line)

        report_path = os.path.join(install_path, "vrgdg_musubi_install_report.txt")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report_lines) + "\n")

        _emit(log_lines, f"[VRGDG] Wrote verification report: {report_path}")
        return checks, report_path

    def _install_musubi_tuner(target_root, branch, torch_index_url):
        log_lines = []
        target_root = os.path.normpath(str(target_root or "").strip())
        branch = str(branch or "ltx-2-dev").strip() or "ltx-2-dev"
        torch_index_url = str(torch_index_url or "").strip() or "https://download.pytorch.org/whl/cu124"

        if not target_root:
            raise ValueError("target_root is required.")

        parent_root = os.path.abspath(target_root)
        os.makedirs(parent_root, exist_ok=True)

        install_path = _resolve_unique_install_path(parent_root)
        temp_root = tempfile.mkdtemp(prefix="vrgdg_musubi_")
        zip_path = os.path.join(temp_root, f"musubi-tuner-{branch}.zip")
        extract_root = os.path.join(temp_root, f"musubi-tuner-{branch}")
        repo_url = f"https://github.com/AkaneTendo25/musubi-tuner/archive/refs/heads/{branch}.zip"

        try:
            _emit(log_lines, f"[VRGDG] Target parent folder: {parent_root}")
            _emit(log_lines, f"[VRGDG] Install folder resolved to: {install_path}")
            _emit(log_lines, f"[VRGDG] Branch: {branch}")

            _download_file(repo_url, zip_path, log_lines)

            _emit(log_lines, f"[VRGDG] Extracting archive to: {extract_root}")
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_root)

            extracted_dirs = [
                entry.path
                for entry in os.scandir(extract_root)
                if entry.is_dir()
            ]
            if not extracted_dirs:
                raise RuntimeError("The downloaded archive did not contain a top-level folder.")

            source_dir = extracted_dirs[0]
            _emit(log_lines, f"[VRGDG] Moving source folder into place: {source_dir}")
            shutil.move(source_dir, install_path)

            python_command, version_text = _resolve_supported_python()
            _emit(log_lines, f"[VRGDG] Using Python interpreter: {version_text}")

            _run_command(python_command + ["-m", "venv", "venv"], install_path, log_lines)

            venv_python = os.path.join(install_path, "venv", "Scripts", "python.exe")
            if not os.path.isfile(venv_python):
                raise RuntimeError(f"Could not find venv python: {venv_python}")

            _run_command(
                [venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                install_path,
                log_lines,
            )
            _run_command(
                [venv_python, "-m", "pip", "install", "torch", "torchvision", "--index-url", torch_index_url],
                install_path,
                log_lines,
            )
            _run_command(
                [venv_python, "-m", "pip", "install", "-e", "."],
                install_path,
                log_lines,
            )

            checks, report_path = _run_install_verification(install_path, venv_python, version_text, log_lines)
            if any((not item.get("ok")) and item.get("required") for item in checks):
                raise RuntimeError(
                    "Musubi-Tuner installed, but required verification checks failed. "
                    f"See report: {report_path}"
                )

            _emit(log_lines, "[VRGDG] Musubi-Tuner installation completed successfully.")
            _emit(log_lines, f"[VRGDG] Final install path: {install_path}")

            return {
                "ok": True,
                "install_path": os.path.normpath(install_path),
                "python_version": version_text,
                "report_path": os.path.normpath(report_path),
                "checks": checks,
                "messages": log_lines,
            }
        finally:
            if os.path.isdir(temp_root):
                shutil.rmtree(temp_root, ignore_errors=True)

    def _download_model_artifacts(install_path, model_family, checkpoint_repo, checkpoint_filename, gemma_repo, gemma_root_name):
        model_root = os.path.join(install_path, "models")
        checkpoints_dir = os.path.join(model_root, "checkpoints")
        gemma_dir = os.path.join(model_root, gemma_root_name)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(gemma_dir, exist_ok=True)

        download_script = f"""
import os
from huggingface_hub import hf_hub_download, snapshot_download

install_path = {install_path!r}
checkpoints_dir = {checkpoints_dir!r}
gemma_dir = {gemma_dir!r}
model_family = {model_family!r}
checkpoint_repo = {checkpoint_repo!r}
checkpoint_filename = {checkpoint_filename!r}
gemma_repo = {gemma_repo!r}

def _download_checkpoint():
    print(f"[VRGDG] Downloading checkpoint {{checkpoint_filename}} from {{checkpoint_repo}}")
    path = hf_hub_download(
        repo_id=checkpoint_repo,
        filename=checkpoint_filename,
        local_dir=checkpoints_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[VRGDG] Checkpoint ready: {{path}}")

def _download_gemma():
    print(f"[VRGDG] Downloading Gemma snapshot from {{gemma_repo}}")
    path = snapshot_download(
        repo_id=gemma_repo,
        local_dir=gemma_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[VRGDG] Gemma snapshot ready: {{path}}")

if str(model_family).lower().startswith("z"):
    print("[VRGDG] Downloading Z-Image Turbo snapshot bundle from Tongyi-MAI/Z-Image-Turbo")
    path = snapshot_download(
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        local_dir=gemma_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[VRGDG] Z-Image Turbo snapshot ready: {{path}}")
else:
    _download_checkpoint()
    _download_gemma()
print(f"[VRGDG] Model download root: {{os.path.join(install_path, 'models')}}")
"""
        return download_script, checkpoints_dir, gemma_dir

    def _find_first_matching_safetensor(root_dir, include_any=None, exclude_any=None):
        include_any = [str(item).lower() for item in (include_any or []) if str(item or "").strip()]
        exclude_any = [str(item).lower() for item in (exclude_any or []) if str(item or "").strip()]
        if not os.path.isdir(root_dir):
            return ""

        candidates = []
        for current_root, _dirs, files in os.walk(root_dir):
            for name in files:
                if not name.lower().endswith(".safetensors"):
                    continue
                lowered = name.lower()
                if include_any and not any(token in lowered for token in include_any):
                    continue
                if exclude_any and any(token in lowered for token in exclude_any):
                    continue
                candidates.append(os.path.join(current_root, name))

        if not candidates:
            return ""

        def _sort_key(path):
            name = os.path.basename(path).lower()
            split_rank = 0 if re.search(r"00001-of-\d+", name) else 1
            return (split_rank, len(path), path.lower())

        candidates.sort(key=_sort_key)
        return os.path.normpath(candidates[0])

    def _run_model_downloads(install_path, venv_python, log_lines, download_models, model_family):
        if not download_models:
            return {
                "downloaded": False,
                "checkpoint_path": "",
                "assets_root": "",
            }

        model_family_key = str(model_family or "LTX 2.3").strip().lower()
        checkpoint_repo = "Lightricks/LTX-2.3"
        checkpoint_filename = "ltx-2.3-22b-dev.safetensors"
        gemma_repo = "google/gemma-3-12b-it"
        gemma_root_name = "gemma-3-12b-it"
        if model_family_key.startswith("z"):
            checkpoint_repo = "Tongyi-MAI/Z-Image-Turbo"
            checkpoint_filename = "model.safetensors"
            gemma_repo = "Tongyi-MAI/Z-Image-Turbo"
            gemma_root_name = "zimage_turbo"
        script, checkpoint_dir, gemma_dir = _download_model_artifacts(
            install_path,
            model_family,
            checkpoint_repo,
            checkpoint_filename,
            gemma_repo,
            gemma_root_name,
        )
        _emit(log_lines, "[VRGDG] Auto-download enabled: fetching required model files.")
        _run_command([venv_python, "-c", script], install_path, log_lines)
        if model_family_key.startswith("z"):
            checkpoint_path = _find_first_matching_safetensor(
                gemma_dir,
                include_any=["00001-of-", "dit", "transformer", "model"],
                exclude_any=["vae", "text", "qwen", "tokenizer"],
            )
            if not checkpoint_path:
                checkpoint_path = gemma_dir
            vae_path = _find_first_matching_safetensor(
                gemma_dir,
                include_any=["vae"],
                exclude_any=["qwen", "text", "tokenizer"],
            )
            if not vae_path:
                vae_path = gemma_dir
            text_encoder_path = _find_first_matching_safetensor(
                gemma_dir,
                include_any=["qwen", "text", "llm"],
                exclude_any=["vae"],
            )
            if not text_encoder_path:
                text_encoder_path = gemma_dir
            return {
                "downloaded": True,
                "checkpoint_path": os.path.normpath(checkpoint_path),
                "assets_root": os.path.normpath(gemma_dir),
                "vae_path": os.path.normpath(vae_path),
                "text_encoder_path": os.path.normpath(text_encoder_path),
            }
        return {
            "downloaded": True,
            "checkpoint_path": os.path.normpath(os.path.join(checkpoint_dir, checkpoint_filename)),
            "assets_root": os.path.normpath(gemma_dir),
        }

    def _get_huggingface_token():
        candidates = [
            os.environ.get("HF_TOKEN", ""),
            os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
            os.environ.get("HF_HUB_TOKEN", ""),
        ]
        token_paths = [
            os.path.join(os.path.expanduser("~"), ".huggingface", "token"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token"),
        ]
        for token_path in token_paths:
            if os.path.isfile(token_path):
                try:
                    with open(token_path, "r", encoding="utf-8") as handle:
                        candidates.append(handle.read().strip())
                except Exception:
                    pass
        for candidate in candidates:
            token = str(candidate or "").strip()
            if token:
                return token
        return ""

    def _probe_huggingface_endpoint(url, token, log_lines, label):
        headers = {"User-Agent": "VRGDG-Musubi-Installer/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = getattr(response, "status", 200)
                _emit(log_lines, f"[VRGDG] HF preflight passed: {label} (HTTP {status})")
                return
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"Please log into Hugging Face and accept the required license before downloading {label}."
                ) from exc
            raise RuntimeError(
                f"Hugging Face preflight failed for {label} with HTTP {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Hugging Face preflight failed for {label}: {exc.reason}") from exc

    def _preflight_huggingface_downloads(log_lines, model_family):
        token = _get_huggingface_token()
        model_family_key = str(model_family or "LTX 2.3").strip().lower()
        _emit(log_lines, "[VRGDG] Running non-blocking Hugging Face preflight before model download.")
        try:
            if model_family_key.startswith("z"):
                _probe_huggingface_endpoint(
                    "https://huggingface.co/api/models/Tongyi-MAI/Z-Image-Turbo",
                    token,
                    log_lines,
                    "Z-Image Turbo checkpoint bundle",
                )
            else:
                _probe_huggingface_endpoint(
                    "https://huggingface.co/api/models/Lightricks/LTX-2.3",
                    token,
                    log_lines,
                    "LTX-2.3 checkpoint",
                )
                _probe_huggingface_endpoint(
                    "https://huggingface.co/api/models/google/gemma-3-12b-it",
                    token,
                    log_lines,
                    "Gemma 3 12B IT",
                )
        except Exception as exc:
            _emit(log_lines, f"[VRGDG] Hugging Face preflight warning: {exc}")

    async def _run_musubi_install_job(payload):
        acquired = _VRGDG_MUSUBI_INSTALL_LOCK.acquire(blocking=False)
        if not acquired:
            return {"ok": False, "error": "An installation is already running."}

        try:
            target_root = str(payload.get("target_root", "") or "").strip()
            branch = str(payload.get("branch", "ltx-2-dev") or "ltx-2-dev").strip() or "ltx-2-dev"
            torch_index_url = str(
                payload.get("torch_index_url", "https://download.pytorch.org/whl/cu124")
                or "https://download.pytorch.org/whl/cu124"
            ).strip()
            download_models_flag = bool(payload.get("download_models", False))
            action = str(payload.get("action", "") or "").strip()
            if not action:
                action = "install_and_download" if download_models_flag else "install_tuner"
            model_family = str(payload.get("model_family", "LTX 2.3") or "LTX 2.3").strip() or "LTX 2.3"

            if action == "download_models":
                await asyncio.to_thread(_preflight_huggingface_downloads, [], model_family)
                install_path = _resolve_existing_install_path(target_root)
                if not install_path:
                    raise RuntimeError(
                        "Could not find an existing Musubi-Tuner install under target_root. "
                        "Install Musubi-Tuner first, then use the model download button."
                    )
                venv_python = os.path.join(install_path, "venv", "Scripts", "python.exe")
                if not os.path.isfile(venv_python):
                    raise RuntimeError(
                        f"Could not find the Musubi-Tuner virtual environment at: {venv_python}"
                    )
                result = {
                    "ok": True,
                    "install_path": os.path.normpath(install_path),
                    "python_version": "",
                    "report_path": os.path.join(install_path, "vrgdg_musubi_install_report.txt"),
                    "checks": [],
                    "messages": [f"[VRGDG] Reusing existing Musubi-Tuner install: {install_path}"],
                }
                model_result = await asyncio.to_thread(
                    _run_model_downloads,
                    install_path,
                    venv_python,
                    result["messages"],
                    True,
                    model_family,
                )
                result.update(model_result)
                model_section = [
                    "",
                    "Model Downloads:",
                    f"- Model family: {model_family}",
                    f"- Checkpoint path: {model_result.get('checkpoint_path', '')}",
                    f"- Model assets root: {model_result.get('assets_root', '')}",
                ]
                if model_result.get("vae_path"):
                    model_section.append(f"- VAE path: {model_result.get('vae_path', '')}")
                if model_result.get("text_encoder_path"):
                    model_section.append(f"- Text encoder path: {model_result.get('text_encoder_path', '')}")
                result["messages"].extend(model_section)
                if result.get("report_path"):
                    with open(result["report_path"], "a", encoding="utf-8") as handle:
                        handle.write("\n".join(model_section) + "\n")
                result["status"] = "Selected model assets downloaded successfully."
            elif action == "install_and_download":
                await asyncio.to_thread(_preflight_huggingface_downloads, [], model_family)
                result = await asyncio.to_thread(_install_musubi_tuner, target_root, branch, torch_index_url)
                install_path = result.get("install_path", "")
                venv_python = os.path.join(install_path, "venv", "Scripts", "python.exe")
                if not os.path.isfile(venv_python):
                    raise RuntimeError(
                        f"Could not find the Musubi-Tuner virtual environment at: {venv_python}"
                    )
                model_result = await asyncio.to_thread(
                    _run_model_downloads,
                    install_path,
                    venv_python,
                    result["messages"],
                    True,
                    model_family,
                )
                result.update(model_result)
                model_section = [
                    "",
                    "Model Downloads:",
                    f"- Model family: {model_family}",
                    f"- Checkpoint path: {model_result.get('checkpoint_path', '')}",
                    f"- Model assets root: {model_result.get('assets_root', '')}",
                ]
                if model_result.get("vae_path"):
                    model_section.append(f"- VAE path: {model_result.get('vae_path', '')}")
                if model_result.get("text_encoder_path"):
                    model_section.append(f"- Text encoder path: {model_result.get('text_encoder_path', '')}")
                result["messages"].extend(model_section)
                if result.get("report_path"):
                    with open(result["report_path"], "a", encoding="utf-8") as handle:
                        handle.write("\n".join(model_section) + "\n")
                result["status"] = "Musubi-Tuner installed successfully."
                result["gemma_root"] = result.get("assets_root", "")
            else:
                result = await asyncio.to_thread(_install_musubi_tuner, target_root, branch, torch_index_url)
                result["status"] = "Musubi-Tuner installed successfully."
            return result
        finally:
            if acquired:
                _VRGDG_MUSUBI_INSTALL_LOCK.release()

    @server_instance.routes.post("/vrgdg/musubi/install")
    async def vrgdg_install_musubi(request):
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body."}, status=400)

        try:
            result = await _run_musubi_install_job(payload)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        status_code = 200 if result.get("ok") else 500
        if "error" in result and not result.get("ok"):
            status_code = 409 if "already running" in str(result.get("error", "")).lower() else 500
        return web.json_response(result, status=status_code)

    _VRGDG_MUSUBI_INSTALL_JOB = _run_musubi_install_job

    _VRGDG_MUSUBI_INSTALL_ROUTE_REGISTERED = True


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
    VIDEO_EXTENSIONS = {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".avi",
        ".m4v",
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
                "gemma_recovery_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental. If enabled, the node will keep the normal Gemma cache path first, then try alternate cache settings if that stage fails."
                }),
                "gemma_load_in_4bit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loads Gemma in 4-bit mode during text encoder caching. This lowers VRAM more than 8-bit, but can be slower or less stable."
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

    @staticmethod
    def _extract_command_exit_code(error):
        match = re.search(r"exit code (\d+)", str(error))
        return int(match.group(1)) if match else None

    def _build_text_encoder_cache_command(
        self,
        python_exe,
        dataset_config,
        ltx2_checkpoint,
        gemma_root,
        ltx_mode,
        gemma_load_in_8bit,
        mixed_precision="bf16",
        gemma_load_in_4bit=False,
    ):
        command = [
            python_exe,
            "ltx2_cache_text_encoder_outputs.py",
            "--dataset_config",
            dataset_config,
            "--ltx2_checkpoint",
            ltx2_checkpoint,
            "--gemma_root",
            gemma_root,
        ]
        if gemma_load_in_4bit:
            command.append("--gemma_load_in_4bit")
        elif gemma_load_in_8bit:
            command.append("--gemma_load_in_8bit")
        command.extend(
            [
                "--device",
                "cuda",
                "--mixed_precision",
                str(mixed_precision or "bf16"),
                "--ltx2_mode",
                ltx_mode,
                "--batch_size",
                "1",
            ]
        )
        return command

    def _run_text_encoder_cache_stage(
        self,
        stage_number,
        total_stages,
        title,
        python_exe,
        dataset_config,
        ltx2_checkpoint,
        gemma_root,
        ltx_mode,
        gemma_load_in_8bit,
        cwd,
        log_handle,
        detail_lines=None,
        gemma_load_in_4bit=False,
    ):
        command = self._build_text_encoder_cache_command(
            python_exe,
            dataset_config,
            ltx2_checkpoint,
            gemma_root,
            ltx_mode,
            gemma_load_in_8bit,
            gemma_load_in_4bit=gemma_load_in_4bit,
        )
        self._run_stage_command(stage_number, total_stages, title, command, cwd, log_handle, detail_lines)

    def _run_text_encoder_cache_stage_with_recovery(
        self,
        stage_number,
        total_stages,
        title,
        python_exe,
        dataset_config,
        ltx2_checkpoint,
        gemma_root,
        ltx_mode,
        gemma_load_in_8bit,
        recovery_mode,
        cwd,
        log_handle,
        detail_lines=None,
        gemma_load_in_4bit=False,
    ):
        try:
            self._run_text_encoder_cache_stage(
                stage_number,
                total_stages,
                title,
                python_exe,
                dataset_config,
                ltx2_checkpoint,
                gemma_root,
                ltx_mode,
                gemma_load_in_8bit,
                cwd,
                log_handle,
                detail_lines,
                gemma_load_in_4bit=gemma_load_in_4bit,
            )
            return
        except RuntimeError as exc:
            if not recovery_mode:
                raise
            last_exc = exc

        gemma_flag_label = "--gemma_load_in_4bit" if gemma_load_in_4bit else "--gemma_load_in_8bit"
        retry_plans = [
            ("fp16", f"experimental recovery: retrying with mixed_precision=fp16 and {gemma_flag_label}"),
            ("bf16", f"experimental recovery: retrying with mixed_precision=bf16 and {gemma_flag_label}"),
        ]

        for mixed_precision, label in retry_plans:
            retry_message = f"[VRGDG] {title} failed. {label}."
            print(retry_message)
            log_handle.write(retry_message + "\n")
            log_handle.flush()

            retry_command = self._build_text_encoder_cache_command(
                python_exe,
                dataset_config,
                ltx2_checkpoint,
                gemma_root,
                ltx_mode,
                gemma_load_in_8bit,
                mixed_precision,
                gemma_load_in_4bit=gemma_load_in_4bit,
            )
            try:
                self._run_stage_command(stage_number, total_stages, title, retry_command, cwd, log_handle, detail_lines)
                return
            except RuntimeError as exc:
                last_exc = exc

        raise RuntimeError(
            f"{title} failed after experimental recovery attempts. "
            "Check the log for the exact Gemma cache command outputs."
        ) from last_exc

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

    def _has_latent_cache_files(self, cache_dir):
        architecture = "ltx2"
        if not os.path.isdir(cache_dir):
            return False
        for entry in os.scandir(cache_dir):
            if not entry.is_file():
                continue
            if entry.name.endswith(f"_{architecture}.safetensors") and not entry.name.endswith(f"_{architecture}_te.safetensors") and not entry.name.endswith(f"_{architecture}_audio.safetensors"):
                return True
        return False

    def _has_complete_text_encoder_cache_files(self, cache_dir):
        architecture = "ltx2"
        if not os.path.isdir(cache_dir):
            return False

        latent_cache_files = []
        for entry in os.scandir(cache_dir):
            if not entry.is_file():
                continue
            if entry.name.endswith(f"_{architecture}.safetensors") and not entry.name.endswith(f"_{architecture}_te.safetensors") and not entry.name.endswith(f"_{architecture}_audio.safetensors"):
                latent_cache_files.append(entry.path)

        if not latent_cache_files:
            return False

        seen_text_cache_files = set()
        for latent_cache_file in latent_cache_files:
            expected_text_cache = self._expected_text_cache_path_from_latent_cache_file(cache_dir, latent_cache_file)
            if not expected_text_cache:
                continue
            normalized_expected = os.path.normpath(expected_text_cache)
            if normalized_expected in seen_text_cache_files:
                continue
            seen_text_cache_files.add(normalized_expected)
            if not os.path.exists(expected_text_cache):
                return False

        return True

    def _expected_text_cache_path_from_latent_cache_file(self, cache_dir, latent_cache_file):
        architecture = "ltx2"
        basename = os.path.basename(latent_cache_file)
        if not basename.endswith(f"_{architecture}.safetensors"):
            return ""
        tokens = basename.split("_")
        if len(tokens) < 4:
            return ""

        # Image latents are typically: {item_key}_{WxH}_{arch}.safetensors
        # Video/AV latents are typically: {item_key}_{start-end}_{WxH}_{arch}.safetensors
        # Keep the same item_key extraction logic the Musubi loader uses.
        if len(tokens) >= 5 and re.fullmatch(r"\d+-\d+(?:-\d+)?", tokens[-3]):
            item_key = "_".join(tokens[:-3])
        else:
            item_key = "_".join(tokens[:-2])

        if not item_key:
            return ""
        return os.path.join(cache_dir, f"{item_key}_{architecture}_te.safetensors")

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
        messages = ["[VRGDG] Clearing ComfyUI and CUDA memory before text encoder cache stage."]
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

    def _write_audio_dataset_config(
        self,
        path,
        audio_directory,
        cache_dir,
        target_resolution,
        num_repeats,
        audio_bucket_strategy,
        audio_bucket_interval,
    ):
        content = (
            "[general]\n"
            f"resolution = [{int(target_resolution)}, {int(target_resolution)}]\n"
            'caption_extension = ".txt"\n'
            "batch_size = 1\n"
            "enable_bucket = true\n"
            "bucket_no_upscale = false\n\n"
            "[[datasets]]\n"
            f'audio_directory = "{self._quote(audio_directory)}"\n'
            f'cache_directory = "{self._quote(cache_dir)}"\n'
            f"num_repeats = {int(num_repeats)}\n"
            f'audio_bucket_strategy = "{str(audio_bucket_strategy)}"\n'
            f"audio_bucket_interval = {float(audio_bucket_interval)}\n"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _write_audio_training_config(
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
        lora_target_preset,
        fp8_base,
        fp8_scaled,
        audio_only_sequence_resolution,
        ltx2_audio_only_model,
    ):
        checkpoint_name = os.path.basename(str(checkpoint or "")).lower()
        effective_fp8_scaled = bool(fp8_scaled) and ("fp8" not in checkpoint_name)
        content = (
            "# Auto-generated by VRGDG LTX audio chunk trainer\n"
            f"# total_target_steps_from_workflow = {int(total_target_steps)}\n"
            f"# chunk_target_steps_this_run = {int(max_train_steps)}\n"
            f"# save_interval_per_run = {int(steps_per_run)}\n"
            f'ltx2_checkpoint = "{self._quote(checkpoint)}"\n'
            f'gemma_root = "{self._quote(gemma_root)}"\n'
            f'dataset_config = "{self._quote(dataset_config)}"\n\n'
            'ltx_mode = "audio"\n'
            'ltx_version = "2.3"\n'
            'ltx_version_check_mode = "error"\n'
            f'lora_target_preset = "{lora_target_preset}"\n'
            f"ltx2_audio_only_model = {'true' if ltx2_audio_only_model else 'false'}\n"
            f"audio_only_sequence_resolution = {int(audio_only_sequence_resolution)}\n\n"
            "cache_text_encoder_outputs = true\n"
            "cache_text_encoder_outputs_to_disk = false\n\n"
            f"fp8_base = {'true' if fp8_base else 'false'}\n"
            f"fp8_scaled = {'true' if effective_fp8_scaled else 'false'}\n"
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

    def _build_audio_filename_prefix(self, audio_output_subfolder, output_name, step_number):
        return f"{audio_output_subfolder}/{self._safe_name(output_name, 'LTXAudioRun')}_step_{int(step_number)}"

    def _get_or_create_audio_output_subfolder(self, config_dir, dataset_audio_dir):
        subfolder_file = os.path.join(config_dir, "audio_output_subfolder.txt")
        if os.path.isfile(subfolder_file):
            with open(subfolder_file, "r", encoding="utf-8") as handle:
                existing_subfolder = handle.read().strip()
            if existing_subfolder:
                return existing_subfolder

        dataset_label = self._get_dataset_label(dataset_audio_dir)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        relative_subfolder = f"{dataset_label}_{stamp}"

        with open(subfolder_file, "w", encoding="utf-8") as handle:
            handle.write(relative_subfolder)

        output_root = folder_paths.get_output_directory()
        os.makedirs(os.path.join(output_root, relative_subfolder), exist_ok=True)
        return relative_subfolder

    def _get_or_create_audio_filename_prefix(self, config_dir, dataset_audio_dir, output_name, step_number):
        prefix_file = os.path.join(config_dir, "audio_filename_prefix.txt")
        current_prefix = self._build_audio_filename_prefix(
            self._get_or_create_audio_output_subfolder(config_dir, dataset_audio_dir),
            output_name,
            step_number,
        )
        with open(prefix_file, "w", encoding="utf-8") as handle:
            handle.write(current_prefix)
        return current_prefix

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

    def _resolve_musubi_script_root(self, musubi_root, required_scripts):
        musubi_root = self._norm(musubi_root)
        required_scripts = [str(script or "").strip() for script in required_scripts if str(script or "").strip()]
        if not required_scripts:
            return musubi_root

        def _has_required_scripts(root):
            if not os.path.isdir(root):
                return False
            return all(os.path.isfile(os.path.join(root, script)) for script in required_scripts)

        if _has_required_scripts(musubi_root):
            return musubi_root

        parent_root = os.path.dirname(musubi_root)
        if os.path.isdir(parent_root):
            for entry in os.scandir(parent_root):
                if not entry.is_dir():
                    continue
                if _has_required_scripts(entry.path):
                    print(
                        f"[VRGDG] Resolved Musubi script root from {musubi_root} to {entry.path} "
                        f"because the configured root did not contain all required Musubi scripts."
                    )
                    return os.path.normpath(entry.path)

        missing_text = ", ".join(required_scripts)
        raise ValueError(
            f"musubi_root does not contain the required scripts: {missing_text}. "
            f"Checked: {musubi_root}"
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
        output_lines = []
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="")
            output_lines.append(line.rstrip("\n"))
        process.wait()
        if process.returncode != 0:
            tail = "\n".join(output_lines[-40:]).strip()
            message = f"Command failed with exit code {process.returncode}: {' '.join(command)}"
            if tail:
                message += f"\n--- output tail ---\n{tail}"
            raise RuntimeError(message)

    def _should_build_cache(self, cache_strategy, cache_dir):
        if cache_strategy == "force":
            return True
        if cache_strategy == "skip":
            return False
        if not os.path.isdir(cache_dir):
            return True
        return not (self._has_latent_cache_files(cache_dir) and self._has_complete_text_encoder_cache_files(cache_dir))

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
        gemma_recovery_mode,
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
        gemma_load_in_4bit=False,
        _autochunk_mode=False,
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
        if not os.path.isfile(ltx2_checkpoint):
            raise ValueError(f"ltx2_checkpoint does not exist: {ltx2_checkpoint}")
        if not os.path.isdir(gemma_root):
            raise ValueError(f"gemma_root does not exist: {gemma_root}")

        musubi_root = self._resolve_musubi_script_root(
            musubi_root,
            [
                "ltx2_cache_latents.py",
                "ltx2_cache_text_encoder_outputs.py",
                "ltx2_train_network.py",
            ],
        )

        gemma_load_in_4bit = bool(gemma_load_in_4bit)
        gemma_load_in_8bit = not gemma_load_in_4bit
        gemma_load_mode = "4bit" if gemma_load_in_4bit else "8bit"

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
            print(f"[VRGDG] gemma_load_mode={gemma_load_mode}")
            print(f"[VRGDG] gemma_recovery_mode={gemma_recovery_mode}")
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
                if gemma_recovery_mode:
                    self._run_text_encoder_cache_stage_with_recovery(
                        2,
                        total_stages,
                        "Cache text encoder outputs",
                        python_exe,
                        dataset_config,
                        ltx2_checkpoint,
                        gemma_root,
                        "video",
                        gemma_load_in_8bit,
                        gemma_recovery_mode,
                        musubi_root,
                        log_handle,
                        [
                            f"Gemma root: {gemma_root}",
                            "This is usually the slowest setup stage.",
                            "You should see per-item progress from the text encoder cache script.",
                        ],
                        gemma_load_in_4bit=gemma_load_in_4bit,
                    )
                else:
                    self._run_text_encoder_cache_stage(
                        2,
                        total_stages,
                        "Cache text encoder outputs",
                        python_exe,
                        dataset_config,
                        ltx2_checkpoint,
                        gemma_root,
                        "video",
                        gemma_load_in_8bit,
                        musubi_root,
                        log_handle,
                        [
                            f"Gemma root: {gemma_root}",
                            "This is usually the slowest setup stage.",
                            "You should see per-item progress from the text encoder cache script.",
                        ],
                        gemma_load_in_4bit=gemma_load_in_4bit,
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


class VRGDG_LTXAudioOnlyLoraTrainChunk(VRGDG_LTXLoraTrainChunk):
    DESCRIPTION = "Runs one LTX-2.3 audio-only LoRA training chunk using musubi-tuner on spoken dialogue or music clips."

    AUDIO_PROFILE_PRESETS = {
        "voice_test": {
            "audio_only_target_resolution": 64,
            "audio_only_target_fps": 25.0,
            "audio_only_sequence_resolution": 64,
            "steps_per_run": 100,
            "total_target_steps": 100,
            "network_dim": 16,
            "network_alpha": 16,
            "blocks_to_swap": 0,
            "lora_target_preset": "audio",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "1e-4",
            "learning_rate": 1e-4,
            "num_repeats": 1,
            "audio_bucket_strategy": "pad",
            "audio_bucket_interval": 2.0,
            "ltx2_audio_only_model": True,
        },
        "voice": {
            "audio_only_target_resolution": 64,
            "audio_only_target_fps": 25.0,
            "audio_only_sequence_resolution": 64,
            "steps_per_run": 100,
            "total_target_steps": 800,
            "network_dim": 16,
            "network_alpha": 16,
            "blocks_to_swap": 2,
            "lora_target_preset": "audio",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "1e-4",
            "learning_rate": 1e-4,
            "num_repeats": 2,
            "audio_bucket_strategy": "pad",
            "audio_bucket_interval": 2.0,
            "ltx2_audio_only_model": True,
        },
        "music": {
            "audio_only_target_resolution": 64,
            "audio_only_target_fps": 25.0,
            "audio_only_sequence_resolution": 128,
            "steps_per_run": 200,
            "total_target_steps": 1200,
            "network_dim": 32,
            "network_alpha": 32,
            "blocks_to_swap": 4,
            "lora_target_preset": "audio",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "7e-5",
            "learning_rate": 7e-5,
            "num_repeats": 4,
            "audio_bucket_strategy": "truncate",
            "audio_bucket_interval": 4.0,
            "ltx2_audio_only_model": True,
        },
        "voice_fast": {
            "audio_only_target_resolution": 64,
            "audio_only_target_fps": 25.0,
            "audio_only_sequence_resolution": 64,
            "steps_per_run": 100,
            "total_target_steps": 800,
            "network_dim": 16,
            "network_alpha": 16,
            "blocks_to_swap": 2,
            "lora_target_preset": "audio",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "1e-4",
            "learning_rate": 1e-4,
            "num_repeats": 2,
            "audio_bucket_strategy": "pad",
            "audio_bucket_interval": 2.0,
            "ltx2_audio_only_model": True,
        },
        "voice_balanced": {
            "audio_only_target_resolution": 64,
            "audio_only_target_fps": 25.0,
            "audio_only_sequence_resolution": 128,
            "steps_per_run": 200,
            "total_target_steps": 1200,
            "network_dim": 32,
            "network_alpha": 32,
            "blocks_to_swap": 4,
            "lora_target_preset": "audio",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "7e-5",
            "learning_rate": 7e-5,
            "num_repeats": 4,
            "audio_bucket_strategy": "truncate",
            "audio_bucket_interval": 4.0,
            "ltx2_audio_only_model": True,
        },
    }

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "model",
        "latest_state_path",
        "log_path",
        "dataset_audio_dir",
        "output_name",
        "completed_steps",
        "total_target_steps",
    )
    FUNCTION = "run"
    CATEGORY = "VRGDG/Training"

    @classmethod
    def _profile_names(cls):
        return ["voice_test", "voice", "music", "voice_fast", "voice_balanced"]

    @classmethod
    def _profile_defaults(cls, profile):
        return dict(cls.AUDIO_PROFILE_PRESETS.get(str(profile or "voice"), cls.AUDIO_PROFILE_PRESETS["voice"]))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base model to return downstream with the latest trained LoRA optionally applied."}),
                "workspace_dir": ("STRING", {"default": "", "multiline": False, "tooltip": "Working folder for cache, logs, config files, checkpoints, and training state."}),
                "run_name": ("STRING", {"default": "LTXAudioChunkRun", "multiline": False, "tooltip": "Name prefix used for the log file."}),
                "output_name": ("STRING", {"default": "LTXAudioChunkRun", "multiline": False, "tooltip": "Name prefix used for saved LoRA files and state folders."}),
                "audio_profile": (cls._profile_names(), {"default": "voice_test", "tooltip": "voice_test is a short spoken-dialogue sanity check. voice is the spoken-dialogue preset. music uses a larger temporal budget and more capacity. voice_fast and voice_balanced are legacy aliases."}),
                "use_profile_defaults": ("BOOLEAN", {"default": True, "tooltip": "When enabled, the selected profile controls the audio recipe bundle instead of the manual fields below."}),
                "audio_only_target_resolution": ("INT", {"default": 64, "min": 32, "max": 4096, "step": 1, "tooltip": "Square target resolution used for audio-only latent geometry."}),
                "audio_only_target_fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.1, "tooltip": "Target FPS used to derive frame count from audio duration during caching."}),
                "audio_only_sequence_resolution": ("INT", {"default": 64, "min": 0, "max": 8192, "step": 1, "tooltip": "Virtual sequence resolution used for shifted_logit_normal in audio mode. Set 0 to use cached virtual geometry."}),
                "steps_per_run": ("INT", {"default": 100, "min": 1, "max": 100000, "step": 1, "tooltip": "How many steps to train before saving and stopping this chunk."}),
                "total_target_steps": ("INT", {"default": 800, "min": 1, "max": 1000000, "step": 1, "tooltip": "Total training budget across all chunks."}),
                "network_dim": ("INT", {"default": 16, "min": 1, "max": 2048, "step": 1, "tooltip": "LoRA rank."}),
                "network_alpha": ("INT", {"default": 16, "min": 1, "max": 2048, "step": 1, "tooltip": "LoRA alpha."}),
                "blocks_to_swap": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1, "tooltip": "How many transformer blocks to offload to CPU."}),
                "lora_target_preset": (["audio", "t2v", "v2v", "full"], {"default": "audio", "tooltip": "audio is the intended preset for audio-only training."}),
                "fp8_base": ("BOOLEAN", {"default": True, "tooltip": "Use the FP8 base-model loading path."}),
                "fp8_scaled": ("BOOLEAN", {"default": True, "tooltip": "Quantize non-FP8 checkpoints into FP8 at load time."}),
                "ltx2_audio_only_model": ("BOOLEAN", {"default": True, "tooltip": "Force loading the physically audio-only transformer variant. Keep this on for audio-only training."}),
                "clear_memory_before_gemma": ("BOOLEAN", {"default": True, "tooltip": "Unloads ComfyUI models and clears memory before Gemma caching."}),
                "gemma_recovery_mode": ("BOOLEAN", {"default": False, "tooltip": "Experimental. If enabled, the node will try alternate Gemma cache settings after the normal path fails."}),
                "learning_rate_preset": (["Custom", "1e-4", "7e-5", "5e-5", "3e-5", "1e-5"], {"default": "1e-4", "tooltip": "Quick preset for the training learning rate."}),
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-6, "tooltip": "Custom learning rate used only when the preset is set to Custom."}),
                "num_repeats": ("INT", {"default": 2, "min": 1, "max": 1000, "step": 1, "tooltip": "How many times each audio-caption pair is repeated in the dataset."}),
                "audio_bucket_strategy": (["pad", "truncate"], {"default": "pad", "tooltip": "Audio duration bucketing strategy."}),
                "audio_bucket_interval": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 120.0, "step": 0.1, "tooltip": "Audio bucket step size in seconds."}),
                "cache_strategy": (["auto", "force", "skip"], {"default": "auto", "tooltip": "Auto rebuilds cache only when the node detects it is missing or stale."}),
                "copy_latest_to_comfy_loras": ("BOOLEAN", {"default": True, "tooltip": "Copies the latest Comfy-compatible LoRA into the ComfyUI loras folder after training."}),
                "keep_only_comfy_lora": ("BOOLEAN", {"default": False, "tooltip": "Deletes the standard .safetensors LoRA files after a matching .comfy.safetensors file exists."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Strength used if the node applies the latest LoRA back onto the output model."}),
                "create_captions": ("BOOLEAN", {"default": False, "tooltip": "Creates missing .txt caption files automatically from the caption_text field."}),
                "caption_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Fallback caption text used when a source audio clip has no caption file."}),
                "add_trigger_word": ("BOOLEAN", {"default": False, "tooltip": "Prepends trigger_text to every caption."}),
                "trigger_text": ("STRING", {"default": "", "multiline": False, "tooltip": "Trigger word or phrase to prepend to captions when add_trigger_word is enabled."}),
                "musubi_root": ("STRING", {"default": "A:/MUSUBI/musubi-tuner-ltx2", "multiline": False, "tooltip": "Root folder of your musubi-tuner-ltx2 install."}),
                "ltx2_checkpoint": ("STRING", {"default": "A:/MUSUBI/models/ltx/ltx-2.3-22b-dev.safetensors", "multiline": False, "tooltip": "Path to the base LTX checkpoint used for caching and training."}),
                "gemma_root": ("STRING", {"default": "A:/MUSUBI/models/gemma3", "multiline": False, "tooltip": "Folder containing the Gemma model files used for text encoder caching."}),
                "gemma_load_in_4bit": ("BOOLEAN", {"default": False, "tooltip": "Loads Gemma in 4-bit mode instead of 8-bit."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional connected AUDIO input. If provided, it is copied into a managed audio dataset folder and used instead of audio_directory."}),
                "audio_directory": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional folder containing audio files with matching .txt captions. Leave empty if you connect an AUDIO input."}),
            },
        }

    def _count_audio_dataset_files(self, audio_dir):
        audio_count = 0
        caption_count = 0
        if not os.path.isdir(audio_dir):
            return audio_count, caption_count
        for entry in os.scandir(audio_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                audio_count += 1
            elif ext == ".txt":
                caption_count += 1
        return audio_count, caption_count

    def _resolve_audio_input_name(self, audio):
        if not isinstance(audio, dict):
            return "audio_input"
        for key in ("filename", "file_name", "name"):
            value = str(audio.get(key) or "").strip()
            if value:
                return self._safe_name(value, "audio_input")
        for key in ("file_path", "audio_file_path", "path"):
            value = str(audio.get(key) or "").strip()
            if value:
                stem = os.path.splitext(os.path.basename(value))[0]
                if stem:
                    return self._safe_name(stem, "audio_input")
        return "audio_input"

    def _resolve_audio_input_source_path(self, audio):
        if not isinstance(audio, dict):
            return ""
        for key in ("file_path", "audio_file_path", "path"):
            value = str(audio.get(key) or "").strip()
            if value and os.path.isfile(value):
                return os.path.normpath(value)
        return ""

    def _normalize_audio_waveform(self, audio):
        if not isinstance(audio, dict):
            return None, None
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            return None, None
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim != 3:
            raise ValueError(f"Audio waveform must be 1D, 2D, or 3D, got {tuple(waveform.shape)}")
        return waveform.float(), int(sample_rate)

    def _ensure_audio_captions(self, audio_dir, create_captions, caption_text, add_trigger_word, trigger_text):
        audio_entries = [
            entry
            for entry in os.scandir(audio_dir)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in AUDIO_EXTENSIONS
        ]
        base_caption = str(caption_text or "").strip()
        created_count = 0
        updated_count = 0
        for entry in audio_entries:
            stem = os.path.splitext(entry.name)[0]
            caption_path = os.path.join(audio_dir, f"{stem}.txt")

            existing_caption = ""
            if os.path.isfile(caption_path):
                with open(caption_path, "r", encoding="utf-8") as handle:
                    existing_caption = handle.read().strip()

            caption_body = existing_caption if existing_caption else base_caption
            final_caption = self._compose_caption_text(caption_body, add_trigger_word, trigger_text)
            if not final_caption and not existing_caption and not create_captions and not base_caption:
                final_caption = ""

            if existing_caption == final_caption:
                continue

            with open(caption_path, "w", encoding="utf-8") as handle:
                handle.write(final_caption)

            if existing_caption:
                updated_count += 1
            else:
                created_count += 1

        print(
            f"[VRGDG] Audio caption prep complete. created={created_count} updated={updated_count} "
            f"trigger={'on' if add_trigger_word else 'off'}"
        )

    def _prepare_audio_dataset_directory(
        self,
        audio_directory,
        audio,
        create_captions,
        caption_text,
        add_trigger_word,
        trigger_text,
        workspace_dir,
        run_name,
    ):
        if audio is not None:
            managed_dir = self._ensure_dir(os.path.join(workspace_dir, "audio_inputs", run_name, "audio"))
            for entry in os.scandir(managed_dir):
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in AUDIO_EXTENSIONS or ext == ".txt":
                    os.remove(entry.path)

            input_name = self._resolve_audio_input_name(audio)
            source_path = self._resolve_audio_input_source_path(audio)
            dest_audio_path = os.path.join(managed_dir, f"{input_name}.wav")
            if source_path:
                source_ext = os.path.splitext(source_path)[1].lower()
                if source_ext in AUDIO_EXTENSIONS:
                    dest_audio_path = os.path.join(managed_dir, f"{input_name}{source_ext}")
                    shutil.copy2(source_path, dest_audio_path)
                else:
                    waveform, sample_rate = self._normalize_audio_waveform(audio)
                    if waveform is None or sample_rate is None:
                        raise ValueError("Invalid AUDIO input. Expected waveform and sample_rate.")
                    if torchaudio is None:
                        raise ImportError("torchaudio is required to save AUDIO inputs to a managed dataset folder.")
                    save_waveform = waveform.detach().cpu()
                    if save_waveform.ndim == 3:
                        save_waveform = save_waveform[0]
                    torchaudio.save(dest_audio_path, save_waveform, int(sample_rate))
            else:
                waveform, sample_rate = self._normalize_audio_waveform(audio)
                if waveform is None or sample_rate is None:
                    raise ValueError("Invalid AUDIO input. Expected waveform and sample_rate.")
                if torchaudio is None:
                    raise ImportError("torchaudio is required to save AUDIO inputs to a managed dataset folder.")
                save_waveform = waveform.detach().cpu()
                if save_waveform.ndim == 3:
                    save_waveform = save_waveform[0]
                torchaudio.save(dest_audio_path, save_waveform, int(sample_rate))

            self._ensure_audio_captions(
                managed_dir,
                True,
                caption_text if str(caption_text or "").strip() else input_name,
                add_trigger_word,
                trigger_text,
            )
            print(f"[VRGDG] Managed audio dataset prepared from AUDIO input: {managed_dir}")
            return managed_dir

        audio_directory = str(audio_directory or "").strip()
        if not audio_directory:
            raise ValueError("Provide either a connected AUDIO input or an audio_directory path.")
        audio_directory = self._norm(audio_directory)
        if not os.path.isdir(audio_directory):
            raise ValueError(f"audio_directory does not exist: {audio_directory}")

        self._ensure_audio_captions(
            audio_directory,
            create_captions,
            caption_text,
            add_trigger_word,
            trigger_text,
        )
        return audio_directory

    def run(
        self,
        model,
        workspace_dir,
        run_name,
        output_name,
        audio_profile,
        use_profile_defaults,
        audio_only_target_resolution,
        audio_only_target_fps,
        audio_only_sequence_resolution,
        steps_per_run,
        total_target_steps,
        network_dim,
        network_alpha,
        blocks_to_swap,
        lora_target_preset,
        fp8_base,
        fp8_scaled,
        ltx2_audio_only_model,
        clear_memory_before_gemma,
        gemma_recovery_mode,
        learning_rate_preset,
        learning_rate,
        num_repeats,
        audio_bucket_strategy,
        audio_bucket_interval,
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
        gemma_load_in_4bit=False,
        audio=None,
        audio_directory="",
    ):
        workspace_dir = self._norm(workspace_dir)
        musubi_root = self._norm(musubi_root)
        ltx2_checkpoint = self._norm(ltx2_checkpoint)
        gemma_root = self._norm(gemma_root)
        gemma_load_in_4bit = bool(gemma_load_in_4bit)
        audio_profile = str(audio_profile or "voice_fast").strip()
        profile_defaults = self._profile_defaults(audio_profile) if bool(use_profile_defaults) else {}
        if profile_defaults:
            audio_only_target_resolution = profile_defaults["audio_only_target_resolution"]
            audio_only_target_fps = profile_defaults["audio_only_target_fps"]
            audio_only_sequence_resolution = profile_defaults["audio_only_sequence_resolution"]
            steps_per_run = profile_defaults["steps_per_run"]
            total_target_steps = profile_defaults["total_target_steps"]
            network_dim = profile_defaults["network_dim"]
            network_alpha = profile_defaults["network_alpha"]
            blocks_to_swap = profile_defaults["blocks_to_swap"]
            lora_target_preset = profile_defaults["lora_target_preset"]
            fp8_base = profile_defaults["fp8_base"]
            fp8_scaled = profile_defaults["fp8_scaled"]
            learning_rate_preset = profile_defaults["learning_rate_preset"]
            learning_rate = profile_defaults["learning_rate"]
            num_repeats = profile_defaults["num_repeats"]
            audio_bucket_strategy = profile_defaults["audio_bucket_strategy"]
            audio_bucket_interval = profile_defaults["audio_bucket_interval"]
            ltx2_audio_only_model = profile_defaults["ltx2_audio_only_model"]

        gemma_load_in_8bit = not gemma_load_in_4bit
        gemma_load_mode = "4bit" if gemma_load_in_4bit else "8bit"
        run_name = self._safe_name(run_name, "LTXAudioChunkRun")
        output_name = self._safe_name(output_name, run_name)
        effective_learning_rate = self._resolve_learning_rate(learning_rate_preset, learning_rate)

        workspace_dir = self._ensure_dir(workspace_dir)
        if not os.path.isdir(musubi_root):
            raise ValueError(f"musubi_root does not exist: {musubi_root}")
        if not os.path.isfile(ltx2_checkpoint):
            raise ValueError(f"ltx2_checkpoint does not exist: {ltx2_checkpoint}")
        if not os.path.isdir(gemma_root):
            raise ValueError(f"gemma_root does not exist: {gemma_root}")

        dataset_audio_dir = self._prepare_audio_dataset_directory(
            audio_directory,
            audio,
            create_captions,
            caption_text,
            add_trigger_word,
            trigger_text,
            workspace_dir,
            run_name,
        )

        audio_only_target_resolution = int(audio_only_target_resolution)
        audio_only_target_fps = float(audio_only_target_fps)
        audio_only_sequence_resolution = int(audio_only_sequence_resolution)
        audio_bucket_strategy = str(audio_bucket_strategy or "pad").strip().lower()
        if audio_bucket_strategy not in {"pad", "truncate"}:
            raise ValueError(f"audio_bucket_strategy must be 'pad' or 'truncate', got '{audio_bucket_strategy}'")
        audio_bucket_interval = float(audio_bucket_interval)

        python_exe, accelerate_exe, env_source = self._resolve_musubi_executables(musubi_root)

        cache_dir = self._ensure_dir(os.path.join(workspace_dir, "cache"))
        output_dir = self._ensure_dir(os.path.join(workspace_dir, "output"))
        logs_dir = self._ensure_dir(os.path.join(workspace_dir, "logs"))
        config_dir = self._ensure_dir(os.path.join(workspace_dir, "config"))
        dataset_config = os.path.join(config_dir, "dataset-audio.toml")
        training_config = os.path.join(config_dir, "training_args_audio.toml")
        cache_signature_path = os.path.join(cache_dir, "ltx2_audio_cache_signature.json")
        current_cache_signature = {
            "dataset_audio_dir": dataset_audio_dir,
            "audio_only_target_resolution": audio_only_target_resolution,
            "audio_only_target_fps": audio_only_target_fps,
            "audio_only_sequence_resolution": audio_only_sequence_resolution,
            "audio_bucket_strategy": audio_bucket_strategy,
            "audio_bucket_interval": audio_bucket_interval,
            "num_repeats": int(num_repeats),
            "lora_target_preset": str(lora_target_preset),
            "fp8_base": bool(fp8_base),
            "fp8_scaled": bool(fp8_scaled),
            "ltx2_audio_only_model": bool(ltx2_audio_only_model),
            "use_profile_defaults": bool(use_profile_defaults),
            "audio_profile": audio_profile,
            "ltx2_mode": "audio",
        }

        latest_state_path, completed_steps = self._latest_state_dir(output_dir, output_name)
        if completed_steps >= int(total_target_steps):
            raise RuntimeError(
                f"Training complete: reached {completed_steps}/{int(total_target_steps)} steps. Stopping workflow."
            )

        next_target_steps = min(completed_steps + int(steps_per_run), int(total_target_steps))

        self._write_audio_dataset_config(
            dataset_config,
            dataset_audio_dir,
            cache_dir,
            audio_only_target_resolution,
            num_repeats,
            audio_bucket_strategy,
            audio_bucket_interval,
        )
        self._write_audio_training_config(
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
            lora_target_preset,
            fp8_base,
            fp8_scaled,
            audio_only_sequence_resolution,
            ltx2_audio_only_model,
        )

        log_path = os.path.join(logs_dir, f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(log_path, "w", encoding="utf-8") as log_handle:
            log_handle.write(f"dataset_audio_dir={dataset_audio_dir}\n")
            log_handle.write(f"workspace_dir={workspace_dir}\n")
            log_handle.write(f"completed_steps={completed_steps}\n")
            log_handle.write(f"next_target_steps={next_target_steps}\n\n")
            log_handle.flush()

            audio_count, caption_count = self._count_audio_dataset_files(dataset_audio_dir)
            cache_file_count_before = self._count_cache_files(cache_dir)
            cache_signature_matches = False
            if os.path.isfile(cache_signature_path):
                try:
                    with open(cache_signature_path, "r", encoding="utf-8") as handle:
                        cached_signature = json.load(handle)
                    cache_signature_matches = cached_signature == current_cache_signature
                except Exception:
                    cache_signature_matches = False
            if cache_strategy == "force":
                should_build_cache = True
            elif cache_strategy == "skip":
                should_build_cache = False
            else:
                should_build_cache = not self._has_latent_cache_files(cache_dir) or not self._has_complete_text_encoder_cache_files(cache_dir)
                if self._has_latent_cache_files(cache_dir) and self._has_complete_text_encoder_cache_files(cache_dir) and not cache_signature_matches:
                    should_build_cache = True
            total_stages = 3 if should_build_cache else 1

            print(f"[VRGDG] dataset_audio_dir={dataset_audio_dir}")
            print(f"[VRGDG] workspace_dir={workspace_dir}")
            print(f"[VRGDG] completed_steps={completed_steps}")
            print(f"[VRGDG] next_target_steps={next_target_steps}")
            print(f"[VRGDG] steps_per_run_and_save={steps_per_run}")
            print(f"[VRGDG] total_target_steps={total_target_steps}")
            print(f"[VRGDG] blocks_to_swap={int(blocks_to_swap)}")
            print(f"[VRGDG] musubi_env_source={env_source}")
            print(f"[VRGDG] musubi_python={python_exe}")
            print(f"[VRGDG] musubi_accelerate={accelerate_exe}")
            print(f"[VRGDG] clear_memory_before_gemma={clear_memory_before_gemma}")
            print(f"[VRGDG] gemma_load_mode={gemma_load_mode}")
            print(f"[VRGDG] gemma_recovery_mode={gemma_recovery_mode}")
            print(f"[VRGDG] ltx2_audio_only_model={bool(ltx2_audio_only_model)}")
            print(f"[VRGDG] dataset summary: audio={audio_count} captions={caption_count}")
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
                    "Cache audio latents",
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
                        "audio",
                        "--audio_only_target_resolution",
                        str(int(audio_only_target_resolution)),
                        "--audio_only_target_fps",
                        str(float(audio_only_target_fps)),
                        "--audio_only_sequence_resolution",
                        str(int(audio_only_sequence_resolution)),
                    ],
                    musubi_root,
                    log_handle,
                    [
                        f"Dataset audio dir: {dataset_audio_dir}",
                        f"Audio files found: {audio_count}",
                        f"Captions found: {caption_count}",
                        f"Cache dir: {cache_dir}",
                    ],
                )
                if clear_memory_before_gemma:
                    self._clear_memory_before_gemma(log_handle)
                if gemma_recovery_mode:
                    self._run_text_encoder_cache_stage_with_recovery(
                        2,
                        total_stages,
                        "Cache audio text encoder outputs",
                        python_exe,
                        dataset_config,
                        ltx2_checkpoint,
                        gemma_root,
                        "audio",
                        gemma_load_in_8bit,
                        gemma_recovery_mode,
                        musubi_root,
                        log_handle,
                        [
                            f"Gemma root: {gemma_root}",
                            "This is usually the slowest setup stage.",
                            "You should see per-item progress from the text encoder cache script.",
                        ],
                        gemma_load_in_4bit=gemma_load_in_4bit,
                    )
                else:
                    self._run_text_encoder_cache_stage(
                        2,
                        total_stages,
                        "Cache audio text encoder outputs",
                        python_exe,
                        dataset_config,
                        ltx2_checkpoint,
                        gemma_root,
                        "audio",
                        gemma_load_in_8bit,
                        musubi_root,
                        log_handle,
                        [
                            f"Gemma root: {gemma_root}",
                            "This is usually the slowest setup stage.",
                            "You should see per-item progress from the text encoder cache script.",
                        ],
                        gemma_load_in_4bit=gemma_load_in_4bit,
                    )
                with open(cache_signature_path, "w", encoding="utf-8") as handle:
                    json.dump(current_cache_signature, handle, indent=2, sort_keys=True)
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
                "--ltx2_mode",
                "audio",
                "--audio_only_sequence_resolution",
                str(int(audio_only_sequence_resolution)),
            ]
            if ltx2_audio_only_model:
                train_command.append("--ltx2_audio_only_model")
            if latest_state_path:
                train_command.extend(["--resume", latest_state_path])

            self._run_stage_command(
                total_stages,
                total_stages,
                "Train audio LoRA",
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
            with open(cache_signature_path, "w", encoding="utf-8") as handle:
                json.dump(current_cache_signature, handle, indent=2, sort_keys=True)

        latest_lora_path, latest_lora_step = self._latest_file(output_dir, output_name, ".safetensors")
        latest_comfy_lora_path, latest_comfy_step = self._latest_file(output_dir, output_name, ".comfy.safetensors")
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
            print(f"[VRGDG] keep_only_comfy_lora deleted {deleted_count} standard LoRA file(s).")

        if copy_latest_to_comfy_loras and latest_comfy_lora_path:
            latest_comfy_lora_path = self._export_latest_to_comfy(latest_comfy_lora_path, output_name)

        applied_lora_path = latest_comfy_lora_path or latest_lora_path
        self._log_message(
            f"[VRGDG] Latest state selected: {os.path.normpath(latest_state_path) if latest_state_path else '(none)'}",
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
            os.path.normpath(dataset_audio_dir),
            output_name,
            int(completed_steps),
            int(total_target_steps),
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
                    "default": "A:/MUSUBI/musubi-tuner-ltx2",
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
                "gemma_recovery_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental. If enabled, the node will try alternate Gemma cache settings after the normal path fails."
                }),
                "gemma_load_in_4bit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loads Gemma in 4-bit mode during text encoder caching. This lowers VRAM more than 8-bit, but can be slower or less stable."
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
        gemma_recovery_mode,
        cache_strategy,
        strength_model,
        musubi_root,
        ltx2_checkpoint,
        gemma_root,
        gemma_load_in_4bit,
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
            gemma_recovery_mode=gemma_recovery_mode,
            gemma_load_in_4bit=gemma_load_in_4bit,
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


class VRGDG_LTXAudioVideoLoraTrainChunk(VRGDG_LTXLoraTrainChunk):
    DESCRIPTION = (
        "Runs one LTX-2.3 audio-video LoRA training chunk using musubi-tuner on short videos with embedded audio."
    )

    AV_PROFILE_PRESETS = {
        "fast_short_clip": {
            "resolution_width": 768,
            "resolution_height": 512,
            "target_frames": "17,33",
            "frame_extraction": "head",
            "max_frames": 49,
            "target_fps": 25.0,
            "steps_per_run": 100,
            "total_target_steps": 400,
            "network_dim": 16,
            "network_alpha": 16,
            "blocks_to_swap": 2,
            "separate_audio_buckets": True,
            "lora_target_preset": "t2v",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "1e-4",
            "learning_rate": 1e-4,
            "num_repeats": 8,
        },
        "step200": {
            "resolution_width": 768,
            "resolution_height": 512,
            "target_frames": "17,33",
            "frame_extraction": "head",
            "max_frames": 49,
            "target_fps": 25.0,
            "steps_per_run": 200,
            "total_target_steps": 200,
            "network_dim": 16,
            "network_alpha": 16,
            "blocks_to_swap": 2,
            "separate_audio_buckets": True,
            "lora_target_preset": "t2v",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "1e-4",
            "learning_rate": 1e-4,
            "num_repeats": 2,
        },
        "balanced": {
            "resolution_width": 768,
            "resolution_height": 512,
            "target_frames": "17,33,49",
            "frame_extraction": "head",
            "max_frames": 97,
            "target_fps": 25.0,
            "steps_per_run": 250,
            "total_target_steps": 1200,
            "network_dim": 32,
            "network_alpha": 32,
            "blocks_to_swap": 4,
            "separate_audio_buckets": True,
            "lora_target_preset": "v2v",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "7e-5",
            "learning_rate": 7e-5,
            "num_repeats": 4,
        },
        "full_video": {
            "resolution_width": 960,
            "resolution_height": 544,
            "target_frames": "1",
            "frame_extraction": "full",
            "max_frames": 129,
            "target_fps": 25.0,
            "steps_per_run": 250,
            "total_target_steps": 3000,
            "network_dim": 32,
            "network_alpha": 32,
            "blocks_to_swap": 4,
            "separate_audio_buckets": True,
            "lora_target_preset": "full",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "7e-5",
            "learning_rate": 7e-5,
            "num_repeats": 1,
        },
        "overnight_full_video": {
            "resolution_width": 960,
            "resolution_height": 544,
            "target_frames": "17",
            "frame_extraction": "chunk",
            "max_frames": 65,
            "target_fps": 25.0,
            "steps_per_run": 250,
            "total_target_steps": 1000,
            "network_dim": 16,
            "network_alpha": 16,
            "blocks_to_swap": 4,
            "separate_audio_buckets": True,
            "lora_target_preset": "v2v",
            "fp8_base": True,
            "fp8_scaled": True,
            "learning_rate_preset": "1e-4",
            "learning_rate": 1e-4,
            "num_repeats": 1,
        },
    }

    @classmethod
    def _profile_names(cls):
        return list(cls.AV_PROFILE_PRESETS.keys())

    @classmethod
    def _profile_defaults(cls, profile):
        return dict(cls.AV_PROFILE_PRESETS.get(str(profile or "fast_short_clip"), cls.AV_PROFILE_PRESETS["fast_short_clip"]))

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
    def _parse_boolish(value):
        return bool(value)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Base model to return downstream with the latest trained LoRA optionally applied."
                }),
                "dataset_videos_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Folder containing your training videos, or a parent folder that will be organized into a videos subfolder."
                }),
                "workspace_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Working folder for cache, logs, config files, checkpoints, and training state."
                }),
                "run_name": ("STRING", {
                    "default": "LTXAVChunkRun",
                    "multiline": False,
                    "tooltip": "Name prefix used for the log file."
                }),
                "output_name": ("STRING", {
                    "default": "LTXAVChunkRun",
                    "multiline": False,
                    "tooltip": "Name prefix used for saved LoRA files and state folders."
                }),
                "av_profile": ([
                    "fast_short_clip",
                    "step200",
                    "balanced",
                    "full_video",
                    "overnight_full_video",
                ], {
                    "default": "fast_short_clip",
                    "tooltip": "Training recipe switch. fast_short_clip is the default quick test for one short video. step200 keeps the same fast_short_clip recipe but sets a 200-step chunk budget with lighter repeats. balanced is a stronger middle ground. full_video trains on the whole clip and is the slowest. overnight_full_video is tuned for a longer clip using smaller frame windows, lower resolution, and light swapping so it can run overnight on a midrange GPU."
                }),
                "use_profile_defaults": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, the selected profile controls the AV recipe bundle instead of the manual fields below. Turn this off if you want to hand-tune every value."
                }),
                "resolution_width": ("INT", {
                    "default": 768, "min": 64, "max": 8192, "step": 1,
                    "tooltip": "Training bucket width in pixels. 768 is a good fast default for a single short clip. Smaller values like 512 are faster; larger values cost more VRAM and time."
                }),
                "resolution_height": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 1,
                    "tooltip": "Training bucket height in pixels. 512 pairs well with 768x512 for a quick short-clip run. Use a lower height for faster experiments."
                }),
                "target_frames": ("STRING", {
                    "default": "17,33",
                    "multiline": False,
                    "tooltip": "Comma-separated frame buckets to train on. For a 10-second clip, 17 and 33 frames are a good fast starting point. The trainer rounds values to the required N*4+1 frame buckets."
                }),
                "frame_extraction": ([
                    "head",
                    "chunk",
                    "slide",
                    "uniform",
                    "full",
                ], {
                    "default": "head",
                    "tooltip": "How frames are selected from each video. head uses only the first target_frames window and is the fastest choice for a short single-clip LoRA. full uses the entire video, but is much slower and should be capped with max_frames."
                }),
                "max_frames": ("INT", {
                    "default": 49, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "Safety cap used when frame_extraction is full. 49 keeps the run short and prevents a 10-second clip from turning into a long heavy sample. It does not affect head/chunk/slide/uniform."
                }),
                "target_fps": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 240.0, "step": 0.1,
                    "tooltip": "Target FPS used during caching/training. 25.0 is the standard LTX rate and works well for most short clips with audio."
                }),
                "steps_per_run": ("INT", {
                    "default": 100, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "How many steps to train before saving and stopping this chunk. 100 is a fast check-run default for a single short clip."
                }),
                "total_target_steps": ("INT", {
                    "default": 400, "min": 1, "max": 1000000, "step": 1,
                    "tooltip": "Total training budget across all chunks. 400 is a fast starter value for one short video; increase later only if you need more adaptation."
                }),
                "network_dim": ("INT", {
                    "default": 16, "min": 1, "max": 2048, "step": 1,
                    "tooltip": "LoRA rank. 16 is a good quick-training default for one short clip. Use 32 if the result is too weak."
                }),
                "network_alpha": ("INT", {
                    "default": 16, "min": 1, "max": 2048, "step": 1,
                    "tooltip": "LoRA alpha. Usually keep this equal to the rank for a simple fast baseline."
                }),
                "blocks_to_swap": ("INT", {
                    "default": 2, "min": 0, "max": 64, "step": 1,
                    "tooltip": "How many transformer blocks to offload to CPU. Lower is faster if you have enough VRAM. Raise this only if you run out of memory."
                }),
                "separate_audio_buckets": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keeps audio and non-audio items in separate batches. Safe to leave on and useful if your dataset ever mixes audio and silent clips."
                }),
                "clear_memory_before_gemma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unloads ComfyUI models and clears memory before Gemma caching. Leave this on unless you know your workspace is already empty."
                }),
                "lora_target_preset": ([
                    "t2v",
                    "v2v",
                    "audio",
                    "full",
                ], {
                    "default": "t2v",
                    "tooltip": "Which transformer layers get LoRA adapters. t2v is the fastest useful baseline. v2v adds FFN layers. full is the heaviest and produces the largest adapter."
                }),
                "fp8_base": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use the FP8 base-model loading path. Leave this on for speed and lower VRAM."
                }),
                "fp8_scaled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Quantize non-FP8 checkpoints into FP8 at load time. Turn this off if your checkpoint file is already FP8, usually visible from the filename."
                }),
                "learning_rate_preset": ([
                    "Custom",
                    "1e-4",
                    "7e-5",
                    "5e-5",
                    "3e-5",
                    "1e-5",
                ], {
                    "default": "1e-4",
                    "tooltip": "Quick preset for the training learning rate. 1e-4 is a strong starter value for a short 10-second clip."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-6,
                    "tooltip": "Custom learning rate used only when the preset is set to Custom."
                }),
                "num_repeats": ("INT", {
                    "default": 8, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "How many times each video-caption pair is repeated in the dataset. Higher repeats help a single clip produce enough training signal without needing a huge number of source videos."
                }),
                "cache_strategy": (["auto", "force", "skip"], {
                    "default": "auto",
                    "tooltip": "Auto rebuilds cache only when the node detects the cache is missing or stale. Force always rebuilds. Skip assumes the cache is already correct."
                }),
                "copy_latest_to_comfy_loras": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Copies the latest Comfy-compatible LoRA into the ComfyUI loras folder after training."
                }),
                "keep_only_comfy_lora": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, deletes the standard .safetensors LoRA files after a matching .comfy.safetensors file exists."
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Strength used if the node applies the latest LoRA back onto the output model."
                }),
                "create_captions": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Creates missing .txt caption files automatically from the caption_text field. Leave this off if your videos already have captions."
                }),
                "caption_text": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Fallback caption text used only when create_captions is on and a video has no caption file."
                }),
                "add_trigger_word": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Prepends trigger_text to every caption. Useful when you want a fixed concept token."
                }),
                "trigger_text": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Trigger word or phrase to prepend to captions when add_trigger_word is enabled."
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
                        "tooltip": "Path to the base LTX audio-video checkpoint used for caching and training."
                    },
                ),
                "gemma_root": ("STRING", {
                    "default": "A:/MUSUBI/models/gemma3", "multiline": False,
                    "tooltip": "Folder containing the Gemma model files used for text encoder caching."
                }),
                "gemma_load_in_4bit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loads Gemma in 4-bit mode instead of 8-bit. This reduces VRAM further, but can be slower or a little less stable."
                }),
                "gemma_recovery_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental. If enabled, the node will try alternate Gemma cache settings after the normal path fails."
                }),
            }
        }

    @staticmethod
    def _parse_target_frames(value):
        text = str(value or "").strip()
        if not text:
            return [1, 17, 33, 49]
        frames = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            frames.append(int(part))
        if not frames:
            raise ValueError("target_frames must contain at least one integer value.")
        return frames

    def _count_video_dataset_files(self, videos_dir):
        video_count = 0
        caption_count = 0
        if not os.path.isdir(videos_dir):
            return video_count, caption_count
        for entry in os.scandir(videos_dir):
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in self.VIDEO_EXTENSIONS:
                video_count += 1
            elif ext == ".txt":
                caption_count += 1
        return video_count, caption_count

    def _ensure_video_captions(self, videos_dir, create_captions, caption_text, add_trigger_word, trigger_text):
        video_entries = [
            entry for entry in os.scandir(videos_dir)
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in self.VIDEO_EXTENSIONS
        ]
        base_caption = str(caption_text or "").strip()

        created_count = 0
        updated_count = 0
        for entry in video_entries:
            stem = os.path.splitext(entry.name)[0]
            caption_path = os.path.join(videos_dir, f"{stem}.txt")

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
            f"[VRGDG] Video caption prep complete. created={created_count} updated={updated_count} "
            f"trigger={'on' if add_trigger_word else 'off'}"
        )

    def _prepare_video_dataset_directory(
        self,
        dataset_root,
        create_captions,
        caption_text,
        add_trigger_word,
        trigger_text,
    ):
        dataset_root = self._norm(dataset_root)
        if not os.path.isdir(dataset_root):
            raise ValueError(f"dataset_videos_dir does not exist: {dataset_root}")

        if os.path.basename(dataset_root).lower() == "videos":
            self._ensure_video_captions(
                dataset_root,
                create_captions,
                caption_text,
                add_trigger_word,
                trigger_text,
            )
            return dataset_root

        videos_dir = os.path.join(dataset_root, "videos")
        if os.path.isdir(videos_dir):
            self._ensure_video_captions(
                videos_dir,
                create_captions,
                caption_text,
                add_trigger_word,
                trigger_text,
            )
            return os.path.normpath(videos_dir)

        os.makedirs(videos_dir, exist_ok=True)

        root_files = [entry for entry in os.scandir(dataset_root) if entry.is_file()]
        video_stems = {
            os.path.splitext(entry.name)[0]
            for entry in root_files
            if os.path.splitext(entry.name)[1].lower() in self.VIDEO_EXTENSIONS
        }

        moved_count = 0
        for entry in root_files:
            ext = os.path.splitext(entry.name)[1].lower()
            stem = os.path.splitext(entry.name)[0]
            should_move = ext in self.VIDEO_EXTENSIONS or (ext == ".txt" and stem in video_stems)
            if not should_move:
                continue

            target_path = os.path.join(videos_dir, entry.name)
            if os.path.exists(target_path):
                continue
            shutil.move(entry.path, target_path)
            moved_count += 1

        self._ensure_video_captions(
            videos_dir,
            create_captions,
            caption_text,
            add_trigger_word,
            trigger_text,
        )
        print(f"[VRGDG] Video dataset prep complete. Using videos folder: {videos_dir} (moved {moved_count} file(s))")
        return os.path.normpath(videos_dir)

    def _video_files_in_directory(self, videos_dir):
        video_entries = []
        if not os.path.isdir(videos_dir):
            return video_entries
        for entry in sorted(os.scandir(videos_dir), key=lambda item: item.name.lower()):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in self.VIDEO_EXTENSIONS:
                video_entries.append(entry.path)
        return video_entries

    def _read_caption_text_for_video(self, video_path, create_captions, caption_text, add_trigger_word, trigger_text):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        caption_path = os.path.join(os.path.dirname(video_path), f"{stem}.txt")
        if os.path.isfile(caption_path):
            with open(caption_path, "r", encoding="utf-8") as handle:
                existing = handle.read().strip()
            return self._compose_caption_text(existing, add_trigger_word, trigger_text)
        if not create_captions:
            return ""
        return self._compose_caption_text(caption_text, add_trigger_word, trigger_text)

    def _probe_video_frame_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for probing: {video_path}")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        if frame_count <= 0:
            raise RuntimeError(f"Could not determine frame count for: {video_path}")
        if fps <= 0:
            fps = 25.0
        return frame_count, fps

    def _split_video_into_chunks(
        self,
        source_video_path,
        chunk_videos_dir,
        chunk_frame_count,
        create_captions,
        caption_text,
        add_trigger_word,
        trigger_text,
        log_handle,
    ):
        ffmpeg_path = self._find_ffmpeg_path()
        source_stem = os.path.splitext(os.path.basename(source_video_path))[0]
        source_ext = os.path.splitext(source_video_path)[1].lower() or ".mp4"
        frame_count, fps = self._probe_video_frame_info(source_video_path)
        caption = self._read_caption_text_for_video(
            source_video_path,
            create_captions,
            caption_text,
            add_trigger_word,
            trigger_text,
        )

        created_chunk_dirs = []
        chunk_index = 0
        for start_frame in range(0, frame_count, max(1, int(chunk_frame_count))):
            end_frame = min(start_frame + max(1, int(chunk_frame_count)), frame_count)
            chunk_index += 1
            chunk_name = f"{source_stem}_chunk{chunk_index:04d}"
            chunk_dir = self._ensure_dir(os.path.join(chunk_videos_dir, chunk_name))
            chunk_video_dir = self._ensure_dir(os.path.join(chunk_dir, "videos"))
            chunk_video_path = os.path.join(chunk_video_dir, f"{chunk_name}{source_ext}")
            chunk_caption_path = os.path.join(chunk_video_dir, f"{chunk_name}.txt")

            if not os.path.isfile(chunk_video_path):
                start_seconds = max(0.0, float(start_frame) / float(fps))
                duration_seconds = max(1.0 / float(fps), float(end_frame - start_frame) / float(fps))
                self._run_command(
                    [
                        ffmpeg_path,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{start_seconds:.6f}",
                        "-t",
                        f"{duration_seconds:.6f}",
                        "-i",
                        source_video_path,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        "-movflags",
                        "+faststart",
                        chunk_video_path,
                    ],
                    chunk_dir,
                    log_handle,
                )

            if caption and not os.path.isfile(chunk_caption_path):
                with open(chunk_caption_path, "w", encoding="utf-8") as handle:
                    handle.write(caption)

            created_chunk_dirs.append(chunk_video_dir)

        print(
            f"[VRGDG] Auto-chunked {os.path.basename(source_video_path)} into {len(created_chunk_dirs)} chunk(s) "
            f"of up to {int(chunk_frame_count)} frame(s)."
        )
        return created_chunk_dirs

    def _prepare_step400_autochunk_dataset(
        self,
        dataset_videos_dir,
        workspace_dir,
        run_name,
        create_captions,
        caption_text,
        add_trigger_word,
        trigger_text,
        log_handle,
    ):
        chunk_source_dirs = []
        chunk_root = self._ensure_dir(os.path.join(workspace_dir, "autochunks", run_name))
        source_videos = self._video_files_in_directory(dataset_videos_dir)
        if not source_videos:
            raise RuntimeError(f"No video files were found for autochunking in: {dataset_videos_dir}")

        target_chunk_frames = 33
        for source_video_path in source_videos:
            chunk_source_dirs.extend(
                self._split_video_into_chunks(
                    source_video_path,
                    chunk_root,
                    target_chunk_frames,
                    create_captions,
                    caption_text,
                    add_trigger_word,
                    trigger_text,
                    log_handle,
                )
            )

        if not chunk_source_dirs:
            raise RuntimeError(f"Autochunking produced no chunk datasets for: {dataset_videos_dir}")

        return chunk_source_dirs, target_chunk_frames

    def _write_av_dataset_config(self, path, dataset_videos_dir, cache_dir, width, height, num_repeats, target_frames, target_fps, frame_extraction, max_frames):
        frame_list = ", ".join(str(int(frame)) for frame in target_frames)
        target_fps_value = float(target_fps)
        content = (
            "[general]\n"
            f"resolution = [{int(width)}, {int(height)}]\n"
            'caption_extension = ".txt"\n'
            "batch_size = 1\n"
            "enable_bucket = true\n"
            "bucket_no_upscale = false\n\n"
            "[[datasets]]\n"
            f'video_directory = "{self._quote(dataset_videos_dir)}"\n'
            f'cache_directory = "{self._quote(cache_dir)}"\n'
            f"target_frames = [{frame_list}]\n"
            f'frame_extraction = "{frame_extraction}"\n'
            f"max_frames = {int(max_frames)}\n"
            f"target_fps = {target_fps_value:.1f}\n"
            f"num_repeats = {int(num_repeats)}\n"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _write_av_training_config(
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
        separate_audio_buckets,
        lora_target_preset,
        fp8_base,
        fp8_scaled,
    ):
        checkpoint_name = os.path.basename(str(checkpoint or "")).lower()
        effective_fp8_scaled = bool(fp8_scaled) and ("fp8" not in checkpoint_name)
        content = (
            "# Auto-generated by VRGDG LTX audio-video chunk trainer\n"
            f"# total_target_steps_from_workflow = {int(total_target_steps)}\n"
            f"# chunk_target_steps_this_run = {int(max_train_steps)}\n"
            f"# save_interval_per_run = {int(steps_per_run)}\n"
            f'ltx2_checkpoint = "{self._quote(checkpoint)}"\n'
            f'gemma_root = "{self._quote(gemma_root)}"\n'
            f'dataset_config = "{self._quote(dataset_config)}"\n\n'
            'ltx_mode = "av"\n'
            'ltx_version = "2.3"\n'
            'ltx_version_check_mode = "error"\n'
            f'lora_target_preset = "{lora_target_preset}"\n\n'
            "cache_text_encoder_outputs = true\n"
            "cache_text_encoder_outputs_to_disk = false\n"
            f"separate_audio_buckets = {'true' if separate_audio_buckets else 'false'}\n\n"
            f"fp8_base = {'true' if fp8_base else 'false'}\n"
            f"fp8_scaled = {'true' if effective_fp8_scaled else 'false'}\n"
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

    def run(
        self,
        model,
        dataset_videos_dir,
        workspace_dir,
        run_name,
        output_name,
        av_profile,
        use_profile_defaults,
        resolution_width,
        resolution_height,
        target_frames,
        frame_extraction,
        max_frames,
        target_fps,
        steps_per_run,
        total_target_steps,
        network_dim,
        network_alpha,
        blocks_to_swap,
        separate_audio_buckets,
        lora_target_preset,
        fp8_base,
        fp8_scaled,
        clear_memory_before_gemma,
        gemma_recovery_mode,
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
        gemma_load_in_4bit=False,
        _autochunk_mode=False,
    ):
        dataset_videos_dir = self._norm(dataset_videos_dir)
        workspace_dir = self._norm(workspace_dir)
        musubi_root = self._norm(musubi_root)
        ltx2_checkpoint = self._norm(ltx2_checkpoint)
        gemma_root = self._norm(gemma_root)
        gemma_load_in_4bit = bool(gemma_load_in_4bit)
        av_profile = str(av_profile or "fast_short_clip").strip()
        profile_defaults = self._profile_defaults(av_profile) if bool(use_profile_defaults) else {}
        if profile_defaults:
            resolution_width = profile_defaults["resolution_width"]
            resolution_height = profile_defaults["resolution_height"]
            target_frames = profile_defaults["target_frames"]
            frame_extraction = profile_defaults["frame_extraction"]
            max_frames = profile_defaults["max_frames"]
            target_fps = profile_defaults["target_fps"]
            steps_per_run = profile_defaults["steps_per_run"]
            total_target_steps = profile_defaults["total_target_steps"]
            network_dim = profile_defaults["network_dim"]
            network_alpha = profile_defaults["network_alpha"]
            blocks_to_swap = profile_defaults["blocks_to_swap"]
            separate_audio_buckets = profile_defaults["separate_audio_buckets"]
            lora_target_preset = profile_defaults["lora_target_preset"]
            fp8_base = profile_defaults["fp8_base"]
            fp8_scaled = profile_defaults["fp8_scaled"]
            learning_rate_preset = profile_defaults["learning_rate_preset"]
            learning_rate = profile_defaults["learning_rate"]
            num_repeats = profile_defaults["num_repeats"]
        gemma_load_in_8bit = not gemma_load_in_4bit
        gemma_load_mode = "4bit" if gemma_load_in_4bit else "8bit"
        run_name = self._safe_name(run_name, "LTXAVChunkRun")
        output_name = self._safe_name(output_name, run_name)
        effective_learning_rate = self._resolve_learning_rate(learning_rate_preset, learning_rate)
        parsed_target_frames = self._parse_target_frames(target_frames)

        dataset_videos_dir = self._prepare_video_dataset_directory(
            dataset_videos_dir,
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

        autochunk_enabled = bool(profile_defaults) and str(av_profile).strip() == "step200"
        autochunk_total_steps = None
        if autochunk_enabled:
            logs_dir = self._ensure_dir(os.path.join(workspace_dir, "logs"))
            master_log_path = os.path.join(
                logs_dir,
                f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_autochunk.log",
            )
            with open(master_log_path, "w", encoding="utf-8") as log_handle:
                chunk_dataset_dirs, chunk_frame_count = self._prepare_step400_autochunk_dataset(
                    dataset_videos_dir,
                    workspace_dir,
                    run_name,
                    create_captions,
                    caption_text,
                    add_trigger_word,
                    trigger_text,
                    log_handle,
                )
                autochunk_total_steps = len(chunk_dataset_dirs) * int(steps_per_run)
                if completed_steps >= int(autochunk_total_steps):
                    raise RuntimeError(
                        f"Training complete: reached {completed_steps}/{int(autochunk_total_steps)} steps. Stopping workflow."
                    )
                chunk_index = min(len(chunk_dataset_dirs) - 1, int(completed_steps) // int(steps_per_run))
                selected_chunk_dir = chunk_dataset_dirs[chunk_index]
                chunk_start_step = int(completed_steps) + 1
                chunk_end_step = min(int(completed_steps) + int(steps_per_run), int(autochunk_total_steps))
                print(
                    f"[VRGDG] step200 autochunk enabled: chunk_frame_count={chunk_frame_count} "
                    f"chunk_count={len(chunk_dataset_dirs)} total_target_steps={int(autochunk_total_steps)}"
                )
                print(
                    f"[VRGDG] step200 autochunk chunk {chunk_index + 1}/{len(chunk_dataset_dirs)} "
                    f"dataset={selected_chunk_dir} steps={chunk_start_step}-{chunk_end_step}"
                )
                log_handle.write(
                    f"step200 autochunk enabled: chunk_frame_count={chunk_frame_count} "
                    f"chunk_count={len(chunk_dataset_dirs)} total_target_steps={int(autochunk_total_steps)}\n"
                )
                log_handle.write(
                    f"step200 autochunk chunk {chunk_index + 1}/{len(chunk_dataset_dirs)} "
                    f"dataset={selected_chunk_dir} steps={chunk_start_step}-{chunk_end_step}\n"
                )
                log_handle.flush()
                dataset_videos_dir = selected_chunk_dir
                total_target_steps = int(autochunk_total_steps)

        python_exe, accelerate_exe, env_source = self._resolve_musubi_executables(musubi_root)

        cache_dir = self._ensure_dir(os.path.join(workspace_dir, "cache"))
        output_dir = self._ensure_dir(os.path.join(workspace_dir, "output"))
        logs_dir = self._ensure_dir(os.path.join(workspace_dir, "logs"))
        config_dir = self._ensure_dir(os.path.join(workspace_dir, "config"))
        dataset_config = os.path.join(config_dir, "dataset-av.toml")
        training_config = os.path.join(config_dir, "training_args_av.toml")
        latest_state_path, completed_steps = self._latest_state_dir(output_dir, output_name)
        autochunk_total_steps = None
        if bool(profile_defaults) and str(av_profile).strip() == "step200":
            master_log_path = os.path.join(
                logs_dir,
                f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_autochunk.log",
            )
            with open(master_log_path, "w", encoding="utf-8") as log_handle:
                chunk_dataset_dirs, chunk_frame_count = self._prepare_step400_autochunk_dataset(
                    dataset_videos_dir,
                    workspace_dir,
                    run_name,
                    create_captions,
                    caption_text,
                    add_trigger_word,
                    trigger_text,
                    log_handle,
                )
                autochunk_total_steps = len(chunk_dataset_dirs) * int(steps_per_run)
                if completed_steps >= int(autochunk_total_steps):
                    raise RuntimeError(
                        f"Training complete: reached {completed_steps}/{int(autochunk_total_steps)} steps. Stopping workflow."
                    )
                chunk_index = min(len(chunk_dataset_dirs) - 1, int(completed_steps) // int(steps_per_run))
                selected_chunk_dir = chunk_dataset_dirs[chunk_index]
                chunk_start_step = int(completed_steps) + 1
                chunk_end_step = min(int(completed_steps) + int(steps_per_run), int(autochunk_total_steps))
                print(
                    f"[VRGDG] step200 autochunk enabled: chunk_frame_count={chunk_frame_count} "
                    f"chunk_count={len(chunk_dataset_dirs)} total_target_steps={int(autochunk_total_steps)}"
                )
                print(
                    f"[VRGDG] step200 autochunk chunk {chunk_index + 1}/{len(chunk_dataset_dirs)} "
                    f"dataset={selected_chunk_dir} steps={chunk_start_step}-{chunk_end_step}"
                )
                log_handle.write(
                    f"step200 autochunk enabled: chunk_frame_count={chunk_frame_count} "
                    f"chunk_count={len(chunk_dataset_dirs)} total_target_steps={int(autochunk_total_steps)}\n"
                )
                log_handle.write(
                    f"step200 autochunk chunk {chunk_index + 1}/{len(chunk_dataset_dirs)} "
                    f"dataset={selected_chunk_dir} steps={chunk_start_step}-{chunk_end_step}\n"
                )
                log_handle.flush()
                dataset_videos_dir = selected_chunk_dir
                total_target_steps = int(autochunk_total_steps)

        cache_signature_path = os.path.join(cache_dir, "ltx2_av_cache_signature.json")
        current_cache_signature = {
            "dataset_videos_dir": dataset_videos_dir,
            "ltx2_checkpoint": ltx2_checkpoint,
            "gemma_root": gemma_root,
            "resolution_width": int(resolution_width),
            "resolution_height": int(resolution_height),
            "target_frames": parsed_target_frames,
            "frame_extraction": str(frame_extraction),
            "max_frames": int(max_frames),
            "target_fps": float(target_fps),
            "num_repeats": int(num_repeats),
            "separate_audio_buckets": bool(separate_audio_buckets),
            "lora_target_preset": str(lora_target_preset),
            "fp8_base": bool(fp8_base),
            "fp8_scaled": bool(fp8_scaled),
            "ltx2_mode": "av",
            "av_profile": av_profile,
            "use_profile_defaults": bool(use_profile_defaults),
        }

        if completed_steps >= int(total_target_steps):
            raise RuntimeError(
                f"Training complete: reached {completed_steps}/{int(total_target_steps)} steps. Stopping workflow."
            )

        next_target_steps = min(completed_steps + int(steps_per_run), int(total_target_steps))
        video_filename_prefix = self._get_or_create_video_filename_prefix(
            config_dir,
            dataset_videos_dir,
            output_name,
            next_target_steps,
        )

        self._write_av_dataset_config(
            dataset_config,
            dataset_videos_dir,
            cache_dir,
            resolution_width,
            resolution_height,
            num_repeats,
            parsed_target_frames,
            target_fps,
            frame_extraction,
            max_frames,
        )
        self._write_av_training_config(
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
            separate_audio_buckets,
            lora_target_preset,
            fp8_base,
            fp8_scaled,
        )

        log_path = os.path.join(
            logs_dir,
            f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        with open(log_path, "w", encoding="utf-8") as log_handle:
            log_handle.write(f"dataset_videos_dir={dataset_videos_dir}\n")
            log_handle.write(f"workspace_dir={workspace_dir}\n")
            log_handle.write(f"completed_steps={completed_steps}\n")
            log_handle.write(f"next_target_steps={next_target_steps}\n\n")
            log_handle.flush()

            video_count, caption_count = self._count_video_dataset_files(dataset_videos_dir)
            cache_file_count_before = self._count_cache_files(cache_dir)
            has_latent_cache_files = self._has_latent_cache_files(cache_dir)
            has_complete_text_cache_files = self._has_complete_text_encoder_cache_files(cache_dir)
            cache_signature_matches = False
            if os.path.isfile(cache_signature_path):
                try:
                    with open(cache_signature_path, "r", encoding="utf-8") as handle:
                        cached_signature = json.load(handle)
                    cache_signature_matches = cached_signature == current_cache_signature
                except Exception:
                    cache_signature_matches = False
            if cache_strategy == "force":
                should_build_latent_cache = True
                should_build_text_cache = True
            elif cache_strategy == "skip":
                should_build_latent_cache = False
                should_build_text_cache = False
            else:
                should_build_latent_cache = not has_latent_cache_files
                should_build_text_cache = should_build_latent_cache or not has_complete_text_cache_files
                if has_latent_cache_files and has_complete_text_cache_files and not cache_signature_matches:
                    should_build_latent_cache = True
                    should_build_text_cache = True
            total_stages = int(should_build_latent_cache) + int(should_build_text_cache) + 1

            print(f"[VRGDG] dataset_videos_dir={dataset_videos_dir}")
            print(f"[VRGDG] workspace_dir={workspace_dir}")
            print(f"[VRGDG] video_filename_prefix={video_filename_prefix}")
            print(f"[VRGDG] completed_steps={completed_steps}")
            print(f"[VRGDG] next_target_steps={next_target_steps}")
            print(f"[VRGDG] steps_per_run_and_save={steps_per_run}")
            print(f"[VRGDG] total_target_steps={total_target_steps}")
            print(f"[VRGDG] av_profile={av_profile}")
            print(f"[VRGDG] use_profile_defaults={bool(use_profile_defaults)}")
            print(f"[VRGDG] target_frames={parsed_target_frames}")
            print(f"[VRGDG] frame_extraction={frame_extraction}")
            print(f"[VRGDG] max_frames={int(max_frames)}")
            print(f"[VRGDG] target_fps={float(target_fps):.1f}")
            print(f"[VRGDG] blocks_to_swap={int(blocks_to_swap)}")
            print(f"[VRGDG] separate_audio_buckets={bool(separate_audio_buckets)}")
            print(f"[VRGDG] lora_target_preset={lora_target_preset}")
            print(f"[VRGDG] fp8_base={bool(fp8_base)} fp8_scaled={bool(fp8_scaled)}")
            print(f"[VRGDG] musubi_env_source={env_source}")
            print(f"[VRGDG] musubi_python={python_exe}")
            print(f"[VRGDG] musubi_accelerate={accelerate_exe}")
            print(f"[VRGDG] clear_memory_before_gemma={clear_memory_before_gemma}")
            print(f"[VRGDG] gemma_load_mode={gemma_load_mode}")
            print(f"[VRGDG] gemma_recovery_mode={gemma_recovery_mode}")
            print(f"[VRGDG] keep_only_comfy_lora={keep_only_comfy_lora}")
            print(
                f"[VRGDG] learning_rate={effective_learning_rate} "
                f"(preset={learning_rate_preset})"
            )
            print(f"[VRGDG] dataset summary: videos={video_count} captions={caption_count}")
            print(
                f"[VRGDG] cache summary: strategy={cache_strategy} "
                f"build_latent={'yes' if should_build_latent_cache else 'no'} "
                f"build_text={'yes' if should_build_text_cache else 'no'} "
                f"existing_cache_files={cache_file_count_before} "
                f"cache_signature_match={cache_signature_matches}"
            )
            if latest_state_path:
                print(f"[VRGDG] resume state detected: {latest_state_path}")
            else:
                print("[VRGDG] resume state detected: none")

            if should_build_latent_cache:
                self._run_stage_command(
                    1,
                    total_stages,
                    "Cache AV latents",
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
                        "av",
                        "--ltx2_audio_source",
                        "video",
                    ],
                    musubi_root,
                    log_handle,
                    [
                        f"Dataset videos dir: {dataset_videos_dir}",
                        f"Videos found: {video_count}",
                        f"Captions found: {caption_count}",
                        f"Cache dir: {cache_dir}",
                    ],
                )
            if should_build_text_cache:
                if clear_memory_before_gemma:
                    self._clear_memory_before_gemma(log_handle)
                if gemma_recovery_mode:
                    self._run_text_encoder_cache_stage_with_recovery(
                        2 if should_build_latent_cache else 1,
                        total_stages,
                        "Cache AV text encoder outputs",
                        python_exe,
                        dataset_config,
                        ltx2_checkpoint,
                        gemma_root,
                        "av",
                        gemma_load_in_8bit,
                        gemma_recovery_mode,
                        musubi_root,
                        log_handle,
                        [
                            f"Gemma root: {gemma_root}",
                            "This is usually the slowest setup stage.",
                            "You should see per-item progress from the text encoder cache script.",
                        ],
                        gemma_load_in_4bit=gemma_load_in_4bit,
                    )
                else:
                    self._run_text_encoder_cache_stage(
                        2 if should_build_latent_cache else 1,
                        total_stages,
                        "Cache AV text encoder outputs",
                        python_exe,
                        dataset_config,
                        ltx2_checkpoint,
                        gemma_root,
                        "av",
                        gemma_load_in_8bit,
                        musubi_root,
                        log_handle,
                        [
                            f"Gemma root: {gemma_root}",
                            "This is usually the slowest setup stage.",
                            "You should see per-item progress from the text encoder cache script.",
                        ],
                        gemma_load_in_4bit=gemma_load_in_4bit,
                    )
                    print(f"[VRGDG] cache summary after build: files={self._count_cache_files(cache_dir)}")
                    with open(cache_signature_path, "w", encoding="utf-8") as handle:
                        json.dump(current_cache_signature, handle, indent=2, sort_keys=True)
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
                "--ltx2_mode",
                "av",
            ]
            if latest_state_path:
                train_command.extend(["--resume", latest_state_path])

            self._run_stage_command(
                total_stages,
                total_stages,
                "Train AV LoRA",
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

            with open(cache_signature_path, "w", encoding="utf-8") as handle:
                json.dump(current_cache_signature, handle, indent=2, sort_keys=True)

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
            print(f"[VRGDG] keep_only_comfy_lora deleted {deleted_count} standard LoRA file(s).")

        exported_comfy_lora = ""
        if copy_latest_to_comfy_loras and latest_comfy_lora_path:
            exported_comfy_lora = self._export_latest_to_comfy(latest_comfy_lora_path, output_name)
            if exported_comfy_lora:
                print(f"[VRGDG] exported latest comfy LoRA to {exported_comfy_lora}")

        applied_lora_path = latest_comfy_lora_path or latest_lora_path
        self._log_message(
            f"[VRGDG] Latest state selected: {os.path.normpath(latest_state_path) if latest_state_path else '(none)'}",
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


class VRGDG_ZImageLoraTrainChunk(VRGDG_LTXLoraTrainChunk):
    DESCRIPTION = (
        "Runs one Z-Image LoRA training chunk using musubi-tuner, optionally caches if needed, "
        "converts the latest LoRA to ComfyUI format, and can apply it to the returned MODEL."
    )

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = (
        "model",
        "latest_state_path",
        "log_path",
        "latest_comfy_lora_path",
        "output_name",
        "completed_steps",
        "total_target_steps",
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
                    "default": "ZImageChunkRun",
                    "multiline": False,
                    "tooltip": "Name prefix used for the log file."
                }),
                "output_name": ("STRING", {
                    "default": "ZImageChunkRun",
                    "multiline": False,
                    "tooltip": "Name prefix used for saved LoRA files and state folders."
                }),
                "resolution_width": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 1,
                    "tooltip": "Training bucket width written to the musubi dataset config."
                }),
                "resolution_height": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 1,
                    "tooltip": "Training bucket height written to the musubi dataset config."
                }),
                "steps_per_run": ("INT", {
                    "default": 250, "min": 1, "max": 100000, "step": 1,
                    "tooltip": "How many steps to train per run, and also when to save the LoRA/state at the end of that run."
                }),
                "total_target_steps": ("INT", {
                    "default": 3000, "min": 1, "max": 1000000, "step": 1,
                    "tooltip": "Training stops once the latest saved step reaches this total."
                }),
                "network_dim": ("INT", {
                    "default": 32, "min": 1, "max": 2048, "step": 1,
                    "tooltip": "LoRA rank."
                }),
                "network_alpha": ("INT", {
                    "default": 32, "min": 1, "max": 2048, "step": 1,
                    "tooltip": "LoRA alpha scaling value."
                }),
                "blocks_to_swap": ("INT", {
                    "default": 4, "min": 0, "max": 28, "step": 1,
                    "tooltip": "Higher values reduce VRAM usage but usually slow training."
                }),
                "clear_memory_before_text_encoder": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Tries to unload ComfyUI models and clear VRAM/RAM before text encoder caching."
                }),
                "learning_rate_preset": ([
                    "Custom",
                    "1e-4",
                    "7e-5",
                    "5e-5",
                    "3e-5",
                    "1e-5",
                ], {
                    "default": "1e-4",
                    "tooltip": "Quick preset for the training learning rate. Choose Custom to use the float input below."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-6,
                    "tooltip": "Custom learning rate used only when the preset is set to Custom."
                }),
                "num_repeats": ("INT", {
                    "default": 1, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "How many times each image-caption pair is repeated in the dataset."
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
                    "tooltip": "If enabled, deletes the standard .safetensors LoRA files after a matching .comfy.safetensors file exists."
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Strength used if the node applies the latest LoRA back onto the output model."
                }),
                "create_captions": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, missing caption txt files are created automatically using the caption text input."
                }),
                "caption_text": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Base caption text used when create_captions is enabled and an image has no caption file."
                }),
                "add_trigger_word": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, the trigger text is prepended to each caption."
                }),
                "trigger_text": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Trigger word or phrase to prepend to captions when add_trigger_word is enabled."
                }),
                "musubi_root": ("STRING", {
                    "default": "A:/MUSUBI/musubi-tuner-ltx2", "multiline": False,
                    "tooltip": "Root folder of your musubi-tuner-ltx2 install."
                }),
                "zimage_checkpoint": ("STRING", {
                    "default": "A:/MUSUBI/models/zimage/zimage-base.safetensors", "multiline": False,
                    "tooltip": "Path to the base Z-Image DiT checkpoint used for caching and training."
                }),
                "vae": ("STRING", {
                    "default": "A:/MUSUBI/models/zimage/vae.safetensors", "multiline": False,
                    "tooltip": "Path to the Z-Image VAE checkpoint."
                }),
                "text_encoder": ("STRING", {
                    "default": "A:/MUSUBI/models/qwen3", "multiline": False,
                    "tooltip": "Path to the Qwen3 text encoder checkpoint or directory."
                }),
                "fp8_base": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable fp8 base model weights during Z-Image training."
                }),
                "fp8_scaled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable scaled fp8 weights during Z-Image training. Requires fp8_base."
                }),
                "fp8_llm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loads the text encoder in fp8 mode during caching to reduce VRAM usage."
                }),
                "use_32bit_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use 32-bit precision for attention computations in the Z-Image model."
                }),
            }
        }

    @staticmethod
    def _get_zimage_cache_architecture():
        return "zi"

    @staticmethod
    def _find_first_matching_safetensor_file(root_dir, include_any=None, exclude_any=None):
        include_any = [str(item).lower() for item in (include_any or []) if str(item or "").strip()]
        exclude_any = [str(item).lower() for item in (exclude_any or []) if str(item or "").strip()]
        if not os.path.isdir(root_dir):
            return ""

        candidates = []
        for current_root, _dirs, files in os.walk(root_dir):
            for name in files:
                if not name.lower().endswith(".safetensors"):
                    continue
                lowered = name.lower()
                if include_any and not any(token in lowered for token in include_any):
                    continue
                if exclude_any and any(token in lowered for token in exclude_any):
                    continue
                candidates.append(os.path.join(current_root, name))

        if not candidates:
            return ""

        def _sort_key(path):
            name = os.path.basename(path).lower()
            split_rank = 0 if re.search(r"00001-of-\d+", name) else 1
            return (split_rank, len(path), path.lower())

        candidates.sort(key=_sort_key)
        return os.path.normpath(candidates[0])

    def _resolve_zimage_model_path(self, path, kind):
        path = self._norm(path)
        if os.path.isfile(path):
            return path
        if not os.path.isdir(path):
            raise ValueError(f"{kind} does not exist: {path}")

        if kind == "vae":
            candidate = self._find_first_matching_safetensor_file(
                path,
                include_any=["vae"],
                exclude_any=["qwen", "text", "tokenizer", "llm"],
            )
        elif kind == "text_encoder":
            candidate = self._find_first_matching_safetensor_file(
                path,
                include_any=["qwen", "text", "llm"],
                exclude_any=["vae"],
            )
        else:
            candidate = self._find_first_matching_safetensor_file(
                path,
                include_any=["00001-of-", "dit", "transformer", "model"],
                exclude_any=["vae", "qwen", "text", "tokenizer", "llm"],
            )

        if candidate:
            return candidate

        if kind == "text_encoder":
            # If the tokenizer/model bundle is installed as a repo snapshot directory,
            # keep the directory path as a fallback for troubleshooting, but the loader
            # still expects a safetensors file if it can be found.
            raise ValueError(
                "text_encoder directory does not contain a recognizable Qwen3 safetensors file. "
                "Point it at the split checkpoint file or the repository root that contains it."
            )

        raise ValueError(
            f"Could not resolve a {kind} safetensors file from directory: {path}. "
            "Point the node at the file directly or the repository root that contains it."
        )

    def _has_latent_cache_files(self, cache_dir):
        architecture = self._get_zimage_cache_architecture()
        if not os.path.isdir(cache_dir):
            return False
        for entry in os.scandir(cache_dir):
            if not entry.is_file():
                continue
            if entry.name.endswith(f"_{architecture}.safetensors") and not entry.name.endswith(f"_{architecture}_te.safetensors"):
                return True
        return False

    def _expected_text_cache_path_from_latent_cache_file(self, cache_dir, latent_cache_file):
        architecture = self._get_zimage_cache_architecture()
        basename = os.path.basename(latent_cache_file)
        if not basename.endswith(f"_{architecture}.safetensors"):
            return ""

        tokens = basename.split("_")
        if len(tokens) < 4:
            return ""

        item_key = "_".join(tokens[:-2])
        if not item_key:
            return ""
        return os.path.join(cache_dir, f"{item_key}_{architecture}_te.safetensors")

    def _has_complete_text_encoder_cache_files(self, cache_dir):
        architecture = self._get_zimage_cache_architecture()
        if not os.path.isdir(cache_dir):
            return False

        latent_cache_files = []
        for entry in os.scandir(cache_dir):
            if not entry.is_file():
                continue
            if entry.name.endswith(f"_{architecture}.safetensors") and not entry.name.endswith(f"_{architecture}_te.safetensors"):
                latent_cache_files.append(entry.path)

        if not latent_cache_files:
            return False

        seen_text_cache_files = set()
        for latent_cache_file in latent_cache_files:
            expected_text_cache = self._expected_text_cache_path_from_latent_cache_file(cache_dir, latent_cache_file)
            if not expected_text_cache:
                continue
            normalized_expected = os.path.normpath(expected_text_cache)
            if normalized_expected in seen_text_cache_files:
                continue
            seen_text_cache_files.add(normalized_expected)
            if not os.path.exists(expected_text_cache):
                return False

        return True

    def _write_zimage_training_config(
        self,
        path,
        dataset_config,
        checkpoint,
        vae,
        text_encoder,
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
        fp8_base,
        fp8_scaled,
        fp8_llm,
        use_32bit_attention,
    ):
        fp8_base = bool(fp8_base)
        fp8_scaled = bool(fp8_scaled)
        fp8_llm = bool(fp8_llm)
        use_32bit_attention = bool(use_32bit_attention)
        content = (
            "# Auto-generated by VRGDG Z-Image chunk trainer\n"
            f"# total_target_steps_from_workflow = {int(total_target_steps)}\n"
            f"# chunk_target_steps_this_run = {int(max_train_steps)}\n"
            f"# save_interval_per_run = {int(steps_per_run)}\n"
            f'dit = "{self._quote(checkpoint)}"\n'
            f'vae = "{self._quote(vae)}"\n'
            f'text_encoder = "{self._quote(text_encoder)}"\n'
            f'dataset_config = "{self._quote(dataset_config)}"\n\n'
            'network_module = "networks.lora_zimage"\n'
            f"network_dim = {int(network_dim)}\n"
            f"network_alpha = {int(network_alpha)}\n\n"
            'cache_text_encoder_outputs = true\n'
            'cache_text_encoder_outputs_to_disk = false\n'
            'sdpa = true\n'
            'gradient_checkpointing = true\n'
            'gradient_accumulation_steps = 1\n'
            f"blocks_to_swap = {int(blocks_to_swap)}\n"
            'optimizer_type = "AdamW8Bit"\n'
            f"learning_rate = {learning_rate}\n"
            'lr_scheduler = "constant_with_warmup"\n'
            "lr_warmup_steps = 100\n\n"
            'timestep_sampling = "shift"\n'
            'weighting_scheme = "none"\n'
            'discrete_flow_shift = 2.0\n\n'
            f"fp8_base = {'true' if fp8_base else 'false'}\n"
            f"fp8_scaled = {'true' if fp8_scaled else 'false'}\n"
            f"fp8_llm = {'true' if fp8_llm else 'false'}\n"
            f"use_32bit_attention = {'true' if use_32bit_attention else 'false'}\n\n"
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

    def _ensure_zimage_checkpointing_compatibility(self, musubi_root, log_path=None):
        zimage_model_path = os.path.normpath(
            os.path.join(musubi_root, "src", "musubi_tuner", "zimage", "zimage_model.py")
        )
        backup_path = zimage_model_path + ".vrgdg_checkpointing_backup"

        self._log_message(
            f"[VRGDG] Z-Image checkpointing compatibility check: {zimage_model_path}",
            log_path,
        )

        if not os.path.isfile(zimage_model_path):
            raise RuntimeError(
                f"Z-Image checkpointing compatibility patch target not found: {zimage_model_path}"
            )

        with open(zimage_model_path, "r", encoding="utf-8") as handle:
            original = handle.read()

        match = re.search(
            r"^\s*def\s+enable_gradient_checkpointing\(\s*self,\s*cpu_offload:\s*bool\s*=\s*False(?P<rest>[^)]*)\):\s*$",
            original,
            flags=re.MULTILINE,
        )
        if not match:
            raise RuntimeError(
                "Could not locate the Z-Image gradient checkpointing method signature to patch."
            )

        signature_tail = match.group("rest") or ""
        if "blocks_to_checkpoint" in signature_tail or "**kwargs" in signature_tail:
            self._log_message(
                "[VRGDG] Z-Image checkpointing compatibility check: already compatible; no patch needed.",
                log_path,
            )
            return False

        if not os.path.isfile(backup_path):
            shutil.copy2(zimage_model_path, backup_path)
            self._log_message(
                f"[VRGDG] Z-Image checkpointing compatibility check: backed up original file to {backup_path}",
                log_path,
            )

        patched = re.sub(
            r"^\s*def\s+enable_gradient_checkpointing\(\s*self,\s*cpu_offload:\s*bool\s*=\s*False\s*\):\s*$",
            "    def enable_gradient_checkpointing(self, cpu_offload: bool = False, blocks_to_checkpoint=None, **kwargs):",
            original,
            count=1,
            flags=re.MULTILINE,
        )

        if patched == original:
            raise RuntimeError(
                "Failed to apply the Z-Image checkpointing compatibility patch; musubi file contents did not change."
            )

        with open(zimage_model_path, "w", encoding="utf-8") as handle:
            handle.write(patched)

        self._log_message(
            "[VRGDG] Z-Image checkpointing compatibility check: patch applied; the local musubi install can now accept blocks_to_checkpoint.",
            log_path,
        )
        return True

    def _convert_lora_to_comfy(self, stage_number, total_stages, python_exe, musubi_root, input_lora_path, output_name, log_handle):
        comfy_output_path = os.path.splitext(input_lora_path)[0] + ".comfy.safetensors"
        self._run_stage_command(
            stage_number,
            total_stages,
            "Convert LoRA for ComfyUI",
            [
                python_exe,
                "convert_lora.py",
                "--input",
                input_lora_path,
                "--output",
                comfy_output_path,
                "--target",
                "other",
            ],
            musubi_root,
            log_handle,
            [
                f"Input LoRA: {input_lora_path}",
                f"Output LoRA: {comfy_output_path}",
                f"Output name: {output_name}",
            ],
        )
        if not os.path.isfile(comfy_output_path):
            raise RuntimeError(f"ComfyUI LoRA conversion did not produce an output file: {comfy_output_path}")
        return os.path.normpath(comfy_output_path)

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
        clear_memory_before_text_encoder,
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
        zimage_checkpoint,
        vae,
        text_encoder,
        fp8_base,
        fp8_scaled,
        fp8_llm=False,
        use_32bit_attention=False,
    ):
        dataset_images_dir = self._norm(dataset_images_dir)
        workspace_dir = self._norm(workspace_dir)
        musubi_root = self._norm(musubi_root)
        zimage_checkpoint = self._resolve_zimage_model_path(zimage_checkpoint, "checkpoint")
        vae = self._resolve_zimage_model_path(vae, "vae")
        text_encoder = self._resolve_zimage_model_path(text_encoder, "text_encoder")
        run_name = self._safe_name(run_name, "ZImageChunkRun")
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
        if bool(fp8_scaled) and not bool(fp8_base):
            raise ValueError("fp8_scaled requires fp8_base to be enabled.")

        musubi_root = self._resolve_musubi_script_root(
            musubi_root,
            [
                "zimage_cache_latents.py",
                "zimage_cache_text_encoder_outputs.py",
                "zimage_train_network.py",
                "convert_lora.py",
            ],
        )

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
        self._write_dataset_config(
            dataset_config,
            dataset_images_dir,
            cache_dir,
            resolution_width,
            resolution_height,
            num_repeats,
        )
        self._write_zimage_training_config(
            training_config,
            dataset_config,
            zimage_checkpoint,
            vae,
            text_encoder,
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
            fp8_base,
            fp8_scaled,
            fp8_llm,
            use_32bit_attention,
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

            self._ensure_zimage_checkpointing_compatibility(musubi_root, log_path)

            image_count, caption_count = self._count_dataset_files(dataset_images_dir)
            cache_file_count_before = self._count_cache_files(cache_dir)
            should_build_cache = self._should_build_cache(cache_strategy, cache_dir)
            total_stages = 4 if should_build_cache else 3

            print(f"[VRGDG] dataset_images_dir={dataset_images_dir}")
            print(f"[VRGDG] workspace_dir={workspace_dir}")
            print(f"[VRGDG] completed_steps={completed_steps}")
            print(f"[VRGDG] next_target_steps={next_target_steps}")
            print(f"[VRGDG] steps_per_run_and_save={steps_per_run}")
            print(f"[VRGDG] total_target_steps={total_target_steps}")
            print(f"[VRGDG] blocks_to_swap={int(blocks_to_swap)}")
            print(f"[VRGDG] musubi_env_source={env_source}")
            print(f"[VRGDG] musubi_python={python_exe}")
            print(f"[VRGDG] musubi_accelerate={accelerate_exe}")
            print(f"[VRGDG] clear_memory_before_text_encoder={clear_memory_before_text_encoder}")
            print(f"[VRGDG] fp8_base={bool(fp8_base)} fp8_scaled={bool(fp8_scaled)} fp8_llm={bool(fp8_llm)}")
            print(f"[VRGDG] use_32bit_attention={bool(use_32bit_attention)}")
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
                        "zimage_cache_latents.py",
                        "--dataset_config",
                        dataset_config,
                        "--vae",
                        vae,
                        "--device",
                        "cuda",
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
                if clear_memory_before_text_encoder:
                    self._clear_memory_before_gemma(log_handle)
                self._run_stage_command(
                    2,
                    total_stages,
                    "Cache text encoder outputs",
                    [
                        python_exe,
                        "zimage_cache_text_encoder_outputs.py",
                        "--dataset_config",
                        dataset_config,
                        "--text_encoder",
                        text_encoder,
                        "--device",
                        "cuda",
                    ]
                    + (["--fp8_llm"] if bool(fp8_llm) else []),
                    musubi_root,
                    log_handle,
                    [
                        f"Text encoder: {text_encoder}",
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

            train_stage_number = 3 if should_build_cache else 2
            convert_stage_number = train_stage_number + 1
            train_command = [
                accelerate_exe,
                "launch",
                "--num_cpu_threads_per_process",
                "1",
                "--mixed_precision",
                "bf16",
                "zimage_train_network.py",
                "--config_file",
                training_config,
                "--dit",
                zimage_checkpoint,
                "--vae",
                vae,
                "--text_encoder",
                text_encoder,
            ]
            if latest_state_path:
                train_command.extend(["--resume", latest_state_path])

            self._run_stage_command(
                train_stage_number,
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
        latest_state_path, latest_state_step = self._latest_state_dir(output_dir, output_name)

        completed_steps = max(latest_lora_step, latest_state_step)
        if completed_steps < next_target_steps:
            raise RuntimeError(
                f"Training chunk did not produce the expected checkpoint. Expected step {next_target_steps}, got {completed_steps}."
            )

        with open(log_path, "a", encoding="utf-8") as log_handle:
            latest_comfy_lora_path = self._convert_lora_to_comfy(
                convert_stage_number,
                total_stages,
                python_exe,
                musubi_root,
                latest_lora_path,
                output_name,
                log_handle,
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
            os.path.normpath(latest_comfy_lora_path) if latest_comfy_lora_path else "",
            output_name,
            int(completed_steps),
            int(total_target_steps),
        )


class VRGDG_ZImageSpeedCharacterLoraTraining(VRGDG_ZImageLoraTrainChunk):
    DESCRIPTION = (
        "Runs the Z-Image trainer with a fast character-LoRA preset using dynamic IMAGE and caption inputs."
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
                    "default": "A:/MUSUBI/Training/ZImageSpeedCharacterLoraTraining",
                    "multiline": False,
                    "tooltip": "Workspace folder for cache, output, logs, config, and the managed dynamic dataset."
                }),
                "run_name": ("STRING", {
                    "default": "ZImageSpeedCharacterLoraTrainingRun",
                    "multiline": False,
                    "tooltip": "Run name used for logs."
                }),
                "output_name": ("STRING", {
                    "default": "ZImageSpeedCharacterLoraTraining",
                    "multiline": False,
                    "tooltip": "LoRA output name used for checkpoints and downstream preview naming."
                }),
                "image_count": ("INT", {
                    "default": 4, "min": 1, "max": cls.MAX_IMAGE_SLOTS, "step": 1,
                    "tooltip": "How many dynamic image inputs and caption fields to show."
                }),
                "resolution_width": ("INT", {
                    "default": 1024, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Training bucket width. Pick the resolution preset you want to train at."
                }),
                "resolution_height": ("INT", {
                    "default": 1024, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Training bucket height. Pick the resolution preset you want to train at."
                }),
                "blocks_to_swap": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "How many transformer blocks to swap to CPU. 0 is fastest if VRAM allows it."
                }),
                "clear_memory_before_text_encoder": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clears Comfy and CUDA memory before the text encoder cache stage."
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
                    "default": "A:/MUSUBI/musubi-tuner-ltx2",
                    "multiline": False,
                    "tooltip": "Root folder of your musubi install."
                }),
                "zimage_checkpoint": ("STRING", {
                    "default": "A:/MUSUBI/models/zimage/zimage-base.safetensors",
                    "multiline": False,
                    "tooltip": "Path to the base Z-Image DiT checkpoint."
                }),
                "vae": ("STRING", {
                    "default": "A:/MUSUBI/models/zimage/vae.safetensors",
                    "multiline": False,
                    "tooltip": "Path to the Z-Image VAE checkpoint."
                }),
                "text_encoder": ("STRING", {
                    "default": "A:/MUSUBI/models/qwen3",
                    "multiline": False,
                    "tooltip": "Path to the Qwen3 text encoder checkpoint or directory."
                }),
                "fp8_base": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable fp8 base model weights during Z-Image training."
                }),
                "fp8_scaled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable scaled fp8 weights during Z-Image training. Requires fp8_base."
                }),
                "fp8_llm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loads the text encoder in fp8 mode during caching to reduce VRAM usage."
                }),
                "use_32bit_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use 32-bit precision for attention computations in the Z-Image model."
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
        clear_memory_before_text_encoder,
        cache_strategy,
        strength_model,
        musubi_root,
        zimage_checkpoint,
        vae,
        text_encoder,
        fp8_base,
        fp8_scaled,
        fp8_llm=False,
        use_32bit_attention=False,
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
            clear_memory_before_text_encoder=clear_memory_before_text_encoder,
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
            zimage_checkpoint=zimage_checkpoint,
            vae=vae,
            text_encoder=text_encoder,
            fp8_base=fp8_base,
            fp8_scaled=fp8_scaled,
            fp8_llm=fp8_llm,
            use_32bit_attention=use_32bit_attention,
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


class VRGDG_MusubiTunerInstaller:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("install_root", "status", "checkpoint_path", "assets_root", "report_path")
    FUNCTION = "run"
    CATEGORY = "VRGDG/Training"
    DESCRIPTION = (
        "Provides buttons for installing Musubi-Tuner, downloading the selected model family, or doing both."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_root": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Point this to the Musubi root folder location where Musubi-Tuner should be installed or already exists, not to an arbitrary file path. The node will create Musubi-tuner here, or VRGDG_Musubi-tuner if that name already exists."
                }),
                "model_family": (["LTX 2.3", "Z-Image Turbo"], {
                    "default": "LTX 2.3",
                    "tooltip": "Select which model family the model-download buttons should fetch."
                }),
            },
            "hidden": {
                "install_root": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Resolved install folder written by the installer button."
                }),
                "checkpoint_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Downloaded checkpoint path written by the installer button."
                }),
                "assets_root_out": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Downloaded model assets folder written by the installer button."
                }),
                "report_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Verification report path written by the installer button."
                }),
                "status_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Status text written by the installer button."
                }),
            }
        }

    @staticmethod
    def _norm(path):
        return os.path.normpath(str(path or "").strip())

    @staticmethod
    def _blankable_norm(path):
        text = str(path or "").strip()
        return os.path.normpath(text) if text else ""

    def run(
        self,
        target_root,
        model_family,
        install_root="",
        checkpoint_path="",
        assets_root_out="",
        report_path="",
        status_text="",
    ):
        # Queue execution is intentionally passive.
        # The installer only runs from the node buttons.
        target_root = self._norm(target_root)
        status = "Use the buttons to install the tuner, install the selected models, or do both."
        return (
            self._blankable_norm(install_root) or target_root,
            str(status_text or "").strip() or status,
            self._blankable_norm(checkpoint_path),
            self._blankable_norm(assets_root_out),
            self._blankable_norm(report_path),
        )


NODE_CLASS_MAPPINGS = {
    "VRGDG_LTXLoraTrainChunk": VRGDG_LTXLoraTrainChunk,
    "VRGDG_LTXAudioVideoLoraTrainChunk": VRGDG_LTXAudioVideoLoraTrainChunk,
    "VRGDG_LTXAudioOnlyLoraTrainChunk": VRGDG_LTXAudioOnlyLoraTrainChunk,
    "VRGDG_ZImageLoraTrainChunk": VRGDG_ZImageLoraTrainChunk,
    "VRGDG_ZImageSpeedCharacterLoraTraining": VRGDG_ZImageSpeedCharacterLoraTraining,
    "VRGDG_SpeedCharacterLoraTraining": VRGDG_SpeedCharacterLoraTraining,
    "VRGDG_LTXPreviewXYZPlot": VRGDG_LTXPreviewXYZPlot,
    "VRGDG_VideoFolderGridPlot": VRGDG_VideoFolderGridPlot,
    "VRGDG_MusubiTunerInstaller": VRGDG_MusubiTunerInstaller,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LTXLoraTrainChunk": "VRGDG LTX LoRA Train Chunk",
    "VRGDG_LTXAudioVideoLoraTrainChunk": "VRGDG LTX Audio Video LoRA Train Chunk",
    "VRGDG_LTXAudioOnlyLoraTrainChunk": "VRGDG LTX Audio Only LoRA Train Chunk",
    "VRGDG_ZImageLoraTrainChunk": "VRGDG Z-Image LoRA Train Chunk",
    "VRGDG_ZImageSpeedCharacterLoraTraining": "VRGDG Z-Image Speed Character Lora Training",
    "VRGDG_SpeedCharacterLoraTraining": "VRGDG LTX Speed Character Lora Training",
    "VRGDG_LTXPreviewXYZPlot": "VRGDG LTX Preview XYZ Plot",
    "VRGDG_VideoFolderGridPlot": "VRGDG Video Folder Grid Plot",
    "VRGDG_MusubiTunerInstaller": "VRGDG Musubi-Tuner Installer",
}

try:
    _ensure_tensorboard_route_registered()
except Exception as exc:
    print(f"[VRGDG] Failed to register TensorBoard route: {exc}")

try:
    _ensure_musubi_install_route_registered()
except Exception as exc:
    print(f"[VRGDG] Failed to register Musubi installer route: {exc}")
