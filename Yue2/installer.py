"""Explicit-button installer and portable settings node for VRGDG YuE2."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path

from aiohttp import web
from server import PromptServer

from .nodes import SHEETSAGE2_CONFIG, YUE2_CONFIG


_INSTALL_LOCK = threading.Lock()
_ROUTE_REGISTERED = False
_SOURCE_ARCHIVE = "https://github.com/multimodal-art-projection/YuE/archive/refs/heads/main.zip"


def _paths(target_root: str) -> dict:
    root = Path(target_root).expanduser().resolve()
    windows = os.name == "nt"
    return {
        "root": root,
        "yue2_python": root / ".venv" / ("Scripts/python.exe" if windows else "bin/python"),
        "sheetsage_python": root / "SheetSage2-venv" / ("Scripts/python.exe" if windows else "bin/python"),
        "model": root / "models" / "YuE2-3B",
        "vae": root / "models" / "YuE2-Vae",
        "sheetsage_model": root / "models" / "SheetSage2",
        "mert_model": root / "models" / "MERT-v2-FullSong",
        "cache": root / "hf-cache",
        "report": root / "vrgdg_yue2_install_report.json",
    }


def _validate_target(value: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Choose a target_root, for example D:\\Yue2.")
    root = Path(text).expanduser().resolve()
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError("target_root must be a dedicated YuE2 folder, not a drive root or home folder.")
    return root


def _emit(lines: list[str], message) -> None:
    value = str(message)
    lines.append(value)
    print(value, flush=True)


def _run(command: list[str], cwd: Path, lines: list[str]) -> None:
    _emit(lines, "$ " + " ".join(map(str, command)))
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        list(map(str, command)), cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        env=environment,
    )
    assert process.stdout is not None
    for output in process.stdout:
        _emit(lines, output.rstrip("\r\n"))
    code = process.wait()
    if code:
        raise RuntimeError(f"Command exited with code {code}: {' '.join(map(str, command))}")


def _python_version(command: list[str]) -> tuple[int, int, str] | None:
    try:
        result = subprocess.run(
            command + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        )
        version = result.stdout.strip().splitlines()[-1]
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
        return (int(match.group(1)), int(match.group(2)), version) if match else None
    except Exception:
        return None


def _find_python(preferred_minors: tuple[int, ...]) -> tuple[list[str], str]:
    candidates: list[list[str]] = []
    if os.name == "nt":
        launcher = shutil.which("py")
        if launcher:
            candidates.extend([[launcher, f"-3.{minor}"] for minor in preferred_minors])
    else:
        for minor in preferred_minors:
            executable = shutil.which(f"python3.{minor}")
            if executable:
                candidates.append([executable])
    candidates.append([sys.executable])
    seen = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        found = _python_version(candidate)
        if found and found[0] == 3 and found[1] in preferred_minors:
            return candidate, f"Python {found[2]}"
    versions = ", ".join(f"3.{minor}" for minor in preferred_minors)
    if os.name == "nt" and set(preferred_minors) == {10, 11}:
        raise RuntimeError(
            "SheetSage2 cover tools require Python 3.10 or 3.11; they cannot use the "
            "Python 3.12/3.13 runtime bundled with ComfyUI. Install Python 3.11.9 from "
            "https://www.python.org/downloads/release/python-3119/ with the Windows py "
            "launcher enabled, confirm `py -3.11 --version` in Command Prompt, restart "
            "ComfyUI, and run this button again."
        )
    raise RuntimeError(f"Could not find a supported Python ({versions}). Install it with the Windows py launcher enabled.")


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            candidate = (destination / member.filename).resolve()
            if destination_resolved not in candidate.parents and candidate != destination_resolved:
                raise RuntimeError(f"Unsafe path in downloaded archive: {member.filename}")
        bundle.extractall(destination)
    directories = [entry for entry in destination.iterdir() if entry.is_dir()]
    if len(directories) != 1:
        raise RuntimeError("The YuE2 source archive did not contain one top-level directory.")
    return directories[0]


def _ensure_source(root: Path, lines: list[str]) -> None:
    if (root / "pyproject.toml").is_file() and (root / "src" / "yue2").is_dir():
        _emit(lines, f"[VRGDG/YuE2 installer] Reusing official source at {root}")
        return
    if root.exists() and any(root.iterdir()):
        allowed = {"SheetSage2-venv", "models", "hf-cache", "vrgdg_yue2_install_report.json"}
        unexpected = sorted(entry.name for entry in root.iterdir() if entry.name not in allowed)
        if unexpected:
            raise RuntimeError(
                f"Target exists but is not a YuE2 source checkout: {root}. "
                f"Unexpected entries: {', '.join(unexpected[:8])}. Choose a new folder or an existing YuE2 root."
            )
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="vrgdg_yue2_source_"))
    try:
        archive = temporary / "YuE-main.zip"
        _emit(lines, f"[VRGDG/YuE2 installer] Downloading official source: {_SOURCE_ARCHIVE}")
        request = urllib.request.Request(_SOURCE_ARCHIVE, headers={"User-Agent": "VRGDG-YuE2-Installer/1.0"})
        with urllib.request.urlopen(request) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        extracted = _safe_extract(archive, temporary / "extract")
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extracted, root, dirs_exist_ok=True)
        _emit(lines, f"[VRGDG/YuE2 installer] Source installed at {root}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _ensure_venv(root: Path, environment: Path, preferred: tuple[int, ...], lines: list[str]) -> Path:
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python.is_file():
        _emit(lines, f"[VRGDG/YuE2 installer] Reusing environment: {environment}")
        return python
    command, version = _find_python(preferred)
    _emit(lines, f"[VRGDG/YuE2 installer] Creating {environment.name} with {version}")
    _run(command + ["-m", "venv", str(environment)], root, lines)
    if not python.is_file():
        raise RuntimeError(f"Virtual-environment Python was not created: {python}")
    return python


def _install_generation(root: Path, lines: list[str]) -> Path:
    _ensure_source(root, lines)
    python = _ensure_venv(root, root / ".venv", (12, 11, 10), lines)
    _run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], root, lines)
    _emit(lines, "[VRGDG/YuE2 installer] Installing CUDA-enabled PyTorch 2.10.0 (CUDA 13.0)")
    _run([
        python, "-m", "pip", "install", "--upgrade", "torch==2.10.0+cu130",
        "--index-url", "https://download.pytorch.org/whl/cu130",
    ], root, lines)
    _run([python, "-m", "pip", "install", "--upgrade", "."], root, lines)
    return python


def _download_snapshots(python: Path, root: Path, snapshots: list[tuple[str, Path]], lines: list[str]) -> None:
    instructions = [(repo, str(path)) for repo, path in snapshots]
    script = (
        "import json; from huggingface_hub import snapshot_download; "
        f"items=json.loads({json.dumps(json.dumps(instructions))}); "
        "[(print('[VRGDG/HF] '+repo+' -> '+path, flush=True), "
        "snapshot_download(repo_id=repo, local_dir=path)) for repo,path in items]"
    )
    _run([python, "-c", script], root, lines)


def _install_generation_models(root: Path, python: Path, lines: list[str]) -> None:
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    _download_snapshots(python, root, [
        ("m-a-p/YuE2-3B", models / "YuE2-3B"),
        ("m-a-p/YuE2-Vae", models / "YuE2-Vae"),
    ], lines)


def _install_cover(root: Path, lines: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    python = _ensure_venv(root, root / "SheetSage2-venv", (10, 11), lines)
    _run([python, "-m", "pip", "install", "--upgrade", "pip", "wheel"], root, lines)
    torch_index = "https://download.pytorch.org/whl/cu129"
    _run([
        python, "-m", "pip", "install", "torch==2.8.0", "torchaudio==2.8.0",
        "--index-url", torch_index,
    ], root, lines)
    _run([
        python, "-m", "pip", "install",
        "transformers==4.45.2", "huggingface-hub==0.36.0", "safetensors==0.5.3",
        "numpy==1.24.3", "scipy==1.13.1", "mir_eval==0.8.2",
        "pretty_midi==0.2.10", "mido==1.3.3", "setuptools==78.1.1",
    ], root, lines)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    _download_snapshots(python, root, [
        ("m-a-p/SheetSage2", models / "SheetSage2"),
        ("m-a-p/MERT-v2-FullSong", models / "MERT-v2-FullSong"),
    ], lines)
    return python


def _has_weights(directory: Path) -> bool:
    return directory.is_dir() and any(directory.rglob("*.safetensors"))


def _run_check(command: list[str], cwd: Path, label: str, required: bool, lines: list[str]) -> dict:
    try:
        _run(command, cwd, lines)
        return {"label": label, "required": required, "ok": True, "message": "PASS"}
    except Exception as exc:
        _emit(lines, f"[VRGDG/YuE2 installer] {'ERROR' if required else 'WARN'}: {label}: {exc}")
        return {"label": label, "required": required, "ok": False, "message": str(exc)}


def _verify(root: Path, lines: list[str], scope: str = "all") -> tuple[list[dict], Path]:
    paths = _paths(str(root))
    checks: list[dict] = []
    generation_required = scope in {"generation", "all"}
    cover_required = scope in {"cover", "all"}
    yue_python = paths["yue2_python"]
    cover_python = paths["sheetsage_python"]
    checks.append(_run_check([
        str(yue_python), "-c",
        "import torch,yue2,transformers,numpy; "
        "print('yue2',yue2.__file__); print('torch',torch.__version__,'cuda',torch.cuda.is_available()); "
        "assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()",
    ], root, "YuE2 imports, CUDA, and BF16", generation_required, lines) if yue_python.is_file() else {
        "label": "YuE2 environment", "required": generation_required, "ok": False, "message": f"Missing {yue_python}"})
    checks.append(_run_check([
        str(cover_python), "-c",
        "import torch,torchaudio,transformers,numpy,pretty_midi,mir_eval; "
        "print('torch',torch.__version__,'cuda',torch.cuda.is_available()); assert torch.cuda.is_available()",
    ], root, "SheetSage2 imports and CUDA", cover_required, lines) if cover_python.is_file() else {
        "label": "SheetSage2 environment", "required": cover_required, "ok": False, "message": f"Missing {cover_python}"})
    for label, key, required in (
        ("YuE2-3B weights", "model", generation_required),
        ("YuE2-Vae weights", "vae", generation_required),
        ("SheetSage2 weights", "sheetsage_model", cover_required),
        ("MERT-v2-FullSong weights", "mert_model", cover_required),
    ):
        location = paths[key]
        checks.append({"label": label, "required": required, "ok": _has_weights(location), "message": str(location)})
    ffmpeg = shutil.which("ffmpeg")
    checks.append({
        "label": "FFmpeg on PATH", "required": False, "ok": bool(ffmpeg),
        "message": ffmpeg or "Not found; ComfyUI WAV handoff may still work, but direct compressed-file decoding can fail.",
    })
    report = paths["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "schema": "vrgdg-yue2-install-report-v1", "target_root": str(root),
        "paths": {key: str(value) for key, value in paths.items()}, "checks": checks,
    }, indent=2), encoding="utf-8")
    _emit(lines, f"[VRGDG/YuE2 installer] Verification report: {report}")
    return checks, report


def _install(payload: dict) -> dict:
    acquired = _INSTALL_LOCK.acquire(blocking=False)
    if not acquired:
        raise RuntimeError("Another YuE2 installation job is already running.")
    lines: list[str] = []
    try:
        root = _validate_target(payload.get("target_root", ""))
        action = str(payload.get("action", "install_all"))
        if action not in {"install_generation", "install_cover", "install_all", "verify"}:
            raise ValueError(f"Unknown YuE2 installer action: {action}")
        root.parent.mkdir(parents=True, exist_ok=True)
        if action in {"install_generation", "install_all"}:
            generation_python = _install_generation(root, lines)
            _install_generation_models(root, generation_python, lines)
        if action in {"install_cover", "install_all"}:
            _install_cover(root, lines)
        scope = {
            "install_generation": "generation", "install_cover": "cover",
            "install_all": "all", "verify": "all",
        }[action]
        checks, report = _verify(root, lines, scope=scope)
        failed = [check for check in checks if check.get("required") and not check.get("ok")]
        paths = _paths(str(root))
        result = {
            "ok": not failed,
            "status": "YuE2 installation and verification completed." if not failed else "Installation completed with required verification failures.",
            "messages": lines, "checks": checks, "report_path": str(report),
            **{key: str(value) for key, value in paths.items() if key != "root"},
            "target_root": str(root),
        }
        if failed:
            result["error"] = "; ".join(f"{item['label']}: {item['message']}" for item in failed)
        return result
    finally:
        _INSTALL_LOCK.release()


def _register_route() -> None:
    global _ROUTE_REGISTERED
    if _ROUTE_REGISTERED:
        return
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return

    @server.routes.post("/vrgdg/yue2/install")
    async def install_yue2(request):
        try:
            payload = await request.json()
            result = await asyncio.to_thread(_install, payload)
            return web.json_response(result, status=200 if result.get("ok") else 500)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    _ROUTE_REGISTERED = True


class VRGDG_YuE2Installer:
    RETURN_TYPES = (YUE2_CONFIG, SHEETSAGE2_CONFIG, "STRING", "STRING")
    RETURN_NAMES = ("yue2_config", "sheetsage2_config", "status", "report_path")
    FUNCTION = "settings"
    CATEGORY = "VRGDG/Audio/YuE2"
    DESCRIPTION = "Installs and verifies isolated YuE2 and SheetSage2 runtimes using explicit buttons."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_root": ("STRING", {
                    "default": "", "placeholder": r"D:\Yue2",
                    "tooltip": "Dedicated installation folder. Install buttons create/reuse the YuE2 source, environments, models, cache, and report under this root.",
                }),
                "runtime_mode": (["isolated_process", "in_process_experimental"], {
                    "default": "isolated_process",
                    "tooltip": "Isolated process is recommended and prevents YuE2 dependency conflicts with ComfyUI. In-process mode ignores yue2_python and requires compatible packages inside ComfyUI.",
                }),
                "yue2_python": ("STRING", {
                    "default": "", "placeholder": "Blank = <target_root>\\.venv\\Scripts\\python.exe",
                    "tooltip": "Optional YuE2 Python override. Leave blank to use the isolated environment created under target_root.",
                }),
                "model": ("STRING", {
                    "default": "", "placeholder": "Blank = <target_root>\\models\\YuE2-3B",
                    "tooltip": "Optional YuE2-3B model folder or Hugging Face repository override. Leave blank for the model downloaded by this installer.",
                }),
                "vae": ("STRING", {
                    "default": "", "placeholder": "Blank = <target_root>\\models\\YuE2-Vae",
                    "tooltip": "Optional YuE2 VAE decoder folder or repository override. YuE2-Vae is the recommended listening decoder.",
                }),
                "cache_dir": ("STRING", {
                    "default": "", "placeholder": "Blank = <target_root>\\hf-cache",
                    "tooltip": "Hugging Face download/module cache. Leave blank to keep the cache inside target_root.",
                }),
                "device": (["cuda", "auto", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Generation device. CUDA is required for practical full-song generation; auto selects CUDA when available; CPU is primarily for diagnostics.",
                }),
                "memory_budget_gib": ("FLOAT", {
                    "default": 24.0, "min": 4.0, "max": 128.0, "step": 0.5,
                    "tooltip": "Total YuE2 memory budget in GiB. YuE2 keeps a 2 GiB reserve. Try 30 on a 32 GiB GPU for long covers; reduce it when sharing the GPU.",
                }),
                "backend": (["torch", "torch-eager"], {
                    "default": "torch",
                    "tooltip": "torch uses fast CUDA graphs and is recommended. torch-eager avoids graph capture for troubleshooting but is substantially slower.",
                }),
                "quantization": (["none", "fp8"], {
                    "default": "none",
                    "tooltip": "Model-weight precision. none uses the validated BF16 path. FP8 is experimental and can change quality, compatibility, and speed.",
                }),
                "offload_ar": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Moves unused autoregressive modules to system RAM during acoustic synthesis to reduce VRAM pressure. It can prevent OOM errors but adds transfer time.",
                }),
                "local_files_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, workers never contact Hugging Face and require complete local model folders. Disable only when intentionally allowing model downloads at runtime.",
                }),
                "unload_comfy_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Asks ComfyUI to unload its currently loaded models before starting YuE2 or SheetSage2, freeing VRAM for music generation.",
                }),
                "verify_hashes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Re-hashes every YuE2 model weight before each run. This improves integrity checking but can add minutes on slower drives; the installer already verifies required files.",
                }),
                "sheetsage2_python": ("STRING", {
                    "default": "", "placeholder": "Blank = <target_root>\\SheetSage2-venv\\Scripts\\python.exe",
                    "tooltip": "Optional SheetSage2 Python override. Keep it separate from YuE2 because their Torch, Transformers, and NumPy requirements differ.",
                }),
                "sheetsage2_model": ("STRING", {
                    "default": "", "placeholder": "Blank = <target_root>\\models\\SheetSage2",
                    "tooltip": "Optional SheetSage2 model folder override. SheetSage2 converts source audio into the ABC score used by the cover workflow.",
                }),
                "mert_model": ("STRING", {
                    "default": "", "placeholder": "Blank = <target_root>\\models\\MERT-v2-FullSong",
                    "tooltip": "Optional MERT-v2-FullSong parent-model override used internally by SheetSage2 transcription.",
                }),
                "sheetsage2_device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device used only for source-song transcription. CUDA is faster; CPU avoids transcription VRAM use but is much slower.",
                }),
            },
            "hidden": {
                "status_text": ("STRING", {"default": ""}),
                "report_path_saved": ("STRING", {"default": ""}),
            },
        }

    def settings(self, target_root, runtime_mode, yue2_python, model, vae, cache_dir,
                 device, memory_budget_gib, backend, quantization, offload_ar,
                 local_files_only, unload_comfy_models, verify_hashes,
                 sheetsage2_python, sheetsage2_model, mert_model, sheetsage2_device,
                 status_text="", report_path_saved=""):
        root = _validate_target(target_root)
        paths = _paths(str(root))
        yue2_python = str(yue2_python or "").strip() or str(paths["yue2_python"])
        model = str(model or "").strip() or str(paths["model"])
        vae = str(vae or "").strip() or str(paths["vae"])
        cache_dir = str(cache_dir or "").strip() or str(paths["cache"])
        sheetsage2_python = str(sheetsage2_python or "").strip() or str(paths["sheetsage_python"])
        sheetsage2_model = str(sheetsage2_model or "").strip() or str(paths["sheetsage_model"])
        mert_model = str(mert_model or "").strip() or str(paths["mert_model"])
        yue2 = {
            "schema": "vrgdg-yue2-config-v1", "runtime_mode": runtime_mode,
            "python_executable": yue2_python, "model": model,
            "vae": vae, "device": device, "memory_budget_gib": float(memory_budget_gib),
            "backend": backend, "quantization": quantization, "offload_ar": bool(offload_ar),
            "local_files_only": bool(local_files_only),
            "unload_comfy_models": bool(unload_comfy_models),
            "cache_dir": cache_dir, "verify_hashes": bool(verify_hashes),
        }
        sheetsage = {
            "schema": "vrgdg-sheetsage2-config-v1",
            "python_executable": sheetsage2_python,
            "model": sheetsage2_model, "parent_model": mert_model,
            "cache_dir": cache_dir, "device": sheetsage2_device,
            "local_files_only": bool(local_files_only),
            "unload_comfy_models": bool(unload_comfy_models),
        }
        status = str(status_text or "Use an installer button, then connect these configuration outputs.")
        return yue2, sheetsage, status, str(report_path_saved or paths["report"])


NODE_CLASS_MAPPINGS = {"VRGDG_YuE2Installer": VRGDG_YuE2Installer}
NODE_DISPLAY_NAME_MAPPINGS = {"VRGDG_YuE2Installer": "VRGDG YuE2 Installer + Settings"}

_register_route()
