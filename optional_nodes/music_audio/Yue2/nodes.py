"""YuE2 nodes with isolated-process and experimental in-process backends."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path

import folder_paths
import torch


YUE2_CONFIG = "VRGDG_YUE2_CONFIG"
SHEETSAGE2_CONFIG = "VRGDG_SHEETSAGE2_CONFIG"
DEFAULT_MODEL = "m-a-p/YuE2-3B"
DEFAULT_VAE = "m-a-p/YuE2-Vae"
DEFAULT_SHEETSAGE2_MODEL = "m-a-p/SheetSage2"
CUSTOM_STYLE_PRESET = "Custom / Keep typed style"
STYLE_PRESETS = {
    "Rock / Classic Rock": "English, classic rock, powerful warm lead vocal, overdriven electric guitars, Hammond organ, electric bass and live acoustic drums, anthemic hooks, dynamic analog studio production, 118 BPM",
    "Rock / Indie Rock": "English, indie rock, intimate slightly raw lead vocal, jangly electric guitars, melodic bass and energetic live drums, bittersweet memorable melody, roomy natural production, 126 BPM",
    "Rock / Arena Rock": "English, arena rock, soaring commanding lead vocal, wide distorted guitars, driving bass and thunderous live drums, huge singalong chorus, polished expansive production, 124 BPM",
    "Rock / Blues Rock": "English, blues rock, gritty soulful lead vocal, expressive electric guitar, Hammond organ, thick bass and loose live drums, swaggering groove and blues-inflected melody, warm vintage production, 104 BPM",
    "Rock / Pop Punk": "English, pop punk rock, urgent youthful lead vocal, bright distorted guitars, driving bass and fast punchy live drums, concise melodic verses and explosive singalong chorus, polished energetic production, 168 BPM",
    "Pop / Warm Piano Pop": "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, polished intimate production, 88 BPM",
    "Pop / Synth Pop": "English, cinematic synth-pop, expressive female lead vocal, analog synthesizers, pulsing electric bass and punchy electronic drums, emotional uplifting hook, wide polished production, 112 BPM",
    "Pop / Dance Pop": "English, modern dance-pop, confident bright lead vocal, glossy synthesizers, deep bass and crisp four-on-the-floor drums, immediate melodic chorus, energetic radio-ready production, 124 BPM",
    "Pop / Indie Pop": "English, dreamy indie pop, soft conversational lead vocal, clean guitar, warm synthesizers, melodic bass and light organic drums, wistful hook and airy layered production, 102 BPM",
    "Pop / Power Ballad": "English, dramatic pop power ballad, vulnerable lead vocal rising to a soaring chorus, grand piano, cinematic strings, electric guitar, bass and large live drums, emotional slow build, polished production, 76 BPM",
    "Country / Modern Country": "English, modern country, clear heartfelt lead vocal, acoustic guitar, clean electric guitar, pedal steel, warm bass and punchy live drums, conversational verses and a broad memorable chorus, polished Nashville production, 96 BPM",
    "Country / Honky-Tonk": "English, traditional honky-tonk country, lively twangy lead vocal, Telecaster guitar, fiddle, pedal steel, upright-style bass and shuffling drums, playful dancehall energy, vintage production, 132 BPM",
    "Country / Country Pop": "English, country pop, bright expressive lead vocal, acoustic guitar, subtle banjo, clean electric guitar, rounded bass and modern drums, uplifting melodic chorus, glossy spacious production, 108 BPM",
    "Country / Americana": "English, rootsy Americana, weathered intimate lead vocal, fingerpicked acoustic guitar, mandolin, fiddle, upright bass and restrained live drums, reflective storytelling melody, natural room production, 84 BPM",
    "Country / Outlaw Country": "English, outlaw country, low rugged lead vocal, dry acoustic guitar, gritty electric guitar, pedal steel, steady bass and sparse live drums, defiant storytelling and dusty analog production, 92 BPM",
    "Rap / Boom Bap": "English, boom bap rap, focused articulate rapper, chopped soul samples, dusty piano, deep bass and hard swung drums, dense internal-rhyme cadence and a memorable hook, gritty warm production, 92 BPM",
    "Rap / Trap": "English, modern trap rap, confident rhythmic rapper with restrained melodic ad-libs, dark bell synths, sliding sub bass and crisp hi-hat patterns, spacious verses and forceful hook, polished heavy production, 142 BPM",
    "Rap / Conscious": "English, conscious rap, thoughtful expressive rapper with clear diction, warm jazz piano, subtle guitar, rounded bass and laid-back drums, narrative verses and soulful refrain, organic detailed production, 88 BPM",
    "Rap / Melodic": "English, melodic rap, emotive lead alternating sung hooks and agile rap verses, atmospheric synthesizers, guitar textures, deep sub bass and half-time drums, bittersweet modern production, 138 BPM",
    "Rap / Cinematic": "English, cinematic rap, commanding dramatic rapper, low strings, brass accents, dark piano, massive bass and hard orchestral drums, escalating verses and triumphant hook, wide theatrical production, 96 BPM",
    "Hip Hop / Golden Age": "English, golden-age hip hop, charismatic rhythmic lead vocal, funk and jazz sample textures, upright-style bass, scratches and punchy breakbeat drums, call-and-response hook, warm tape production, 98 BPM",
    "Hip Hop / Neo-Soul": "English, neo-soul hip hop, smooth sung lead with relaxed rap passages, Rhodes piano, mellow guitar, warm bass and pocket drums, rich harmony and intimate late-night production, 86 BPM",
    "Hip Hop / Alternative": "English, alternative hip hop, expressive unconventional lead vocal, warped keyboards, textured samples, elastic bass and off-kilter drums, surprising structure and experimental spacious production, 94 BPM",
    "Hip Hop / West Coast": "English, West Coast hip hop, laid-back confident rapper, bright synth lead, clean electric piano, deep rounded bass and crisp bouncing drums, relaxed melodic hook, sunlit polished production, 96 BPM",
    "Hip Hop / Lo-Fi": "English, lo-fi hip hop song, soft close-miked lead vocal with relaxed rap phrasing, dusty Rhodes, muted guitar, mellow bass and swung vinyl-textured drums, understated hook and hazy intimate production, 82 BPM",
    "Metal / Heavy Metal": "English, traditional heavy metal, powerful high-register lead vocal, twin distorted guitars, galloping bass and forceful live drums, heroic riffs and anthemic chorus, clear muscular production, 148 BPM",
    "Metal / Thrash Metal": "English, thrash metal, aggressive shouted melodic lead vocal, fast palm-muted guitars, cutting bass and relentless double-kick drums, angular riffs and urgent chorus, tight raw production, 190 BPM",
    "Metal / Metalcore": "English, modern metalcore, intense harsh verses and soaring clean chorus vocals, downtuned guitars, heavy bass and precise double-kick drums, dramatic breakdowns and polished dense production, 154 BPM",
    "Metal / Symphonic Metal": "English, symphonic metal, commanding operatic female lead vocal, heavy guitars, orchestral strings, choir, deep bass and cinematic live drums, grand melodic chorus and expansive production, 132 BPM",
    "Metal / Doom Metal": "English, doom metal, deep mournful lead vocal, massive slow distorted guitars, ominous organ, heavy bass and deliberate drums, bleak sustained melody and cavernous analog production, 62 BPM",
    "90s Alternative / Grunge": "English, 1990s grunge-inspired alternative rock, raw anguished lead vocal, thick detuned guitars, gritty bass and explosive live drums, quiet-loud dynamics and an unpolished room sound, 112 BPM",
    "90s Alternative / Shoegaze": "English, 1990s shoegaze-inspired alternative, distant breathy lead vocal, layered washed-out guitars, melodic bass and driving live drums, dreamy bittersweet melody and dense immersive production, 104 BPM",
    "90s Alternative / Britpop": "English, 1990s Britpop-inspired alternative rock, charismatic melodic lead vocal, bright electric guitars, tuneful bass and lively drums, witty verses and a sweeping singalong chorus, crisp guitar-forward production, 122 BPM",
    "90s Alternative / College Rock": "English, 1990s college alternative rock, earnest understated lead vocal, chiming guitars, melodic bass and loose live drums, literate verses and hooky chorus, dry natural production, 116 BPM",
    "90s Alternative / Industrial": "English, 1990s industrial alternative rock, tense gritty lead vocal, distorted guitars, abrasive synthesizers, mechanical bass and pounding programmed drums, dark repetitive hook and aggressive layered production, 118 BPM",
}
STYLE_PRESET_NAMES = [CUSTOM_STYLE_PRESET, *STYLE_PRESETS]
_PIPELINES = {}
_PIPELINE_LOCK = threading.RLock()


def _relay_log(path: Path, state: dict, label: str, final: bool = False) -> bool:
    """Relay newly appended worker output to the ComfyUI console."""
    if not path.is_file():
        return False
    key = str(path)
    entry = state.setdefault(key, {"offset": 0, "pending": b""})
    with path.open("rb") as stream:
        stream.seek(entry["offset"])
        chunk = stream.read()
        entry["offset"] = stream.tell()
    if not chunk and not final:
        return False
    data = entry["pending"] + chunk
    lines = data.split(b"\n")
    if final:
        complete, entry["pending"] = lines, b""
    else:
        complete, entry["pending"] = lines[:-1], lines[-1]
    wrote = False
    for raw in complete:
        line = raw.rstrip(b"\r").decode("utf-8", errors="replace").strip()
        if line:
            print(f"[{label}] {line}", flush=True)
            wrote = True
    return wrote


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "song").strip())
    return value.strip("._")[:80] or "song"


def _new_output_dir(prefix: str) -> Path:
    root = Path(folder_paths.get_output_directory()) / "Yue2"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return root / f"{stamp}-{_safe_name(prefix)}-{uuid.uuid4().hex[:8]}"


def _prepare_comfy_memory(enabled: bool):
    def report(label):
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                print(
                    f"[VRGDG/YuE2] GPU memory {label}: {free / 1024**3:.1f} GiB free / "
                    f"{total / 1024**3:.1f} GiB total.", flush=True,
                )
        except Exception as exc:
            print(f"[VRGDG/YuE2] Could not query GPU memory {label}: {exc}", flush=True)

    report("before ComfyUI unload")
    if not enabled:
        return
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
        report("after ComfyUI unload")
    except Exception as exc:
        print(f"[VRGDG/YuE2] Could not request ComfyUI model unload: {exc}")


def _check_interrupted():
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except AttributeError:
        return


def _is_interrupted() -> bool:
    try:
        import comfy.model_management as model_management

        return bool(model_management.processing_interrupted())
    except (AttributeError, ImportError):
        return False


def _audio_from_file(path: str):
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(path)
    except Exception:
        try:
            import soundfile as sf
        except Exception as exc:
            raise RuntimeError("Reading YuE2 audio requires torchaudio or soundfile.") from exc
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T.copy())
    return {"waveform": waveform.float().unsqueeze(0), "sample_rate": int(sample_rate)}


def _audio_to_pcm16_wav(audio: dict, path: Path):
    """Write the first ComfyUI AUDIO batch item without adding dependencies."""
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Connect a ComfyUI AUDIO output to source_audio.")
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu()
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError(f"Unsupported ComfyUI waveform shape: {tuple(waveform.shape)}")
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError(f"Invalid audio sample rate: {sample_rate}")
    pcm = (waveform.clamp(-1.0, 1.0).transpose(0, 1) * 32767.0).round().to(torch.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(int(pcm.shape[1]))
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.numpy().tobytes())


def _config_key(config: dict):
    return tuple(
        config.get(name)
        for name in (
            "model", "vae", "device", "memory_budget_gib", "backend",
            "quantization", "offload_ar", "local_files_only", "cache_dir", "verify_hashes",
            "ode_steps",
        )
    )


def _load_in_process(config: dict):
    try:
        from yue2 import YuE2Pipeline
        from yue2.protocol import GenerationConfig
    except Exception as exc:
        raise RuntimeError(
            "YuE2 is not importable by ComfyUI. Use isolated_process with a YuE2 "
            "environment, or install yue2-infer without replacing ComfyUI's Torch stack."
        ) from exc

    key = _config_key(config)
    with _PIPELINE_LOCK:
        pipe = _PIPELINES.get(key)
        if pipe is not None:
            return pipe
        kwargs = {
            "vae": config["vae"],
            "device": config["device"],
            "memory_budget_gib": config["memory_budget_gib"],
            "backend": config["backend"],
            "quantization": config["quantization"],
            "offload_ar": config["offload_ar"],
            "local_files_only": config["local_files_only"],
            "verify_hashes": config.get("verify_hashes", False),
            "progress": True,
            "generation_config": GenerationConfig(ode_steps=int(config.get("ode_steps", 32))),
        }
        if config["cache_dir"]:
            kwargs["cache_dir"] = config["cache_dir"]
        pipe = YuE2Pipeline.from_pretrained(config["model"], **kwargs)
        _PIPELINES[key] = pipe
        return pipe


def _request(style, lyrics, cot, seed, cfg_scale, abc="", song_id="song", ode_steps=None):
    style = str(style or "").strip()
    lyrics = str(lyrics or "").strip()
    if not style:
        raise ValueError("YuE2 requires a non-empty style prompt.")
    if not lyrics:
        raise ValueError("YuE2 requires non-empty lyrics.")
    values = {
        "style": style,
        "lyrics": lyrics,
        "cot": cot,
        "seed": int(seed),
        "id": _safe_name(song_id),
    }
    if abc:
        values["abc"] = str(abc)
    if float(cfg_scale) >= 0:
        values["cfg_scale"] = float(cfg_scale)
    if ode_steps is not None:
        ode_steps = int(ode_steps)
        if not 1 <= ode_steps <= 128:
            raise ValueError("YuE2 ODE steps must be between 1 and 128.")
        values["_ode_steps"] = ode_steps
    return values


def _run_isolated(config: dict, operation: str, request: dict, output_dir: Path):
    executable = str(config.get("python_executable", "")).strip()
    if not executable:
        raise ValueError(
            "isolated_process requires python_executable from the YuE2 virtual environment."
        )
    executable_path = Path(os.path.expandvars(os.path.expanduser(executable)))
    if not executable_path.is_file():
        raise FileNotFoundError(f"YuE2 Python executable was not found: {executable_path}")

    control_dir = output_dir.parent / f".{output_dir.name}-worker"
    control_dir.mkdir(parents=True, exist_ok=False)
    request_path = control_dir / "request.json"
    response_path = control_dir / "response.json"
    stdout_path = control_dir / "stdout.log"
    stderr_path = control_dir / "stderr.log"
    payload = {
        "operation": operation,
        "config": config,
        "request": request,
        "output_dir": str(output_dir),
    }
    request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    worker = Path(__file__).with_name("worker.py")
    command = [str(executable_path), "-u", str(worker), "--request", str(request_path), "--response", str(response_path)]
    worker_env = os.environ.copy()
    worker_env["PYTHONUNBUFFERED"] = "1"

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
            env=worker_env,
        )
        print(
            f"[VRGDG/YuE2] Started worker PID {process.pid}. Live progress follows. "
            f"Logs: {control_dir}", flush=True,
        )
        relay_state = {}
        started = time.monotonic()
        last_message = started
        try:
            while process.poll() is None:
                _check_interrupted()
                wrote = _relay_log(stdout_path, relay_state, "VRGDG/YuE2 stdout")
                wrote |= _relay_log(stderr_path, relay_state, "VRGDG/YuE2 progress")
                now = time.monotonic()
                if wrote:
                    last_message = now
                elif now - last_message >= 15:
                    print(
                        f"[VRGDG/YuE2] Worker PID {process.pid} is alive; elapsed "
                        f"{now - started:.0f}s. Waiting for the next model update...",
                        flush=True,
                    )
                    last_message = now
                time.sleep(0.2)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        finally:
            _relay_log(stdout_path, relay_state, "VRGDG/YuE2 stdout", final=True)
            _relay_log(stderr_path, relay_state, "VRGDG/YuE2 progress", final=True)
        print(
            f"[VRGDG/YuE2] Worker PID {process.pid} exited with code {process.returncode} "
            f"after {time.monotonic() - started:.1f}s.", flush=True,
        )

    if not response_path.is_file():
        details = stderr_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"YuE2 worker exited without a response (code {process.returncode}).\n{details}")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if process.returncode or response.get("status") != "complete":
        details = response.get("traceback") or response.get("error") or stderr_path.read_text(
            encoding="utf-8", errors="replace"
        )[-6000:]
        raise RuntimeError(f"YuE2 worker failed:\n{details}")
    return response


def _run_in_process(config: dict, operation: str, request: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    pipe = _load_in_process(config)
    if operation == "plan":
        plan = pipe.plan(cancelled=_is_interrupted, **request)
        plan.save(output_dir)
        return {
            "status": "complete", "operation": "plan", "abc": plan.abc or "",
            "truncated": bool(plan.truncated), "timing": plan.timing,
            "output_dir": str(output_dir),
        }
    song = pipe(cancelled=_is_interrupted, **request)
    saved = song.save_artifacts(output_dir)
    return {
        "status": "complete", "operation": "generate", "abc": song.abc or "",
        "truncated": song.truncated, "sample_rate": int(song.sample_rate),
        "audio_path": str(output_dir / "audio.flac"), "output_dir": str(output_dir),
        "result": saved,
    }


def _execute(config: dict, operation: str, request: dict, prefix: str):
    if not isinstance(config, dict) or config.get("schema") != "vrgdg-yue2-config-v1":
        raise ValueError("Connect a VRGDG YuE2 Settings node to yue2_config.")
    config = dict(config)
    request = dict(request)
    config["ode_steps"] = int(request.pop("_ode_steps", config.get("ode_steps", 32)))
    _prepare_comfy_memory(bool(config.get("unload_comfy_models", True)))
    output_dir = _new_output_dir(prefix)
    if config["runtime_mode"] == "isolated_process":
        return _run_isolated(config, operation, request, output_dir)
    return _run_in_process(config, operation, request, output_dir)


def _run_sheetsage2(config: dict, audio: dict, melody_only: bool, prefix: str):
    if not isinstance(config, dict) or config.get("schema") != "vrgdg-sheetsage2-config-v1":
        raise ValueError("Connect a VRGDG SheetSage2 Settings node to sheetsage2_config.")
    executable = Path(os.path.expandvars(os.path.expanduser(config["python_executable"])))
    if not executable.is_file():
        raise FileNotFoundError(f"SheetSage2 Python executable was not found: {executable}")

    _prepare_comfy_memory(bool(config.get("unload_comfy_models", True)))
    output_dir = _new_output_dir(prefix + "-score")
    control_dir = output_dir.parent / f".{output_dir.name}-sheetsage-worker"
    control_dir.mkdir(parents=True, exist_ok=False)
    source_path = control_dir / "source.wav"
    _audio_to_pcm16_wav(audio, source_path)

    request_path = control_dir / "request.json"
    response_path = control_dir / "response.json"
    stdout_path = control_dir / "stdout.log"
    stderr_path = control_dir / "stderr.log"
    request_path.write_text(json.dumps({
        "config": config,
        "audio_path": str(source_path),
        "output_dir": str(output_dir),
        "melody_only": bool(melody_only),
    }, indent=2), encoding="utf-8")
    worker = Path(__file__).with_name("sheetsage_worker.py")
    command = [str(executable), "-u", str(worker), "--request", str(request_path), "--response", str(response_path)]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    worker_env = os.environ.copy()
    cache_dir = str(config.get("cache_dir", "")).strip()
    if cache_dir:
        worker_env["HF_HOME"] = cache_dir
        worker_env["HF_MODULES_CACHE"] = str(Path(cache_dir) / "modules")
    worker_env["PYTHONUNBUFFERED"] = "1"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, creationflags=creationflags, env=worker_env
        )
        print(
            f"[VRGDG/SheetSage2] Started worker PID {process.pid}. Live progress follows. "
            f"Logs: {control_dir}", flush=True,
        )
        relay_state = {}
        started = time.monotonic()
        last_message = started
        try:
            while process.poll() is None:
                _check_interrupted()
                wrote = _relay_log(stdout_path, relay_state, "VRGDG/SheetSage2 stdout")
                wrote |= _relay_log(stderr_path, relay_state, "VRGDG/SheetSage2 progress")
                now = time.monotonic()
                if wrote:
                    last_message = now
                elif now - last_message >= 15:
                    print(
                        f"[VRGDG/SheetSage2] Worker PID {process.pid} is alive; elapsed "
                        f"{now - started:.0f}s. Waiting for the next model update...",
                        flush=True,
                    )
                    last_message = now
                time.sleep(0.2)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        finally:
            _relay_log(stdout_path, relay_state, "VRGDG/SheetSage2 stdout", final=True)
            _relay_log(stderr_path, relay_state, "VRGDG/SheetSage2 progress", final=True)
        print(
            f"[VRGDG/SheetSage2] Worker PID {process.pid} exited with code {process.returncode} "
            f"after {time.monotonic() - started:.1f}s.", flush=True,
        )
    if not response_path.is_file():
        details = stderr_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"SheetSage2 worker exited without a response (code {process.returncode}).\n{details}")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if process.returncode or response.get("status") != "complete":
        details = response.get("traceback") or response.get("error") or stderr_path.read_text(
            encoding="utf-8", errors="replace"
        )[-6000:]
        raise RuntimeError(f"SheetSage2 worker failed:\n{details}")
    return response


class VRGDG_SheetSage2Settings:
    RETURN_TYPES = (SHEETSAGE2_CONFIG, "STRING")
    RETURN_NAMES = ("sheetsage2_config", "summary")
    FUNCTION = "build"
    CATEGORY = "VRGDG/Audio/YuE2"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "python_executable": ("STRING", {"default": "", "placeholder": r"C:\path\to\SheetSage2-venv\Scripts\python.exe", "tooltip": "Python executable from the separate SheetSage2 environment used to transcribe source audio."}),
            "model": ("STRING", {"default": DEFAULT_SHEETSAGE2_MODEL, "tooltip": "Local SheetSage2 model snapshot or Hugging Face repository ID."}),
            "parent_model": ("STRING", {"default": "m-a-p/MERT-v2-FullSong", "tooltip": "MERT-v2-FullSong encoder used internally by SheetSage2."}),
            "cache_dir": ("STRING", {"default": "", "placeholder": r"C:\path\to\hf-cache", "tooltip": "Hugging Face cache used for model files and trusted remote model code."}),
            "device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "CUDA is recommended for transcription. CPU is supported but much slower."}),
            "local_files_only": ("BOOLEAN", {"default": True, "tooltip": "Prevent network access and require complete local SheetSage2/MERT snapshots."}),
            "unload_comfy_models": ("BOOLEAN", {"default": True, "tooltip": "Unload ComfyUI models before transcription to free GPU memory."}),
        }}

    def build(self, python_executable, model, parent_model, cache_dir, device,
              local_files_only, unload_comfy_models):
        config = {
            "schema": "vrgdg-sheetsage2-config-v1",
            "python_executable": str(python_executable or "").strip(),
            "model": str(model or DEFAULT_SHEETSAGE2_MODEL).strip(),
            "parent_model": str(parent_model or "m-a-p/MERT-v2-FullSong").strip(),
            "cache_dir": str(cache_dir or "").strip(),
            "device": device,
            "local_files_only": bool(local_files_only),
            "unload_comfy_models": bool(unload_comfy_models),
        }
        return config, f"SheetSage2 | {config['model']} + {config['parent_model']} | {device} | isolated process"


class VRGDG_SheetSage2Transcribe:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "AUDIO")
    RETURN_NAMES = ("abc_score", "artifact_directory", "metadata_json", "source_audio")
    FUNCTION = "transcribe"
    CATEGORY = "VRGDG/Audio/YuE2"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sheetsage2_config": (SHEETSAGE2_CONFIG, {"tooltip": "Configuration from YuE2 Installer + Settings or SheetSage2 Settings."}),
            "source_audio": ("AUDIO", {"tooltip": "Source recording to analyze for melody, timing, key, and structure."}),
            "melody_only": ("BOOLEAN", {"default": True, "tooltip": "Export chord-free melody conditioning for a style-changing cover. Disable to retain detected harmony too."}),
            "filename_prefix": ("STRING", {"default": "cover", "tooltip": "Name prefix for the transcription artifact folder."}),
        }}

    def transcribe(self, sheetsage2_config, source_audio, melody_only, filename_prefix):
        result = _run_sheetsage2(sheetsage2_config, source_audio, melody_only, filename_prefix)
        abc = str(result.get("abc", "")).strip()
        if not abc:
            raise RuntimeError("SheetSage2 completed but did not return an ABC score.")
        return abc, result["output_dir"], json.dumps(result, indent=2), source_audio


class VRGDG_YuE2Settings:
    RETURN_TYPES = (YUE2_CONFIG, "STRING")
    RETURN_NAMES = ("yue2_config", "summary")
    FUNCTION = "build"
    CATEGORY = "VRGDG/Audio/YuE2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "runtime_mode": (["isolated_process", "in_process_experimental"], {"default": "isolated_process", "tooltip": "Isolated mode protects ComfyUI from YuE2 dependency conflicts. In-process mode is experimental."}),
                "python_executable": ("STRING", {"default": "", "placeholder": r"C:\path\to\yue2\.venv\Scripts\python.exe", "tooltip": "Python executable containing the official yue2-infer package; required for isolated mode."}),
                "model": ("STRING", {"default": DEFAULT_MODEL, "tooltip": "YuE2-3B local model folder or Hugging Face repository ID."}),
                "vae": ("STRING", {"default": DEFAULT_VAE, "tooltip": "YuE2 VAE decoder folder or repository ID used to convert latents into 48 kHz stereo audio."}),
                "device": (["cuda", "auto", "cpu"], {"default": "cuda", "tooltip": "CUDA is required for practical generation. Auto chooses an available device; CPU is mainly diagnostic."}),
                "memory_budget_gib": ("FLOAT", {"default": 24.0, "min": 4.0, "max": 128.0, "step": 0.5, "tooltip": "YuE2 GPU-memory budget including a 2 GiB safety reserve. Long covers on a 32 GiB GPU may need 30."}),
                "backend": (["torch", "torch-eager"], {"default": "torch", "tooltip": "torch uses fast CUDA graphs. torch-eager is a slower troubleshooting fallback."}),
                "quantization": (["none", "fp8"], {"default": "none", "tooltip": "none uses validated BF16 weights. FP8 is experimental and can affect compatibility and quality."}),
                "offload_ar": ("BOOLEAN", {"default": False, "tooltip": "Move unused AR modules to system RAM during synthesis to save VRAM at the cost of transfer time."}),
                "local_files_only": ("BOOLEAN", {"default": False, "tooltip": "Require local files and prevent Hugging Face network downloads during generation."}),
                "unload_comfy_models": ("BOOLEAN", {"default": True, "tooltip": "Unload other ComfyUI models before launching YuE2 to maximize available VRAM."}),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": "", "placeholder": "Optional Hugging Face cache directory", "tooltip": "Optional directory for downloaded Hugging Face snapshots and metadata."}),
                "verify_hashes": ("BOOLEAN", {"default": False, "tooltip": "Hash all model weights before every run. More integrity checking, but slow on large weights and non-SSD storage."}),
            },
        }

    def build(self, runtime_mode, python_executable, model, vae, device,
              memory_budget_gib, backend, quantization, offload_ar,
              local_files_only, unload_comfy_models, cache_dir="", verify_hashes=False):
        config = {
            "schema": "vrgdg-yue2-config-v1",
            "runtime_mode": runtime_mode,
            "python_executable": str(python_executable or "").strip(),
            "model": str(model or DEFAULT_MODEL).strip(),
            "vae": str(vae or DEFAULT_VAE).strip(),
            "device": device,
            "memory_budget_gib": float(memory_budget_gib),
            "backend": backend,
            "quantization": quantization,
            "offload_ar": bool(offload_ar),
            "local_files_only": bool(local_files_only),
            "unload_comfy_models": bool(unload_comfy_models),
            "cache_dir": str(cache_dir or "").strip(),
            "verify_hashes": bool(verify_hashes),
        }
        summary = (
            f"YuE2 {runtime_mode} | {model} | {vae} | {device} | "
            f"{memory_budget_gib:g} GiB | {backend} | {quantization} | "
            f"verify_hashes={bool(verify_hashes)}"
        )
        return config, summary


class _YuE2GenerationInputs:
    @classmethod
    def common_inputs(cls):
        return {
            "yue2_config": (YUE2_CONFIG, {"tooltip": "Runtime configuration from YuE2 Installer + Settings or YuE2 Settings."}),
            "style": ("STRING", {"default": "English, cinematic pop, expressive lead vocal, piano, bass and drums", "multiline": True, "tooltip": "Describe language, genre, vocal character, instruments, musical feel/production, and BPM. Do not put lyrics here."}),
            "lyrics": ("STRING", {"default": "[Verse]\nWrite your lyrics here.\n\n[Chorus]\nWrite your chorus here.", "multiline": True, "tooltip": "The actual words to sing, organized with section tags such as [Verse], [Chorus], [Bridge], and [Outro]."}),
            "planning_mode": (["full", "melody", "off"], {"default": "full", "tooltip": "full plans melody and chords; melody plans without chords and is recommended for covers; off skips symbolic planning."}),
            "seed": ("INT", {"default": 831001, "min": 0, "max": 0x7FFFFFFFFFFFFFFF, "tooltip": "Random seed for reproducible comparisons. Different prompts can still produce different audio with the same seed."}),
            "cfg_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.01, "tooltip": "Style/lyric classifier-free guidance. -1 uses YuE2 defaults; approximately 1.2–1.4 can strengthen style but may reduce quality."}),
            "filename_prefix": ("STRING", {"default": "song", "tooltip": "Safe name prefix for the unique output artifact directory."}),
            "style_preset": (STYLE_PRESET_NAMES, {"default": CUSTOM_STYLE_PRESET, "tooltip": "Select a prepared genre prompt. The style box is filled automatically and remains editable."}),
        }


def _resolve_style(style, style_preset):
    if style_preset != CUSTOM_STYLE_PRESET:
        preset = STYLE_PRESETS.get(style_preset)
        if preset:
            return preset
    value = str(style or "").strip()
    if not value:
        raise ValueError("YuE2 style must not be empty.")
    return value


class VRGDG_YuE2Generate(_YuE2GenerationInputs):
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("audio", "abc_score", "artifact_directory", "metadata_json", "truncated")
    FUNCTION = "generate"
    CATEGORY = "VRGDG/Audio/YuE2"

    @classmethod
    def INPUT_TYPES(cls):
        values = cls.common_inputs()
        items = list(values.items())
        cfg_index = next(index for index, (name, _) in enumerate(items) if name == "cfg_scale") + 1
        items.insert(cfg_index, ("ode_steps", ("INT", {
            "default": 32, "min": 1, "max": 128, "step": 1,
            "tooltip": "Number of midpoint ODE steps used only during final audio synthesis. 32 is YuE2's official quality default; 16 is faster with lower fidelity, while 48–64 is slower with diminishing returns. This does not change planning or semantic token count.",
        })))
        return {"required": dict(items)}

    def generate(self, yue2_config, style, lyrics, planning_mode, seed, cfg_scale, ode_steps,
                 filename_prefix, style_preset=CUSTOM_STYLE_PRESET):
        style = _resolve_style(style, style_preset)
        request = _request(style, lyrics, planning_mode, seed, cfg_scale,
                           song_id=filename_prefix, ode_steps=ode_steps)
        result = _execute(yue2_config, "generate", request, filename_prefix)
        audio = _audio_from_file(result["audio_path"])
        truncated = result.get("truncated", False)
        if isinstance(truncated, dict):
            truncated = any(bool(value) for value in truncated.values())
        return audio, result.get("abc", ""), result["output_dir"], json.dumps(result, indent=2), bool(truncated)


class VRGDG_YuE2Plan(_YuE2GenerationInputs):
    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("abc_score", "plan_directory", "metadata_json", "truncated")
    FUNCTION = "plan"
    CATEGORY = "VRGDG/Audio/YuE2"

    @classmethod
    def INPUT_TYPES(cls):
        values = cls.common_inputs()
        values["planning_mode"] = (["full", "melody"], {
            "default": "full",
            "tooltip": "Choose whether the planned ABC score contains melody plus chords (full) or melody only. Melody-only is useful when you want to change the harmony or style later.",
        })
        return {"required": values}

    def plan(self, yue2_config, style, lyrics, planning_mode, seed, cfg_scale,
             filename_prefix, style_preset=CUSTOM_STYLE_PRESET):
        style = _resolve_style(style, style_preset)
        request = _request(style, lyrics, planning_mode, seed, cfg_scale, song_id=filename_prefix)
        result = _execute(yue2_config, "plan", request, filename_prefix + "-plan")
        return result.get("abc", ""), result["output_dir"], json.dumps(result, indent=2), bool(result.get("truncated", False))


class VRGDG_YuE2RenderABC:
    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("audio", "abc_score", "artifact_directory", "metadata_json", "truncated")
    FUNCTION = "render"
    CATEGORY = "VRGDG/Audio/YuE2"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yue2_config": (YUE2_CONFIG, {"tooltip": "Runtime configuration from YuE2 Installer + Settings or YuE2 Settings."}),
                "abc_score": ("STRING", {"default": "", "multiline": True, "tooltip": "Existing ABC score to render. For covers, connect the chord-free score from SheetSage2."}),
                "style": ("STRING", {"default": "English, cinematic pop, expressive lead vocal", "multiline": True, "tooltip": "Target language, genre, vocal character, instruments, production feel, and BPM. Do not paste lyrics here."}),
                "lyrics": ("STRING", {"default": "[Verse]\nWrite your lyrics here.", "multiline": True, "tooltip": "Actual sectioned words to sing. For covers, phrasing and syllable counts should roughly match the supplied melody."}),
                "planning_mode": (["full", "melody"], {"default": "full", "tooltip": "Use melody for a chord-free style-changing cover. Use full when the supplied ABC intentionally includes melody and harmony."}),
                "seed": ("INT", {"default": 831001, "min": 0, "max": 0x7FFFFFFFFFFFFFFF, "tooltip": "Random seed used for semantic generation and synthesis."}),
                "cfg_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.01, "tooltip": "-1 uses the native default. Values around 1.2–1.4 may strengthen target-style conditioning."}),
                "ode_steps": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1, "tooltip": "Number of midpoint ODE steps used only during final audio synthesis. 32 is the official quality default; 16 is faster with lower fidelity, while 48–64 is slower with diminishing returns. It does not change melody planning or semantic token count."}),
                "filename_prefix": ("STRING", {"default": "song-edited", "tooltip": "Safe prefix for the generated artifact directory."}),
                "style_preset": (STYLE_PRESET_NAMES, {"default": CUSTOM_STYLE_PRESET, "tooltip": "Select a target genre preset to fill the style box; editing the result switches back to Custom."}),
            }
        }

    def render(self, yue2_config, abc_score, style, lyrics, planning_mode, seed,
               cfg_scale, ode_steps, filename_prefix, style_preset=CUSTOM_STYLE_PRESET):
        if not str(abc_score or "").strip():
            raise ValueError("Render ABC requires a non-empty ABC score.")
        style = _resolve_style(style, style_preset)
        request = _request(style, lyrics, planning_mode, seed, cfg_scale, abc=abc_score,
                           song_id=filename_prefix, ode_steps=ode_steps)
        result = _execute(yue2_config, "generate", request, filename_prefix)
        audio = _audio_from_file(result["audio_path"])
        truncated = result.get("truncated", False)
        if isinstance(truncated, dict):
            truncated = any(bool(value) for value in truncated.values())
        return audio, result.get("abc", abc_score), result["output_dir"], json.dumps(result, indent=2), bool(truncated)


class VRGDG_YuE2Unload:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "VRGDG/Audio/YuE2"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True, "tooltip": "Release cached experimental in-process YuE2 models and empty the CUDA cache. Isolated workers exit automatically."})}}

    def unload(self, unload):
        if not unload:
            return ("YuE2 unload skipped.",)
        with _PIPELINE_LOCK:
            count = len(_PIPELINES)
            for pipe in list(_PIPELINES.values()):
                try:
                    pipe.close()
                except Exception as exc:
                    print(f"[VRGDG/YuE2] Pipeline close warning: {exc}")
            _PIPELINES.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (f"Unloaded {count} in-process YuE2 pipeline(s). Isolated workers exit after each run.",)


NODE_CLASS_MAPPINGS = {
    "VRGDG_SheetSage2Settings": VRGDG_SheetSage2Settings,
    "VRGDG_SheetSage2Transcribe": VRGDG_SheetSage2Transcribe,
    "VRGDG_YuE2Settings": VRGDG_YuE2Settings,
    "VRGDG_YuE2Generate": VRGDG_YuE2Generate,
    "VRGDG_YuE2Plan": VRGDG_YuE2Plan,
    "VRGDG_YuE2RenderABC": VRGDG_YuE2RenderABC,
    "VRGDG_YuE2Unload": VRGDG_YuE2Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_SheetSage2Settings": "VRGDG SheetSage2 Settings",
    "VRGDG_SheetSage2Transcribe": "VRGDG SheetSage2 Transcribe Cover",
    "VRGDG_YuE2Settings": "VRGDG YuE2 Settings",
    "VRGDG_YuE2Generate": "VRGDG YuE2 Generate Song",
    "VRGDG_YuE2Plan": "VRGDG YuE2 Create Plan",
    "VRGDG_YuE2RenderABC": "VRGDG YuE2 Render ABC",
    "VRGDG_YuE2Unload": "VRGDG YuE2 Unload Models",
}
