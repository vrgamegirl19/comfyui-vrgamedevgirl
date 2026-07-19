import base64
import io
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone

from PIL import Image

try:
    from server import PromptServer
except Exception:
    PromptServer = None

from .VRGDG_WorkflowRunnerNodes import _resolve_comfy_image_path


T2I_INSTRUCTIONS = """You are a text-to-image prompt generator that creates one single detailed image prompt from an art style and a short concept. Preserve the subject, action, setting, important objects, mood, and viewpoint. Treat the art style as the governing direction for every visible element. Describe recognizable medium, materials, textures, construction, shapes, palette, lighting, environment, composition, and atmosphere. Add concrete visual details without changing the concept. Write one continuous paragraph of plain text. Output exactly one complete prompt and nothing else. Do not use headings, labels, bullets, commentary, or generic quality phrases."""

CAPTION_INSTRUCTIONS = """You are writing captions for text-to-video style LoRA training. Begin with the supplied trigger word exactly as written, followed by a comma, then reproduce the supplied style phrase exactly, followed by a comma. Describe only visibly supported subject, appearance, wardrobe, pose, setting, props, lighting, palette, materials, textures, framing, camera qualities, and visual style. Do not invent story, motion, identity, relationships, unseen details, or unverifiable locations. Use dense natural tag-like prose in one line, 35 to 80 words. Do not use headings, labels, explanations, confidence notes, or phrases such as 'image of' or 'this image shows'. Return only the caption."""


def _clean_text(value):
    text = str(value or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", text, flags=re.I).strip()
    return text


def _runner(payload):
    return str(payload.get("llm_runner") or "lm_studio").strip().lower()


def _gemma_local(payload, prompt, images=None):
    from .LLM import VRGDG_LLM_Multi, VRGDG_SuperGemmaGGUFChat
    model_file = str(payload.get("gemma_model") or "").strip()
    mmproj_file = str(payload.get("gemma_mmproj") or "").strip()
    if not model_file:
        raise ValueError("Choose a local Gemma model.")
    runner = VRGDG_SuperGemmaGGUFChat()
    kwargs = {}
    helper = VRGDG_LLM_Multi()
    for index, image in enumerate(images or [], 1):
        kwargs[f"image{index}"] = helper._pil_to_tensor(image.convert("RGB"))
    text, _used_model, status = runner.generate_prompt(
        model_file=model_file,
        mmproj_file=mmproj_file,
        task_preset="custom",
        user_input="",
        custom_instructions=str(prompt or ""),
        trigger_word="",
        image_count=len(kwargs),
        advanced=True,
        unload_after_run=bool(payload.get("gemma_unload_after", False)),
        n_ctx=int(payload.get("gemma_n_ctx") or 8192),
        n_gpu_layers=int(payload.get("gemma_n_gpu_layers") or 99),
        n_threads=int(payload.get("gemma_n_threads") or 8),
        chat_format="",
        temperature=float(payload.get("temperature", 0.55)),
        top_p=0.95,
        max_new_tokens=int(payload.get("max_tokens", 1400)),
        **kwargs,
    )
    if str(status or "").strip().lower() != "ok":
        raise RuntimeError(str(status or "Local Gemma failed."))
    return _clean_text(text)


def _lm_studio(payload, prompt, images=None):
    base = str(payload.get("lmstudio_base_url") or "http://127.0.0.1:1234/v1").rstrip("/")
    model = str(payload.get("lmstudio_model") or "").strip()
    if not model:
        raise ValueError("Choose an LM Studio model in Advanced Settings.")
    content = prompt
    if images:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            image = image.convert("RGB")
            if image.height > 768:
                width = max(1, round(image.width * 768 / image.height))
                resampling = getattr(Image, "Resampling", Image)
                image = image.resize((width, 768), resampling.LANCZOS)
            buf = io.BytesIO()
            image.save(buf, "JPEG", quality=90)
            data = base64.b64encode(buf.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": float(payload.get("temperature", 0.55)),
        "max_tokens": int(payload.get("max_tokens", 1400)),
        "stream": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    key = str(payload.get("lmstudio_api_key") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach LM Studio at {base}.") from exc
    choices = data.get("choices") or []
    text = choices[0].get("message", {}).get("content", "") if choices else ""
    if not str(text).strip():
        raise RuntimeError("LM Studio returned no text.")
    return _clean_text(text)


def _api_llm(payload, prompt, images=None):
    from .LLM import VRGDG_LLM_Multi
    key = str(payload.get("api_key") or "").strip()
    provider = str(payload.get("provider") or "openai").strip()
    model = str(payload.get("model") or "").strip()
    custom = str(payload.get("custom_model") or "").strip()
    if not key:
        raise ValueError("Enter the LLM API key in Advanced Settings.")
    kwargs = {"api_key": key, "provider": provider, "model": model, "prompt": prompt, "custom_model": custom}
    runner = VRGDG_LLM_Multi()
    for index, image in enumerate(images or [], 1):
        kwargs[f"image{index}"] = runner._pil_to_tensor(image.convert("RGB"))
    text, _provider, _model, status, _image = runner.generate_text(**kwargs)
    if str(status or "").lower().startswith("error"):
        raise RuntimeError(str(status))
    if not str(text or "").strip():
        raise RuntimeError("The LLM API returned no text.")
    return _clean_text(text)


def _run_llm(payload, prompt, images=None):
    if _runner(payload) == "gemma_local":
        return _gemma_local(payload, prompt, images)
    if _runner(payload) == "llm_api":
        return _api_llm(payload, prompt, images)
    return _lm_studio(payload, prompt, images)


def _llm_choices(_payload=None):
    from .LLM import VRGDG_LLM_Multi, VRGDG_SuperGemmaGGUFChat
    return {
        "gemma_models": VRGDG_SuperGemmaGGUFChat._list_local_gemma_gguf_choices(),
        "gemma_mmproj": VRGDG_SuperGemmaGGUFChat._list_local_mmproj_choices(),
        "provider_models": VRGDG_LLM_Multi.PROVIDER_MODELS,
        "default_models": VRGDG_LLM_Multi.DEFAULT_MODEL,
    }


def _lm_studio_models(payload):
    base = str(payload.get("lmstudio_base_url") or "http://127.0.0.1:1234/v1").rstrip("/")
    headers = {"Accept": "application/json"}
    key = str(payload.get("lmstudio_api_key") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(f"{base}/models", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
    except Exception as exc:
        raise RuntimeError(f"Could not list LM Studio models at {base}.") from exc
    return {"models": [str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict) and item.get("id")]}


def _json_object(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("The LLM did not return the requested JSON object.")
    return json.loads(match.group(0))


def _safe_dataset_folder(path):
    raw_path = str(path or "").strip()
    if not raw_path:
        raise ValueError("Choose a dataset folder.")
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(raw_path)))
    os.makedirs(path, exist_ok=True)
    return path


def _project_folders(path):
    root = _safe_dataset_folder(path)
    dataset = os.path.join(root, "dataset")
    project_files = os.path.join(root, "project_files")
    os.makedirs(dataset, exist_ok=True)
    os.makedirs(project_files, exist_ok=True)
    return root, dataset, project_files


def _identity(payload):
    style = str(payload.get("art_style") or "").strip()
    if not style:
        raise ValueError("Describe the art style first.")
    dataset_type = str(payload.get("dataset_type") or "style")
    if dataset_type == "character":
        request = "Create a LoRA character identity. The phrase should concisely describe the character's stable identifying visible traits, without fixing pose, action, camera, lighting, or background."
    else:
        request = "Create a LoRA style identity. The phrase should concisely describe the medium, construction, textures, shapes, palette, and lighting."
    prompt = f"""{request}\nUser description:\n{style}\n\nReturn JSON only with trigger_word and trigger_phrase. trigger_word must be one invented memorable ASCII word, 7-20 characters, letters and numbers only, not a normal dictionary word. trigger_phrase must be a reusable comma-separated visual description of 15-40 words. Do not include the trigger word in the phrase."""
    data = _json_object(_run_llm(payload, prompt))
    trigger = re.sub(r"[^A-Za-z0-9]", "", str(data.get("trigger_word") or ""))[:20]
    phrase = " ".join(str(data.get("trigger_phrase") or "").split())
    if len(trigger) < 5 or not phrase:
        raise ValueError("The LLM returned an invalid style identity. Try again.")
    return {"trigger_word": trigger, "trigger_phrase": phrase}


def _concepts(payload):
    style = str(payload.get("art_style") or "").strip()
    count = max(1, min(200, int(payload.get("count") or 20)))
    dataset_type = str(payload.get("dataset_type") or "style")
    if dataset_type == "character":
        goal = "Create varied scenes for the same character. Vary pose, expression, action, camera angle, shot distance, clothing when appropriate, environment, lighting, and composition while keeping the character identity consistent. Do not redescribe or replace the character in each line; write the scene assignment."
    elif dataset_type == "ic_pair":
        goal = "Create varied source-image scenes on which the requested edit can be learned. Vary subjects, camera angles, environments, lighting, and composition. Each source must make the edit visually testable."
    else:
        goal = "Use diverse subjects, environments, compositions, lighting, colors, scales, and actions so the style is learned rather than one subject."
    prompt = f"""Create exactly {count} image concepts for a {dataset_type} LoRA dataset. User description: {style}\n{goal} Each concept must be visually clear, self-contained, and one short line. Return only the concepts, one per line, with no numbering, bullets, headings, or commentary."""
    text = _run_llm(payload, prompt)
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line][:count]
    if not lines:
        raise ValueError("The LLM returned no concepts.")
    return {"concepts": lines}


def _image_prompt(payload):
    style = str(payload.get("art_style") or "").strip()
    concept = str(payload.get("concept") or "").strip()
    phrase = str(payload.get("trigger_phrase") or "").strip()
    dataset_type = str(payload.get("dataset_type") or "style")
    if dataset_type == "character":
        prompt = f"""Write one detailed image-generation prompt for a consistent-character LoRA dataset. Character description: {style}\nStable character identity phrase: {phrase}\nScene assignment: {concept}\nThe generator may receive a character reference image or turnaround. Explicitly instruct it to preserve the exact same character identity, facial structure, distinguishing traits, hair, proportions, and recurring design while placing that character naturally in the requested new scene. Vary only what the scene requires. Describe pose, expression, action, framing, environment, lighting, and composition. Output one plain-text paragraph only."""
    else:
        prompt = f"{T2I_INSTRUCTIONS}\n\nArt style: {style}\nConcept: {concept}\nThe output must begin naturally with this exact reusable style phrase: {phrase}"
    text = " ".join(_run_llm(payload, prompt).split())
    return {"prompt": text}


def _caption(payload):
    info = payload.get("image") or {}
    path = _resolve_comfy_image_path(info)
    with Image.open(path) as image:
        image.load()
        prompt = f"{CAPTION_INSTRUCTIONS}\n\nTrigger word: {payload.get('trigger_word', '')}\nStyle phrase: {payload.get('trigger_phrase', '')}"
        text = " ".join(_run_llm(payload, prompt, [image.copy()]).split())
    trigger = str(payload.get("trigger_word") or "").strip()
    phrase = str(payload.get("trigger_phrase") or "").strip()
    required_prefix = f"{trigger}, {phrase},"
    if not text.lower().startswith(required_prefix.lower()):
        remainder = text
        if remainder.lower().startswith((trigger + ",").lower()):
            remainder = remainder[len(trigger) + 1:].strip()
        if phrase and remainder.lower().startswith((phrase + ",").lower()):
            remainder = remainder[len(phrase) + 1:].strip()
        text = f"{required_prefix} {remainder}".strip()
    return {"caption": text}


def _save_pair(payload):
    project_root, folder, project_files = _project_folders(payload.get("dataset_folder"))
    index = max(1, int(payload.get("index") or 1))
    stem = f"image_{index:03d}"
    source = _resolve_comfy_image_path(payload.get("image") or {})
    image_path = os.path.join(folder, stem + ".png")
    caption_path = os.path.join(folder, stem + ".txt")
    with Image.open(source) as image:
        image.save(image_path, "PNG")
    with open(caption_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(str(payload.get("caption") or "").strip() + "\n")
    manifest_path = os.path.join(project_files, "dataset.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    manifest.update({
        "art_style": payload.get("art_style", ""),
        "trigger_word": payload.get("trigger_word", ""),
        "trigger_phrase": payload.get("trigger_phrase", ""),
        "generator": payload.get("generator", "zimage"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    items = manifest.setdefault("items", [])
    record = {"index": index, "concept": payload.get("concept", ""), "prompt": payload.get("prompt", ""), "caption": payload.get("caption", ""), "image": f"../dataset/{stem}.png", "text": f"../dataset/{stem}.txt", "seed": payload.get("seed")}
    items[:] = [item for item in items if int(item.get("index", -1)) != index]
    items.append(record)
    items.sort(key=lambda item: int(item.get("index", 0)))
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return {"project_root": project_root, "dataset_folder": folder, "project_files_folder": project_files, "image_path": image_path, "caption_path": caption_path, "manifest_path": manifest_path}


def _save_ic_pair(payload):
    project_root, dataset_folder, project_files = _project_folders(payload.get("dataset_folder"))
    reference_dir = os.path.join(dataset_folder, "references")
    target_dir = os.path.join(dataset_folder, "targets")
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    index = max(1, int(payload.get("index") or 1))
    stem = f"pair_{index:03d}"
    reference_path = os.path.join(reference_dir, stem + ".png")
    target_path = os.path.join(target_dir, stem + ".png")
    instruction_path = os.path.join(target_dir, stem + ".txt")
    for info, path in ((payload.get("reference") or {}, reference_path), (payload.get("target") or {}, target_path)):
        with Image.open(_resolve_comfy_image_path(info)) as image:
            image.save(path, "PNG")
    instruction = " ".join(str(payload.get("instruction") or "").split())
    with open(instruction_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(instruction + "\n")
    metadata_path = os.path.join(project_files, "dataset.json")
    records = []
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
        except Exception:
            records = []
    if not isinstance(records, list):
        records = []
    record = {
        "caption": instruction,
        "video": f"../dataset/targets/{stem}.png",
        "reference_video": f"../dataset/references/{stem}.png",
        "experimental_one_frame_ic_lora": True,
    }
    records = [item for item in records if item.get("video") != record["video"]]
    records.append(record)
    with open(metadata_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    return {"project_root": project_root, "dataset_folder": dataset_folder, "project_files_folder": project_files, "reference_path": reference_path, "target_path": target_path, "instruction_path": instruction_path, "metadata_path": metadata_path}


def _pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        try:
            return str(filedialog.askdirectory(title="Choose LoRA Dataset Project Folder", mustexist=False) or "").strip()
        finally:
            root.destroy()
    except Exception:
        script = "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.OpenFileDialog; $d.Title='Choose LoRA Dataset Project Folder'; $d.CheckFileExists=$false; $d.ValidateNames=$false; $d.FileName='Select this folder'; if($d.ShowDialog() -eq 'OK'){[Console]::Write((Split-Path -Parent $d.FileName))}"
        result = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script], capture_output=True, text=True, timeout=180)
        return str(result.stdout or "").strip()


def _route(handler):
    async def wrapped(request):
        from aiohttp import web
        try:
            payload = await request.json() if request.can_read_body else {}
            return web.json_response({"ok": True, **handler(payload)})
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
    return wrapped


_ROUTES_READY = False
def _ensure_routes():
    global _ROUTES_READY
    if _ROUTES_READY or PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return
    routes = PromptServer.instance.routes
    routes.post("/vrgdg/lora_dataset/identity")(_route(_identity))
    routes.post("/vrgdg/lora_dataset/concepts")(_route(_concepts))
    routes.post("/vrgdg/lora_dataset/image_prompt")(_route(_image_prompt))
    routes.post("/vrgdg/lora_dataset/caption")(_route(_caption))
    routes.post("/vrgdg/lora_dataset/save_pair")(_route(_save_pair))
    routes.post("/vrgdg/lora_dataset/save_ic_pair")(_route(_save_ic_pair))
    routes.post("/vrgdg/lora_dataset/pick_folder")(_route(lambda _p: {"path": _pick_folder()}))
    routes.post("/vrgdg/lora_dataset/open_folder")(_route(lambda p: (_open_folder(p))))
    routes.post("/vrgdg/lora_dataset/image_source")(_route(_image_source))
    routes.post("/vrgdg/lora_dataset/llm_choices")(_route(_llm_choices))
    routes.post("/vrgdg/lora_dataset/lm_studio_models")(_route(_lm_studio_models))
    _ROUTES_READY = True


def _open_folder(payload):
    path = _safe_dataset_folder(payload.get("path"))
    os.startfile(path)
    return {"path": path}


def _image_source(payload):
    return {"path": _resolve_comfy_image_path(payload.get("image") or {})}


class VRGDG_LoraDatasetCreatorUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Opens the standalone VRGDG LoRA Dataset Creator."

    def noop(self):
        return ("Open the LoRA Dataset Creator UI to build image/caption pairs.",)


_ensure_routes()
NODE_CLASS_MAPPINGS = {"VRGDG_LoraDatasetCreatorUI": VRGDG_LoraDatasetCreatorUI}
NODE_DISPLAY_NAME_MAPPINGS = {"VRGDG_LoraDatasetCreatorUI": "VRGDG LoRA Dataset Creator UI"}
