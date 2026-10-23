"""Vision-assisted MiniMax enhancement, separate from the existing text-only task."""

import json
import math
import re

from PIL import Image, ImageDraw, ImageOps

from ..builder.media import _image_from_prompt_payload
from ..minimax.prompt_enhance import merge_reference_enhanced_prompt
from .builder_instructions import _effective_builder_instruction
from .builder_runner import (
    _EXTERNAL_LLM_RUNNERS, _builder_local_llm, _builder_local_mmproj_file,
    _builder_local_model_file, _llm_runner_from_payload, _resolve_mmproj_dropdown_path,
    _runner_output_token_limit, _try_run_remote_vision,
)
from .output_checks import _extract_json_object_from_text
from .prompts.minimax import (
    MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT,
    MINIMAX_H3_PROMPT_ENHANCE_WITH_REFERENCES_INSTRUCTIONS,
)


def prepare_reference_images(references):
    """Keep every reference visible, including API runners that accept at most four images."""
    images = [
        _image_from_prompt_payload(item.get("path", ""), item.get("data", ""),
                                   f"Enhance reference {item['picture_label']}").convert("RGB")
        for item in references
    ]
    if len(images) <= 4:
        return images, "\n".join(
            f"Attached image {index + 1} = {item['picture_label']} ({item['label']})."
            for index, item in enumerate(references)
        )
    # No silent [:4] truncation: tile all references, with their immutable labels, into <=4 sheets.
    sheets = []
    group_size = math.ceil(len(images) / 4)
    for start in range(0, len(images), group_size):
        group = images[start:start + group_size]
        sheet = Image.new("RGB", (1536, 808 * math.ceil(len(group) / 2)), "#202020")
        draw = ImageDraw.Draw(sheet)
        for offset, image in enumerate(group):
            item = references[start + offset]
            x, y = (offset % 2) * 768, (offset // 2) * 808
            draw.text((x + 12, y + 10), f"{item['picture_label']} / {item['label']}", fill="white")
            tile = ImageOps.contain(image, (768, 768))
            sheet.paste(tile, (x + (768 - tile.width) // 2, y + 40 + (768 - tile.height) // 2))
        sheets.append(sheet)
    return sheets, "Attached images are labeled contact sheets. Each tile heading gives its exact picture/subject assignment. Inspect every tile separately; do not treat tiles as one scene."


def run_reference_vision(payload, instruction, images):
    temperature = float(payload.get("temperature") or 0.35)
    top_p = float(payload.get("top_p") or 0.9)
    tokens = _runner_output_token_limit(payload, int(payload.get("max_new_tokens") or 4500))
    if _llm_runner_from_payload(payload) in _EXTERNAL_LLM_RUNNERS:
        return _try_run_remote_vision(
            {**payload, "max_new_tokens": tokens}, instruction, images,
            temperature=temperature, top_p=top_p, max_new_tokens=tokens,
        )
    llm = _builder_local_llm(payload)
    model_file = _builder_local_model_file(payload, str(payload.get("model_file") or ""))
    if not model_file.lower().endswith(".gguf"):
        raise ValueError("Choose a vision-capable GGUF model and its mmproj for enhancement with reference images.")
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = _resolve_mmproj_dropdown_path(llm, _builder_local_mmproj_file(payload, str(payload.get("mmproj_file") or "")))
    settings = dict(model_path=model_path, mmproj_path=mmproj_path,
                    n_ctx=int(payload.get("n_ctx") or 16000), n_gpu_layers=int(payload.get("n_gpu_layers") or 99),
                    n_threads=int(payload.get("n_threads") or 8), chat_format=str(payload.get("chat_format") or ""))
    unload = bool(payload.get("unload_after", True))
    try:
        model = llm._load_gguf_model(**settings)
        text = llm._run_gguf_vision_pipeline(
            model=model, pil_images=images, instruction_text=instruction,
            temperature=temperature, top_p=top_p, max_new_tokens=tokens,
            seed=int(payload["seed"]) if payload.get("seed") is not None else None,
        )
        return text, {"runner": "builtin", "used_model": model_path, "unloaded": unload}
    finally:
        if unload:
            from .cache import _clear_vrgdg_llm_caches
            llm._unload_gguf_model(**settings)
            _clear_vrgdg_llm_caches(clear_cuda_cache=True, clear_hf_pipeline_cache=False)


def enhance_with_references(payload):
    current = str(payload.get("current_prompt") or "").strip()
    if not current:
        raise ValueError("This scene has no MiniMax H3 prompt to enhance. Create the prompt first.")
    references = payload.get("image_references")
    if not isinstance(references, list) or not 1 <= len(references) <= 9:
        raise ValueError("Enhancement with images needs 1–9 assigned character/location reference images.")
    labels, pictures = set(), set()
    for item in references:
        if not isinstance(item, dict):
            raise ValueError("Invalid enhancement image reference.")
        label, picture = item.get("label", ""), item.get("picture_label", "")
        if (not re.fullmatch(r"<(?:Subject|Picture) [1-9]\d*>", label)
                or not re.fullmatch(r"<Picture [1-9]\d*>", picture)
                or label in labels or picture in pictures):
            raise ValueError("Enhancement reference labels must be unique and match the assigned subject/picture order.")
        if not (item.get("path") or item.get("data")) or str(item.get("path", "")).startswith("refmod://"):
            raise ValueError(f"{label} has no inspectable reference image. Assign an image or enhance without images.")
        labels.add(label)
        pictures.add(picture)
    if not any(item.get("describe") for item in references):
        raise ValueError("No mapped character/location references are available to describe. Use reference-to-video or enhance without images.")
    instruction = _effective_builder_instruction(
        payload, "minimax_h3_prompt_enhance_with_references",
        MINIMAX_H3_PROMPT_ENHANCE_WITH_REFERENCES_INSTRUCTIONS,
    )
    if MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT not in instruction:
        instruction += "\n\n" + MINIMAX_H3_ENHANCE_CONTINUITY_CONTRACT
    images, image_map = prepare_reference_images(references)
    mapping = [{key: item.get(key) for key in ("label", "picture_label", "name", "kind", "purpose", "describe")}
               for item in references]
    text, info = run_reference_vision(
        payload, f"{instruction}\n\nIMAGE ASSIGNMENTS:\n{image_map}\n{json.dumps(mapping, ensure_ascii=False)}"
                 f"\n\nCURRENT PROMPT:\n{current}", images,
    )
    prompt = merge_reference_enhanced_prompt(current, _extract_json_object_from_text(text), references)
    return {"prompt": prompt, "characters": len(prompt), "used_image_reference": True,
            "reference_image_count": len(references), "runner": info.get("runner", "builtin"),
            "used_model": info.get("used_model", ""), "unloaded": info.get("unloaded", False)}
