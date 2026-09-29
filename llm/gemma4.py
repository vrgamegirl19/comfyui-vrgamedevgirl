"""Gemma 4 helper generation for the Video Builder and Prompt Creator: style, story, subjects, lyrics,
concept-to-image/video and detail expansion, run on the selected LLM runner or the built-in SuperGemma GGUF."""

import os
import re

from .builder_runner import _llm_runner_from_payload, _run_builder_text_llm
from .prompts.gemma4 import (
    _VRGDG_GEMMA4_STYLE_INSTRUCTIONS,
    _VRGDG_GEMMA4_STORY_INSTRUCTIONS,
    _VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS,
    _VRGDG_GEMMA4_LYRICS_INSTRUCTIONS,
    _VRGDG_GEMMA4_T2I_FROM_CONCEPT_INSTRUCTIONS,
    _VRGDG_GEMMA4_T2V_FROM_CONCEPT_INSTRUCTIONS,
    _VRGDG_GEMMA4_ADVANCED_PROMPT_DETAIL_INSTRUCTIONS,
    _VRGDG_GEMMA4_LOCATION_DETAIL_INSTRUCTIONS,
)


def _clean_gemma4_text(value):
    text = str(value or "").strip()
    text = re.sub(r"^\s*```(?:text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _prompt_creator_custom_instruction(payload, key, default_text):
    project_folder = os.path.abspath(str(payload.get("project_folder", "") or "").strip().strip('"'))
    if not project_folder:
        return str(default_text or "")
    safe_key = re.sub(r"[^a-z0-9_]+", "_", str(key or "").strip().lower()).strip("_")
    if not safe_key:
        return str(default_text or "")
    path = os.path.join(project_folder, "project_context", "custom_llm_instructions", f"{safe_key}.txt")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                text = handle.read().strip()
            if text:
                return text
        except Exception:
            pass
    return str(default_text or "")


def _build_gemma4_prompt(target, payload):
    target = str(target or "").strip()
    notes = str(payload.get("notes", "") or "").strip()
    lyrics = str(payload.get("lyrics", "") or "").strip()
    style_theme = str(payload.get("style_theme", "") or "").strip()
    story_idea = str(payload.get("story_idea", "") or "").strip()

    if target == "builder_style_theme":
        idea = notes or lyrics or style_theme or story_idea
        instructions = _prompt_creator_custom_instruction(payload, "style_theme", _VRGDG_GEMMA4_STYLE_INSTRUCTIONS)
        prompt = (
            f"{instructions}\n\n"
            "Create the style/theme block from normal user ideas instead of lyrics.\n"
            "Use the user's rough idea as the full creative direction.\n"
            "If the idea is short, infer a useful cinematic visual style without adding extra sections.\n\n"
            f"User idea:\n{idea}"
        )
        return prompt

    if target == "builder_story_idea":
        idea = notes or story_idea or lyrics
        instructions = _prompt_creator_custom_instruction(payload, "story_idea", _VRGDG_GEMMA4_STORY_INSTRUCTIONS)
        prompt = (
            f"{instructions}\n\n"
            "Create the story idea from normal user ideas instead of lyrics.\n"
            "Treat the user idea as the full creative foundation.\n"
            "Do not ask for lyrics. Output only the story concept.\n\n"
            f"User idea:\n{idea}"
        )
        if style_theme:
            prompt += f"\n\nStyle/theme:\n{style_theme}"
        return prompt

    if target == "builder_subjects_and_scenes":
        idea = notes or story_idea or lyrics
        instructions = _prompt_creator_custom_instruction(payload, "subject_locations", _VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS)
        prompt = (
            f"{instructions}\n\n"
            "Create the subject and location list from normal user ideas instead of lyrics.\n"
            "Use the user idea as the highest priority creative direction.\n"
        )
        if style_theme:
            prompt += f"\n\nStyle/theme:\n{style_theme}"
        prompt += f"\n\nStory or user idea:\n{idea}"
        return prompt

    if target == "style_theme":
        instructions = _prompt_creator_custom_instruction(payload, "style_theme", _VRGDG_GEMMA4_STYLE_INSTRUCTIONS)
        prompt = f"{instructions}\n\nfull lyrics:\n{lyrics}"
        if notes:
            prompt += f"\n\nother notes:\n{notes}"
        return prompt

    if target == "story_idea":
        instructions = _prompt_creator_custom_instruction(payload, "story_idea", _VRGDG_GEMMA4_STORY_INSTRUCTIONS)
        prompt = f"{instructions}\n\nLyrics:\n{lyrics}"
        if style_theme:
            prompt += f"\n\nStyle/theme:\n{style_theme}"
        if notes:
            prompt += f"\n\nOptional notes:\n{notes}"
        return prompt

    if target == "subjects_and_scenes":
        instructions = _prompt_creator_custom_instruction(payload, "subject_locations", _VRGDG_GEMMA4_SUBJECTS_INSTRUCTIONS)
        prompt = f"{instructions}"
        if notes:
            prompt += f"\n\nUser notes - highest priority:\n{notes}"
        prompt += f"\n\nStory idea:\n{story_idea}"
        return prompt

    if target == "song_lyrics":
        duration = str(payload.get("duration", "") or "").strip()
        instructions = _prompt_creator_custom_instruction(payload, "full_lyrics", _VRGDG_GEMMA4_LYRICS_INSTRUCTIONS)
        prompt = f"{instructions}"
        if duration:
            prompt += f"\n\nRequested duration seconds:\n{duration}"
        prompt += f"\n\nSong idea and notes:\n{notes}"
        if not notes:
            prompt += "\nCreate an original short song about refusing to give up."
        return prompt

    if target == "text_to_image_from_concept":
        concept_prompt = str(payload.get("concept_prompt", "") or "").strip()
        if not concept_prompt:
            raise ValueError("No concept prompt was provided for text-to-image generation.")
        extra_user_input = str(payload.get("extra_user_input", "") or "").strip()
        prompt = (
            f"{_VRGDG_GEMMA4_T2I_FROM_CONCEPT_INSTRUCTIONS}\n\n"
            f"Story idea:\n{story_idea}\n\n"
            f"Style/theme:\n{style_theme}\n\n"
            f"Current visual prompt:\n{concept_prompt}"
        )
        if extra_user_input:
            prompt += f"\n\nExtra user input:\n{extra_user_input}"
        return prompt

    if target == "text_to_video_from_concept":
        concept_prompt = str(payload.get("concept_prompt", "") or "").strip()
        if not concept_prompt:
            raise ValueError("No concept prompt was provided for text-to-video generation.")
        subjects_and_scenes = str(payload.get("subjects_and_scenes", "") or "").strip()
        extra_user_input = str(payload.get("extra_user_input", "") or "").strip()
        prompt = (
            f"{_VRGDG_GEMMA4_T2V_FROM_CONCEPT_INSTRUCTIONS}\n\n"
            f"Subject and location list:\n{subjects_and_scenes}\n\n"
            f"Style/theme:\n{style_theme}\n\n"
            f"Concept prompt:\n{concept_prompt}"
        )
        if extra_user_input:
            prompt += f"\n\nUser input:\n{extra_user_input}"
        return prompt

    if target == "advanced_prompt_detail":
        label = str(payload.get("label", "") or "").strip() or "Custom"
        prompts = payload.get("prompts") or []
        if not isinstance(prompts, list):
            prompts = []
        prompt_lines = []
        for index, item in enumerate(prompts, start=1):
            text = str(item or "").strip()
            if text:
                prompt_lines.append(f"{index}. {text}")
        if not prompt_lines:
            raise ValueError("No scene prompts were provided for Gemma4 advanced list generation.")
        prompt = (
            f"{_VRGDG_GEMMA4_ADVANCED_PROMPT_DETAIL_INSTRUCTIONS}\n\n"
            f"Detail label:\n{label}\n\n"
        )
        if notes:
            prompt += f"Optional user guidance for all lists:\n{notes}\n\n"
        prompt += f"Scene prompts:\n" + "\n".join(prompt_lines)
        return prompt

    if target == "location_description_detail":
        location_name = str(payload.get("location_name", "") or "").strip()
        location_description = str(payload.get("location_description", "") or notes).strip()
        if not location_name or not location_description:
            raise ValueError("A location label and short description are required.")
        instructions = _prompt_creator_custom_instruction(
            payload,
            "location_description_detail",
            _VRGDG_GEMMA4_LOCATION_DETAIL_INSTRUCTIONS,
        )
        return (
            f"{instructions}\n\n"
            f"Location label:\n{location_name}\n\n"
            f"Short location description:\n{location_description}"
        )

    raise ValueError(f"Unsupported Gemma4 target: {target}")


def _run_gemma4_prompt(payload):
    from .gguf import VRGDG_SuperGemmaGGUFChat

    target = str(payload.get("target", "") or "").strip()
    model_file = str(payload.get("model_file", "") or "").strip()
    runner = _llm_runner_from_payload(payload)
    if not model_file and runner == "builtin":
        raise ValueError("No Gemma4 model_file was selected.")

    prompt = _build_gemma4_prompt(target, payload)
    if not prompt.strip():
        raise ValueError("Gemma4 prompt was empty.")

    # Match the known-good SuperGemma settings used in the workflow node.
    n_ctx = int(payload.get("n_ctx") or 13000)
    n_gpu_layers = int(payload.get("n_gpu_layers") or 99)
    n_threads = int(payload.get("n_threads") or 8)
    chat_format = str(payload.get("chat_format", "") or "").strip()
    temperature = float(payload.get("temperature") or 0.75)
    top_p = float(payload.get("top_p") or 0.95)
    max_new_tokens = int(payload.get("max_new_tokens") or 32000)
    unload_after = bool(payload.get("unload_after"))

    if runner != "builtin":
        text, run_info = _run_builder_text_llm(
            payload,
            prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            label="Gemma4",
            preserve_paragraphs=target in {
                "builder_style_theme",
                "style_theme",
                "builder_story_idea",
                "story_idea",
                "builder_subjects_and_scenes",
                "subjects_and_scenes",
                "song_lyrics",
            },
        )
        text = _clean_gemma4_text(text)
        if not text:
            raise ValueError("Gemma4 returned an empty response.")
        return {
            "text": text,
            "used_model": run_info.get("used_model", ""),
            "runner": run_info.get("runner", runner),
            "unloaded": False,
        }

    llm = VRGDG_SuperGemmaGGUFChat()
    model_path = llm._resolve_dropdown_path(model_file, llm.MISSING_MODEL_OPTION)
    mmproj_path = ""
    model = None
    try:
        model = llm._load_gguf_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            chat_format=chat_format,
            mmproj_path=mmproj_path,
        )
        text = llm._run_gguf_text_pipeline(
            model=model,
            instruction_text=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        text = _clean_gemma4_text(text)
        if not text:
            raise ValueError("Gemma4 returned an empty response.")
        return {
            "text": text,
            "used_model": model_path,
            "unloaded": unload_after,
        }
    finally:
        if unload_after and model_path:
            llm._unload_gguf_model(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                chat_format=chat_format,
                mmproj_path=mmproj_path,
            )
