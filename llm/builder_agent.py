"""The Video Builder agent reply."""

import json
import re

from .output_checks import _extract_json_object_from_text
from .builder_runner import _clean_lm_studio_plain_text, _run_builder_text_llm


def _generate_builder_agent_reply(payload):
    context = payload.get("context") or {}
    if not isinstance(context, dict):
        context = {}
    auto_apply = bool(payload.get("auto_apply"))
    agent_purpose = str(payload.get("agent_purpose") or "scene_work").strip().lower()
    if agent_purpose not in {"walkthrough", "scene_work", "story_builder", "troubleshoot"}:
        agent_purpose = "scene_work"
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    cleaned_messages = []
    for item in messages[-10:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            cleaned_messages.append({"role": role, "content": content[:3000]})
    latest_user = str(payload.get("message") or "").strip()
    if latest_user:
        cleaned_messages.append({"role": "user", "content": latest_user[:3000]})
    if not cleaned_messages:
        raise ValueError("Type a message for the Builder Agent first.")

    context_text = json.dumps(context, ensure_ascii=False, indent=2)[:9000]
    conversation_text = "\n\n".join(
        f"{item['role'].title()}:\n{item['content']}"
        for item in cleaned_messages
    )
    instruction = (
        "You are the VRGDG Music Video Builder Agent, a local assistant inside a scene-by-scene music video workflow.\n"
        "Help the user think through lyrics, scene concepts, image prompts, video prompts, shot choices, continuity, and troubleshooting.\n"
        "Be practical, concise, and specific to the provided active scene context.\n"
        "Keep the user-facing reply short: usually 1-5 sentences. Do not write strategy essays, option menus, or long explanations unless the user asks.\n"
        "When suggesting prompt text, provide copy-ready text clearly, but keep normal chat friendly.\n"
        "If the context is missing something, say what is missing and make a reasonable suggestion from what is available.\n\n"
        f"Agent purpose: {agent_purpose}.\n"
        "Read project_status and scene_directory first. If project_status.scene_count is greater than 0, scenes already exist; do not say no scenes have been created and do not ask to create initial scenes from timeline markers unless the user explicitly asks for that.\n"
        "If purpose is walkthrough, act like intake plus step-by-step onboarding. First identify what the user is making: music video, short film, ad/social clip, visualizer, or something else. Use project_status before choosing the next step. If project_status.has_audio is false, ask whether they plan to use custom audio and point them to Choose Audio if yes; do not ask for lyrics/SRT before resolving the audio plan. If audio exists, ask how they want scenes created: import Prompt Creator outputs, load SRT/lyrics, manually create scenes, or plan scenes with the agent. Only ask about lyrics when the user is making a music video or lyric-driven project. For short films/ads, ask for script, scene beats, product/message, or visual outline instead. Tell them one small UI step at a time and why. Do not generate prompts or run models unless the user explicitly asks. Prefer one next step over a full tutorial.\n"
        "If purpose is troubleshoot, focus on diagnosing the user's stated problem from project_status, active scene context, and available get actions.\n"
        "If purpose is scene_work, help create/update scene notes, prompts, images, and videos using the supported actions.\n\n"
        "If purpose is story_builder, act like a scene planner for complex multi-character projects. This workflow may not use Prompt Creator, SRT, or existing lyric segments. First check project_status.has_audio and story_source.has_source. For music videos/songs, if no audio exists, ask for global/timeline audio before asking for lyrics. If no story source exists, ask the user to paste lyrics/script/story source and use save_story_source in Auto Apply mode when they provide it. Use get_story_source when you need the saved lyrics/script instead of keeping it in chat memory. Help define characters, assign which character(s) appear in each scene, match the lyrics/story beat, and create compact per-scene planning notes. Prefer structured scene plans over long prose. When Auto Apply is enabled, use create_story_scenes_from_source if the user asks to create starter scenes from saved lyrics/script, then use set_scene_plan for requested scenes. Do not generate final image/video prompts unless the user explicitly asks; write planning notes that the existing prompt generators can use.\n\n"
        "Return only valid JSON with this shape:\n"
        "{\n"
        "  \"reply\": \"short message to show the user\",\n"
        "  \"actions\": []\n"
        "}\n\n"
        "Supported actions are:\n"
        "- {\"type\":\"select_scene\",\"scene_id\":\"...\"} or {\"type\":\"select_scene\",\"scene_number\":2}\n"
        "- {\"type\":\"get_scene_lyrics\",\"scene_id\":\"...\"} or {\"type\":\"get_scene_lyrics\",\"scene_number\":2}\n"
        "- {\"type\":\"get_scene_context\",\"scene_id\":\"...\"} or {\"type\":\"get_scene_context\",\"scene_number\":2}\n"
        "- {\"type\":\"get_story_source\"}\n"
        "- {\"type\":\"get_context_prompts\",\"context_type\":\"all|theme_style|story_idea|subject_scene\"}\n"
        "- {\"type\":\"get_selected_timeline_range\"}\n"
        "- {\"type\":\"save_story_source\",\"text\":\"lyrics/script/story source text\"}\n"
        "- {\"type\":\"set_context_prompt\",\"context_type\":\"theme_style|story_idea|subject_scene\",\"text\":\"...\",\"mode\":\"replace|append\"}\n"
        "- {\"type\":\"create_story_scenes_from_source\",\"scene_count\":12}\n"
        "- {\"type\":\"create_concept_prompts\",\"source_mode\":\"all|director|scene|timeline|director_scene\",\"scope\":\"all|selected|range\",\"batch_size\":5,\"use_story\":true,\"use_theme\":true}\n"
        "- {\"type\":\"create_motion_notes\",\"source_mode\":\"concept|concept_director|concept_timeline|all\",\"scope\":\"all|selected|range\",\"batch_size\":5,\"use_story\":true,\"use_theme\":true}\n"
        "- {\"type\":\"set_active_scene_to_selected_range\",\"scene_id\":\"optional target scene\"}\n"
        "- {\"type\":\"create_scene_from_selected_range\",\"label\":\"...\",\"director_note\":\"...\",\"scene_notes\":\"...\",\"flux_notes\":\"...\",\"nb_notes\":\"...\",\"video_notes\":\"...\"}\n"
        "- {\"type\":\"split_selected_range_into_scenes\",\"scene_count\":3,\"label_prefix\":\"Scene\",\"director_notes\":[\"...\"],\"scene_notes\":[\"...\"]}\n"
        "- {\"type\":\"split_scene_into_subscenes\",\"scene_id\":\"...\",\"scene_number\":4,\"scene_count\":3,\"label_prefix\":\"Scene 4\"}\n"
        "- {\"type\":\"merge_scenes\",\"scene_numbers\":[2,3,4],\"label\":\"Scene 2\"}\n"
        "- {\"type\":\"renumber_scene_labels\"}\n"
        "- {\"type\":\"normalize_dual_vocal_director_notes\",\"replacement\":\"male and female\"}\n"
        "- {\"type\":\"replace_director_note_text\",\"find\":\"male singer and female singer\",\"replace\":\"female and male\"}\n"
        "- {\"type\":\"assign_selected_range_note\",\"label\":\"Female vocal\",\"marker_type\":\"female vocal|male vocal|chorus|verse|beat|note\",\"note\":\"...\"}\n"
        "- {\"type\":\"sync_existing_scenes_to_timeline_markers\",\"create_missing\":true}\n"
        "- {\"type\":\"set_scene_notes\",\"scene_number\":2,\"text\":\"...\"}\n"
        "- {\"type\":\"set_flux_notes\",\"scene_number\":2,\"text\":\"...\"}\n"
        "- {\"type\":\"set_nb_notes\",\"scene_number\":2,\"text\":\"...\"}\n"
        "- {\"type\":\"set_video_notes\",\"scene_number\":2,\"text\":\"...\"}\n"
        "- {\"type\":\"set_scene_plan\",\"scene_number\":2,\"director_note\":\"...\",\"scene_notes\":\"...\",\"flux_notes\":\"...\",\"nb_notes\":\"...\",\"video_notes\":\"...\"}\n"
        "- {\"type\":\"set_image_model_mode\",\"image_mode\":\"zimage|flux_klein|nano_banana|ernie_image\"}\n"
        "- {\"type\":\"set_video_model_mode\",\"video_mode\":\"i2v|id_lora|t2v|rtv|ingredients\"}\n"
        "- {\"type\":\"request_reference_images\",\"scene_id\":\"...\",\"image_mode\":\"nano_banana|flux_klein\"}\n"
        "- {\"type\":\"generate_image_prompt_for_current_mode\",\"scene_id\":\"...\",\"image_mode\":\"optional zimage|flux_klein|nano_banana|ernie_image\"}\n"
        "- {\"type\":\"run_image_for_current_mode\",\"scene_id\":\"...\",\"image_mode\":\"optional zimage|flux_klein|nano_banana|ernie_image\"}\n"
        "- {\"type\":\"generate_video_prompt_for_current_mode\",\"scene_id\":\"...\",\"video_mode\":\"optional i2v|id_lora|t2v|rtv|ingredients\"}\n\n"
        "- {\"type\":\"run_video_for_current_mode\",\"scene_id\":\"...\",\"video_mode\":\"optional i2v|id_lora|t2v|rtv|ingredients\"}\n\n"
        "Use note actions to capture the creative direction you discuss with the user. Use set_scene_plan when planning character assignments, story beats, scene concepts, and motion notes for one or more scenes. Keep each set_scene_plan field short and direct.\n"
        "Use create_concept_prompts when the user asks to create, generate, build, or update concept prompts from director notes, scene notes, timeline notes, story idea, or theme/style. This writes generated concept prompts into scene notes in batches.\n"
        "Use create_motion_notes when the user asks to create, generate, build, or update I2V/T2V motion notes from concept prompts, director notes, timeline notes, story idea, or theme/style. This writes generated motion notes into the I2V motion notes fields/file in batches.\n"
        "Use get_context_prompts to read the global Theme/style, Story idea, and Subject/scene context prompts. Use set_context_prompt when the user asks to update those global context prompt files. Choose context_type theme_style for style/mood/look, story_idea for plot/concept/narrative, and subject_scene for characters, subjects, locations, and recurring scene details.\n"
        "Use selected_timeline_range and timeline_markers when the user is timing vocals, choruses, dialogue, or music sections. If they say this selected range is where someone sings, use assign_selected_range_note. If they ask to make a scene cover it, use set_active_scene_to_selected_range. Only use create_scene_from_selected_range or split_selected_range_into_scenes when selected_timeline_range is not null and the user explicitly asks to create/split the selected in/out range. If the user asks to split one existing scene, such as 'split scene 4 into 3', use split_scene_into_subscenes with that scene number and count. If the user asks to combine, merge, join, or consolidate scenes, use merge_scenes with the scene numbers. If the user asks to update/reword/fix director notes, do note actions only; do not sync or change timings. If they ask to replace director note text, such as 'when a director note says X change it to Y', use replace_director_note_text. If they ask that all both-character or dual-vocal director notes say a specific phrase such as 'male and female' or 'female and male', use normalize_dual_vocal_director_notes with that replacement phrase. If timeline markers already exist and the user asks to update, align, match, sync, or split existing scene segments from those markers, or answers yes after you suggest updating scenes from markers, use sync_existing_scenes_to_timeline_markers. If the user asks to fix missing/skipped/duplicate scene numbers or renumber labels, use renumber_scene_labels. Do not use split_selected_range_into_scenes for marker-based timing, one-scene splits, or merge/combine requests.\n"
        "Use generate actions when the user asks you to make/create/update the actual image or video prompt. Use run_image_for_current_mode only when the user asks to run/create/generate the actual image, not just the text prompt.\n"
        "If the user asks for multiple steps, include all requested actions in order. Example: prompt, then image, then video means generate_image_prompt_for_current_mode, run_image_for_current_mode, generate_video_prompt_for_current_mode, run_video_for_current_mode.\n"
        "Nano B and Flux/Klein can run with or without reference images. If no references exist, use the normal generate/run actions anyway and the app will use text-only prompting. Only request reference images when the user specifically asks to use references or the scene concept requires matching an exact character/location.\n"
        "Do not write final image/video prompts yourself as actions. Do not use unsupported set_image_prompt, set_flux_prompt, set_nb_prompt, or set_video_prompt actions.\n"
        "Never put JSON, action arrays, code fences, or internal notes inside the reply string. The reply must be a short user-facing sentence; all work must be in the actions array.\n"
        "Do not claim which image/video generator ran. The app will report the actual mode after it runs the action.\n"
        f"Agent mode: {'AUTO APPLY. If the user asks you to do/apply/update/fill/write prompts or notes, include supported note/generate actions for the requested scene fields.' if auto_apply else 'MANUAL. Never include actions. Do not say you applied or updated the project; only suggest copy-ready text.'}\n"
        "Prefer scene_number for scene-targeted actions. Use scene_id only when copying an exact internal id from the context JSON. If the target scene is unclear, ask one short clarifying question and return no actions.\n"
        "In Auto Apply mode, the application will apply actions after your response. In Manual mode, actions must always be an empty list.\n\n"
        "Active context JSON:\n"
        f"{context_text}\n\n"
        "Conversation:\n"
        f"{conversation_text}\n\n"
        "Return the JSON now."
    )
    text, info = _run_builder_text_llm(
        payload,
        instruction,
        temperature=float(payload.get("temperature") or 0.55),
        top_p=float(payload.get("top_p") or 0.95),
        max_new_tokens=int(payload.get("max_new_tokens") or 500),
        label="Builder Agent",
        preserve_paragraphs=True,
    )
    raw = _clean_lm_studio_plain_text(text)
    try:
        parsed = _extract_json_object_from_text(raw)
    except Exception:
        parsed = {"reply": raw, "actions": []}
    reply = _clean_lm_studio_plain_text(str(parsed.get("reply") or "")).strip()
    actions = parsed.get("actions") if isinstance(parsed, dict) else []
    if not isinstance(actions, list) or not auto_apply:
        actions = []
    cleaned_actions = []
    allowed_types = {
        "set_scene_notes",
        "set_scene_plan",
        "select_scene",
        "get_scene_lyrics",
        "get_scene_context",
        "get_story_source",
        "get_context_prompts",
        "get_selected_timeline_range",
        "save_story_source",
        "set_context_prompt",
        "create_story_scenes_from_source",
        "create_concept_prompts",
        "create_motion_notes",
        "set_active_scene_to_selected_range",
        "create_scene_from_selected_range",
        "split_selected_range_into_scenes",
        "split_scene_into_subscenes",
        "merge_scenes",
        "renumber_scene_labels",
        "normalize_dual_vocal_director_notes",
        "replace_director_note_text",
        "assign_selected_range_note",
        "sync_existing_scenes_to_timeline_markers",
        "set_flux_notes",
        "set_nb_notes",
        "set_video_notes",
        "set_image_model_mode",
        "set_video_model_mode",
        "request_reference_images",
        "generate_image_prompt_for_current_mode",
        "run_image_for_current_mode",
        "generate_video_prompt_for_current_mode",
        "run_video_for_current_mode",
    }
    allowed_image_modes = {"zimage", "flux_klein", "nano_banana", "ernie_image"}
    allowed_video_modes = {"i2v", "id_lora", "t2v", "rtv", "ingredients"}
    max_actions = 60 if agent_purpose == "story_builder" else 12
    plan_fields = {"director_note", "scene_notes", "flux_notes", "nb_notes", "video_notes"}
    for action in actions[:max_actions]:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type") or "").strip()
        scene_id = str(action.get("scene_id") or "").strip()
        action_text_raw = str(action.get("text") or "").strip()
        image_mode = str(action.get("image_mode") or "").strip()
        video_mode = str(action.get("video_mode") or "").strip()
        context_type = str(action.get("context_type") or "").strip().lower()
        context_mode = str(action.get("mode") or "").strip().lower()
        scene_count = action.get("scene_count")
        try:
            scene_count = int(scene_count) if scene_count is not None else None
        except Exception:
            scene_count = None
        plan_values = {
            field: str(action.get(field) or "").strip()[:5000]
            for field in plan_fields
            if str(action.get(field) or "").strip()
        }
        scene_number = action.get("scene_number")
        if scene_number is None:
            scene_target_text = " ".join(
                str(action.get(field) or "")
                for field in (
                    "scene_id",
                    "scene",
                    "scene_label",
                    "scene_name",
                    "target_scene",
                    "target",
                    "label",
                    "name",
                    "title",
                )
            )
            scene_match = re.search(r"\bscene\s*(\d+)(?:\b|\s|\.|:|-)", scene_target_text, re.I)
            if scene_match:
                scene_number = scene_match.group(1)
        try:
            scene_number = int(scene_number) if scene_number is not None else None
        except Exception:
            scene_number = None
        no_scene_actions = {
            "set_image_model_mode",
            "set_video_model_mode",
            "get_story_source",
            "get_context_prompts",
            "save_story_source",
            "set_context_prompt",
            "create_story_scenes_from_source",
            "create_concept_prompts",
            "create_motion_notes",
            "get_selected_timeline_range",
            "create_scene_from_selected_range",
            "split_selected_range_into_scenes",
            "merge_scenes",
            "renumber_scene_labels",
            "normalize_dual_vocal_director_notes",
            "replace_director_note_text",
            "assign_selected_range_note",
            "sync_existing_scenes_to_timeline_markers",
        }
        if action_type in allowed_types and (scene_id or scene_number is not None or action_type in no_scene_actions):
            if action_type == "request_reference_images" and not scene_id:
                continue
            cleaned_action = {
                "type": action_type,
            }
            if scene_id:
                cleaned_action["scene_id"] = scene_id[:160]
            if scene_number is not None:
                cleaned_action["scene_number"] = scene_number
            if action_text_raw:
                cleaned_action["text"] = action_text_raw[:50000 if action_type == "save_story_source" else 5000]
            if image_mode in allowed_image_modes:
                cleaned_action["image_mode"] = image_mode
            if video_mode in allowed_video_modes:
                cleaned_action["video_mode"] = video_mode
            if scene_count is not None:
                cleaned_action["scene_count"] = max(1, min(120, scene_count))
            if action_type == "get_context_prompts":
                if context_type not in {"all", "theme_style", "story_idea", "subject_scene"}:
                    context_type = "all"
                cleaned_action["context_type"] = context_type
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "set_context_prompt":
                if context_type not in {"theme_style", "story_idea", "subject_scene"}:
                    continue
                if not action_text_raw:
                    continue
                cleaned_action["context_type"] = context_type
                cleaned_action["text"] = action_text_raw[:50000]
                cleaned_action["mode"] = context_mode if context_mode in {"replace", "append"} else "replace"
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "assign_selected_range_note":
                for field in ("label", "marker_type", "note"):
                    value = str(action.get(field) or "").strip()
                    if value:
                        cleaned_action[field] = value[:1000]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "create_scene_from_selected_range":
                for field in ("label", "director_note", "scene_notes", "flux_notes", "nb_notes", "video_notes", "note"):
                    value = str(action.get(field) or "").strip()
                    if value:
                        cleaned_action[field] = value[:5000]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "split_selected_range_into_scenes":
                label_prefix = str(action.get("label_prefix") or "").strip()
                if label_prefix:
                    cleaned_action["label_prefix"] = label_prefix[:160]
                for field in ("director_notes", "scene_notes"):
                    values = action.get(field)
                    if isinstance(values, list):
                        cleaned_action[field] = [str(value or "").strip()[:5000] for value in values[:120]]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "split_scene_into_subscenes":
                label_prefix = str(action.get("label_prefix") or "").strip()
                if label_prefix:
                    cleaned_action["label_prefix"] = label_prefix[:160]
                for field in ("director_notes", "scene_notes"):
                    values = action.get(field)
                    if isinstance(values, list):
                        cleaned_action[field] = [str(value or "").strip()[:5000] for value in values[:120]]
                if scene_count is None or scene_count < 2:
                    continue
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "merge_scenes":
                numbers = action.get("scene_numbers") or action.get("scenes") or []
                cleaned_numbers = []
                if isinstance(numbers, list):
                    for value in numbers[:120]:
                        try:
                            number = int(value)
                        except Exception:
                            continue
                        if number > 0 and number not in cleaned_numbers:
                            cleaned_numbers.append(number)
                for src, dst in (("start_scene_number", "start_scene_number"), ("end_scene_number", "end_scene_number")):
                    try:
                        value = int(action.get(src))
                    except Exception:
                        value = None
                    if value is not None and value > 0:
                        cleaned_action[dst] = value
                label = str(action.get("label") or "").strip()
                if label:
                    cleaned_action["label"] = label[:160]
                if cleaned_numbers:
                    cleaned_action["scene_numbers"] = cleaned_numbers
                if len(cleaned_numbers) < 2 and not (cleaned_action.get("start_scene_number") and cleaned_action.get("end_scene_number")):
                    continue
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "renumber_scene_labels":
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "normalize_dual_vocal_director_notes":
                replacement = str(action.get("replacement") or "male and female").strip()
                cleaned_action["replacement"] = (replacement or "male and female")[:160]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "replace_director_note_text":
                find_text = str(action.get("find") or action.get("from") or "").strip()
                replace_text = str(action.get("replace") or action.get("to") or "").strip()
                if not find_text or not replace_text:
                    continue
                cleaned_action["find"] = find_text[:500]
                cleaned_action["replace"] = replace_text[:500]
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "get_story_source":
                cleaned_actions.append(cleaned_action)
                continue
            if action_type in {"save_story_source", "create_story_scenes_from_source"} and action_type == "save_story_source" and not action_text_raw:
                continue
            if action_type == "save_story_source":
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "create_story_scenes_from_source":
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "create_concept_prompts":
                source_mode = str(action.get("source_mode") or action.get("source") or "all").strip().lower()
                scope = str(action.get("scope") or "all").strip().lower()
                if source_mode not in {"all", "director", "scene", "timeline", "director_scene"}:
                    source_mode = "all"
                if scope not in {"all", "selected", "range"}:
                    scope = "all"
                try:
                    batch_size = int(action.get("batch_size") or action.get("batch") or 5)
                except Exception:
                    batch_size = 5
                cleaned_action["source_mode"] = source_mode
                cleaned_action["scope"] = scope
                cleaned_action["batch_size"] = max(1, min(10, batch_size))
                cleaned_action["use_story"] = action.get("use_story") is not False
                cleaned_action["use_theme"] = action.get("use_theme") is not False
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "create_motion_notes":
                source_mode = str(action.get("source_mode") or action.get("source") or "concept").strip().lower()
                scope = str(action.get("scope") or "all").strip().lower()
                if source_mode not in {"concept", "concept_director", "concept_timeline", "all"}:
                    source_mode = "concept"
                if scope not in {"all", "selected", "range"}:
                    scope = "all"
                try:
                    batch_size = int(action.get("batch_size") or action.get("batch") or 5)
                except Exception:
                    batch_size = 5
                cleaned_action["source_mode"] = source_mode
                cleaned_action["scope"] = scope
                cleaned_action["batch_size"] = max(1, min(10, batch_size))
                cleaned_action["use_story"] = action.get("use_story") is not False
                cleaned_action["use_theme"] = action.get("use_theme") is not False
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "sync_existing_scenes_to_timeline_markers":
                cleaned_action["create_missing"] = action.get("create_missing") is not False
                cleaned_actions.append(cleaned_action)
                continue
            if action_type == "set_scene_plan":
                for field, value in plan_values.items():
                    cleaned_action[field] = value
                if not plan_values and not action_text_raw:
                    continue
                cleaned_actions.append(cleaned_action)
                continue
            if action_type in {"set_scene_notes", "set_flux_notes", "set_nb_notes", "set_video_notes"} and not cleaned_action.get("text"):
                alias_fields = {
                    "set_scene_notes": ("scene_notes", "notes", "concept", "concept_prompt", "prompt"),
                    "set_flux_notes": ("flux_notes", "notes", "prompt"),
                    "set_nb_notes": ("nb_notes", "nano_banana_notes", "notes", "prompt"),
                    "set_video_notes": ("video_notes", "i2v_notes", "motion_notes", "notes", "prompt"),
                }[action_type]
                for field in alias_fields:
                    value = str(action.get(field) or "").strip()
                    if value:
                        cleaned_action["text"] = value[:5000]
                        break
            if action_type.startswith("set_") and not action_text_raw:
                if action_type == "set_image_model_mode" and cleaned_action.get("image_mode"):
                    cleaned_actions.append(cleaned_action)
                    continue
                if action_type == "set_video_model_mode" and cleaned_action.get("video_mode"):
                    cleaned_actions.append(cleaned_action)
                    continue
                if action_type in {"set_scene_notes", "set_flux_notes", "set_nb_notes", "set_video_notes"} and cleaned_action.get("text"):
                    cleaned_actions.append(cleaned_action)
                    continue
                continue
            cleaned_actions.append(cleaned_action)
    action_hint_text = "\n\n".join(
        [item.get("content", "") for item in cleaned_messages[-6:]]
        + ([reply] if reply else [])
    )
    latest_user_text = str(latest_user or cleaned_messages[-1].get("content", "") if cleaned_messages else "").strip()
    latest_merge_intent = re.search(r"\b(combine|merge|join|consolidate)\b", latest_user_text, re.I)
    latest_split_intent = re.search(r"\b(split|break|divide|sub[-\s]?scenes?|sub[-\s]?segments?)\b", latest_user_text, re.I)
    latest_director_note_text_intent = re.search(r"\b(director\s+notes?|notes?|reword|rename|text)\b", latest_user_text, re.I)
    latest_dual_vocal_note_intent = (
        latest_director_note_text_intent and
        re.search(r"\b(male\s+and\s+female|both\s+(?:characters|singers|vocals?)|dual[-\s]?vocal)\b", latest_user_text, re.I)
    )
    if auto_apply and latest_director_note_text_intent:
        cleaned_actions = [
            action for action in cleaned_actions
            if action.get("type") not in {
                "sync_existing_scenes_to_timeline_markers",
                "split_selected_range_into_scenes",
                "split_scene_into_subscenes",
                "merge_scenes",
            }
        ]
    if auto_apply and latest_dual_vocal_note_intent and not any(action.get("type") == "normalize_dual_vocal_director_notes" for action in cleaned_actions):
        exact_replace_match = re.search(r"\b(?:director\s+notes?|director\s+note|note)\s+says?\s+[\"']([^\"']+)[\"'].*?\b(?:changed?\s+to|to|instead)\s+[\"']([^\"']+)[\"']", latest_user_text, re.I)
        if exact_replace_match and not any(action.get("type") == "replace_director_note_text" for action in cleaned_actions):
            cleaned_actions.append(
                {
                    "type": "replace_director_note_text",
                    "find": exact_replace_match.group(1).strip(),
                    "replace": exact_replace_match.group(2).strip(),
                }
            )
        replacement_match = re.search(r"\b(?:says?|to|instead|changed?\s+to)\s+[\"']?((?:fe)?male\s+and\s+(?:fe)?male)[\"']?", latest_user_text, re.I)
        replacement = replacement_match.group(1).strip().lower() if replacement_match else "male and female"
        cleaned_actions.append({"type": "normalize_dual_vocal_director_notes", "replacement": replacement})
    if auto_apply and latest_split_intent:
        cleaned_actions = [
            action for action in cleaned_actions
            if action.get("type") != "merge_scenes"
        ]
    if auto_apply and latest_merge_intent:
        cleaned_actions = [
            action for action in cleaned_actions
            if action.get("type") not in {"split_scene_into_subscenes", "split_selected_range_into_scenes"}
        ]
    merge_intent = latest_merge_intent or (
        not latest_split_intent and
        re.fullmatch(r"\s*(yes|yep|yeah|ok|okay|please do it|do it|go ahead|sure)\.?\s*", latest_user_text, re.I) and
        re.search(r"\b(combine|merge|join|consolidate)\b", action_hint_text, re.I)
    )
    if auto_apply and merge_intent:
        cleaned_actions = [
            action for action in cleaned_actions
            if action.get("type") not in {"split_scene_into_subscenes", "split_selected_range_into_scenes"}
        ]
        has_merge_action = any(action.get("type") == "merge_scenes" for action in cleaned_actions)
        if not has_merge_action:
            range_match = re.search(r"\bscenes?\s+(\d+)\s*(?:-|to|through)\s*(\d+)\b", action_hint_text, re.I)
            scene_numbers = []
            if range_match:
                start_num = int(range_match.group(1))
                end_num = int(range_match.group(2))
                if start_num <= end_num:
                    scene_numbers = list(range(start_num, end_num + 1))
            if not scene_numbers:
                sequence_match = re.search(r"\bscenes?\s+((?:\d+\s*(?:,|and)?\s*){2,})", action_hint_text, re.I)
                if sequence_match:
                    scene_numbers = [int(value) for value in re.findall(r"\d+", sequence_match.group(1))]
            if len(scene_numbers) >= 2:
                scene_numbers = list(dict.fromkeys(scene_numbers))
                cleaned_actions.append(
                    {
                        "type": "merge_scenes",
                        "scene_numbers": scene_numbers,
                        "label": f"Scene {scene_numbers[0]}",
                    }
                )
                if not reply:
                    reply = f"Merging scenes {', '.join(str(value) for value in scene_numbers)}."
    if auto_apply and not cleaned_actions:
        split_hint_text = latest_user_text if latest_split_intent else action_hint_text
        if not merge_intent and re.search(r"\b(split|break|divide|sub[-\s]?scenes?|sub[-\s]?segments?)\b", split_hint_text, re.I):
            scene_matches = re.findall(r"\bscene\s+(\d+)\b", split_hint_text, re.I)
            count_matches = []
            count_matches.extend(
                int(match)
                for match in re.findall(r"\b(?:into|in|to)\s+(\d+)\s*(?:sub[-\s]?scenes?|scenes?|segments?)\b", split_hint_text, re.I)
                if str(match).isdigit()
            )
            count_matches.extend(
                int(match)
                for match in re.findall(r"\b(\d+)\s*(?:sub[-\s]?scenes?|sub[-\s]?segments?)\b", split_hint_text, re.I)
                if str(match).isdigit()
            )
            latest_user_number = re.fullmatch(r"\s*(\d{1,2})\s*", latest_user or "")
            if latest_user_number:
                count_matches.append(int(latest_user_number.group(1)))
            if scene_matches and count_matches:
                inferred_scene = int(scene_matches[-1])
                inferred_count = max(2, min(24, int(count_matches[-1])))
                cleaned_actions.append(
                    {
                        "type": "split_scene_into_subscenes",
                        "scene_number": inferred_scene,
                        "scene_count": inferred_count,
                        "label_prefix": f"Scene {inferred_scene}",
                    }
                )
                if not reply:
                    reply = f"Splitting Scene {inferred_scene} into {inferred_count} sub-scenes."
    if not reply:
        reply = "Done." if cleaned_actions else "I can help with that."
    return {"reply": reply, "actions": cleaned_actions, **info}
