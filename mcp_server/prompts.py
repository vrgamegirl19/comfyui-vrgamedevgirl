"""MCP Prompt templates (Section 8.3)."""

import os
from typing import Any, Dict, List, Optional


PROMPT_DESCRIPTORS: List[Dict[str, Any]] = [
    {
        "name": "make_music_video",
        "description": "Guided workflow from audio track to final stitched video with checkpoints.",
        "arguments": [
            {"name": "project_name", "description": "Name for the project", "required": True},
            {"name": "audio_file", "description": "Path to the music audio file", "required": True},
            {"name": "lyrics", "description": "Full lyrics text", "required": False},
            {"name": "character_name", "description": "Name of the main character", "required": False},
            {"name": "character_image", "description": "Path to the character image", "required": False},
            {"name": "location_style_theme", "description": "Where and what look, e.g. 'Los Angeles nightlife, neon, night time'", "required": False},
            {"name": "story_idea", "description": "One or two sentences to build the story from", "required": False},
        ],
    },
    {
        "name": "chat_make_music_video",
        "description": "Chat with the user: ask for the song, lyrics, character and style, confirm, then build the whole video.",
        "arguments": [],
    },
    {
        "name": "review_scene",
        "description": "View image or video contact sheet, critique against prompt, and propose edits.",
        "arguments": [
            {"name": "project_id", "description": "Project ID", "required": True},
            {"name": "scene_id", "description": "Scene segment ID to review", "required": True},
        ],
    },
    {
        "name": "fix_failed_render",
        "description": "Inspect job log, diagnose cause of failure, and propose or execute remediation.",
        "arguments": [
            {"name": "job_id", "description": "ID of the failed job", "required": True},
            {"name": "project_id", "description": "Project ID", "required": True},
        ],
    },
    {
        "name": "polish_timeline",
        "description": "Inspect scene durations, detect gaps, short scenes, and suggest merges or snaps.",
        "arguments": [
            {"name": "project_id", "description": "Project ID to inspect", "required": True},
        ],
    },
]


def list_prompts() -> List[Dict[str, Any]]:
    return PROMPT_DESCRIPTORS


def get_prompt(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = arguments or {}
    p_name = str(name).strip().lower()

    if p_name == "make_music_video":
        form = "\n".join([
            "Fill-in values for this project:",
            f"- project_name: {args.get('project_name', 'MyMusicVideo')}",
            f"- audio_file: {args.get('audio_file', '')}",
            f"- character name: {args.get('character_name', '(not given)')}",
            f"- character image file: {args.get('character_image', '(not given)')}",
            f"- location style theme: {args.get('location_style_theme', '(not given)')}",
            f"- story idea: {args.get('story_idea', '(not given; write one from the lyrics and theme)')}",
            "- lyrics:",
            str(args.get("lyrics", "(not given)")),
        ])
        playbook_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MUSIC_VIDEO_AGENT_PROMPT.md")
        try:
            with open(playbook_path, "r", encoding="utf-8") as handle:
                playbook = handle.read()
        except OSError:
            playbook = "Playbook file MUSIC_VIDEO_AGENT_PROMPT.md was not found. Read resource vrgdg://docs/endpoints and build the video step by step."
        instructions = f"{form}\n\n---\n\n{playbook}"
        return {
            "description": "Make Music Video Pipeline",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": instructions},
                }
            ],
        }

    if p_name == "chat_make_music_video":
        chat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CHAT_AGENT_PROMPT.md")
        try:
            with open(chat_path, "r", encoding="utf-8") as handle:
                chat_prompt = handle.read()
        except OSError:
            chat_prompt = "CHAT_AGENT_PROMPT.md was not found. Ask the user for the song, lyrics, character and style, then follow resource vrgdg://docs/music-video-playbook."
        return {
            "description": "Chat Make Music Video",
            "messages": [{"role": "user", "content": {"type": "text", "text": chat_prompt}}],
        }

    if p_name == "review_scene":
        pid = args.get("project_id", "")
        sid = args.get("scene_id", "")
        msg = (
            f"Please review scene '{sid}' in project '{pid}'.\n"
            "1. Fetch the scene details via `scene_get`.\n"
            "2. Inspect the scene preview using `asset_view` (with kind='contact_sheet' or 'thumbnail').\n"
            "3. Compare the visual output against the scene's `t2i_prompt` and `i2v_prompt`.\n"
            "4. If improvements are needed, suggest an updated prompt and call `scene_update`."
        )
        return {
            "description": "Review Scene",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": msg},
                }
            ],
        }

    if p_name == "fix_failed_render":
        jid = args.get("job_id", "")
        pid = args.get("project_id", "")
        msg = (
            f"Investigate failed job '{jid}' in project '{pid}'.\n"
            "1. Inspect job status and error message via `job_get` or resource `vrgdg://jobs/{job_id}/log`.\n"
            "2. Check if the error is PREDECESSOR_MISSING or LATENT_STALE, and call `latents_status_rebuild` if needed.\n"
            "3. If recoverable output exists in scratch backup, call `video_recover`.\n"
            "4. Otherwise, adjust settings via `project_update_settings` and call `job_retry`."
        )
        return {
            "description": "Fix Failed Render",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": msg},
                }
            ],
        }

    if p_name == "polish_timeline":
        pid = args.get("project_id", "")
        msg = (
            f"Inspect and polish timeline for project '{pid}'.\n"
            "1. Call `scene_list` to examine start and end times of all scenes.\n"
            "2. Identify scenes with duration < 1.0s or unintended gaps between scenes.\n"
            "3. Use `scene_split_merge_move_resize` with op='merge' or op='resize' to resolve timing anomalies."
        )
        return {
            "description": "Polish Timeline",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": msg},
                }
            ],
        }

    raise KeyError(f"Unknown prompt name: {name}")
