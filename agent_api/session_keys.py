"""Top-level keys of the Video Builder project session (``vrgdg_builder_session.json``).

``BUILDER_SESSION_DATA_KEYS`` is the Python twin of the object ``currentSessionData()`` returns in
``web/music_video_builder/session.mjs``: every Builder save writes all of them.
``tests/test_agent_api_project_include_fresh.py`` fails when the two drift apart, so a new Builder session
key has to be added here too. ``SERVER_SESSION_KEYS`` are the keys the save itself adds
(``builder.project._save_builder_session_unlocked`` and the save payload).
"""

from typing import Any, Dict, FrozenSet

BUILDER_SESSION_DATA_KEYS = (
    "segments", "audio_clips", "speaking_audio_defaults", "overlay_segments", "overlay_track", "active_track", "timing_frozen", "srt_mode",
    "prompt_json_path", "i2v_motion_json_path", "image_trigger_phrase", "video_trigger_phrase",
    "default_facial_performance", "default_facial_performance_custom", "use_i2v_prompt_enhancement_pass",
    "fail_on_invalid_prompt_formats", "omit_lyrics_from_video_prompts", "use_structured_outputs", "continuity_mode", "auto_img2img_start_step", "auto_img2img_creativity",
    "auto_chain_last_frame", "image_continuity_enabled", "image_continuity_strength", "auto_chain_style",
    "auto_chain_direction", "auto_chain_transition_lora_prompt", "auto_chain_transition_trigger",
    "use_vrgdg_text_context", "theme_style_path", "story_idea_path", "subject_scene_path", "text_gemma_runner",
    "gemma_context_limit", "gemma_output_token_limit", "gemma_gpu_layers", "lm_studio_base_url",
    "lm_studio_model", "lm_studio_api_key", "lm_studio_context_limit", "lm_studio_output_token_limit",
    "llm_api_provider", "llm_api_model", "llm_api_key_project", "elevenlabs_api_key_project", "own_server_url", "own_server_model",
    "own_server_api_key_project", "own_server_output_token_limit", "own_server_timeout",
    "notification_settings", "automatic_memory_cleanup", "scene_render_wait_hours", "waveform_mode",
    "snap_to_beats", "show_beat_markers", "audio_mask_auto", "show_timeline_scene_notes", "show_timeline_video_notes",
    "show_timeline_lyric_notes", "show_timeline_emotion_tags", "selected_timeline_range", "timeline_markers", "active_timeline_marker_id",
    "audio_duration", "audio_peaks", "beat_markers", "detected_tempo_bpm", "beat_calibration",
    "left_panel_width", "left_panel_collapsed", "right_panel_collapsed", "llm_popout_open", "llm_popout_width",
    "llm_popout_height", "llm_popout_x", "llm_popout_y", "left_panel_tab", "right_panel_width",
    "timeline_panel_height", "timeline_zoom", "auto_save_enabled", "video_type", "video_engine",
    "minimax_h3_settings", "minimax_h3_two_pass", "minimax_h3_three_pass", "minimax_h3_advanced_two_pass",
    "image_model_mode", "zimage_settings", "reference_krea2_settings", "flux_klein_settings",
    "flow_gpt_browser_settings", "nb_image_settings", "ernie_image_settings", "krea2_2pass_settings",
    "use_flux_global_image_ingredients", "flux_global_image_ingredients", "flux_reference_builder",
    "id_lora_reference_builder", "lyric_mapper", "z_enhance_settings", "video_model_mode",
    "i2v_video_settings", "prompt_tools_hint_prefs", "builder_agent_messages", "builder_agent_auto_apply",
    "builder_agent_purpose", "builder_agent_reference_images", "builder_story_source_path",
    "builder_story_reference_images", "builder_story_reference_notes", "builder_story_layer",
    "builder_storyboard_defaults", "auto_build_preparation", "wizard_beta_draft", "render_logs",
    "active_render_log_id",
)

SERVER_SESSION_KEYS = (
    "project_name", "project_folder", "audio_path", "updated", "revision", "builder_save_revision",
    "project_context_files",
)

KNOWN_SESSION_KEYS: FrozenSet[str] = frozenset(BUILDER_SESSION_DATA_KEYS + SERVER_SESSION_KEYS)

# What a known key reads as before the project has saved it. Objects the API already reads with an
# ``or {}`` fallback (the ``references`` and ``story`` include groups, the storyboard defaults) start
# empty; everything else is null ("not saved yet").
UNSAVED_SESSION_DEFAULTS: Dict[str, Any] = {
    "flux_reference_builder": {},
    "builder_story_layer": {},
    "builder_storyboard_defaults": {},
}
