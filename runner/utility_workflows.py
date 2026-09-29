"""Utility graphs: transcription and clear memory."""

import copy
import os

from .paths import _bool_payload, _float_payload, _int_payload
from .api_graph import _api_node_id_by_class, _load_api_template, _set_api_input


def _clear_memory_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "ClearMemory_API.json",
    )


def _transcribe_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "LTX2.3_Transcribe_API.json",
    )


def _timestamped_transcribe_api_template_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Workflows",
        "UsedForUIDoNotTouch",
        "LTX2.3_Transcribe_2_API.json",
    )


def _build_clear_memory_prompt():
    workflow_path, prompt = _load_api_template(_clear_memory_api_template_path())
    return {
        "workflow_path": workflow_path,
        "prompt": prompt,
    }


def _patch_transcribe_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    srt_path = os.path.abspath(str(payload.get("srt_path", "") or "").strip().strip('"'))
    if not audio_path:
        raise ValueError("Audio file path is empty.")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")
    if not srt_path:
        raise ValueError("SRT file path is empty.")
    if not os.path.isfile(srt_path):
        raise FileNotFoundError(f"SRT file was not found: {srt_path}")

    extractor_id = _api_node_id_by_class(prompt, "VRGDG_ManualLyricsExtractor_SRT_Advanced", "960")
    stems_id = _api_node_id_by_class(prompt, "VRGDG_GetStems", "28:114")
    _set_api_input(prompt, stems_id, "audio_file_path", audio_path)
    _set_api_input(prompt, extractor_id, "srt_path", srt_path)
    _set_api_input(prompt, extractor_id, "reference_lyrics", str(payload.get("reference_lyrics", "") or ""))
    _set_api_input(prompt, extractor_id, "language", str(payload.get("language", "") or "english"))
    _set_api_input(prompt, extractor_id, "strict_reference_text", bool(payload.get("strict_reference_text", True)))
    _set_api_input(prompt, extractor_id, "fill_aggressiveness", _int_payload(payload, "fill_aggressiveness", 1, minimum=0, maximum=3))
    _set_api_input(prompt, extractor_id, "preserve_nonvocal_segments", bool(payload.get("preserve_nonvocal_segments", True)))
    _set_api_input(prompt, extractor_id, "alignment_min_words", _int_payload(payload, "alignment_min_words", 1, minimum=1, maximum=10))
    model_name = str(payload.get("model_name", "") or "large-v3").strip()
    if model_name:
        _set_api_input(prompt, extractor_id, "model_name", model_name)
    return prompt


def _build_transcribe_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_transcribe_api_template_path())
    patched_prompt = _patch_transcribe_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
    }


def _patch_timestamped_transcribe_api_prompt(prompt, payload):
    prompt = copy.deepcopy(prompt)
    audio_path = os.path.abspath(str(payload.get("audio_path", "") or "").strip().strip('"'))
    if not audio_path:
        raise ValueError("Audio file path is empty.")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file was not found: {audio_path}")

    extractor_id = _api_node_id_by_class(prompt, "VRGDG_TimestampedLyricsExtractor", "962")
    stems_id = _api_node_id_by_class(prompt, "VRGDG_GetStems", "28:114")
    _set_api_input(prompt, stems_id, "audio_file_path", audio_path)
    _set_api_input(prompt, extractor_id, "reference_lyrics", str(payload.get("reference_lyrics", "") or ""))
    _set_api_input(prompt, extractor_id, "language", str(payload.get("language", "") or "english"))
    segment_mode = str(payload.get("segment_mode", "") or "reference_lines").strip()
    if segment_mode not in {"whisper_chunks", "reference_lines", "exact_reference_lines", "reference_stanzas", "reference_scene_words"}:
        segment_mode = "reference_lines"
    _set_api_input(prompt, extractor_id, "segment_mode", segment_mode)
    _set_api_input(prompt, extractor_id, "include_instrumental_gaps", _bool_payload(payload, "include_instrumental_gaps", True))
    _set_api_input(prompt, extractor_id, "instrumental_text", str(payload.get("instrumental_text", "") or "[instrumental]"))
    _set_api_input(prompt, extractor_id, "min_gap_seconds", _float_payload(payload, "min_gap_seconds", 1.0, minimum=0.0, maximum=30.0))
    _set_api_input(prompt, extractor_id, "min_scene_seconds", _float_payload(payload, "min_scene_seconds", 1.0, minimum=1.0, maximum=30.0))
    _set_api_input(prompt, extractor_id, "max_scene_seconds", _float_payload(payload, "max_scene_seconds", 8.0, minimum=1.0, maximum=60.0))
    _set_api_input(prompt, extractor_id, "vocal_tail_padding_seconds", _float_payload(payload, "vocal_tail_padding_seconds", 0.6, minimum=0.0, maximum=3.0))
    model_name = str(payload.get("model_name", "") or "large-v3").strip()
    if model_name:
        _set_api_input(prompt, extractor_id, "model_name", model_name)
    return prompt


def _build_timestamped_transcribe_api_prompt(payload):
    workflow_path, prompt = _load_api_template(_timestamped_transcribe_api_template_path())
    patched_prompt = _patch_timestamped_transcribe_api_prompt(prompt, payload)
    return {
        "workflow_path": workflow_path,
        "prompt": patched_prompt,
    }
