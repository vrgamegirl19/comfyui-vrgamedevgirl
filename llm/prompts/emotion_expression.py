"""LLM instructions for lyric-aware emotion tags and visible acting."""

from typing import Any, Dict


def emotion_expression_input(segment: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, str]:
    """Resolve the same scene input and inherited facial defaults as the UI."""
    return {
        "emotion_expression_tags": str(segment.get("emotion_expression_tags") or "").strip(),
        "facial_performance": str(segment.get("facial_performance")
                                  or session.get("default_facial_performance") or "").strip(),
        "facial_performance_custom": str(segment.get("facial_performance_custom")
                                         or session.get("default_facial_performance_custom") or "").strip(),
    }


def has_emotion_expression_input(payload: Dict[str, Any]) -> bool:
    """Explicit scene tags override a preset, including Off."""
    return bool(payload.get("emotion_expression_tags") or (
        payload.get("facial_performance") != "off"
        and (payload.get("facial_performance") or payload.get("facial_performance_custom"))))


def emotion_expression_instruction(payload: Dict[str, Any]) -> str:
    """Ask the LLM to choose tags and acting from the lyrics and user's direction."""
    native_speech = (str(payload.get("performance_mode") or payload.get("video_type") or "").strip() == "speaking"
                     and payload.get("audio_mode") == "built_in_audio")
    speech_instruction = (
        "\n\nBUILT-IN H3 SPEECH DELIVERY:\n"
        "Use the scene beat, storyboard, spoken words and requested facial/emotion direction to stage the speaker's visible acting and generated vocal delivery. "
        "Put language and fitting emotion descriptors in the dialogue header, for example <d>[English, curious] exact dialogue.</d> "
        "The says/speaks verb establishes speech; do not substitute a generic speaking descriptor for the actual emotion. "
        "INLINE DELIVERY TAGS ARE REQUIRED: An emotion header alone is incomplete. In every shot with spoken dialogue, "
        "include at least one suitable inline delivery cue inside <d>, chosen from the scene, dialogue meaning and requested emotion. "
        "For several sentences, stage the delivery across relevant phrase boundaries and emphasized words rather than tagging only the header. "
        "Use pauses for changes of thought, breath cues for a motivated breath, and <i>...</i> for the existing words the speaker stresses. "
        "Place each cue at the phrase it affects. Choose from: "
        "<pause>, <long pause>, <breath>, <inhale>, <exhale>, <catches breath>, <deep breath>, "
        "<laughs>, <chuckle>, <sighs>, <uh>, <stutter>, <gasp>, <coughs>, <clears throat>, <sniff>, "
        "<pant>, <pants>, <softer>, <mhm>, <phew>, and <smacks lips></smacks lips>. "
        "For emphasis wrap only 1–4 existing words in <i>...</i>; use <whisper>...</whisper> or <humming>...</humming> for an explicitly requested delivery. "
        "Preserve the supplied spoken words and their order; tags guide performance without adding, repeating or rewriting dialogue. "
        "Preserve existing delivery tags. Keep pauses and breathing within the scene's available duration. "
        "Describe the matching expression and acting in the shot prose, with clear subject ownership. "
        "This inline vocal-delivery policy applies only to generated built-in speech; supplied audio keeps its existing vocal delivery unchanged."
    ) if native_speech else ""
    if not has_emotion_expression_input(payload):
        return speech_instruction
    preset = str(payload.get("facial_performance") or "natural").replace("_", " ")
    custom = (str(payload.get("facial_performance_custom") or "").strip()
              if payload.get("facial_performance") in (None, "", "custom") else "")
    tags = str(payload.get("emotion_expression_tags") or "").strip()
    lyric = str(payload.get("lyric_text") or "").strip()
    return (
        "EMOTION / EXPRESSION — AUTHOR'S ACTING DIRECTION:\n"
        f"Facial preset: {preset}\nCustom facial direction: {custom or '(none)'}\n"
        f"Scene emotion/expression input: {tags or '(inherit facial direction)'}\n"
        f"Lyrics available for interpretation:\n{lyric or '(no vocal line assigned)'}\n\n"
        "Interpret the lyric's meaning, scene action, and the requested emotion together. Explicit scene "
        "emotion input takes priority over the preset; Custom facial text such as Angry is also an emotion "
        "request. Choose fitting bracket descriptors such as [angry, singing] and write specific visible "
        "expressions and acting that suit THIS lyric, rather than copying a stock phrase or the raw input. "
        "The requested emotion remains authoritative: a sad lyric with Angry can be bitter or confrontational. "
        "For 'start happy, then end sad', stage a readable progression with the appropriate tags at each "
        "phase; never flatten the request into one simultaneous [happy, sad] state. "
        "Keep each singer's expressions attached to their assigned cue and subject. Instrumental/B-roll "
        "shots may have visual emotion but never singing tags or vocal performance. Scenes without people "
        "use atmosphere and physical visuals rather than inventing a face. "
        "For speaking scenes use speech emotion descriptors instead of singing descriptors. "
        "For supplied audio, describe visible acting and preserve the existing vocals; do not request "
        "changes to the vocal delivery. "
        "For multi-shot prompts, never describe mouth, lip, or jaw movement. For one shot, articulation "
        "is permitted only while vocals are audible. When lyric omission is enabled, keep the lyrics as "
        "interpretation context only: use [emotion, singing] next to performance synchronized with <Audio 1>, "
        "without lyric quotations or <d> tags. When lyrics are included in a single H3 shot, retain the exact "
        "words and language descriptor and add emotion descriptors inside its cue, for example "
        "<d>[English, angry, singing] exact supplied words.</d>. Choose the actual lyric language. "
        "Write the emotion tags and acting inside the shot prose; never output this instruction block."
    ) + speech_instruction
