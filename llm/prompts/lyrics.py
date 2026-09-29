"""Lyric prompts: batch repair of Whisper transcript segments."""


_WHISPER_BATCH_REPAIR_INSTRUCTIONS = r"""Repair a small batch of Whisper lyric segments by aligning them to a nearby real lyric window.

INPUTS
You will receive:
1. TARGET_WHISPER_SEGMENTS: the exact segment keys to repair.
2. REAL_LYRIC_WINDOW: nearby real lyrics in song order. Use only these lyric words for corrections.
3. PREVIOUS_REPAIRED_CONTEXT: already repaired segments just before this batch, for continuity only.

TASK
Return one corrected value for each TARGET_WHISPER_SEGMENTS key.

STRICT RULES
- Use only words from REAL_LYRIC_WINDOW for sung lyrics.
- You may fix punctuation, capitalization, spacing, and obvious partial-word breaks.
- Do not invent new lyrics.
- Do not use visual story, style/theme, mood, color, or setting words.
- If a Whisper segment is filler such as "Thank you." and no lyric from REAL_LYRIC_WINDOW belongs there, output "[instrumental]".
- INTRO FILLER EXCEPTION: When a short Whisper segment (1-3 words) at the very start of the song contains the beginning of a longer sung line whose remaining words appear in the next segment, that short segment is Whisper picking up the leading edge of the first vocal entry during a silent or instrumental intro. Output "[instrumental]" for it, even though its individual words appear in REAL_LYRIC_WINDOW. The test: would the full line in REAL_LYRIC_WINDOW still make sense if these few words moved to the start of the next segment? If yes, the short segment is intro filler.
- OUTRO FILLER EXCEPTION: Same pattern at the end of the song. A short trailing fragment (1-3 words) that duplicates the last few words of the previous segment's sung phrase, or appears after the last full sung line, is outro filler. Output "[instrumental]".
- If a filler segment clearly sits where a lyric from REAL_LYRIC_WINDOW belongs, use the matching real lyric words.
- Keep the segment count and key names exactly the same as TARGET_WHISPER_SEGMENTS.
- Keep the batch in song order. Do not jump backward to earlier lyrics outside REAL_LYRIC_WINDOW.
- Do not copy the whole lyric window into every segment.

OUTPUT
Return valid JSON only.
No markdown.
No explanation.
Use double quotes.
No trailing commas.
No line breaks inside string values.

FORMAT
{
  "segment10": "corrected lyric chunk",
  "segment11": "corrected lyric chunk"
}"""
