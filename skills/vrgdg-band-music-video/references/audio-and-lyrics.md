# Audio, lyric timing, voices and the project track

## 1. Why each step exists

| Step | Tool | Why |
|---|---|---|
| Split the song | `split_stems.py` (Demucs htdemucs_ft) | H3 lip-syncs any visible face to any voice in its audio. Scenes without the singer render against a vocal-free copy, so bandmates and extras never mouth her lines. The final stitch still uses the real song. |
| Time every word | `elevenlabs.py stt` on the **vocal stem**, then `align_lyrics.py` | Per-shot lyric cues need word-level times. Scribe on the isolated vocals is more accurate than on the full mix. |
| Map vocal activity | `vocal_activity.py` | Scribe cuts held notes short and misses backing vocals. The stem's level shows where singing really is, for cuts and for the instrumental scenes. |
| Spoken intro and outro | `elevenlabs.py voices / tts / sfx / check`, then `build_track.py` | Extra story moments with no music. Eleven v4 voices plus ElevenLabs sound effects, mixed into one project track. |

## 2. Stems (Demucs)

```
"<ComfyUI>/python_embeded/python.exe" scripts/split_stems.py "<work>/source/song.wav" "<work>/audio/stems/song_vocals.wav" "<work>/audio/stems/song_no_vocals.wav"
```
- **Which Python:** run it with ComfyUI's embedded Python, which has demucs, torch and soundfile. The `demucs` command
  line can't save files there (torchaudio's FFmpeg backend fails), so the script calls the model directly.
- **Speed:** about 15 s on an RTX 5090, using the GPU. Run it only while ComfyUI is idle.
- **Output:** two 16-bit WAVs at the song's own sample rate and length, so the full tracks line up exactly.

## 3. Lyric timing

```
python scripts/elevenlabs.py --work W stt audio/stems/song_vocals.wav audio/stt/vocals_scribe.json
python scripts/align_lyrics.py --work W --words audio/stt/vocals_scribe.json        -> lyrics_timed.json
python scripts/vocal_activity.py audio/stems/song_vocals.wav --offset 16 [--detail 48 56]
```
- **Model:** Scribe `scribe_v2` with word timestamps. A 2:19 song costs a few credits and takes about a minute.
- **The aligner:**
  - keeps the official lyric spelling: Scribe heard "weight" for "wait" and "raw" for "rude";
  - interpolates any words Scribe missed;
  - lists sung words that aren't in the lyrics file. Tonight that was the first chorus's repeated "I am never going
    back". Add real repeats to lyrics.txt and run it again, because a singer on screen needs words for every sung
    moment.
- **Times are full-track times:** the song time plus intro_seconds. Everything downstream (timeline.json,
  shot_lyrics.py, the cue check) uses full-track times.
- **Check gaps before cutting in them.** The first "back" in "I am never going back" is held for ~5 s; the
  transcript showed a gap there, but the stem's level shows singing.

## 4. ElevenLabs: the account and its rules

- **The key:** config `elevenlabs_key_file` and `elevenlabs_key_index`. The user's file has two keys and the user
  said to use the second one (index 1). Never print the key.
- **Speech model:** Eleven v4 (`eleven_v4`) for all speech. The user asked for the newest model; check
  https://elevenlabs.io/docs/models for a newer one.
- **TLS:** use urllib with `ssl.create_default_context()`. requests/certifi in ComfyUI's Python fail TLS
  verification on this machine.
- **Starter tier limits:**
  - 10 voice slots, all in use. Never delete or replace the user's voices.
  - Use **Voice Library voices by `voice_id` directly**: Eleven v4 speaks them without adding them to the account.
    Find them with `elevenlabs.py voices --accent british --gender male --age young` or `--search cockney`.
  - `mp3_44100_192` is refused (HTTP 403 "only available on the Creator tier"), so use `mp3_44100_128`.
- **Check the account first:** `elevenlabs.py test` shows the tier, characters left and voice slots. The whole
  FUCK THE STATIC project (9 lines, 4 effects, the Scribe runs, the checks and retries) used about 1,100 characters.

## 5. Writing and checking the spoken lines

- **lines.json** (assets/lines_example.json) holds `voices` (who → voice_id and name) and `lines` (key, who, text).
- **Line length:** one short line per shot, under 3 s each. Fit the line to the character, and make the outro pay
  off the story (the boots in the wardrobe, the table splinters, "Same time next week?").
- **v4 bracket tags such as `[chuckles]`, `[laughs]`, `[smirking]`, `[shouting]`, `[grinning]`** are performed as
  sounds, not read aloud. Scribe confirmed this.
- **Check every take:** `elevenlabs.py check audio/voice/*.mp3` transcribes each one.
  - **Misheard words:** Scribe heard "Tenner" as "Tennis", so the line became "Ten quid says we get thrown out in an
    hour". Rewrite a line until it transcribes correctly, because H3's lip-sync and the viewer both need clear words.
  - **Where speech starts and ends:** use the printed times for the cue windows in the prompt.
- **Accents tonight:**
  - the singer: a Cockney female (Lizzie);
  - the bandmates: young London and East London males (Harry, Ali, Prince);
  - the police officer: a rough Cockney (Eastend Steve).
  - One voice per character, never shared.

## 6. Sound effects

- **sfx.json** (assets/sfx_example.json): key, text, seconds, influence.
- **Effects tonight:** a night street outside a house party (16 s), a door bursting open onto the party roar (4 s), a
  dawn rooftop (18 s), and a group bursting into laughter (4 s).
- **Check them too** with `elevenlabs.py check`. It should show only event tags such as [cheering] or
  [birds chirping], with no words. Speech inside ambience makes H3 animate mouths.
- **Loudness when mixed:**
  - ambience beds −27 to −30 LUFS under the dialogue;
  - one-shot effects (the door) about −15;
  - laughter about −17.

## 7. Building the project track

```
python scripts/build_track.py --work W          (reads track_plan.json)
```
- **track_plan.json** (assets/track_plan_example.json) sets the intro and outro lengths and places items at times,
  with trims and fades.
- **Voice levels:** voices sit at −16 LUFS, matching the −16 LUFS song, so the intro doesn't jump in level against
  the music.
- **Outputs:**
  - `audio/full_track.wav`: the intro, the song untouched, then the outro. This is the project audio and the final
    soundtrack.
  - `audio/full_track_no_vocals.wav`: the same with the vocal-free song. Renders for `inst` scenes use it.
- **Set intro_seconds** in config.json to the same value as the plan, so the lyric offsets match.
- **Verified:** rebuilding tonight's track from the plan matched the original sample for sample.
