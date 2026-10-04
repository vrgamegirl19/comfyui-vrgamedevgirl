---
name: vrgdg-h3-film
description: Make a complete AI short film or narrative video in ComfyUI with the VRGDG Video Builder and MiniMax H3 (built-in voices and sound), Z-Image reference images, a MiniMax Music 3 or YuE2 score, automatic QA and a final edit. Use when the user asks Claude to create, render, fix or assemble a short film, story video or dialogue scene with the VRGDG nodes or Video Builder.
---

# VRGDG Video Builder + MiniMax H3 short-film pipeline

This skill drives the user's own ComfyUI through HTTP: the Video Builder's routes for images, H3 scenes, trims and
stitching, plus core nodes for music. Every script is in `scripts/` next to this file; run them with any Python 3.9+
(the QA and title scripts need the Python that check_env.py reports as `qa_python`, usually ComfyUI's own).

## Ground rules

- **Read-only:** treat the ComfyUI install and every custom node as read-only. Read, understand, call and use them,
  but never edit, patch, update, reinstall or delete anything. Put all new files in the film's project folder.
- **Installs only with the user's OK:** never download models or install packages on your own. If check_env.py
  reports something missing:
  1. Run `python scripts/install_missing.py --comfy-root "<ComfyUI folder>"` (report only; it changes nothing).
  2. Show the user what's missing, the sizes and where each file goes.
  3. Only after the user clearly agrees, re-run it with `--yes`. Add `--groups h3,zimage,music3` for the groups they
     chose, `--nodes` for missing custom-node packs, and `--whisper` for the dialogue checks.
  4. If node packs were installed, ask the user to restart ComfyUI, then run check_env.py again.

  The installer only adds missing files and folders; it never overwrites, updates or deletes anything.
- **One fix at a time:** when the user reviews the film, fix one note at a time and show the result, unless they ask
  for autonomous work.
- **ComfyUI only needs to be running:** the server console is enough; the browser page isn't needed. If a scene
  logs "was interrupted in ComfyUI", something cancelled it. Re-run it; it isn't a model failure.

## Workflow

1. **Check the install.**
   `python scripts/check_env.py --comfy-root "<ComfyUI folder>" [--yue2-root "<YuE2 folder>"]`
   - Ask the user where ComfyUI is if you can't find it.
   - Only pass `--yue2-root` if they installed YuE2 with the VRGDG YuE2 Installer node.
   - It picks the H3, Z-Image and Music 3 models, the music engine (`music3`, `yue2` or `none`), the QA Python and
     ffmpeg. Fix every PROBLEM line before going on, and tell the user about the warnings.
2. **Create the project.** `python scripts/new_project.py "<project folder>"`
   This writes `settings.json` plus starter `screenplay.json`, `image_specs.json` and `score.json` files. Read the
   starter screenplay; it shows every feature.
3. **Write the story and the screenplay,** following the rules below. Keep scenes at most 8 s, with at most 3 people
   per shot.
4. **Make reference images.** `python scripts/gen_refs.py "<project>"` makes one image per seed. View them, pick
   the best, and copy each pick to the `REF_*.png` path that the screenplay names. Character references go on a plain
   grey studio backdrop, knees up and facing the camera.
5. **Build and lint the prompts.** `python scripts/build_prompts.py "<project>"`
   Every PROBLEM must be fixed and the lint must pass before rendering. Run it without piping, so a failure stops
   you. Take the warnings seriously.
6. **Render a test first.** `python scripts/render_scenes.py "<project>" 1 3` for 2–4 representative scenes. Each
   takes about 7–10 minutes on a 32 GB GPU. Run long batches in the background and watch for `scene N: saved` /
   `ERROR` lines in `logs/render.log`.
7. **QA every take:** `python scripts/qa.py "<project>" <n> ...`, run with `qa_python`. It checks:
   - the lines against the script, using Whisper;
   - spoken tag words;
   - each voice's sex, from pitch;
   - lines running into the end of the clip;
   - an 8-frame review sheet.

   View the review sheet image every time, and fix what you see (next section).
   - **Fix every flaw, including small ones.** Re-roll any take with a visible flaw: a stray prop (an object from the
     dialogue showing up in the background, say), a character looking into the lens instead of at the listener, a prop
     in the wrong place, the wrong expression. Never accept a take as "minor" and list it in the report. Keep
     re-rolling (each time with a prompt change) until the take is clean. Only after 3 failed attempts on the same
     flaw, tell the user what you tried and ask before going on.
8. **Render the rest,** then check every scene-to-scene cut: `python scripts/cut_sheet.py "<project>"`. Compare
   positions, props and who is present.
9. **Make the score.** Edit `score.json`, then `python scripts/gen_music.py "<project>"`. Check the cues for
   accidental vocals with `qa.py --music`. With no music engine, the user can drop their own audio in `audio/score/`.
10. **Assemble:** `python scripts/assemble.py "<project>"`. It does the Builder stitch for the picture, then a ducked
    score, title and end card, mastered to -14 LUFS.
    - **Dialogue track:** the script builds it sample-accurately from each clip's own audio. The stitch's audio
      drifts about 5 ms per scene.
    - **Sync check:** it ends with "sync check: worst audio offset N ms". Anything over 40 ms means a clip's audio is
      wrong; fix it before delivering.
    - **Verify the final's audio sync yourself; never deliver on the script's word alone.** After every
      re-stitch, run `python scripts/sync_check.py "<project>"`. For every scene, it finds that clip's own audio
      inside the final film by cross-correlation. It prints each scene's offset in ms and fails on any scene over
      40 ms or with a weak match. Fix any failure before delivering, and report the worst offset.
    - **Reporting:** watch the result as frames and report honestly. The report should contain zero known visual
      flaws: anything you noticed should already be fixed (see step 7).
    - **Partial exports:** when the user asks for part of the film (for example from scene N for their own editor), cut
      it from this final, never from an older export.
11. **Social media kit:** `python scripts/social_kit.py "<project>"` creates `SocialMediaSharing/` with
    `poster_refs/` (each cast member's reference image, numbered in the order to attach them) and `facts.json` (scenes,
    runtime, render hours, re-shoot notes). Then write these text files into `SocialMediaSharing/`:
    - `poster_prompt.txt`: a text-to-image prompt for ChatGPT image generation that uses the `poster_refs` images.
      - **Short film:** a 16:9 theatrical movie poster in the film's genre style.
      - **Music video:** 16:9 single or album key art suited to a YouTube thumbnail.
      - **Contents:** name each attached image by number with that character's look, then composition, style, and the
        exact poster text (title, a tagline, "A FILM BY CLAUDE" or the artist credit). End it with "Use only the text
        listed above."
    - `youtube_description.txt`: a title line, a one-line hook, a short synopsis, then a ━━━ divider and three
      sections. 🎬 HOW THIS WAS MADE splits into "▶ MY PART" (the user's brief and tools) and "▶ WHAT CLAUDE DID ON ITS
      OWN" (story, characters, screenplay, rendering hours, QA, the real re-shoots from facts.json, score and edit).
      Then 🛠 TOOLS and a CAST line. End with the user's links: the GitHub node pack
      (https://github.com/vrgamegirl19/comfyui-vrgamedevgirl) and the skill download if they gave one.
    - `reddit_post.txt`: the same content as **plain text for Reddit's rich text editor**: emoji section headings,
      "•" bullets, bare URLs, and no Markdown symbols.
    - `short_post.txt`: a version that works on every platform. It starts with a post of at most 280 characters for X
      (hook, link placeholder, 2–4 hashtags), followed by a 2–3 sentence version for Instagram, TikTok, Facebook and
      Threads.
    - **Facts:** use facts.json and the screenplay. Never invent numbers, and never describe the user's references as
      real people or pets unless the user said so.

## Prompt rules (the lint enforces the hard ones)

- **Positive-only.** Never write no, not, never, without or nothing outside the dialogue, because H3 renders whatever
  is named. Describe only what IS on screen: "her gaze rests on the window", never "she isn't looking at the camera".
- **Describe only what is in frame.** Describing an off-screen thing (an AI's glowing screens, a phone in another
  room) makes H3 put it in the shot. Use a text-only location (`"ref": null`) when a plate's layout fights the shot.
- **Voices:**
  - one seed for every scene;
  - one voice description per character, pasted word for word everywhere (`voices` in the screenplay);
  - at most one male and one female speaker per scene, or H3 mixes their voices;
  - a voice-only character (radio, phone, AI, narrator) goes in `voice_only`.
  - **Voice-only lines get lip-synced by whoever holds the source.** With a radio strapped to a character's chest, H3
    makes that character mouth the broadcast. While it plays, frame the speaker grille in close-up, or shoot the
    listeners from behind.
  - **Repeating a broadcast:** for an exact repeat later, render the later scene without the line and lay in the first
    recording through a radio filter (band-pass about 400–3200 Hz plus static). Keep "static" and "crackle" out of a
    silent scene's sound line, because H3 hears a voice in them.
  - **Hand-edited audio** must stay at H3's 32 kHz. assemble.py conforms mismatched clips, but CapCut exports and
    previews should match too.
- **Male voices flip female** when the delivery says soft, gentle, sweet, whisper or lullaby, when only a woman is
  on screen, or when the line runs past a cut to a woman's face. Use "deep, low" wording, show the voice's source in
  frame, and keep the whole line inside one shot.
- **Tags:** `<pause>` and `<softer>` are spoken aloud, and build_prompts converts or removes them.
  `<catches breath>`, `<pants>` and `</whisper>` are also spoken, so the lint rejects them, and a bare "..." gets
  invented words. Use `<breath>` for a pause; `<sighs>`, `<chuckle>`, `<laughs>`, `<gasp>` and `<whisper>` work.
- **Silent people on screen:** each one needs their own sentence, by name: `"{tom}'s lips stay pressed together."`
  A generic "everyone is silent" is ignored.
- **Eyelines:** the speaker looks at the listener, framed from the side or over the shoulder. Give glance directions
  as frame directions ("toward the right edge of the frame"). Save looking into the lens for deliberate direct address.
- **Crowds:** at most 3 described extras in focus, with the rest as soft out-of-focus silhouettes. Otherwise H3
  fills the crowd with cloned faces.
- **Framing:** use medium shots and close-ups whenever someone speaks; use wide shots for silent moments.
- **Leave room after the last line.** H3 stretches dialogue to fill the clip, so the last word lands in the final
  frames and gets cut.
  - build_prompts adds "the last spoken word ends by …, then a silent beat", `tail_hold_seconds` (default 1.0) before
    the end.
  - The lint estimates when the lines finish, at about 2.6 words/s plus 0.4 s per `<breath>`. It warns when the last
    line runs into the hold, and fails when it can't fit at all.
  - Fix it by shortening the line, cutting to it earlier, or lengthening the scene. Plan about 1 s of silence after
    every scene's last word.
  - render_scenes keeps the raw take up to its last frame and only removes the warm-up from the front.
  - build_prompts also adds a `<breath>` after each scene's last line. The breath is padding: if anything is cut, it's
    the breath, not a word.
- **Text:** H3 garbles text. Keep signs, plaques and screens free of readable words ("a smooth, unmarked brass
  plaque").
- **Silent scenes:** keep voice words ("murmur", "rasp of a voice", "whispers") out of the `sound` line, or H3 invents
  gibberish speech.
- **Two-shots:** a person can appear twice. Block them explicitly: "{a} on the left of the frame and {b} on the right,
  just the two of them".
- **Close-ups:** name the environment behind the face (lights, props, weather), or the grey studio backdrop of the
  reference can leak in.

## Fixing takes

- **Re-rolls need a prompt change.** With the fixed seed, the same prompt renders the same take, so reword the shot
  text. Don't change the seed, because the seed holds the voices.
- **Never touch anything that defines a voice.** The `voices` text is a voice ID, pasted verbatim. Never edit it, and
  never paraphrase or add timbre words to it in the delivery (`{say:key|how|...}`): no "his deep gravelly baritone"
  or similar. Re-roll a take by changing the shot description (framing, action, blocking) and leave the delivery as
  first written.
- **If a voice drifts** (pitch well off that character's other scenes, measured with pyin), change only the shot
  description and re-roll. Changing the shot itself can move the voice too, so check the pitch on every re-roll.
- **Misheard word:** if Whisper hears the wrong word (check with a second Whisper model before blaming H3), simplify
  the line, or add a `<breath>` before the troublesome phrase.
- **Line cut off at the end:** `qa.py` flags it. Re-trim from the raw take so the clip ends on the take's last frame:
  `python scripts/retrim.py "<project>" <n> <start_s> [duration_s]`. Never trim into the end of a line; remove dead
  air at a shot cut instead.
- **Stray sound at the start of a silent shot:** re-trim with a later start. If the picture is good but the speech
  runs through it, replace the scene's audio with ambience made by ffmpeg (keep the original in
  `video_clips/rejected/`).
- **Whisper hallucinations:** large-v3 often hears "Thank you" or "Thanks for watching" in silence. Confirm with a
  second model and its no_speech_prob before re-rolling.
- **Shouted male lines** can pitch above 165 Hz. Before re-rolling, check them by ear or with a speaker-embedding
  comparison against the same character's confirmed lines.
- **Continuity between scenes:** a character jumping position or vanishing across a cut is fixed with Builder latent
  continuation.
  1. Set `"continuity": "exact_frame"` on scene n. It always continues from scene n-1.
  2. Run `python scripts/continuity_frame.py "<project>" <n-1>` after any re-trim of n-1.
  3. Start scene n's action from rest (a sudden first move lurches) and open its first line with `<breath>`, because
     the soundtrack starts inside the hidden warm-up.
- **Render runner stopped but ComfyUI finished:** `python scripts/adopt_render.py "<project>" <n>`.
- **Slow renders** (over about 15 minutes): the GPU is probably full. `render_scenes.py` frees memory after each
  scene. Don't run Whisper on the GPU while a scene renders; use `--cpu` on qa.py.
- **Log every re-roll** and its reason in `logs/reroll_list.txt`. Replaced takes go to `video_clips/rejected/`.

## Screenplay format (screenplay.json)

`cast` (with a reference image each), `voice_only`, `voices`, `sex` ("M" or "F"), `locations` (`ref` may be null) and
`styles` (a "default" plus any extras), then `scenes`. Each scene has:

```json
{"loc": "kitchen", "cast": ["anna", "tom"], "dur": 8.0, "summary": "...", "sound": "room tone and effects",
 "style": "default", "continuity": "", "shots": [[0, "Shot text with {anna} ... {say:anna|says warmly, looking at him|Line. <breath> More.}"],
                                                 [4.5, "Close-up of {tom} ... {say:tom|says quietly|Reply.}"]]}
```

The number in each shot is its cut time in seconds. Scenes play in array order, and scene numbers are their positions
(1-based). Don't reorder scenes after rendering: takes, re-trims and latents are filed by number, so append new
scenes at the end and move them in the edit with `play_order` in score.json if needed.

## Score (score.json)

- `cues`: each has a caption, a structure in `lyrics` (`[Intro]\n[Instrumental]...` for instrumentals), seconds and
  2 seeds. Generate both seeds and pick one.
- `placement`: maps a cue file to scenes (`from_scene`, `to_scene`, `offset`, fades). Leave the big reveal or twist
  on room tone; silence lands harder than music.
- **YuE2** is a song model (the VRGDG YuE2 nodes run it in its own isolated install). Use it for a credits song or
  theme with real lyrics, and always check its "instrumentals" for vocals.
  - A song's length follows its lyrics, not `seconds`.
  - About 4–7 minutes per song.
  - It sometimes skips a lyric line, so give the cue 3 seeds and pick the take that sings every line (check with
    Whisper).
  - When a user has both engines, `python scripts/gen_music.py <project> <cue> --engine yue2` picks YuE2 for that
    cue.
