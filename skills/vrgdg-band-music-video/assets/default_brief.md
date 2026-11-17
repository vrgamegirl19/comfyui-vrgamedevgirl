# Default creative brief (when the user's rules don't say otherwise)

Distilled from the user's own rules for NO SAINT OF YOURS (`examples/rules_no_saint_of_yours.txt`, the long
creative direction) and NO PERMISSION TO EXIST (`examples/rules_no_permission_to_exist.txt`). The user's rules for
the current video always win; this fills the gaps.

## The goal
A cinematic, photorealistic music video that feels like a real band's professionally directed production:
energetic full-band performances alternating with the lead singer's solo journey through other locations. Every
band member gets meaningful screen time, every location is used, every performance matches the music, and the
video feels cohesive, energetic and believable.

## Music first
- Analyse the song, lyrics, structure, energy changes and every stem before planning: intros, verses, choruses,
  breaks, solos, outro.
- Band performance for choruses and heavy instrumental passages; the singer's solo journey for quieter verses,
  pre-choruses and the bridge.
- Use the stems to know exactly when each instrument plays or leads. A musician is shown playing only when their
  instrument is audible; when it is silent, describe what they actually do (hands resting, moving with the
  rhythm, looking at a bandmate). Solos go to the musician whose instrument leads.
- The vocal stem decides when anyone sings: singing faces only on sung words; closed mouths, breathing and
  reactions elsewhere.
- Cut with the rhythm, in vocal gaps.

## The band (band location only)
- Fixed layout, kept in every shot: singer centre, lead guitarist on the left of the frame, bassist on the right,
  drummer at the back centre behind the singer, with realistic live-band spacing.
- Alive and authentic: moving with the rhythm, real playing technique, looks and interaction between players.
- Mix full-band wide shots (that move in, never long and static), medium shots, close-ups of riffs, bass and
  drumming, side angles and moving shots. Each musician gets their own moments.
- Instruments always correct, and the same, on every member.

## The singer's solo journey (every other location)
- She is the only band member outside the band location (extra characters are allowed when they make the
  journey more fun, such as a foil she plays off).
- Use every solo location, spread across the song. A journey, not a literal retelling of the lyrics.
- Sings toward the camera with expressive faces and natural lip-sync; walks, turns, pauses, touches and uses the
  place; a range of expressions that match the song. Physically active, never just standing and singing.
- Varied framing: extreme close-ups, medium, full body (never wider), tracking and orbiting shots.

## Microphone
- Band location: the same black handheld microphone in her hand in every performance shot.
- Solo locations: no microphone; her hands are free for gestures and the place.

## Characters (when Claude makes them)
- Krea 2 two-pass (Ref Builder > Create Character) through the API; three-panel sheet in this order: 1 face
  close-up, 2 upper body, 3 full body; solid grey background; no text, borders or divider lines.
- Ages and looks from the brief (both videos: everyone mid-20s, in the song's genre look). Sonnet checks every sheet.
- Location images for the band location and for each solo location.

## Pipeline defaults
- Settings: `config_example.json` (Singularity V2 25 steps; turbo 8-step 768 v1.1 LoRA, 4 steps, denoise 0.30;
  2k; no continuation).
- Scenes 2-8 s, multi-shot where it fits.
- Word timing: ElevenLabs Scribe when a key is given, otherwise the Video Builder's transcribe
  (`builder_transcribe.py`).
- Unattended: every scene through both passes, no draft review; out-of-memory retries; check ComfyUI keeps
  running; at the end an Opus frame review of every scene and a summary of issues for the user.

## Tone
Match the song. A darker, elegant song gets a moodier journey (NO SAINT OF YOURS). A fun, fast song gets more
energy, comedy and a foil character (NO PERMISSION TO EXIST: "more energy and feel more fun because the song
itself is a bit more fun").
