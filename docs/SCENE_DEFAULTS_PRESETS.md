# Scene Defaults presets

The toolbar above the four settings cards provides independently composable Speed and Music style presets. Selecting a preset changes the actual controls and existing default state immediately, with one persistence notification. Every control stays editable. Opening the panel or loading the saved-preset list does not apply a preset automatically.

## Speed

All values are on the existing 0–10 scales.

| Speed | Camera motion | Character motion | Cut frequency |
| --- | --- | --- | --- |
| Static | 0 | 0 | 0 (continuous shot) |
| Slow | 2 | 2 | 1 |
| Medium | 5 | 5 | 4 |
| Fast | 8 | 8 | 7 |

## Music style

Music style changes framing, performance and facial direction, and clears prior custom facial text. It preserves the chosen speed and cut frequency.

| Music style | Auto camera flow | Performance | Facial performance |
| --- | --- | --- | --- |
| Rock | Music video shots | Rock / punk | Rock / intense |
| Metal | Fast energetic | Metal / screaming | Metal / rage |
| Rap | Medium shots | Rap / hip-hop | Rap / high intensity |
| Soft | Medium shots | Ballad / emotional | Love song / tender |
| Pop | Music video shots | Pop performance | Pop / polished stage |

Film planning modes use their supported acting and facial presets: Rock → Punk bar attitude / Defiant controlled; Metal → Tense confrontation / Defiant controlled; Rap → Noir restraint / Defiant controlled; Soft → Emotional confession / Vulnerable confession; Pop → Dialogue naturalism / Default natural. Framing remains as above.

Applying any built-in or named preset sets **video aesthetic to Default, temporal/world effect to Off, and shot FX to Off**. These can be edited afterward. Manually editing a preset's component controls clears its toolbar selection so the label cannot imply those settings still match.

## Named presets

Enter a name and choose **Save preset**. Select it from **Saved presets** and choose **Load preset** to restore it. Existing names require confirmation before replacement; names that sanitize to the same filename cannot overwrite one another accidentally.

Snapshots include the camera flow (including a custom camera list), both motion speeds, cut frequency, performance/facial settings and custom facial text, consistency phrase, story arc detail, and Image Prep settings. The effect/aesthetic baseline above is enforced when saving and loading. Inactive custom video aesthetic and effect configuration are not part of the snapshot. Project metadata, scenes, reference assets and generated prompts are excluded.

Files are saved globally beside the existing video profiles:

`<ComfyUI output>/VRGDG_Video_Profiles/scene_defaults/<name>.json`

Existing scene overrides are preserved. Use the existing **Fill Missing** or **Replace All** controls when applying defaults to scene fields. Prompts update on their usual generation path; selecting or loading a preset does not regenerate them.

When scene cards are checked, the rows with existing apply actions also show **Apply to selected**: Image aesthetic, Video aesthetic, Still shot flow, Auto camera flow, Performance style, and Facial performance (including its custom text). This replaces that setting on the checked scenes only. Clearing selection hides the extra buttons. Fill Missing and Replace All retain their all-scene scope. All Scene Defaults action buttons use compact padding and text sizes.

Selection uses the existing scene-card checkboxes, including Select All for visible scenes. Flow application retains original timeline indexes. Empty or stale selection never falls back to applying all scenes. Explicit scene video aesthetics and their custom wording take precedence over the global default, and remain editable in the scene editor. Settings without an existing per-scene apply action retain their global behavior; story arc detail remains a whole-story control.

Implementation lives in `web/storyboard_builder/defaults_presets.mjs` (UI/mappings), `builder/scene_defaults_presets.py` (atomic storage/validation), and `builder/routes.py` (thin REST handlers). These are Builder routes, outside the Agent API v1 export catalog; route contracts are recorded in `AGENT_GUIDE.md`.
