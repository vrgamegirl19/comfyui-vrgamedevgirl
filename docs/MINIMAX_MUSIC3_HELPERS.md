# MiniMax Music3 tuning helpers

This package adds practical controls around ComfyUI's built-in MiniMax Music3 nodes. The helpers make its tuning controls easier to describe, repeat, compare, and combine.

## Included workflow

Import `Workflows/MiniMax_Music3_Advanced_Tuning_VRGDG.json` after restarting ComfyUI. The workflow is prefilled with the local Music3 model filenames and includes notes directly on the canvas.

The optional two-plan blend branch is disabled by default. It will not add a second AR-generation pass unless you explicitly enable and connect it.

## The four tuning stages

1. **Caption and lyrics** determine the requested genre, voice qualities, instruments, production, tempo, emotional arc, and song sections.
2. **AR `cfg_scale` and `top_k`** affect creation of Music3's hidden acoustic plan. `cfg_scale` changes adherence; `top_k` changes token-choice diversity.
3. **KSampler CFG and steps** affect diffusion rendering of the completed acoustic plan.
4. **Conditioning Strength** scales Music3's model-specific `conditioning_scale` metadata. `1.0` is native behavior. Change this gently because excessive strength can produce harshness, static, or collapse.

Keep the same AR and sampler seed when comparing settings. Change one control family at a time.

## Helper nodes

### VRGDG Music3 Caption Builder

Builds a consistent, readable Music3 caption from a preset and editable fields. Selecting a preset visibly fills the singer/name phrase, style label, genre, mood, instruments, vocal qualities, production, tempo, song arc, and avoidance fields. Every populated field remains editable afterward. Selecting `Custom / fields only` preserves the current values so they can be used as a starting point.

The separate Vocal Profile dropdown automatically updates the singer/name phrase and vocal-qualities fields. It includes general female and male leads, alto/mezzo, soprano, baritone, tenor, rock/rasp, soft/intimate, androgynous/neutral, female-and-male duet, style-default, and custom options.

The preset library includes alternative rock, singer-songwriter, post-grunge, indie rock, modern country, outlaw country/Americana, country rock, classic rock, hard rock, punk, heavy metal, alternative metal, R&B/soul, contemporary R&B, neo-soul, hip-hop/rap, boom-bap, melodic rap, trap, modern pop, pop rock, indie pop, folk/acoustic, blues rock, funk, and synthwave. `Custom / fields only` preserves the currently displayed fields for manual editing.

### VRGDG Music3 Tuning Presets

Outputs coordinated values for AR guidance, AR diversity, diffusion guidance, steps, and conditioning strength. These presets describe generation behavior and remain independent from the Caption Builder's musical style:

Hover over the preset control to see the selected preset's expected sound tendency and tradeoffs. The `notes` output also includes that description followed by every exact numerical value and what it controls.

| Preset | AR CFG | top_k | Sampler CFG | Steps | Cond. strength |
|---|---:|---:|---:|---:|---:|
| Balanced / Built-in Baseline | 1.50 | 50 | 1.70 | 30 | 1.00 |
| Balanced Quality | 1.60 | 45 | 1.70 | 36 | 1.00 |
| Raw Live Rock | 1.70 | 40 | 1.80 | 36 | 1.10 |
| Lyrics + Structure Focus | 2.00 | 24 | 1.75 | 36 | 1.05 |
| Creative Variations | 1.15 | 100 | 1.40 | 28 | 0.90 |
| Intimate + Dry | 1.80 | 30 | 1.60 | 34 | 1.05 |
| Aggressive + Dense | 1.55 | 70 | 1.90 | 40 | 1.15 |
| Conservative / Stable | 1.50 | 30 | 1.45 | 36 | 0.90 |
| Strict Lyrics | 2.10 | 20 | 1.75 | 36 | 1.05 |
| Strong Song Structure | 1.95 | 28 | 1.80 | 40 | 1.10 |
| Creative Melody | 1.20 | 90 | 1.50 | 32 | 0.95 |
| Maximum Arrangement Variation | 1.05 | 120 | 1.35 | 30 | 0.85 |
| Stable / Low Artifact | 1.50 | 30 | 1.45 | 38 | 0.90 |
| Strong Acoustic Plan | 1.70 | 40 | 1.85 | 38 | 1.20 |
| Loose Acoustic Plan | 1.25 | 80 | 1.40 | 30 | 0.80 |
| Detailed Rendering | 1.50 | 50 | 1.75 | 48 | 1.00 |
| Long-Song Stability | 1.80 | 32 | 1.60 | 44 | 1.00 |
| Fast Draft | 1.50 | 50 | 1.50 | 20 | 0.95 |
| High-Energy / Dense | 1.60 | 65 | 1.95 | 42 | 1.15 |
| Soft / Restrained | 1.70 | 35 | 1.45 | 36 | 0.95 |

### VRGDG Music3 Lyrics Prepare + Validate

Normalizes recognized section labels to square brackets and strips accidental metadata lines. Numbered labels such as `[Verse 1]` and `[Verse 2]` are valid and useful. Recognized tags include Intro, Verse, Pre-Chorus, Chorus, Post-Chorus, Bridge, Refrain, Hook, Break, Breakdown, Interlude, Instrumental, Solo, and Outro.

Unknown bracketed phrases are preserved but reported because the model may try to sing them. Put genre, production, and vocal descriptions in the caption instead of the lyrics. `[Final Chorus]` is also recognized.

### VRGDG Music3 Conditioning Strength

Scales the Music3-specific `conditioning_scale` stored with the acoustic conditioning. It does not multiply the hidden tensor itself. Start at `1.0`, use increments of `0.05`, and normally remain near `0.85–1.25`.

### VRGDG Music3 Seed Bank

Produces four deterministic seed choices. Seed A is connected to both the AR planner and KSampler in the included workflow. That makes parameter comparisons repeatable.

### VRGDG Music3 Conditioning Blend (Experimental)

Linearly blends two hidden acoustic plans before diffusion. Use the same lyrics, maximum duration, and preferably the same seed for both plans. This is not audio mixing. Incompatible plans can become incoherent, so begin around `blend_b = 0.15–0.35`.

## Practical test order

1. Generate the Balanced baseline with a fixed seed.
2. Compare Creative Melody using that same seed.
3. Edit only the caption's production and vocal language.
4. If lyrics drift, try Strict Lyrics; if sections drift, try Strong Song Structure.
5. If the result is too rigid, raise `top_k` or use Creative Melody.
6. Only then adjust Conditioning Strength in `0.05` steps.

If high values make noise, return to the last clean setting. Numerical guidance can improve or worsen adherence, so compare each change against a fixed-seed baseline.
