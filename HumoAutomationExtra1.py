any_typ = "*"

class VRGDG_MusicVideoPromptCreatorJson:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": "The Woman.",
                }),
                "song_theme_style": ("STRING", {
                    "multiline": True,
                    "default": "cinematic realism, emotional storytelling, soft surrealism, naturalistic tone, dreamlike nostalgia, modern drama, poetic symbolism, intimate atmosphere",
                }),
                "pipe_separated_lyrics": ("STRING", {
                    "multiline": True,
                    "default": "line1 | line2 | line3",
                }),
                "word_count_min": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 200,
                }),
                "word_count_max": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                }),

                # Existing modes
                "list_handling_mode": ([
                    "Strict Cycle (use each once, then repeat)",
                    "Reference Guide (LLM creates variations inspired by list)",
                    "Random Selection (pick randomly from list)",
                    "Free Interpretation (LLM can ignore or combine items)"
                ], {
                    "default": "Reference Guide (LLM creates variations inspired by list)"
                }),

                "prompt_structure_mode": ([
                    "Character-Focused (character always leads)",
                    "Environment-Focused (setting always leads)",
                    "Action-Focused (movement always leads)",
                    "Cycle Through Patterns (rotate 3 structures)",
                    "Dynamic Choice (LLM picks best structure per lyric)"
                ], {
                    "default": "Character-Focused (character always leads)"
                }),

                # Existing inputs for main mode
                "environment": ("STRING", {
                    "multiline": True,
                    "default": "open field at dusk, dimly lit bedroom, empty city street at night, forest clearing with morning fog, seaside cliff at golden hour, rainy urban alley, sunlit living room, desert road at sunrise",
                }),
                "lighting": ("STRING", {
                    "multiline": True,
                    "default": "warm amber glow, cool window light, neon reflections, diffused morning light, soft backlight haze, flickering streetlights, gentle afternoon sun, pink-orange dawn light",
                }),
                "camera_motion": ("STRING", {
                    "multiline": True,
                    "default": "zoom in, zoom out, tilt down, rotate around, tilt up, pan, track",
                }),
                "physical_interaction": ("STRING", {
                    "multiline": True,
                    "default": "walking through tall grass, lying on bed staring upward, leaning against a wall in stillness, reaching toward sunlight, hair moving in wind, footsteps in puddles, brushing hand across furniture, standing motionless in breeze",
                }),
                "facial_expression": ("STRING", {
                    "multiline": True,
                    "default": "Intense raw emotion",
                }),
                "shots": ("STRING", {
                    "multiline": True,
                    "default": "Close up, medium, wide angle, over the shoulder, point of view, overhead, ground level",
                }),
                "outfit_rules": ("STRING", {
                    "multiline": True,
                    "default": "a white dress",
                }),
                "character_visibility": ("STRING", {
                    "multiline": True,
                    "default": "mostly visible, half-shadowed, silhouetted, reflected or obscured, seen from behind, partially out of frame, emerging from light, fading into darkness",
                }),

                # ✅ NEW: Story Mode toggle
                "story_mode": ("BOOLEAN", {
                    "default": False,
                    "label": "Advanced Story Mode (Beta)",  # This changes the visible name
                    "tooltip": "Advanced — using Gemini PRO is recommended",  # This adds hover text

                    
                }),

                "signal": (any_typ,),
            },
            "optional": {
                "custom_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "label": "For advanced users only — if you enter text here, this input will completely override the node. No other settings, modes, or parameters will be used. Only your custom instructions will be sent to the LLM.",
                    "tooltip": "Optional: Enter your own custom prompt instructions. If filled, this will override the normal system instructions."
                }), 
                "Summary_File_Path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "label": "Optional Summary JSON (from previous run)",
                    "tooltip": "If provided, this summary will override the theme/style in story mode."
                }),
                "summary_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "label": "Summary index number",
                    "tooltip": "Which summary file to load (e.g. 2 loads summary2.json)"
                }),        
                "total_sets": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "label": "Total number of runs (sets)",
                    "tooltip": "Used to determine if the current run is the last one (which does not require summary output)."
                }),
                "groups_in_last_set": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 16,
                    "label": "Number of Prompts in Final Chapter",
                    "tooltip": "Used in story mode to tell the LLM how many prompts to generate in the final run (e.g., if fewer than 16)."
                }),


                
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated_string",)

    FUNCTION = "build_prompt_instructions"
    CATEGORY = "VRGDG/Prompt Tools"

    def build_prompt_instructions(
        self,
        character_description,
        song_theme_style,
        pipe_separated_lyrics,
        word_count_min,
        word_count_max,
        list_handling_mode,
        prompt_structure_mode,
        environment,
        lighting,
        camera_motion,
        physical_interaction,
        facial_expression,
        shots,
        outfit_rules,
        character_visibility,
        story_mode,
        signal,
        custom_instructions=None,
        Summary_File_Path="",
        summary_index=0,
        total_sets=1,
        groups_in_last_set=16,

    ):
        
        # Custom instruction override
        if custom_instructions and custom_instructions.strip():
            return (custom_instructions.strip(),)
        
        # ============================================================
        # STORY MODE LOGIC
        # ============================================================
        if story_mode:
            summary_data = {}

            print(f"[VRGDG] STORY MODE ENABLED — Total Chapters: {total_sets}, Current Index: {summary_index}")

            # ------------------------------------------------------------
            # Load previous summary (only if index >= 1)
            # ------------------------------------------------------------
            if summary_index > 0 and Summary_File_Path and os.path.isdir(Summary_File_Path):
                summary_file = os.path.join(Summary_File_Path, f"summary{summary_index - 1}.json")
                print(f"[VRGDG] Attempting to load summary from previous chapter ({summary_index}) → {summary_file}")

                if os.path.isfile(summary_file):
                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            summary_data = json.load(f)
                        print(f"[VRGDG] ✅ Successfully loaded previous summary file: summary{summary_index - 1}.json")
                    except Exception as e:
                        print(f"[VRGDG] ❌ ERROR: Failed to read summary file ({summary_file}): {str(e)}")
                        song_theme_style += f"\n(Note: Failed to read summary file: {str(e)})"
                else:
                    print(f"[VRGDG] ⚠️ WARNING: Expected summary file not found at: {summary_file}")
            else:
                if not Summary_File_Path:
                    print("[VRGDG] ⚠️ WARNING: No Summary_File_Path provided.")
                elif not os.path.isdir(Summary_File_Path):
                    print(f"[VRGDG] ⚠️ WARNING: Provided Summary_File_Path is not a directory: {Summary_File_Path}")
                else:
                    print("[VRGDG] ℹ️ INFO: Starting first run (index 0) — using user-defined theme.")

            # ------------------------------------------------------------
            # Override theme if previous summary exists
            # ------------------------------------------------------------
            if summary_data:
                song_theme_style = "\n".join([
                    f"scene_summary: {summary_data.get('scene_summary', '')}",
                    f"character_arc: {summary_data.get('character_arc', '')}",
                    f"narrative_thread: {summary_data.get('narrative_thread', '')}",
                    f"next_scene_suggestion: {summary_data.get('next_scene_suggestion', '')}"
                ]).strip()

                # Add structured chapter context for the LLM
                song_theme_style += f"""

STORY STRUCTURE CONTEXT:
You are working on a multi-chapter cinematic story.
Total chapters: {total_sets}.
Current chapter: {summary_index + 1}.
The section above describes the previous chapter.
Continue the story visually and emotionally into this new chapter.
Evolve tone, setting, and emotion naturally — do not repeat the previous summary verbatim.
"""
                print(f"[VRGDG] 🧠 STORY CONTEXT ADDED — Chapter {summary_index + 1} of {total_sets}")

            # ------------------------------------------------------------
            # Fallback (no summary found)
            # ------------------------------------------------------------
            elif not song_theme_style.strip():
                song_theme_style = "(derive a suitable cinematic theme and tone based on the lyrical content)"
                print("[VRGDG] ⚠️ No previous summary found — using fallback theme.")

            # ------------------------------------------------------------
            # Inject summary instruction for all runs except final one
            # ------------------------------------------------------------
            summary_instruction = ""
            if total_sets is not None and summary_index < total_sets - 1:
                summary_instruction = """
After the 16 prompts, also include a "summary" block using this format:
"summary": {
"scene_summary": "...",
"character_arc": "...",
"narrative_thread": "...",
"next_scene_suggestion": "..."
}
"""
                print("[VRGDG] 🧩 Summary block injection active for this run.")
            else:
                print("[VRGDG] 🏁 Final chapter detected — skipping summary generation.")

            # ------------------------------------------------------------
            # Handle prompt count (only different in final chapter)
            # ------------------------------------------------------------
            prompts_this_run = 16  # default for all runs
            if total_sets is not None and summary_index == total_sets - 1:
                try:
                    prompts_this_run = int(groups_in_last_set)
                except Exception:
                    prompts_this_run = 16
                print(f"[VRGDG] FINAL CHAPTER DETECTED — {prompts_this_run} prompts will be generated in this run.")
            else:
                print("[VRGDG] Standard chapter — 16 prompts will be generated.")

            # ------------------------------------------------------------
            # Add final chapter closure directive (only on last)
            # ------------------------------------------------------------
            if total_sets is not None and summary_index == total_sets - 1:
                song_theme_style += f"""

FINAL CHAPTER INSTRUCTION:
This is the FINAL CHAPTER of the story.
Generate exactly {prompts_this_run} cinematic prompts to conclude the narrative.
Ensure the final prompt provides emotional and visual closure.
"""

            # ------------------------------------------------------------
            # Build final instruction text for the LLM
            # ------------------------------------------------------------
            story_mode_instructions = f"""
        TASK: Generate cinematic text-to-video prompts for a music video

        CHARACTER TYPE (INPUT 1):
        {character_description.strip()}

        Reminder for final output format:
        Output each prompt in JSON format using numbered keys.
        Do NOT nest any additional JSON objects or fields inside each key.
        Each key must be sequential ("prompt1", "prompt2", etc.)
        Output ONLY the JSON — no code fences, no commentary.
        Do not invent or assume any character details such as hair color, skin tone, age, ethnicity, eye color, body type, or other visual features unless they are explicitly included in the character description.
        {summary_instruction}

        Example JSON: wrapped in curly brackets

        "prompt1": "First cinematic visual based on first lyric line...",
        "prompt2": "Second cinematic visual based on next lyric line...",
        "prompt3": "Third cinematic visual..."

        SONG THEME / CREATIVE DIRECTION (INPUT 2):
        {song_theme_style.strip()}

        LYRIC SEGMENTS (PIPE SEPARATED - INPUT 3):
        {pipe_separated_lyrics.strip()}

        NOTES:
        - There are exactly {prompts_this_run} lyric segments separated by '|'.
        - Treat them as one continuous narrative journey.
        - Read all lyric segments together first to understand the emotional and visual arc.
        - Then generate {prompts_this_run} cinematic prompts, one per lyric segment.
        - Each prompt must be 40–50 words and in JSON format.
        - Output should contain ONLY the {prompts_this_run} prompts joined by '|', with no JSON, code, or commentary.
        - Do not include or repeat any words, text, or lyrics in the prompts — only describe cinematic visuals.

        The following detailed cinematic generation rules and structure guidelines define how the LLM should create these prompts:

        {self._cinematic_task_instructions()}"""

            print(f"[VRGDG] ✅ Story mode instructions built successfully — prompts_this_run = {prompts_this_run}")
            return (story_mode_instructions.strip(),)


        # ============================================================
        # EXISTING MODE LOGIC (UNCHANGED)
        # ============================================================
        # Your entire current implementation remains as-is
        # (copied from your original class exactly)
        # The return below is the existing prompt-building instruction text
        # from your previous logic.
        # ============================================================

        # Generate list handling instructions based on mode
        if "Strict Cycle" in list_handling_mode:
            list_instructions = """8. List Handling:
- If multiple options are provided for any of the below categories, treat them as a list.
- Cycle through list items across prompts in order.
- Do not repeat an item until all others have been used.
- Once all have been used, restart the cycle.
- Each prompt must use exactly one item from each category."""
        elif "Reference Guide" in list_handling_mode:
            list_instructions = """8. List Handling:
- The categories below are INSPIRATION and REFERENCE GUIDES.
- Use them as starting points to create variations and similar ideas.
- Feel free to combine elements or create new options in the same style.
- Prioritize what works best for each lyric fragment and the overall narrative flow.
- Maintain variety across prompts - avoid repeating the exact same choices.
- Stay true to the overall aesthetic and mood of the provided examples."""
        elif "Random Selection" in list_handling_mode:
            list_instructions = """8. List Handling:
- If multiple options are provided for any category, select randomly from the list.
- Items can repeat across prompts - there is no cycling requirement.
- Prioritize what works best for each lyric fragment and the overall narrative flow.
- Ensure overall variety across the full sequence of prompts.
- Each prompt should feel fresh even if some elements repeat."""
        else:  # Free Interpretation
            list_instructions = """8. List Handling:
- The categories below are LOOSE GUIDELINES ONLY.
- You may use them as-is, combine them, modify them, or create entirely new options.
- Prioritize what works best for each lyric fragment and the overall narrative flow.
- Feel free to ignore any category if it doesn't serve the visual storytelling.
- Creativity and coherence are more important than strict adherence to the lists."""

        # ✅ Generate prompt structure instructions based on mode
        if "Character-Focused" in prompt_structure_mode:
            structure_instructions = f"""Prompt Structure (for every lyric fragment, {word_count_min}–{word_count_max} words):
-Start with the Shot Type
-Then add in the Character and Outfit if any
-Then add their Physical Interaction
-Then add the Environment
-Then add the Lighting
-Then add the Camera Motion
-Then provide the Cinematic Detail
-Then mention the Facial Expression / Emotion"""
        elif "Environment-Focused" in prompt_structure_mode:
            structure_instructions = f"""Prompt Structure (for every lyric fragment, {word_count_min}–{word_count_max} words):
-Start with the Shot Type
-Then establish the Environment
-Then add the Lighting
-Then add the Camera Motion
-Then introduce the Character and Outfit if any
-Then add their Physical Interaction
-Then provide the Cinematic Detail
-Then mention the Facial Expression / Emotion"""
        elif "Action-Focused" in prompt_structure_mode:
            structure_instructions = f"""Prompt Structure (for every lyric fragment, {word_count_min}–{word_count_max} words):
-Start with the Shot Type
-Then begin with the Physical Interaction
-Then add the Character and Outfit if any
-Then add the Environment
-Then add the Lighting
-Then add the Camera Motion
-Then provide the Cinematic Detail
-Then mention the Facial Expression / Emotion"""
        elif "Cycle Through Patterns" in prompt_structure_mode:
            structure_instructions = f"""Prompt Structure (for every lyric fragment, {word_count_min}–{word_count_max} words):
Cycle through these 3 patterns in order:

Pattern 1:
-Start with the Shot Type
-Then add in the Character and Outfit if any
-Then add their Physical Interaction
-Then add the Environment
-Then add the Lighting
-Then add the Camera Motion
-Then provide the Cinematic Detail
-Then mention the Facial Expression / Emotion

Pattern 2:
-Start with the Shot Type
-Then establish the Environment
-Then add the Lighting
-Then add the Camera Motion
-Then introduce the Character and Outfit if any
-Then add their Physical Interaction
-Then provide the Cinematic Detail
-Then mention the Facial Expression / Emotion

Pattern 3:
-Start with the Shot Type
-Then begin with the Physical Interaction
-Then add the Character and Outfit if any
-Then add the Environment
-Then add the Lighting
-Then add the Camera Motion
-Then provide the Cinematic Detail
-Then mention the Facial Expression / Emotion"""
        else:  # Dynamic Choice
            structure_instructions = f"""Prompt Structure (for every lyric fragment, {word_count_min}–{word_count_max} words):
Choose the most effective structure for each lyric fragment.
All prompts must include: Shot Type, Character + Outfit, Physical Interaction, Environment, Lighting, Camera Motion, Cinematic Detail, and Facial Expression / Emotion.
Vary the order to best serve each moment."""

        # Generate instructions text
        instructions_text = f"""TASK: You are a professional music video director and cinematographer creating a highly detailed prompt sequence for a text-to-video AI model based on lyrics.

CORE CONTEXT:
Character Type:
{character_description.strip()}

Reminder for output format:
Output the prompts as a valid JSON object.
Do NOT nest any additional JSON objects or fields inside each key.
Use sequential keys ("prompt1", "prompt2", ...).
Do not include code formatting, markdown, or any text outside of the JSON.
Do not invent or assume any character details such as hair color, skin tone, age, ethnicity, eye color, body type, or other visual features unless they are explicitly included in the character description.
Send output in json format-
Example: these wrapped in curly brackets

  "prompt1": "A close-up of a woman in soft morning light...",
  "prompt2": "A medium shot of her walking across a foggy bridge...",
  "prompt3": "A wide shot of the city fading into dusk..."



Song Theme / Creative Direction:
{song_theme_style.strip()}

INSTRUCTIONS:

1. Read ALL Lyric Fragments First:
- The lyrics below represent one continuous story told across multiple segments.
- Each lyric fragment is ~4 seconds of screen time.
- Short phrases may lack context on their own - you must infer the full action across connected segments.

2. Understand the Full Visual Narrative:
Before writing prompts, determine:
- What is happening physically in this story?
- What is the progression of actions and environments?
- How do locations connect?
- What visual motifs emerge?
- What is the emotional arc?

3. Output Format:
- One prompt per lyric fragment
- Each prompt: {word_count_min}–{word_count_max} words
- Use natural flowing sentences
- Output as json format
- DO NOT output JSON, markdown, or commentary
- ONLY output the prompts

4. {structure_instructions}

5. Self-Contained Descriptions:
Each prompt must stand completely alone. The AI model has NO memory.

FORBIDDEN REFERENCES (never use these):
- "the same woman/man/character"
- "she/he/they/it" (unless restated in full within same prompt)
- "still", "continues", "next", "after", "before", "from earlier"
- Any reference to prior prompts

REQUIRED IN EVERY PROMPT:
- Fully describe character + outfit
- Fully describe environment
- Fully describe lighting

6. Continuity Through Repetition:
- Repeat key elements to maintain continuity
- Do NOT reference earlier prompts - just restate
- Keep outfit and character consistent unless narrative demands change

7. Visual Categories to Include:

Character Visibility Options:
{character_visibility.strip()}

Shot Types:
{shots.strip()}

Environments:
{environment.strip()}

Lighting:
{lighting.strip()}

Camera Motion:
{camera_motion.strip()}

Physical Interactions:
{physical_interaction.strip()}

Facial Expression / Visual Emotion:
{facial_expression.strip()}

Outfit Rules:
{outfit_rules.strip()}

{list_instructions}

9. Language Style:
- Clear, cinematic, visual language
- No poetic metaphors or abstract moods
- No semicolons, colons, or quotation marks
- Emotion shown only through visual elements (composition, light, camera work, gestures)

10. Subject Attributes and Actions:
- All descriptive fragments (body parts, clothing, posture, gestures, accessories, facial features) are bound to the subject unless explicitly separate
- Do not create duplicates or disembodied elements
- Example: "a woman, hands on table" should be "A woman with her hands resting on the table"

LYRIC SEGMENTS (PIPE SEPARATED):
{pipe_separated_lyrics.strip()}

FINAL REMINDERS:
- Read all lyrics first - plan the continuous visual story
- Each segment = ~4 seconds
- Repeat key elements every time
- Never reference earlier prompts
- Each prompt must stand alone
- Make it cinematic: camera, lighting, composition
- Follow user specifications above all else
- Do not invent or assume any character details such as hair color, skin tone, age, ethnicity, eye color, body type, or other visual features unless they are explicitly included in the character description.
- Only send in json format: 
- Example: in curly brackets

  "prompt1": "A wide cinematic shot of a woman standing under neon lights...",
  "prompt2": "A close-up as she turns her face toward the light...",
  "prompt3": "A slow pan as rain falls softly in the background..."

-Do NOT nest any additional JSON objects or fields inside each key.  
"""

        return (instructions_text,)

    def _cinematic_task_instructions(self):
        return """
INPUT: Lyric Segments in pipe separated format

Example:

INPUT 2: PIPE Separated Lyric Segments
segment1|segment2|segment3|etc 

INPUT (OPTIONAL): Prompt Structure Pattern

The user may provide a specific pattern for structuring prompts. If provided, use it exactly as specified.
If NO pattern is provided, use the DEFAULT PATTERN below.

DEFAULT PATTERN — Cycle Through Patterns

Cycle through these 4 patterns for variety (40–50 words each):

Pattern 1:
Shot Type → Character + Outfit → Physical Interaction → Environment → Lighting → Camera Motion → Cinematic Detail → Visual Expression

Pattern 2:
Shot Type → Environment → Lighting → Character + Outfit → Physical Interaction → Camera Motion → Cinematic Detail → Visual Expression

Pattern 3:
Shot Type → Physical Interaction → Character + Outfit → Environment → Lighting → Camera Motion → Cinematic Detail → Visual Expression

Pattern 4:
Shot Type → Environment → Physical Interaction → Character + Outfit → Lighting → Camera Motion → Cinematic Detail → Visual Expression

USER OVERRIDE RULE (CRITICAL)

If the user provides ANY details in other inputs

User-provided details ALWAYS override defaults.

Follow their specifications exactly, even if they contradict these rules.

Only fall back to default rules for aspects not specified by the user.

CRITICAL UNDERSTANDING: THE FULL VISUAL NARRATIVE

Before creating prompts, you MUST:

Read all lyric segments completely

They are fragments of ONE continuous visual story.

Each segment = ~4 seconds of video/audio

Understand context-

Short phrases may lack meaning on their own.

Infer the full action across connected segments.

Identify the visual story-

What is happening physically?

What is the progression of actions and environments?

How do locations and visual motifs connect?

PROMPT CREATION RULES-
Rule 1: Self-Contained Descriptions

Each prompt must stand alone. The video model has no memory.

❌ Forbidden references:

"the same woman/man/character"

"she/he/they/it" (unless restated in full within the same prompt)

"still", "continues", "next", "after", "before", "from earlier", etc.

Any reference to prior prompts.

✓ Required:

Fully describe character + outfit every time.

Fully describe environment every time.

Fully describe lighting every time.

Rule 2: Continuity via Repetition

Repeat key elements to maintain continuity (character + outfit, environment, lighting).

Do NOT reference earlier prompts — just restate.

Rule 3: Prompt Structure

Each prompt should flow naturally as sentences and include the following example, but the order depends on the prompt structure.

Shot type

Character + outfit

Physical action

Environment

Lighting

Camera movement

Cinematic details (visual details, atmosphere)

Visual expression (body language, gestures, gaze, etc.)

Rule 4: Word Count

Each prompt: 40–50 words

Natural sentences (no bullet lists)

Rule 5: Language Style

Clear, cinematic, visual language

No poetic metaphors or abstract moods

No semicolons, colons, or quotation marks

Emotion shown only through visual elements (composition, light, camera work, gestures)

Rule 6: Subject Attributes and Actions

All descriptive fragments (body parts, clothing, posture, gestures, accessories, facial features, etc.) are automatically bound to the subject unless explicitly described as separate objects.

Do not generate duplicates or disembodied elements (e.g., "eyes on the table" → subject's gaze is directed at the table, not detached eyes).

Fragments should be understood as conditions or states of the subject, not standalone entities.

Example of correct wording:

"a woman sits, hands on the table" should be:  A woman sits with her hands resting on the table.

"a man, cigarette in mouth" should be: A man smoking a cigarette.

"a child, eyes on the book" should be: A child looking down at the book.

-------------------

EXAMPLE WORKFLOW

INPUT:
Character: "a woman"
Lyrics segments:
a woman|sips black|liquid out of a glass


SONG THEME / CREATIVE DIRECTION:
cinematic realism, emotional storytelling, soft surrealism, naturalistic tone, modern drama, poetic symbolism, intimate atmosphere

Extra notes/Input: 

Setting: stark white room

Action: woman drinking dark liquid from glass

Sequence: establish → hesitation → act

e shot of a stark white room bathed in soft ambient light. A woman in a white dress sits at a table...|A medium close-up from a side angle reveals a women holding a glass cup of black liquid up to her mouth...|Close-up of a woman in a white dress drinking black liquid out of a glass cup..|etc

FINAL REMINDERS

Read all lyrics first — plan the continuous visual story.

Each segment = ~4 seconds.

Repeat key elements every time.

Never reference earlier prompts.

Each prompt must stand alone.

Make it cinematic: camera, lighting, composition.

Follow user overrides above all else.

Output each prompt as a JSON object with sequential keys.
Do not include markdown, code fences, or commentary.
Do not invent or assume any character details such as hair color, skin tone, age, ethnicity, eye color, body type, or other visual features unless they are explicitly included in the character description.

Example:
{
  "prompt1": "A wide cinematic shot of a woman standing under neon lights...",
  "prompt2": "A close-up as she turns her face toward the light...",
  "prompt3": "A slow pan as rain falls softly in the background..."
}

-Do NOT nest any additional JSON objects or fields inside each key.
>"""

import json
import os
import re

class VRGDG_PromptSplitterJson:
    RETURN_TYPES = tuple(["STRING"] * 17)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 17)] + ["summary_output"])
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": '{"prompt1": "example text", "prompt2": "another one", "summary": {"scene_summary": "...", "character_arc": "...", "narrative_thread": "...", "next_scene_suggestion": "..."}}',
                    },
                ),
            },
            "optional": {
                "file_path": ("STRING", {"multiline": False, "default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999}),
            },
        }

    def _clean_json_text(self, text):
        """
        Cleans common formatting issues from LLM output so it's more likely to parse correctly.
        """
        print("[VRGDG] 🧽 Cleaning raw prompt text before parsing...")

        # Remove markdown fences and artifacts
        text = text.strip()
        text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip(), flags=re.MULTILINE)

        # Normalize smart quotes to standard quotes
        text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

        # Fix keys that might start with symbols (*prompt8 → "prompt8")
        text = re.sub(r"([^\w\"])(prompt\d+)\":", r'"\2":', text)

        # Fix missing quotes on keys (prompt1: → "prompt1":)
        text = re.sub(r'(?<!")(\bprompt\d+\b)(?=\s*:)', r'"\1"', text)

        # Remove trailing commas before closing braces/brackets
        text = re.sub(r",(\s*[}\]])", r"\1", text)

        # Remove any control characters or escape codes
        text = re.sub(r"[\x00-\x1f]+", " ", text)

        # Ensure the text starts and ends properly
        if not text.strip().startswith("{"):
            text = "{" + text
        if not text.strip().endswith("}"):
            text = text.rstrip(",") + "}"

        return text.strip()

    def split_prompt(self, prompt_text, file_path=None, index=0, **kwargs):
        parts = []
        summary_text = ""

        print(f"[VRGDG] 🌀 Starting Prompt Splitter — Index: {index}")
        try:
            # --- Step 1: Clean incoming text ---
            cleaned_text = self._clean_json_text(prompt_text)

            # --- Step 2: Attempt to parse JSON ---
            try:
                data = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                print(f"[VRGDG] ❌ JSON parsing failed even after cleanup: {e}")
                error_message = (
                    "❌ ERROR: The JSON prompt structure sent to the Prompt Splitter is invalid.\n"
                    "Please refresh and try again — the LLM incorrectly formatted the JSON output.\n"
                    f"(Parsing error: {str(e)})"
                )
                print(f"[VRGDG] {error_message}")
                raise ValueError(error_message)

            if not isinstance(data, dict):
                raise ValueError("❌ ERROR: The JSON root must be an object with key-value pairs.")

            print(f"[VRGDG] ✅ JSON parsed successfully. Keys found: {list(data.keys())}")

            # --- Save cleaned JSON for reference ---
            if file_path:
                try:
                    os.makedirs(file_path, exist_ok=True)
                    prompt_filename = os.path.join(file_path, f"prompt{index}.json")
                    with open(prompt_filename, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(f"[VRGDG] 💾 Cleaned JSON saved to {prompt_filename}")
                except Exception as e:
                    print(f"[VRGDG] ⚠️ Failed to save cleaned JSON file: {e}")

            # --- Extract summary if available ---
            summary_data = data.get("summary")
            if summary_data:
                summary_text = json.dumps(summary_data, indent=2)
                print(f"[VRGDG] 📘 Summary section found. Saving summary{index}.json...")
                if file_path:
                    try:
                        summary_filename = os.path.join(file_path, f"summary{index}.json")
                        with open(summary_filename, "w", encoding="utf-8") as f:
                            json.dump(summary_data, f, indent=2)
                        print(f"[VRGDG] 💾 Summary saved to {summary_filename}")
                    except Exception as e:
                        print(f"[VRGDG] ⚠️ Failed to save summary JSON: {e}")

            # --- Extract and flatten prompt strings ---
            for key, value in data.items():
                if key.startswith("summary"):
                    continue

                if isinstance(value, dict):
                    flattened = " ".join(
                        str(v) for v in value.values() if isinstance(v, (str, int, float))
                    ).strip()
                    print(f"[VRGDG] ⚠️ Flattened nested prompt for '{key}' -> '{flattened[:60]}...'")
                    parts.append(flattened)
                elif isinstance(value, list):
                    flattened = " ".join(str(v) for v in value if isinstance(v, (str, int, float))).strip()
                    parts.append(flattened)
                elif isinstance(value, (str, int, float)):
                    parts.append(str(value).strip())
                else:
                    print(f"[VRGDG] ⚠️ Unexpected type for '{key}': {type(value)} — skipping.")
                    parts.append("")

        except Exception as e:
            # 🚨 Hard stop — invalid JSON, force error message downstream
            print(f"[VRGDG] ❌ Critical error: {e}")
            error_msg = (
                "❌ The JSON prompt structure sent to the Prompt Splitter is invalid. "
                "Please refresh and try again — the LLM incorrectly formatted the JSON output."
            )
            parts = [error_msg] * 16
            summary_text = error_msg
            # Stop further processing by returning early
            return tuple(parts + [summary_text])

        # --- Step 3: Normalize output count ---
        outputs = [parts[i] if i < len(parts) else "" for i in range(16)]
        outputs.append(summary_text)

        print(f"[VRGDG] 🧩 Split complete. {len(parts)} prompts extracted. Summary included: {'Yes' if summary_text else 'No'}")
        print(f"[VRGDG] ✅ Returning structured outputs for downstream nodes.\n")

        return tuple(outputs)



    



NODE_CLASS_MAPPINGS = {

     "VRGDG_MusicVideoPromptCreatorV3": VRGDG_MusicVideoPromptCreatorJson,
     "VRGDG_PromptSplitterJson":VRGDG_PromptSplitterJson


 



}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MusicVideoPromptCreatorV3": "🌀 VRGDG_MusicVideoPromptCreatorJson",
    "VRGDG_PromptSplitterJson":"VRGDG_PromptSplitterJson"

}
