# 🎬 AI Music Video Workflow (ComfyUI) V8

This workflow takes a **reference image** and an **audio file** to automatically generate a stylized **AI-driven music video**.  
It splits audio into lyric-based snippets, generates visual prompts, and combines everything into a synced final video.

---
Watch example video's I have created using this workflow.

[YouTube Playlist](https://www.youtube.com/playlist?list=PLQ0zxAQhttlZpolPMJTeQQjafa__MaD2v)

Join the community! Get support, share your gens, see latest updates and get tips and tricks!

[discord server here](https://discord.gg/FJ9VvCDXw3)

---

## 🟦 Step 1: Upload Reference Image 👉
The reference image is your **main character** throughout the music video.

- Use **Load Image** to import your reference.  
- The **Background Removal** node removes the image background.  
- Use **Resize Image v2** to adjust to video aspect ratio:  
  - `Crop` if it already matches.  
  - `Pad` if not, to avoid cutting details.  

✅ A clean, resized preview of your performer will be shown here.

---

## 🟦 Step 2: Choose Audio File 👉
Load your track and prepare it for lyric syncing.

- Import your audio file with **Load Audio**.  
- The **Mel-Band RoFormer** nodes separate **vocals** from **instruments**.  
- If you already have a **vocals-only file**, bypass the separation nodes.  

✅ This ensures lyrics sync properly with generated scenes.

---

## 🟦 Step 3: Update Folder name 👉
Choose the folder name. Not a path anymore. The folder will automaticly be creaed in the comfyUI/output folder.
- I recommend using the song name as the folder name.
- If you forgot to change this on a new run, it "should" create a new folder with the same name and add V2 at the end to prevent failure.
- Just keep an eye on the idex and if this does not change from 0 on the next run cancel the process and update the folder name.

---

## 🟦 Step 4: Prompt Creator 👉
This node structures prompts that guide video generation.

### How to Use
- Use **short, comma-separated phrases** (e.g., `rain-slicked street, neon lights`).  
- **Avoid long sentences** — think like a **shot list**.  
- Stick with defaults if unsure.

### Key Fields
- **character_description:** Performer’s look (e.g., `A woman`, `tattoos`).  
- **Overall Theme/Style:** Mood (e.g., `cinematic, melancholic, dreamlike`).  
- **pipe_separated_lyrics:** Auto-filled from transcription.  
- **word_count_min / word_count_max:** Default `30–50`.  
- **environment:** Settings (`abandoned house, moonlit street`).  
- **lighting:** Style (`moody shadows, neon glow`).  
- **camera_motion:** Movement (`dolly, pan, zoom`).  
- **physical_interaction:** Gestures (`touch walls, walk through smoke`).  
- **facial_expression:** Emotions (`intense, calm`).  
- **shots:** Shot types (`close-up, wide angle`).  
- **outfit_rules:** Look (`white dress, leather jacket`).  
- **character_visibility:** How often performer appears.
- **New drop down option called "list handling".** This will tell the LLM how to use your comma separted lists. **Strict Cycle** will use each one once then repeat. **Reference Guide** will use everything as a guide/reference along with the lyrics. **Random Selection** will grab from the list at random. **Free Interperation**. The LLM can ignore or combine things from your list. 

✅ The filled-out fields are expanded by the **LLM Node** into cinematic scene prompts, then split across video chunks by the **Prompt Splitter Node**.

---

# 📌 Additional Details & Nodes

### 🎵 Audio Transcription (HUMO Transcribe V3)
- Breaks audio into lyric-synced snippets.  
- **enable_lyrics:** Toggle transcription.  
- **lyric_overlap:** Smooth transitions between lines.  
- **fallback_words:** Backup words if transcription fails.  
- **context_strings:** Add clarity/corrections.  

### 📖 ReadMe Node
- Explains run setup.  
- Will tell you how many runs are needed, how many have been auto Q'ed
- Will tell you which groups to disable being your last run  

### 🤖 LLM Node
- Expands structured prompt fields into cinematic descriptions.  
- Connect **concatenated_string → LLM input**.  
- Output must go into **Prompt Splitter**.  
- You can swap models (Gemini recommended).

### 🎬 Combine Videos Node
- Collects durations from transcription and merges video chunks.  
- Default FPS = `25`.  
- ⚠️ Must match FPS in **Video Combine nodes**.

### 🔊 Final Audio Setup
- Uses the full track for the final render.  
- If skipping vocal separation:  
  1. Load full song in **AudioFile2**.  
  2. Switch `get_audioFile` to `AudioFile2`.  

### 📂 Create Final video Note
- Collects video chunks from output folder.  
- The final video will be saved as FINAL_VIDEO.mp4 in the same folder as the chunks.
- There is a note that has the file path the main video will be found

---

# 🚀 Tips & Notes
- **First run auto-queues** additional runs.  
- **Index counter** tracks progress (`0 = run 1`, `1 = run 2`, etc).  
- **Creative control:** Focus on the **Prompt Creator** fields.  

---

# ✅ Summary
1. Upload your **reference image**.  
2. Load your **audio file**.  
3. Set your **Folder Name** correctly.  
4. Fill in the **Prompt Creator** for style and scenes.  

Everything else (transcription, prompts, video combining, audio syncing) runs automatically to give you a final  **AI-generated music video**.

---

# High level overview of the main node in this workflow-
# 🎧 VRGDG_LoadAudioSplit_HUMO_TranscribeV3

A powerful **ComfyUI custom node** that automates **audio segmentation, transcription, and project management** for HuMo / VRGDG-style workflows.

This node takes an audio clip, splits it into 16 evenly timed segments, optionally transcribes each segment using **OpenAI Whisper**, manages metadata and output folders, and provides visual workflow instructions through **ComfyUI popups**.

---

## 🧩 Overview

`VRGDG_LoadAudioSplit_HUMO_TranscribeV3` simplifies large audio-to-video pipelines by:
- Splitting an input audio file into **scene-length chunks** (e.g., 4 seconds each).
- Managing **multi-run workflows** for long audio clips.
- Automatically **detecting project state** and versioning output folders.
- Optionally **transcribing each scene** using OpenAI Whisper-Large-V3.
- Sending **color-coded popup instructions** to ComfyUI.

---

## ⚙️ Inputs

| Name | Type | Description |
|------|------|--------------|
| `audio` | `AUDIO` | The input audio clip (waveform + sample rate). |
| `trigger` | `any` | A signal input used for chaining workflow steps. |
| `scene_duration_seconds` | `FLOAT` | Duration of each scene (default: 4.0s). |
| `folder_path` | `STRING` | Output folder for project files. |
| `enable_auto_queue` | `BOOLEAN` | Automatically queue next runs if needed. |
| `language` | `STRING` | Language for Whisper transcription (default: English). |
| `enable_lyrics` | `BOOLEAN` | Enable automatic transcription. |
| `use_context_only` | `BOOLEAN` | Use provided context fields instead of transcription. |
| `overlap_lyric_seconds` | `FLOAT` | Overlap between segments for smoother lyrics merging. |
| `fallback_words` | `STRING` | Backup words for empty transcriptions. |
| `context_1`–`context_16` | `STRING` | Optional per-scene text prompts. |

---

## 📤 Outputs

| Output | Type | Description |
|---------|------|-------------|
| `meta` | `DICT` | Timing, project metadata, and segment information. |
| `total_duration` | `FLOAT` | Total duration of the input audio. |
| `lyrics_string` | `STRING` | Combined transcription or lyric text. |
| `index` | `INT` | Current set index (run number). |
| `start_time` | `STRING` | Start time of the current audio window. |
| `end_time` | `STRING` | End time of the current audio window. |
| `instructions` | `STRING` | Textual guide for what to do next. |
| `total_sets` | `INT` | Total number of runs needed to process the full audio. |
| `groups_in_last_set` | `INT` | How many groups are used in the final run. |
| `frames_per_scene` | `INT` | Adjusted frame count per scene (HuMo-compatible). |
| `audio_meta` | `DICT` | Frame and duration data for this set. |
| `output_folder` | `STRING` | Path to the active output directory. |
| `audio_1`–`audio_16` | `AUDIO` | Individual audio segments per scene. |
| `signal_out` | `any` | Pass-through signal for workflow chaining. |

---

## 🧠 How It Works

### 1. **Audio Analysis**
- Determines sample rate, duration, and frame counts.
- Adjusts frame counts to match **HuMo** (rounded to `(4n + 1)` frames).
- Splits the waveform into **16 equal-length segments**.

### 2. **Metadata Management**
- Stores project info in `.project_metadata.json`.
- Detects if the same audio is being reprocessed.
- Creates versioned folders (`_v2`, `_v3`, etc.) if a new audio file is detected.

### 3. **Set Indexing**
- Counts how many audio sets (`*-audio.mp4`) already exist.
- Determines the **current run index** and **how many total runs** are required.

### 4. **Auto Queueing**
- Optionally adds future runs to ComfyUI’s execution queue.
- Avoids re-queuing during intermediate runs.

### 5. **Transcription (Optional)**
- Uses `openai/whisper-large-v3` to generate text from each segment.
- Supports multilingual transcription.
- Can merge overlapping lyrics for smoother transitions.
- Cleans duplicates and truncates overly long transcriptions.
- If `use_context_only=True`, skips Whisper and uses user-provided context.

### 6. **Instruction Generation**
- Creates human-readable instructions such as:
  - “Mute groups 15–16 and re-run.”
  - “Final run in progress.”
  - “All 3 runs auto-queued.”
- Sends **color-coded popups** to ComfyUI UI:
  - 🔴 **Red** – Action required / cancel & reconfigure  
  - 🟡 **Yellow** – Progress reminder  
  - 🟢 **Green** – Final run complete  
  - 🔵 **Blue / Info** – Normal updates

---

## 🧾 Internal Helpers

| Method | Purpose |
|---------|----------|
| `_count_index_from_folder()` | Detects current run index by scanning output folder. |
| `_calculate_sets()` | Calculates total sets, frames, and user instructions. |
| `_maybe_auto_queue()` | Handles auto-queuing of next runs in ComfyUI. |
| `_get_or_create_project_metadata()` | Reads or creates `.project_metadata.json`. |
| `_save_project_metadata()` | Writes metadata to disk. |
| `_get_smart_output_folder()` | Cleans names, creates folders, handles versioning. |
| `_send_popup_notification()` | Sends color-coded popups via `PromptServer`. |
| `_adjust_frames_for_humo()` | Rounds frames to `(4n + 1)` for HuMo compatibility. |
| `run()` | Main execution method combining all logic above. |

---

# 🧩 Notes

- **HuMo Frame Sync**: Frames per scene are automatically rounded up for HuMo compatibility.  
- **Auto Versioning**: If new audio differs, `_v2`, `_v3`, etc. folders are created automatically.  
- **Fallback Behavior**: If Whisper fails or is disabled, fallback words (e.g., *“thinking”*, *“walking”*) are used.  
- **Language Support**: 100+ languages supported by Whisper-Large-V3.  

---

# 🧠 In Summary

`VRGDG_LoadAudioSplit_HUMO_TranscribeV3` is an all-in-one automation node that:

- Splits, transcribes, and manages long audio files  
- Tracks project state intelligently  
- Automatically queues runs  
- Provides clear, visual workflow feedback inside **ComfyUI**  

It’s built for **HuMo** and **VRGDG** production pipelines where multi-segment audio needs to stay perfectly synchronized across scenes and renders.


---

# ✅ Notes:
- Making changes to any other nodes can cause the workflow to break. Please try with the default settings first before making changes.
