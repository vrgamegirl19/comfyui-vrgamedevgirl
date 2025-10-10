# 🎬 AI Music Video Workflow (ComfyUI) V7

This workflow takes a **reference image** and an **audio file** to automatically generate a stylized **AI-driven music video**.  
It splits audio into lyric-based snippets, generates visual prompts, and combines everything into a synced final video.

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

## 🟦 Step 3: Update File Path 👉
Choose the folder where your video output files will be saved.

- Must be inside your **`comfyui/output`** directory.  
- Folder should be empty before starting.  
- OS-specific examples:  
  - **Windows:** `A:/Comfy_UI/output/my_folder`  
  - **Linux/macOS:** `/home/user/comfyui/output/my_folder`  
  - **RunPod:** `/workspace/comfyui/output/my_folder`  

⚠️ If the **Current Index** stays at `0`, your path is incorrect.

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

✅ The filled-out fields are expanded by the **LLM Node** into cinematic scene prompts, then split across video chunks by the **Prompt Splitter Node**.

---

# 📌 Additional Details & Nodes

### 🎵 Audio Transcription (HUMO Transcribe V3)
- Breaks audio into lyric-synced snippets.  
- **enable_lyrics:** Toggle transcription.  
- **lyric_overlap:** Smooth transitions between lines.  
- **fallback_words:** Backup words if transcription fails.  
- **context_strings:** Add clarity/corrections.  
- Each run = ~62s of video (auto-queues 3 runs).  

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

### 📂 Conditional Video Loader
- Collects video chunks from output folder.  
- **Threshold:** Minimum clips before combining.  
- **Batch Size:** Controls memory usage (smaller = safer, larger = faster).  

---

# 🚀 Tips & Notes
- **First run auto-queues** additional runs.  
- **Index counter** tracks progress (`0 = run 1`, `1 = run 2`, etc).  
- **Creative control:** Focus on the **Prompt Creator** fields.  

---

# ✅ Summary
1. Upload your **reference image**.  
2. Load your **audio file**.  
3. Set your **file path** correctly.  
4. Fill in the **Prompt Creator** for style and scenes.  

Everything else (transcription, prompts, video combining, audio syncing) runs automatically to give you a final  **AI-generated music video**.

---

# ✅ Notes:
- Keep at 25 Frames per second to keep audio in sync with the mouth. Changing this could cause sync issues.
