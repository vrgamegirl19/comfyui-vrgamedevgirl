# 🎬 AI Music Video Workflow (ComfyUI) V8

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

# ✅ Notes:
- Making changes to any other nodes can cause the workflow to break. Please try with the default settings first before making changes.
