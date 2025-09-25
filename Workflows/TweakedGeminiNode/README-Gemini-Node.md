# 📖 README – Custom Gemini Node (with Lyrics String Input)

## 🔧 What This Is
This is a **modified version of the Gemini LLM node** for ComfyUI.  
The modification adds a **string input field** so you can pass the **lyrics directly as text** into the node.  

Previously, `nodes_gemini.py` had no input for lyrics.  
With this version, the node can now take:  
- The **reference image** (as before),  
- The **lyrics string** (new feature),  
- Any **custom instructions**.  

This allows Gemini to generate prompts for each video segment using **both lyrics and the reference image**.
NOTE: When you update ComyUI this node will get overwritten so will have to be replaced again. 
---

## 📂 Installation

1. Go to your ComfyUI install folder:  
   ```
   ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\comfy_api_nodes
   ```

2. Find the existing file:  
   ```
   nodes_gemini.py
   ```

3. **Backup the original file** (recommended).  
   - Example: rename it to `nodes_gemini_backup.py`.

4. Copy **my modified `nodes_gemini.py`** into the same folder.  
   - Overwrite the old version when prompted.

5. Restart **ComfyUI**.

---

## 🚀 Usage

1. You will now see a new **string input field** at the bottom of the Gemini node.
   - Connect the **lyrics string** output from `VRGDG_LoadAudioSplit_HUMO_TranscribeV2` into this new input if its not connected.
   - Keep the reference image input connected as before.  
2. The node will now combine:  
   - **Lyrics string**  
   - **Reference image**  
   - **Custom instructions**  
   and output a structured prompt for the rest of your workflow.

---

## ⚠️ Notes

- This **replaces the stock Gemini node** (`nodes_gemini.py`). If you want to revert, restore your backup.  
- Tested with workflows using **Wan 2.1 Humo + VRGDG custom nodes**.  
- Make sure only **one copy** of `nodes_gemini.py` exists in the folder to avoid conflicts.  
