## License

This ComfyUI custom node is licensed under AGPL-3.0.

You may not use this code in a closed-source paid app, private hosted service, SaaS product, or commercial wrapper unless you comply with the AGPL-3.0 license and provide the complete corresponding source code under the same license.

# 🎮 VRGameDevGirl’s Video / Image & HUMO Workflow Nodes for ComfyUI

Custom ComfyUI nodes for **music video workflows** and **high-quality video/image enhancement**.  
Includes HUMO-based multi-scene workflows, audio splitting, video combining, and realtime-ready enhancement tools.

---

# 🎬 AI Music Video Workflow (ComfyUI)

Turn a **reference image** and an **audio track** into a fully **AI-generated music video**.  
The workflow automatically splits lyrics, generates scene prompts, and syncs everything into a cinematic final video.  

## 🚀 How it Works
1. Upload a **reference image** (your main character).  
2. Import your **audio file** (with lyric syncing).  
3. Set a **folder name** for outputs.  
4. Customize the **Prompt Creator** to define style, mood, and scenes.  

✨ Everything else — transcription, scene building, video chunks, and final rendering — runs automatically.  
The result: a stylized, synced **AI-driven music video**.

---

## 🌟 Features

- 🎞️ **Fast Film Grain** (`FastFilmGrain`)  
  Add controllable, grayscale or color grain for cinematic texture.  
  ➕ *Now includes a `batch_size` setting to reduce out-of-memory (OOM) issues on long or high-res videos. If you have a strong GPU, try `batch_size = 8`. If you hit OOM, lower the value (e.g., to 4 or 2).*

- 🎨 **Color Match to Reference** (`ColorMatchToReference`)  
  Align image tones to a reference image using LAB color matching.  
  ➕ *Also includes a `batch_size` setting to help manage VRAM usage. Start at 8 if your GPU allows — lower it if you get OOM errors.*

- 🎯 **Fast Unsharp Sharpen** (`FastUnsharpSharpen`)  
  Simple and efficient sharpening using unsharp masking.

- 🌀 **Fast Laplacian Sharpen** (`FastLaplacianSharpen`)  
  Edge-based sharpening via Laplacian kernel for crisp detail.

- 📏 **Fast Sobel Sharpen** (`FastSobelSharpen`)  
  Gradient-based edge enhancement using Sobel filters.

- ⚡ Optimized for **image or video workflows** and **per-frame processing** in ComfyUI.

- 💻 Designed to be efficient on modern GPUs.

---

## 📦 Installation

### 🧰 Using ComfyUI Manager (recommended)
1. Open ComfyUI.
2. Go to the **Manager** tab → **Install Custom Nodes**.
3. Search: `vrgamedev` or use this Git URL:  
   ```
   https://github.com/vrgamegirl19/comfyui-vrgamedevgirl
   ```

### 🖐️ Manual Install
1. Clone or download this repo to your `ComfyUI/custom_nodes` directory.
2. Restart ComfyUI.

---

## ✨ Requirements

A `requirements.txt` file is now needed.

To install the required packages, run this from inside the `python_embeded` folder:

```
python.exe -m pip install -r ..\custom_nodes\comfyui-vrgamedevgirl\requirements.txt

```

`requirements.txt` (in the repo) includes:

```

kornia

librosa

imageio

```

## 🧠 Node Details

### ✅ Fast Film Grain (`FastFilmGrain`)
Adds customizable film grain to each frame.  
Grain can be grayscale or saturated RGB, and sized to match your output resolution.

**Inputs:**
- `images`: Frame tensor input.
- `grain_intensity`: Blend amount (0 to 1).
- `saturation_mix`: 0 = grayscale grain, 1 = full RGB noise.

🟢 *Video-safe and very fast.*

---

### 🎨 Color Match to Reference (`ColorMatchToReference`)
Matches image color distribution to a reference image using LAB space normalization.

**Inputs:**
- `images`: Your video frames.
- `reference_image`: A single image to match tone and color against.
- `match_strength`: Blend between original and matched (0.0–1.0).

---

### 🎯 Fast Unsharp Sharpen (`FastUnsharpSharpen`)
Applies unsharp masking to enhance edges with a fast, low-cost blur pass.

**Inputs:**
- `images`: Input image tensor.
- `strength`: Sharpening amount (0.0 to 2.0)

🚀 Lightweight and ideal for subtle sharpening.

---

### 🌀 Fast Laplacian Sharpen (`FastLaplacianSharpen`)
Enhances edges by applying a Laplacian kernel to bring out high-frequency detail.

**Inputs:**
- `images`: Input image tensor.
- `strength`: Sharpening amount (0.0 to 2.0)

🧪 Gives a more "punchy" sharpen effect great for detail recovery.

---

### 📏 Fast Sobel Sharpen (`FastSobelSharpen`)
Uses Sobel filters to detect image gradients and amplify edge contrast.

**Inputs:**
- `images`: Input image tensor.
- `strength`: Sharpening amount (0.0 to 2.0)

🧠 Sharpens by boosting directional edge response — great for outlines and detail clarity.

---

## 🛠️ Roadmap

- [x] 🎞️ Fast Film Grain (`FastFilmGrain`)
- [x] 🎨 Color Match To Reference (`ColorMatchToReference`)
- [x] 📏 Fast Sobel Sharpen (`FastSobelSharpen`)
- [x] 🌀 Fast Laplacian Sharpen (`FastLaplacianSharpen`)
- [x] 🎯 Fast Unsharp Sharpen (`FastUnsharpSharpen`)
- [ ] 🌫️ Local Contrast / Dehaze
- [ ] 🎛️ LUT Loader or Approximate Match

---

## 📁 Folder Structure

```
comfyui-vrgamedevgirl/
│
├── init.py
├── nodes.py
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## 🧑‍💻 Author

**VRGameDevGirl**  
✨ Custom tools for cinematic AI workflows  
💌 Questions or collabs? Reach out via GitHub

---

## 📜 License

This project is licensed under the MIT License.
