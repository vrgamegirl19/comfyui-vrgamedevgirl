# 🎮 VRGameDevGirl’s Video Enhancement Nodes for ComfyUI

Custom ComfyUI nodes for high-quality, frame-by-frame video enhancement.  
Includes realtime-ready nodes for film grain, color tone matching, and more to come!

---

## 🌟 Features

- 🎞️ **Fast Film Grain**: Add controllable, grayscale or color grain for cinematic texture.
- 🎨 **Color Match to Reference**: Align image tones to a reference image for consistent color grading.
- ⚡ Optimized for **video workflows** and **per-frame processing** in ComfyUI.
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

If not using the Manager, install dependencies manually:

```bash
pip install torch kornia
```

Or:

```bash
pip install -r requirements.txt
```

---

## 🧠 Node Details

### ✅ Fast Film Grain Node
Adds customizable film grain to each frame.  
Grain can be grayscale or saturated RGB, and sized to match your output resolution.

**Inputs:**
- `images`: Frame tensor input.
- `grain_intensity`: Blend amount (0 to 1).
- `saturation_mix`: 0 = grayscale grain, 1 = full RGB noise.

🟢 *Video-safe and very fast.*

---

### 🎨 Color Match to Reference
Matches image color distribution to a reference image using LAB space normalization.

**Inputs:**
- `images`: Your video frames.
- `reference_image`: A single image to match tone and color against.
- `match_strength`: Blend between original and matched (0.0–1.0).

⚠️ Make sure all images are 4D tensors: `[batch, height, width, channels]`.  
Uses the same reference image across all frames for consistency.

---

## 🛠️ Roadmap

- [x] Film grain (grayscale and RGB)
- [x] Color match (LAB-based)
- [ ] Sharpness control node
- [ ] Local contrast / dehaze
- [ ] LUT loader or approximate match

---

## 📁 Folder Structure

```
comfyui-vrgamedevgirl/
│
├── custom_nodes/
│   └── VRGameDevGirl/
│       ├── __init__.py
│       └── nodes.py
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
