# Wan Humo Image Edit Workflow (ComfyUI)

Image-to-Image editing workflow using **Wan Humo** inside **ComfyUI** using multiple reference images and a prompt to generate edited results.

This workflow allows you to quickly create edited images using reference inputs, prompts, and seed variation. Takes between 50 to 70 seconds

---

# Features

- Image-to-Image editing with Wan Humo
- Supports **multiple reference images**
- Seed control for different variations
- Adjustable resolution
- Optional **automatic prompt generation**
- Works with **manual prompts or AI-generated prompts**
- Optional **local Ollama prompt generator**
- Designed for **stable results using default settings**

---

# Example Results

Different seeds and prompts will generate different outputs using the same reference images.

You can experiment with:

- Prompt changes
- Different seeds
- Different reference image combinations

---

# Basic Usage

1. Load the workflow in **ComfyUI**
2. Load your **reference images**
3. Enter a **prompt**
4. Set **seed / width / height**
5. Run the workflow

The workflow will generate a new edited image using the references and prompt.

---

# Reference Images

The workflow supports **multiple reference images**.

Default setup includes **5 image inputs**.

Unused inputs can be **bypassed**.

To add more:

1. Duplicate an image input node
2. Connect it to the image input chain
3. Increase the **input count**

In theory the number of reference images is **unlimited**, but most edits work well with a few references.

---

# Settings

### Recommended Settings

It is recommended to **leave the default settings unchanged**.

Changing them can cause issues:

Higher settings may cause:

- Duplicate characters
- Unstable composition

Lower settings may cause:

- Blurry results
- Lower quality images

---

# Seed

Changing the seed will generate different results.

Useful for:

- Testing variations
- Exploring different compositions
- Improving prompt outcomes

---

# Block Swap (VRAM Control)

The workflow includes a **Block Swap** setting.

This helps manage VRAM usage.

If you encounter **Out Of Memory (OOM)** errors:

Increase the **Block Swap value**.

---

# Custom Node Requirement

This workflow includes **one custom node**.

Wan Humo normally expects an **audio input** because it is designed for lip sync.

Since this workflow only generates images, the node provides **silence audio** to prevent errors.

If you do not have my recent custom node installed:

ComfyUI will prompt you to install or update them.

---

# Optional Prompt Generator (GPT)

You can optionally use a **prompt generator GPT** to help write prompts.

It converts a short description into a structured prompt.

Example input:

```
Close up of a woman looking in the mirror from over the shoulder
```

The GPT generates a full prompt suitable for the workflow.

Use it here:
[Custom GPT](https://chatgpt.com/g/g-69a36026b41c8191a1f41b4c2ac85cca-wan-humo-image-edit-prompt-enhancer)

This is **optional** but helps produce better results.

---

# Optional Local Prompt Generator (Ollama)

An optional node allows you to generate prompts locally using **Ollama**.

Example model used:

- Qwen 2.5 VL 7B

This model can analyze images and generate prompts automatically.

Usage:

1. Install Ollama
2. Pull a supported model
3. Select the model in the node
4. Provide a short idea of what you want

Example:

```
Woman standing in front of a mirror, over the shoulder view
```

The model will generate a prompt which feeds directly into the workflow.

---

# Image Behavior Notes

Some edits may move objects in the scene.

Example issues:

- Objects shifting position
- Scene composition changing

This behavior is common in many image editing models and may not always be avoidable.

Prompt adjustments and seed changes can sometimes improve results.

---

# Tips for Better Results

- Use **clear prompts**
- Use **relevant reference images**
- Test **multiple seeds**
- Avoid excessive settings changes
- Keep references consistent with your prompt

---

# Workflow Summary

1. Load references
2. Enter prompt
3. Adjust seed or resolution if needed
4. Run workflow
5. Iterate using different seeds or prompts

---

# Feedback

If you encounter issues with:

- The workflow
- The custom nodes
- The Ollama prompt generator

Please report them so improvements can be made.
