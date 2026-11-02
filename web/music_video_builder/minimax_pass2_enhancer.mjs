import { makeCheckbox, makeField, makeSearchableLoraPicker } from "./controls.mjs";
import { DEFAULT_MINIMAX_H3_SETTINGS } from "./minimax_h3.mjs";

export function createMiniMaxPass2Enhancer() {
  const miniMaxPass2Enhancer = {
    enabled: makeCheckbox("Use NVIDIA Enhancer on Pass 2", false),
    dmad: makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_enhancer_dmad_name),
    refiner: makeSearchableLoraPicker(DEFAULT_MINIMAX_H3_SETTINGS.two_pass_enhancer_refiner_name),
    wrapper: document.createElement("div"),
    details: document.createElement("div"),
    status: document.createElement("div"),
  };
  miniMaxPass2Enhancer.wrapper.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  miniMaxPass2Enhancer.details.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  miniMaxPass2Enhancer.status.setAttribute("role", "status");
  miniMaxPass2Enhancer.status.style.cssText = "font-size:11px;color:#facc15;";
  const enhancerNote = document.createElement("div");
  enhancerNote.textContent = "Pass 2 uses DMAD → Open-H3 Refiner at strength 1 each, er_sde, and fixed sigmas 0.7, 0.494118, 0 (2 steps). Replaces the Pass 2 Turbo LoRA and sampling settings. Pass 1 stays as configured. Place both files in ComfyUI/models/loras, then refresh the model list. Use the ComfyUI-format DMAD file.";
  enhancerNote.style.cssText = "font-size:11px;color:#a1a1aa;line-height:1.45;";
  const enhancerDownloads = document.createElement("div");
  enhancerDownloads.style.cssText = "display:flex;flex-wrap:wrap;gap:12px;";
  for (const [label, url] of [
    ["Download DMAD LoRA (ComfyUI)", "https://huggingface.co/ZhengmingYu/DMAD/resolve/main/minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors?download=true"],
    ["Download Open-H3 Refiner LoRA", "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors?download=true"],
  ]) {
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = label;
    link.style.cssText = "font-size:11px;color:#67e8f9;";
    enhancerDownloads.append(link);
  }
  miniMaxPass2Enhancer.details.append(
    enhancerDownloads, enhancerNote,
    makeField("Pass 2 DMAD LoRA", miniMaxPass2Enhancer.dmad.wrapper),
    makeField("Pass 2 Open-H3 Refiner LoRA", miniMaxPass2Enhancer.refiner.wrapper),
    miniMaxPass2Enhancer.status,
  );
  miniMaxPass2Enhancer.wrapper.append(miniMaxPass2Enhancer.enabled.wrapper, miniMaxPass2Enhancer.details);
  return miniMaxPass2Enhancer;
}

// Display the fixed recipe without changing the user's saved normal-pass values.
export function syncMiniMaxPass2Enhancer(controls, settings, {
  twoPass, threePass, twoPassControls, turboPicker, turboStrength,
}) {
  const available = twoPass && !threePass;
  const active = available && settings.two_pass_use_enhancer;
  controls.wrapper.style.display = available ? "flex" : "none";
  controls.enabled.input.checked = settings.two_pass_use_enhancer;
  controls.dmad.input.value = settings.two_pass_enhancer_dmad_name;
  controls.refiner.input.value = settings.two_pass_enhancer_refiner_name;
  controls.details.style.display = active ? "flex" : "none";
  turboPicker.input.disabled = active;
  turboStrength.disabled = active;
  if (controls.turboLayout) controls.turboLayout.style.display = active ? "none" : "grid";
  if (controls.turboDownload) controls.turboDownload.style.display = active ? "none" : "";
  for (const control of twoPassControls) {
    for (const field of [control.steps, control.denoise, control.sampler, control.scheduler]) {
      const overridden = active && control === twoPassControls[1];
      field.disabled = overridden;
      field.parentElement.style.display = overridden ? "none" : "";
    }
  }
  return active;
}
