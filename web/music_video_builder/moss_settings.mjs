// The settings MOSS-Image is run with, and where they are kept. One place for everything that uses MOSS-Image: the Reference Builder's
// generator, the Thumbnail dialog and the MOSS-Image tab of Models > Select models. They share one saved object, so a model picked in
// one place is the model used in the others. No browser code besides the storage lookup, so tests/moss_settings.cjs can run it.

// The key the Reference Builder has always used, so settings saved before the Models tab existed are kept.
export const MOSS_REFERENCE_STORAGE_KEY = "vrgdg_moss_image_reference_settings";
export const MOSS_TURBO_LORA = "moss_image_sft_8_steps_dmd_lora.safetensors";
// chinoll/Moss-Image-Edit downloads as adapter_model.safetensors and is renamed to this (see the folder locations in the Models dialog).
export const MOSS_EDIT_LORA = "moss_image_edit_lora.safetensors";
export const MOSS_PRESETS = {
  turbo: { lora_name: MOSS_TURBO_LORA, steps: 8, cfg: 1 },
  quality: { lora_name: "[none]", steps: 50, cfg: 3 },
};
export const DEFAULT_MOSS_REFERENCE_SETTINGS = {
  unet_name: "moss_image_sft_bf16.safetensors",
  clip_name: "qwen3.5_9b_int8_convrot.safetensors",
  vae_name: "flux2-vae.safetensors",
  preset: "turbo",
  ...MOSS_PRESETS.turbo,
  edit_lora_name: MOSS_EDIT_LORA,
  width: 1920,
  height: 1088,
  seed: 1,
  seed_mode: "random",
};

function mossStorage() {
  try {
    return globalThis.localStorage || null;
  } catch (_error) {
    return null;
  }
}

// The saved settings over the defaults. Storage that is missing, blocked or holds something else gives the defaults.
export function loadMossReferenceSettings(storage = mossStorage()) {
  try {
    const saved = JSON.parse(storage?.getItem(MOSS_REFERENCE_STORAGE_KEY) || "{}");
    return { ...DEFAULT_MOSS_REFERENCE_SETTINGS, ...(saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {}) };
  } catch (_error) {
    return { ...DEFAULT_MOSS_REFERENCE_SETTINGS };
  }
}

export function saveMossReferenceSettings(settings, storage = mossStorage()) {
  try {
    storage?.setItem(MOSS_REFERENCE_STORAGE_KEY, JSON.stringify(settings));
    return true;
  } catch (_error) {
    // Per-viewer convenience only: when the browser keeps nothing, the settings last until the page is closed.
    return false;
  }
}

function mossText(value, fallback) {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function mossWhole(value, fallback, min, max) {
  const number = Math.floor(Number(value));
  return Number.isFinite(number) ? Math.max(min, Math.min(max, number)) : fallback;
}

// A size MOSS-Image can make: a multiple of 16 (its latent is 16 times smaller), between 256 and 4096.
export function mossSize(value, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return fallback;
  return Math.max(256, Math.min(4096, Math.round(number / 16) * 16));
}

// Settings in a known shape, for the panel that edits them. Empty or wrong values fall back to the defaults.
export function normalizeMossSettings(value) {
  const input = { ...DEFAULT_MOSS_REFERENCE_SETTINGS, ...(value && typeof value === "object" && !Array.isArray(value) ? value : {}) };
  const cfg = Number(input.cfg);
  return {
    unet_name: mossText(input.unet_name, DEFAULT_MOSS_REFERENCE_SETTINGS.unet_name),
    clip_name: mossText(input.clip_name, DEFAULT_MOSS_REFERENCE_SETTINGS.clip_name),
    vae_name: mossText(input.vae_name, DEFAULT_MOSS_REFERENCE_SETTINGS.vae_name),
    lora_name: mossText(input.lora_name, "[none]"),
    edit_lora_name: mossText(input.edit_lora_name, DEFAULT_MOSS_REFERENCE_SETTINGS.edit_lora_name),
    preset: ["turbo", "quality", "custom"].includes(input.preset) ? input.preset : "custom",
    steps: mossWhole(input.steps, DEFAULT_MOSS_REFERENCE_SETTINGS.steps, 1, 200),
    cfg: Number.isFinite(cfg) ? Math.max(1, Math.min(20, cfg)) : DEFAULT_MOSS_REFERENCE_SETTINGS.cfg,
    width: mossSize(input.width, DEFAULT_MOSS_REFERENCE_SETTINGS.width),
    height: mossSize(input.height, DEFAULT_MOSS_REFERENCE_SETTINGS.height),
    seed: mossWhole(input.seed, DEFAULT_MOSS_REFERENCE_SETTINGS.seed, 0, Number.MAX_SAFE_INTEGER),
    seed_mode: input.seed_mode === "fixed" ? "fixed" : "random",
  };
}

// The Reference Builder's MOSS-Image request: the saved settings and a prompt.
export function mossReferencePayload(prompt, settings = {}) {
  const moss = { ...DEFAULT_MOSS_REFERENCE_SETTINGS, ...settings };
  return {
    prompt: String(prompt || "").trim(),
    unet_name: moss.unet_name,
    clip_name: moss.clip_name,
    vae_name: moss.vae_name,
    lora_name: moss.lora_name || "[none]",
    steps: Number(moss.steps) || 8,
    cfg: Number(moss.cfg) || 1,
    width: Number(moss.width) || 1920,
    height: Number(moss.height) || 1088,
    seed: Number(moss.seed) || 0,
    seed_mode: moss.seed_mode || "random",
    batch_size: 1,
  };
}
