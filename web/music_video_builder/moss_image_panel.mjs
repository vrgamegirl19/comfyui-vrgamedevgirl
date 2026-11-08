// The MOSS-Image tab of Models > Select models: the models, LoRAs and sampling MOSS-Image is run with. Choosing "MOSS-Image" in the Image
// tab's model list opens the Models dialog on it. The values are saved as they are changed, in the object the Reference Builder's MOSS-Image
// option and the Thumbnail dialog already read (moss_settings.mjs), so one choice is the choice everywhere.

import { getJson } from "./comfy_api.mjs";
import { BUILDER_FONT_STACK } from "./constants.mjs";
import { makeField, makeInput, makeSearchableLoraPicker, makeSelect, makeSettingsSection } from "./controls.mjs";
import { wireSearchablePicker } from "./model_pickers.mjs";
import {
  loadMossReferenceSettings, MOSS_EDIT_LORA, MOSS_PRESETS, normalizeMossSettings, saveMossReferenceSettings,
} from "./moss_settings.mjs";

// The label of the tab, and the value of the option it adds to the Image tab's model list.
export const MOSS_PANEL_LABEL = "MOSS-Image";
export const MOSS_OPTION_VALUE = "moss_image_settings";

const EDIT_LORA_MISSING = `The Edit LoRA is not in ComfyUI/models/loras yet. Install it with the Edit LoRA button under Models > Download models > Image Models > MOSS-Image, which saves it as ${MOSS_EDIT_LORA}. It needs a ComfyUI with the Text Encode MOSS-Image Edit node (newer than v0.42.1).`;

// Adds "MOSS-Image (models)" to the Image tab's model list, before "+ Custom". MOSS-Image has its models and settings here, not a scene
// generator, so choosing it calls `openModels` and puts the list back on the mode in use. Every other choice goes to the list's own handler.
export function addMossImageOption(select, openModels) {
  const option = document.createElement("option");
  option.value = MOSS_OPTION_VALUE;
  option.textContent = "MOSS-Image (models)";
  select.insertBefore(option, select.querySelector('option[value="custom_image"]'));
  const switchMode = select.onchange;
  select.onchange = (event) => {
    if (select.value !== MOSS_OPTION_VALUE) return switchMode?.call(select, event);
    select.value = select.dataset.activeMode;
    openModels();
    return undefined;
  };
  return option;
}

function mossFileName(value) {
  return String(value || "").replaceAll("\\", "/").split("/").pop();
}

// Whether `name` is one of the installed files (a model in a subfolder matches by its file name).
export function mossIsInstalled(options, name) {
  const wanted = String(name || "").trim();
  return Boolean(wanted) && (options || []).some((item) => item === wanted || mossFileName(item) === mossFileName(wanted));
}

// The installed spelling of `name` when there is one, otherwise `name` itself. A model that is not installed is never swapped for another
// model behind the user's back: it stays as it was saved and the panel says it is missing.
export function mossInstalledSpelling(options, name) {
  const wanted = String(name || "").trim();
  return (options || []).find((item) => item === wanted) || (options || []).find((item) => mossFileName(item) === mossFileName(wanted)) || wanted;
}

function mossNote(text) {
  const note = document.createElement("div");
  note.textContent = text;
  note.style.cssText = `font-family:${BUILDER_FONT_STACK};font-size:12px;line-height:1.45;color:#a1a1aa;`;
  return note;
}

function setMossPickerOptions(picker, options) {
  picker.options = Array.from(new Set((options || []).map((item) => String(item || "").trim()).filter(Boolean)));
  picker.input.value = mossInstalledSpelling(picker.options, picker.input.value);
}

// `fetchJson` and `storage` are only given by tests.
export function createMossImagePanel({ fetchJson = getJson, storage } = {}) {
  const saved = normalizeMossSettings(loadMossReferenceSettings(storage));
  const pickers = {
    unet: makeSearchableLoraPicker(saved.unet_name),
    clip: makeSearchableLoraPicker(saved.clip_name),
    vae: makeSearchableLoraPicker(saved.vae_name),
    lora: makeSearchableLoraPicker(saved.lora_name),
    editLora: makeSearchableLoraPicker(saved.edit_lora_name),
  };
  const preset = makeSelect([
    { value: "turbo", label: "Turbo (8-step LoRA)" },
    { value: "quality", label: "Quality (no LoRA, 50 steps)" },
    { value: "custom", label: "Custom" },
  ], saved.preset);
  const steps = makeInput(String(saved.steps), "number");
  const cfg = makeInput(String(saved.cfg), "number");
  const width = makeInput(String(saved.width), "number");
  const height = makeInput(String(saved.height), "number");
  const seed = makeInput(String(saved.seed), "number");
  const seedMode = makeSelect(["random", "fixed"], saved.seed_mode);
  steps.min = "1";
  cfg.min = "1";
  cfg.step = "0.1";
  seed.min = "0";
  for (const input of [width, height]) {
    input.min = "256";
    input.step = "16";
  }
  const editLoraNote = mossNote("");
  editLoraNote.style.color = "#fcd34d";
  editLoraNote.style.display = "none";
  const status = mossNote("");
  let loraListLoaded = false;

  // What the panel shows, as settings: normalized (a size becomes a multiple of 16), written back so the boxes show what is kept, saved
  // on top of what is stored (so a setting this panel does not show is kept) and returned.
  const read = () => {
    const settings = normalizeMossSettings({
      unet_name: pickers.unet.input.value, clip_name: pickers.clip.input.value, vae_name: pickers.vae.input.value,
      lora_name: pickers.lora.input.value, edit_lora_name: pickers.editLora.input.value, preset: preset.value,
      steps: steps.value, cfg: cfg.value, width: width.value, height: height.value, seed: seed.value, seed_mode: seedMode.value,
    });
    for (const [input, value] of [[steps, settings.steps], [cfg, settings.cfg], [width, settings.width], [height, settings.height], [seed, settings.seed]]) {
      input.value = String(value);
    }
    saveMossReferenceSettings({ ...loadMossReferenceSettings(storage), ...settings }, storage);
    status.textContent = "Saved. The Reference Builder's MOSS-Image option and the Thumbnail dialog use these.";
    return settings;
  };

  const paintEditLoraNote = () => {
    const name = String(pickers.editLora.input.value || "").trim();
    const missing = loraListLoaded && name !== "" && name !== "[none]" && !mossIsInstalled(pickers.editLora.options, name);
    editLoraNote.textContent = missing ? EDIT_LORA_MISSING : "";
    editLoraNote.style.display = missing ? "block" : "none";
  };

  // A preset sets the LoRA, the steps and the CFG it stands for. Changing any of those by hand makes the preset Custom.
  preset.addEventListener("change", () => {
    const chosen = MOSS_PRESETS[preset.value];
    if (chosen) {
      pickers.lora.input.value = chosen.lora_name;
      steps.value = String(chosen.steps);
      cfg.value = String(chosen.cfg);
    }
    read();
  });
  for (const input of [steps, cfg]) {
    input.addEventListener("input", () => { preset.value = "custom"; });
    input.addEventListener("change", read);
  }
  for (const control of [width, height, seed, seedMode]) control.addEventListener("change", read);
  for (const [name, picker] of Object.entries(pickers)) {
    wireSearchablePicker(picker, () => {
      if (name === "lora") preset.value = "custom";
      paintEditLoraNote();
      read();
    });
  }

  // The installed models and LoRAs, from the routes the other model panels use. A failure leaves the saved names as they are.
  const refresh = async () => {
    try {
      const data = await fetchJson("/vrgdg/workflow_runner/i2v_choices");
      setMossPickerOptions(pickers.unet, data.unets);
      setMossPickerOptions(pickers.clip, data.clip);
      setMossPickerOptions(pickers.vae, data.vae);
    } catch (_error) {
      // The pickers keep what they have.
    }
    try {
      const data = await fetchJson("/vrgdg/workflow_runner/lora_list");
      const loras = ["[none]", ...(data.loras || [])];
      setMossPickerOptions(pickers.lora, loras);
      setMossPickerOptions(pickers.editLora, loras);
      loraListLoaded = true;
    } catch (_error) {
      // The pickers keep what they have.
    }
    paintEditLoraNote();
  };

  const content = document.createElement("div");
  content.style.cssText = "display:flex;flex-direction:column;gap:12px;min-width:0;";
  const pair = (...fields) => {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;";
    row.append(...fields);
    return row;
  };
  content.append(
    mossNote("MOSS-Image is used by the Reference Builder's MOSS-Image option and by the Thumbnail dialog. The LoRAs are optional."),
    makeSettingsSection("MOSS-Image models", [
      makeField("Diffusion model", pickers.unet.wrapper),
      makeField("Text encoder", pickers.clip.wrapper),
      makeField("VAE", pickers.vae.wrapper),
    ]),
    makeSettingsSection("LoRAs", [
      makeField("Preset", preset),
      makeField("Sampling LoRA (the Turbo LoRA for 8 steps)", pickers.lora.wrapper),
      makeField("Edit LoRA (chinoll/Moss-Image-Edit)", pickers.editLora.wrapper),
      editLoraNote,
    ]),
    makeSettingsSection("Sampling", [
      pair(makeField("Steps", steps), makeField("CFG", cfg)),
      pair(makeField("Width", width), makeField("Height", height)),
      pair(makeField("Seed", seed), makeField("Seed mode", seedMode)),
    ]),
    status,
  );
  // The Models dialog moves a tab's content in and out of the page and puts it back where it was, so it needs a parent to go back to.
  const holder = document.createElement("div");
  holder.style.display = "none";
  holder.append(content);
  const ready = refresh();

  return {
    modelSelection: { label: MOSS_PANEL_LABEL, content }, holder, read, refresh, ready, pickers,
    controls: { preset, steps, cfg, width, height, seed, seedMode }, editLoraNote, status,
  };
}
