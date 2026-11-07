// How each image workflow is asked to make the thumbnail from the chosen frame. The dialog hands in the Builder's own settings for the
// workflow (the models, LoRAs and sizes the user already set for it) and gets back the build route and its payload. No browser
// code in here, so tests/thumbnail_engines.cjs can run it.
//
// "edit" workflows (NanoBanana, Flux/Klein, MOSS-Image Edit) take the frame as a reference picture and follow an instruction.
// "restyle" workflows (Krea 2, Z-Image, Ernie, MOSS-Image) use it as an image-to-image source, and how far they may move away from it
// is the `change` setting (0 to 1). A restyle keeps the frame's composition and mostly ignores what the prompt asks to add, so only
// an edit workflow can be relied on to draw the title text.

// The build route of each workflow, written out in full so a search for a route finds this module.
const ENGINE_ROUTES = {
  nano_banana: "/vrgdg/workflow_runner/build_nb_image_prompt",
  flux_klein: "/vrgdg/workflow_runner/build_flux_klein_prompt",
  moss_image_edit: "/vrgdg/workflow_runner/build_moss_image_prompt",
  krea2_2pass: "/vrgdg/workflow_runner/build_krea2_2pass_prompt",
  zimage: "/vrgdg/workflow_runner/build_zimage_prompt",
  ernie_image: "/vrgdg/workflow_runner/build_ernie_image_prompt",
  moss_image: "/vrgdg/workflow_runner/build_moss_image_prompt",
};

export const THUMBNAIL_ENGINE_LABELS = {
  overlay: "Overlay: your frame exactly, no AI (grade, glow, title)",
  nano_banana: "NanoBanana (API, edit)",
  flux_klein: "Flux/Klein (local, edit)",
  moss_image_edit: "MOSS-Image Edit (local, edit)",
  krea2_2pass: "Krea 2 (local, restyle)",
  zimage: "Z-Image (local, restyle)",
  ernie_image: "Ernie (local, restyle)",
  moss_image: "MOSS-Image (local, restyle)",
};

// The Overlay engine is not an image workflow: it has its own route, which keeps the frame and puts the grade and the title on top,
// and saves the result in the project itself (so there is no ComfyUI queue and no separate save). `{ url, payload, label }`.
export function thumbnailOverlayRequest({ sourcePath, title, overlay, projectFolder }) {
  const path = String(sourcePath || "").trim();
  if (!path) throw new Error("Choose the frame first: scrub the video to the moment you want and click Use this frame.");
  const folder = String(projectFolder || "").trim();
  if (!folder) throw new Error("Create or load a project before making a thumbnail.");
  const look = overlay && typeof overlay === "object" ? overlay : {};
  return {
    url: "/vrgdg/music_builder/thumbnail/overlay",
    payload: { project_folder: folder, source_path: path, title: String(title || ""), color: look.color, position: look.position, strength: look.strength },
    label: THUMBNAIL_ENGINE_LABELS.overlay,
  };
}

function unitChange(change) {
  if (change === null || change === undefined || change === "") return 0.65;
  const value = Number(change);
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.65;
}

// Z-Image and Ernie start sampling at a step of 8: step 1 changes the frame most, step 8 least.
export function startStepFor(change) {
  return Math.max(1, Math.min(8, Math.round(1 + (1 - unitChange(change)) * 7)));
}

// Krea 2's "denoise when using image to image", 0 to 10: more is a bigger change.
export function creativityFor(change) {
  return Math.round(unitChange(change) * 10);
}

// MOSS-Image's sampler denoise: 1 would ignore the frame, so the range stops short of it.
export function denoiseFor(change) {
  return Math.round(Math.max(0.2, Math.min(0.95, unitChange(change))) * 100) / 100;
}

function positiveNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function loraCountOf(settings, limit) {
  const count = Math.max(0, Math.min(limit, Number(settings.lora_count || 0)));
  return settings.use_loras && count > 0 ? count : 0;
}

// The `lora_N` names and strengths of a workflow whose settings keep them as `loras: [{name, strength}]`. Krea 2 and Z-Image keep a first
// and a second pass strength as well.
function loraFields(settings, count, slots, twoPass) {
  const fields = {};
  for (let index = 0; index < slots; index += 1) {
    const lora = (Array.isArray(settings.loras) ? settings.loras[index] : null) || {};
    const on = index < count;
    const strength = Number(lora.strength ?? 1);
    fields[`lora_${index + 1}`] = on ? (lora.name || "[none]") : "[none]";
    if (twoPass) {
      fields[`first_pass_strength_${index + 1}`] = Number(lora.first_pass_strength ?? lora.strength ?? 0.5);
      fields[`second_pass_strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 1);
      fields[`strength_${index + 1}`] = Number(lora.second_pass_strength ?? lora.strength ?? 1);
    } else {
      fields[`strength_${index + 1}`] = strength;
    }
  }
  return fields;
}

// `{ url, payload, label }` for the build route of `engine`. Throws a message the user can act on when something is missing.
// `settings` is the Builder's settings object for that workflow; `loraPayload(settings)` makes Flux/Klein's LoRA fields.
export function thumbnailEngineRequest({ engine, prompt, sourcePath, change, settings, seed, loraPayload = () => ({}), defaultNbModel = "" }) {
  const route = ENGINE_ROUTES[engine];
  if (!route) throw new Error(`Unknown image workflow: ${engine}`);
  const text = String(prompt || "").trim();
  if (!text) throw new Error("Write the prompt first: click Write prompt with LLM or type one.");
  const path = String(sourcePath || "").trim();
  if (!path) throw new Error("Choose the frame first: scrub the video to the moment you want and click Use this frame.");
  const s = settings && typeof settings === "object" ? settings : {};
  const sourceFrame = [{ path }];
  let payload;

  if (engine === "nano_banana") {
    if (!String(s.api_key || "").trim()) {
      throw new Error("The NanoBanana API key is missing. Enter it in the NanoBanana image settings, or choose another image workflow.");
    }
    payload = { api_key: s.api_key, model: s.model || defaultNbModel, prompt: text, image_ingredients: sourceFrame };
  } else if (engine === "flux_klein") {
    payload = {
      prompt: text, image_ingredients: sourceFrame,
      unet_name: s.unet_name || "", clip_name: s.clip_name || "", vae_name: s.vae_name || "",
      width: positiveNumber(s.width, 1280), height: positiveNumber(s.height, 720), seed,
      ...loraPayload(s),
    };
  } else if (engine === "krea2_2pass") {
    const count = loraCountOf(s, 4);
    payload = {
      prompt: text, unet_name: s.unet_name || "", clip_name: s.clip_name || "", vae_name: s.vae_name || "",
      use_custom_loras: count > 0, lora_count: count,
      aspect_ratio: s.aspect_ratio || "16:9 (Widescreen)", sampler_name: s.sampler_name || "euler_ancestral_cfg_pp",
      cfg: Math.max(1, Math.min(1.2, Number(s.cfg ?? 1.2))),
      seed, seed_mode: "fixed", batch_size: 1,
      use_image_to_image: true, image_to_image_creativity: creativityFor(change), image_to_image_path: path,
      ...loraFields(s, count, 4, true),
    };
  } else if (engine === "zimage") {
    const count = loraCountOf(s, 4);
    payload = {
      prompt: text, unet_name: s.unet_name || "", clip_name: s.clip_name || "", vae_name: s.vae_name || "",
      first_pass_width: positiveNumber(s.first_pass_width, 1280), first_pass_height: positiveNumber(s.first_pass_height, 720),
      second_pass_width: positiveNumber(s.second_pass_width, 1920), second_pass_height: positiveNumber(s.second_pass_height, 1080),
      seed, seed_mode: "fixed", batch_size: 1, use_custom_loras: count > 0, lora_count: count, ltx_two_pass_mode: false,
      use_image_to_image: true, image_to_image_start_at_step: startStepFor(change), image_to_image_path: path,
      ...loraFields(s, count, 4, true),
    };
  } else if (engine === "ernie_image") {
    const count = loraCountOf(s, 4);
    payload = {
      prompt: text, unet_name: s.unet_name || "", clip_name: s.clip_name || "", vae_name: s.vae_name || "",
      width: positiveNumber(s.width, 1280), height: positiveNumber(s.height, 720),
      seed, seed_mode: "fixed", batch_size: 1, use_custom_loras: count > 0, lora_count: count,
      use_image_to_image: true, image_to_image_start_at_step: startStepFor(change), image_to_image_path: path,
      ...loraFields(s, count, 4, false),
    };
  } else if (engine === "moss_image_edit") {
    // The edit LoRA's own sampler (30 steps, CFG 4, skip layer guidance) and negative line are the builder's defaults, so the saved
    // Turbo steps, CFG and LoRA are not sent: they are for text to image and would spoil an edit.
    payload = {
      prompt: text, image_ingredients: sourceFrame, use_edit: true,
      unet_name: s.unet_name, clip_name: s.clip_name, vae_name: s.vae_name, edit_lora_name: s.edit_lora_name || "",
      seed, seed_mode: "fixed",
    };
  } else {
    payload = {
      prompt: text, unet_name: s.unet_name, clip_name: s.clip_name, vae_name: s.vae_name, lora_name: s.lora_name || "[none]",
      steps: positiveNumber(s.steps, 8), cfg: positiveNumber(s.cfg, 1), width: positiveNumber(s.width, 1280), height: positiveNumber(s.height, 720),
      seed, seed_mode: "fixed", batch_size: 1,
      use_image_to_image: true, image_to_image_path: path, denoise: denoiseFor(change),
    };
  }
  return { url: route, payload, label: THUMBNAIL_ENGINE_LABELS[engine] };
}
