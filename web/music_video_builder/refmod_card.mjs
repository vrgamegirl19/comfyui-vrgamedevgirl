import { api } from "../../../scripts/api.js";
import { BUILDER_FONT_STACK } from "./constants.mjs";
import { toast } from "./controls.mjs";
import { DEFAULT_QUALITY, QUALITY_PRESETS } from "./refmod_trim.mjs";

// The RefMod part of a Reference Builder card: choose a saved RefMod instead of an image, set how strongly it is used,
// and (for clothing) say who wears it. Used by the subject and location cards in the RefMod pipeline.
// The card keeps whatever image it had, so switching a project back to the standard pipeline loses nothing.

// Folder (RefMod type) a card of each reference type picks from. A clothing card may also be limited to one set.
export const REFMOD_FOLDERS_BY_REFERENCE_TYPE = {
  character: ["identity"],
  outfit: ["clothing_men", "clothing_women"],
  environment: ["background"],
  style: ["style"],
  object: ["object"],
  prop: ["prop"],
  vehicle: ["vehicle"],
  creature: ["creature"],
  other: ["generic", "pose_motion"],
};
const CLOTHING_FOLDERS_BY_SET = { men: ["clothing_men"], women: ["clothing_women"], all: ["clothing_men", "clothing_women"] };

export function refmodFoldersFor(item, kind = "subject") {
  if (kind === "location") return REFMOD_FOLDERS_BY_REFERENCE_TYPE.environment;
  const type = String(item?.reference_type || "character");
  if (type === "outfit") return CLOTHING_FOLDERS_BY_SET[item?.clothing_set] || CLOTHING_FOLDERS_BY_SET.all;
  return REFMOD_FOLDERS_BY_REFERENCE_TYPE[type] || REFMOD_FOLDERS_BY_REFERENCE_TYPE.other;
}

let libraryPromise = null;

// The saved RefMods, fetched once and reused. Pass true to read the folders again.
export function loadRefmodLibrary(force = false) {
  if (force || !libraryPromise) {
    libraryPromise = api.fetchApi("/vrgdg/refmod/library")
      .then((response) => response.json())
      .then((result) => (result?.ok ? result.refmods || [] : []))
      .catch(() => []);
  }
  return libraryPromise;
}

export function refmodPreviewUrl(name) {
  return api.apiURL(`/vrgdg/refmod/preview?name=${encodeURIComponent(name)}`);
}

const baseName = (name) => String(name || "").split("/").pop();
const prettyName = (name) => baseName(name).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().replace(/\b\w/g, (char) => char.toUpperCase());
const isPlaceholderName = (name) => !String(name || "").trim() || /^(character|location|subject)\s+\d+$/i.test(String(name).trim());

const SELECT_STYLE = `width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:7px;font-family:${BUILDER_FONT_STACK};font-size:12px;`;

function labelled(text, control) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = `display:flex;flex-direction:column;gap:4px;font-family:${BUILDER_FONT_STACK};font-size:11px;color:#cbd5e1;font-weight:600;min-width:0;`;
  const span = document.createElement("span");
  span.textContent = text;
  wrapper.append(span, control);
  return wrapper;
}

function makeNativeSelect(options, value) {
  const select = document.createElement("select");
  select.style.cssText = SELECT_STYLE;
  for (const [optionValue, label] of options) select.append(new Option(label, optionValue));
  select.value = value;
  return select;
}

// Build the RefMod panel for one card.
//   item:       the subject or location card (mutated in place)
//   kind:       "subject" or "location"
//   characters: the character cards (for the "worn by" choice on clothing)
//   onChange:   called after a change; pass true when the card layout should be rebuilt
export function buildRefmodPicker({ item, kind = "subject", characters = [], onChange = () => {} }) {
  const panel = document.createElement("div");
  panel.style.cssText = "border:1px solid #155e75;border-radius:7px;background:#06202e;padding:10px;display:flex;flex-direction:column;gap:10px;";

  const sourceSelect = makeNativeSelect([["image", "Reference image"], ["refmod", "Saved RefMod"]], item.source === "refmod" ? "refmod" : "image");
  sourceSelect.onchange = () => {
    item.source = sourceSelect.value;
    onChange(true);
  };
  const top = document.createElement("div");
  top.style.cssText = "display:grid;grid-template-columns:minmax(150px,220px) 1fr;gap:10px;align-items:end;";
  const hint = document.createElement("div");
  hint.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.4;";
  hint.textContent = item.source === "refmod"
    ? "This card renders from a saved RefMod. Its picture is not used in the RefMod pipeline."
    : "This card renders from its image. In the RefMod pipeline a scene needs a saved RefMod, so pick one or save this image as a RefMod.";
  top.append(labelled("Source", sourceSelect), hint);
  panel.append(top);
  if (item.source !== "refmod") {
    // Turn this card's image into a saved RefMod, in the folder for its type, under the card's name. It is made once and
    // then used by every scene and every project.
    const hasImage = Boolean(String(item.image?.path || "").trim() || String(item.image?.data || "").trim());
    if (hasImage) {
      const folders = refmodFoldersFor(item, kind);
      const folderSelect = makeNativeSelect(folders.map((folder) => [folder, folder]), folders[0]);
      const qualitySelect = makeNativeSelect(QUALITY_PRESETS.map((preset) => [preset.key, preset.label]), DEFAULT_QUALITY);
      qualitySelect.title = QUALITY_PRESETS.map((preset) => `${preset.label}: ${preset.detail}`).join("\n");
      // More images of the same subject make a stronger RefMod. They are uploaded when picked and sent with the save.
      const extraPaths = [];
      const extraInput = document.createElement("input");
      extraInput.type = "file";
      extraInput.accept = "image/*";
      extraInput.multiple = true;
      extraInput.style.display = "none";
      const addMore = document.createElement("button");
      addMore.type = "button";
      addMore.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#27272a;color:#f4f4f5;font-size:12px;font-weight:600;padding:7px 11px;cursor:pointer;";
      const syncAddMore = () => { addMore.textContent = extraPaths.length ? `More images: ${extraPaths.length} (click to clear)` : "Add more images"; };
      syncAddMore();
      addMore.onclick = () => {
        if (extraPaths.length) {
          extraPaths.length = 0;
          syncAddMore();
          return;
        }
        extraInput.click();
      };
      extraInput.onchange = async () => {
        const files = [...extraInput.files];
        extraInput.value = "";
        addMore.disabled = true;
        try {
          for (const [index, file] of files.entries()) {
            addMore.textContent = `Uploading ${index + 1} of ${files.length}...`;
            const form = new FormData();
            form.append("image", file, file.name);
            const response = await api.fetchApi("/vrgdg/refmod/upload_image", { method: "POST", body: form });
            const result = await response.json();
            if (!response.ok || !result.ok) throw new Error(result.error || "The upload failed.");
            extraPaths.push(result.path);
          }
        } catch (error) {
          toast(`Could not add the image: ${error?.message || error}`, true);
        } finally {
          addMore.disabled = false;
          syncAddMore();
        }
      };
      const save = document.createElement("button");
      save.type = "button";
      save.textContent = "Save this image as a RefMod";
      save.style.cssText = "border:1px solid #0891b2;border-radius:6px;background:#06b6d4;color:#082f49;font-size:12px;font-weight:700;padding:7px 11px;cursor:pointer;";
      const saveRow = document.createElement("div");
      saveRow.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;align-items:end;";
      const saveNote = document.createElement("div");
      saveNote.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.4;";
      saveNote.style.gridColumn = "1 / -1";
      saveNote.textContent = "Saved to models/refmods under this card's name. One image gives a much lighter RefMod than several, and a light character can be duplicated by heavier ones in the same scene. Add more images of the same subject, or use RefMods Studio for trimming and per-image control.";
      saveRow.append(labelled("Folder", folderSelect), labelled("Quality", qualitySelect), addMore, save, extraInput, saveNote);
      panel.append(saveRow);
      const send = async (overwrite) => {
        const response = await api.fetchApi("/vrgdg/refmod/from_image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            type: folderSelect.value, name: String(item.name || "").trim(), image: item.image,
            extra_images: extraPaths.map((path) => ({ path })), quality: qualitySelect.value,
            description: String(item.description || "").trim(), overwrite,
          }),
        });
        return { response, result: await response.json() };
      };
      save.onclick = async () => {
        if (!String(item.name || "").trim()) {
          toast("Give the card a name first. The RefMod is saved under it.", true);
          return;
        }
        save.disabled = true;
        save.textContent = "Saving...";
        try {
          let { response, result } = await send(false);
          if (response.status === 409 && result.exists) {
            if (!window.confirm(`A RefMod with this name already exists:

${result.path}

Replace it?`)) return;
            ({ response, result } = await send(true));
          }
          if (!response.ok || !result.ok) throw new Error(result.error || "The RefMod could not be saved.");
          const library = await loadRefmodLibrary(true);
          const saved = library.find((entry) => entry.path === result.path || entry.name === `${result.folder}/${result.name}`);
          item.source = "refmod";
          item.refmod = {
            name: saved?.name || `${result.folder}/${result.name}`, folder: result.folder,
            type: saved?.type || result.folder, kind: saved?.kind === "image" ? "image" : "video",
            tokens: saved?.tokens || result.tokens || 0, frames: saved?.frames || result.latent_frames || 1, strength: 1,
          };
          toast(`Saved as RefMod: ${item.refmod.name}`);
          onChange(true);
        } catch (error) {
          toast(`Could not save the RefMod: ${error?.message || error}`, true);
        } finally {
          save.disabled = false;
          save.textContent = "Save this image as a RefMod";
        }
      };
    }
    return panel;
  }

  // Clothing: which set it is picked from, who wears it, and whether it follows that character into every scene.
  if (kind === "subject" && String(item.reference_type) === "outfit") {
    const setSelect = makeNativeSelect([["all", "All clothing"], ["men", "Men's clothing"], ["women", "Women's clothing"]], item.clothing_set || "all");
    setSelect.onchange = () => {
      item.clothing_set = setSelect.value;
      onChange(true);
    };
    const wearsSelect = makeNativeSelect(
      [["", "Nobody in particular"], ...characters.map((card) => [String(card.id), card.name || "Character"])],
      String(item.wears || ""),
    );
    wearsSelect.onchange = () => {
      item.wears = wearsSelect.value;
      onChange(false);
    };
    const follow = document.createElement("label");
    follow.style.cssText = "display:flex;gap:7px;align-items:center;font-size:12px;color:#e2e8f0;";
    const followInput = document.createElement("input");
    followInput.type = "checkbox";
    followInput.checked = item.follow !== false;
    followInput.onchange = () => {
      item.follow = followInput.checked;
      onChange(false);
    };
    follow.append(followInput, document.createTextNode("Goes with this character into every scene they are in"));
    const clothingRow = document.createElement("div");
    clothingRow.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;align-items:end;";
    clothingRow.append(labelled("Clothing set", setSelect), labelled("Worn by", wearsSelect), follow);
    panel.append(clothingRow);
  }

  const body = document.createElement("div");
  body.style.cssText = "display:grid;grid-template-columns:110px minmax(0,1fr);gap:12px;align-items:start;";
  const preview = document.createElement("div");
  preview.style.cssText = "width:110px;height:110px;border:1px solid #334155;border-radius:6px;background:#020617;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#64748b;font-size:10px;text-align:center;";
  preview.textContent = "No preview";
  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;flex-direction:column;gap:8px;min-width:0;";
  const modSelect = makeNativeSelect([["", "Loading RefMods..."]], "");
  const info = document.createElement("div");
  info.style.cssText = "font-size:11px;color:#a5f3fc;line-height:1.4;";
  const strengthRow = document.createElement("div");
  strengthRow.style.cssText = "display:grid;grid-template-columns:1fr 70px;gap:8px;align-items:center;";
  const strengthSlider = document.createElement("input");
  strengthSlider.type = "range";
  strengthSlider.min = "0";
  strengthSlider.max = "1";
  strengthSlider.step = "0.05";
  const strengthNumber = document.createElement("input");
  strengthNumber.type = "number";
  strengthNumber.min = "0";
  strengthNumber.max = "1";
  strengthNumber.step = "0.05";
  strengthNumber.style.cssText = SELECT_STYLE;
  strengthRow.append(strengthSlider, strengthNumber);
  const strengthField = labelled("Strength (how strongly the RefMod is used in every scene it is in)", strengthRow);
  const useDescription = document.createElement("button");
  useDescription.type = "button";
  useDescription.textContent = "Use the RefMod's description";
  useDescription.style.cssText = "align-self:flex-start;border:1px solid #3f3f46;border-radius:6px;background:#27272a;color:#f4f4f5;font-size:11px;font-weight:600;padding:5px 9px;cursor:pointer;";
  controls.append(labelled("RefMod", modSelect), info, strengthField, useDescription);
  body.append(preview, controls);
  panel.append(body);

  let entries = [];
  const current = () => entries.find((entry) => entry.name === item.refmod?.name) || null;

  const refreshInfo = () => {
    const entry = current();
    const chosen = item.refmod;
    if (!chosen) {
      info.textContent = "Pick a RefMod for this card.";
      preview.replaceChildren(document.createTextNode("No preview"));
      strengthField.style.opacity = "0.5";
      useDescription.style.display = "none";
      return;
    }
    strengthField.style.opacity = "1";
    if (!entry) {
      info.textContent = `${chosen.name} was not found in models/refmods. Pick it again or restore the file.`;
      info.style.color = "#fbbf24";
      useDescription.style.display = "none";
      return;
    }
    info.style.color = "#a5f3fc";
    info.textContent = `${entry.kind === "image" ? "Single image" : `${entry.frames} images`} · ${entry.canvas[0]}x${entry.canvas[1]} · ${entry.tokens.toLocaleString()} tokens`
      + (entry.kind === "image" ? " · labelled <Picture n>" : " · labelled <Video n>");
    useDescription.style.display = entry.description ? "" : "none";
    preview.replaceChildren();
    if (entry.has_preview) {
      const image = document.createElement("img");
      image.src = refmodPreviewUrl(entry.name);
      image.alt = entry.name;
      image.style.cssText = "max-width:100%;max-height:100%;object-fit:contain;";
      image.onerror = () => preview.replaceChildren(document.createTextNode("No preview"));
      preview.append(image);
    } else {
      preview.append(document.createTextNode("No preview yet"));
    }
  };

  const syncStrength = (value) => {
    const strength = Math.min(1, Math.max(0, Number(value)));
    const safe = Number.isFinite(strength) ? strength : 1;
    if (item.refmod) item.refmod.strength = safe;
    strengthSlider.value = String(safe);
    strengthNumber.value = String(Math.round(safe * 100) / 100);
  };
  strengthSlider.oninput = () => { syncStrength(strengthSlider.value); onChange(false); };
  strengthNumber.onchange = () => { syncStrength(strengthNumber.value); onChange(false); };
  syncStrength(item.refmod?.strength ?? 1);

  modSelect.onchange = () => {
    const entry = entries.find((candidate) => candidate.name === modSelect.value);
    if (!entry) {
      item.refmod = undefined;
      refreshInfo();
      onChange(true);
      return;
    }
    const previousStrength = item.refmod?.strength ?? 1;
    item.refmod = {
      name: entry.name, folder: entry.folder, type: entry.type, kind: entry.kind === "image" ? "image" : "video",
      tokens: entry.tokens, frames: entry.frames, strength: previousStrength,
    };
    // A new card takes the RefMod's name and description, so nothing has to be typed or described again.
    if (isPlaceholderName(item.name)) item.name = prettyName(entry.name);
    if (!String(item.description || "").trim() && entry.description) item.description = entry.description;
    refreshInfo();
    onChange(true);
  };
  useDescription.onclick = () => {
    const entry = current();
    if (!entry?.description) return;
    item.description = entry.description;
    onChange(true);
  };

  loadRefmodLibrary().then((library) => {
    const folders = new Set(refmodFoldersFor(item, kind));
    entries = library;
    const choices = library.filter((entry) => folders.has(entry.folder) || entry.name === item.refmod?.name);
    modSelect.replaceChildren(new Option(choices.length ? "(choose a RefMod)" : `No RefMods in ${[...folders].join(" or ")} yet. Make one in RefMods Studio.`, ""));
    for (const entry of choices) {
      modSelect.append(new Option(`${prettyName(entry.name)}  ·  ${entry.folder}  ·  ${entry.tokens.toLocaleString()} tokens`, entry.name));
    }
    modSelect.value = item.refmod?.name || "";
    refreshInfo();
  });
  refreshInfo();
  return panel;
}
