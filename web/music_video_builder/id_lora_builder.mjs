import { makeEditorImageUrl } from "./comfy_api.mjs";
import {
  escapeHtml,
  makeButton,
  makeCheckbox,
  makeEditField,
  makeField,
  makeInput,
  makeSelect,
  makeSubTabs,
  normalizeProjectVideoEngine,
  toast,
} from "./controls.mjs";
import { formatTime } from "./format.mjs";
import { pickPath } from "./project_setup.mjs";
import {
  estimateIdLoraDialogueDuration,
  idLoraSceneEntry,
  normalizeIdLoraReferenceBuilder,
} from "./reference_data.mjs";



export function createIdLoraBuilder({
  activeSegment, autoSaveSessionQuiet, drawWaveform, openFluxReferenceBuilderModal,
  openIngredientsReferenceBuilderModal, pushHistory, render, renderSegments, sceneDisplayName,
  setBaseSegmentDurationRipple, setMiniMaxH3ModeForSegment, state, syncInspector, syncMiniMaxH3Panel,
  syncVideoModePanel,
}) {
  function openIdLoraReferenceBuilderModal() {
    let refs = normalizeIdLoraReferenceBuilder(state.idLoraReferenceBuilder);
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(1380px,calc(100vw - 42px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;box-sizing:border-box;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">ID-LoRA Ref Builder</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Characters, voices, locations, and scene casting for ID-LoRA I2V.</div>`;
    const close = makeButton("Close");
    header.append(title, close);

    const charactersPanel = document.createElement("div");
    const locationsPanel = document.createElement("div");
    const castingPanel = document.createElement("div");
    for (const panel of [charactersPanel, locationsPanel, castingPanel]) {
      panel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #334155;border-radius:0 7px 7px 7px;background:#0b1220;padding:10px;max-height:min(64vh,740px);overflow:auto;";
    }
    const tabs = makeSubTabs([
      { value: "characters", label: "Characters", content: charactersPanel },
      { value: "locations", label: "Locations", content: locationsPanel },
      { value: "casting", label: "Scene Casting", content: castingPanel },
    ]);

    const rowCardStyle = "border:1px solid #334155;border-radius:7px;background:linear-gradient(135deg,#0f172a,#111827);padding:10px;display:flex;flex-direction:column;gap:8px;";
    const fieldGridStyle = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;";
    const idLoraImageSrc = (image = {}) => image?.data || (image?.path ? makeEditorImageUrl(image.path) : "");
    const makeIdLoraThumb = (image = {}, label = "Reference") => {
      const wrap = document.createElement("div");
      wrap.style.cssText = "width:76px;height:58px;border:1px solid #155e75;border-radius:6px;background:#061620;overflow:hidden;display:flex;align-items:center;justify-content:center;color:#67e8f9;font-size:10px;font-weight:900;text-align:center;";
      const src = idLoraImageSrc(image);
      if (src) {
        const img = document.createElement("img");
        img.src = src;
        img.alt = label;
        img.draggable = false;
        img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
        wrap.append(img);
      } else {
        wrap.textContent = "No img";
      }
      return wrap;
    };
    const makeTinyButton = (label) => {
      const button = makeButton(label, "neutral");
      button.style.cssText += "padding:6px 9px;font-size:11px;";
      return button;
    };
    const selectOptionsFromItems = (items, fallbackLabel) => {
      const options = [{ value: "", label: fallbackLabel }];
      for (const item of items || []) {
        const name = String(item.name || "").trim();
        if (name) options.push({ value: item.id, label: name });
      }
      return options;
    };
    const syncSelectOptions = (select, options, value = "") => {
      select.innerHTML = "";
      for (const option of options) {
        const el = document.createElement("option");
        el.value = option.value;
        el.textContent = option.label;
        select.append(el);
      }
      select.value = value;
    };
    const saveRefsQuiet = async (reason = "ID-LoRA ref builder") => {
      state.idLoraReferenceBuilder = normalizeIdLoraReferenceBuilder(refs);
      await autoSaveSessionQuiet(reason);
    };
    const renderCharacters = () => {
      charactersPanel.innerHTML = "";
      const actions = document.createElement("div");
      actions.style.cssText = "display:flex;justify-content:flex-end;";
      const add = makeButton("Add Character", "primary");
      actions.append(add);
      charactersPanel.append(actions);
      add.onclick = () => {
        refs.characters.push({
          id: `id_lora_char_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
          name: `Character ${refs.characters.length + 1}`,
          description: "",
          image: { path: "", data: "", name: "" },
          voice_audio_path: "",
          voice_trim_start: 0,
          voice_trim_duration: 0,
          identity_guidance_scale: 3,
          speech_style: "",
        });
        renderAll();
      };
      if (!refs.characters.length) {
        const empty = document.createElement("div");
        empty.textContent = "Add characters here. Each character can have their own reference voice sample.";
        empty.style.cssText = "border:1px dashed #334155;border-radius:7px;color:#94a3b8;padding:14px;text-align:center;font-size:12px;";
        charactersPanel.append(empty);
      }
      refs.characters.forEach((character, index) => {
        const card = document.createElement("div");
        card.style.cssText = rowCardStyle;
        const top = document.createElement("div");
        top.style.cssText = "display:grid;grid-template-columns:34px 76px minmax(180px,.8fr) minmax(120px,.55fr) minmax(230px,1fr) 100px;gap:10px;align-items:center;min-width:900px;";
        const number = document.createElement("div");
        number.textContent = String(index + 1);
        number.style.cssText = "font-size:18px;font-weight:900;color:#f8fafc;text-align:center;";
        const label = document.createElement("div");
        label.textContent = `Character ${index + 1}`;
        label.style.cssText = "display:none;";
        const remove = makeTinyButton("Remove");
        const name = makeInput(character.name || "");
        const description = document.createElement("textarea");
        description.value = character.description || "";
        description.placeholder = "Character description...";
        description.style.cssText = "width:100%;box-sizing:border-box;min-height:58px;resize:vertical;border:1px solid #334155;border-radius:6px;background:#020617;color:#f8fafc;padding:8px;font-size:12px;";
        const voice = makeInput(character.voice_audio_path || "");
        const voicePick = makeTinyButton("Pick");
        const voiceField = makeEditField("Reference voice sample", voice, voicePick);
        const imagePath = makeInput(character.image?.path || "");
        const imagePick = makeTinyButton("Pick");
        const imageField = makeEditField("Character ref sheet", imagePath, imagePick);
        const imageNote = document.createElement("div");
        imageNote.textContent = "Used as a character reference for Flux/Klein, Nano B, and Flow/GPT scene image creation.";
        imageNote.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.35;margin-top:-4px;";
        const identity = makeInput(String(character.identity_guidance_scale ?? 3), "number");
        identity.step = "0.1";
        const speechStyle = makeInput(character.speech_style || "");
        speechStyle.placeholder = "Optional speech style notes";
        const grid = document.createElement("div");
        grid.style.cssText = "display:grid;grid-template-columns:minmax(260px,1fr) minmax(260px,1fr);gap:8px;";
        grid.append(voiceField, imageField, makeField("Speech style", speechStyle));
        top.append(number, makeIdLoraThumb(character.image || {}, character.name || "Character"), makeField("Character label", name), makeField("Identity scale", identity), makeField("Description", description), remove);
        card.append(top, grid, imageNote);
        charactersPanel.append(card);
        remove.onclick = () => {
          refs.characters.splice(index, 1);
          for (const entry of Object.values(refs.scene_map || {})) {
            if (entry.character_id === character.id) entry.character_id = "";
          }
          renderAll();
        };
        voicePick.onclick = async () => {
          const path = await pickPath("audio", voice);
          if (path) {
            character.voice_audio_path = path;
            await saveRefsQuiet("ID-LoRA character voice picked");
          }
        };
        imagePick.onclick = async () => {
          const path = await pickPath("image", imagePath);
          if (path) {
            character.image = { ...(character.image || {}), path, data: "", name: "" };
            await saveRefsQuiet("ID-LoRA character image picked");
          }
        };
        const update = () => {
          character.name = name.value || "";
          character.description = description.value || "";
          character.voice_audio_path = voice.value || "";
          character.image = { ...(character.image || {}), path: imagePath.value || "", data: "", name: character.image?.name || "" };
          character.voice_trim_start = 0;
          character.voice_trim_duration = 0;
          character.identity_guidance_scale = Number(identity.value || 3);
          character.speech_style = speechStyle.value || "";
        };
        for (const control of [name, description, voice, imagePath, identity, speechStyle]) {
          control.addEventListener("input", update);
          control.addEventListener("change", () => {
            update();
            saveRefsQuiet("ID-LoRA character edited");
          });
        }
      });
    };
    const renderLocations = () => {
      locationsPanel.innerHTML = "";
      const actions = document.createElement("div");
      actions.style.cssText = "display:flex;justify-content:flex-end;";
      const add = makeButton("Add Location", "primary");
      actions.append(add);
      locationsPanel.append(actions);
      add.onclick = () => {
        refs.locations.push({
          id: `id_lora_loc_${Date.now()}_${Math.floor(Math.random() * 10000)}`,
          name: `Location ${refs.locations.length + 1}`,
          description: "",
          image: { path: "", data: "", name: "" },
        });
        renderAll();
      };
      if (!refs.locations.length) {
        const empty = document.createElement("div");
        empty.textContent = "Add locations for image creation and Gemma visual context.";
        empty.style.cssText = "border:1px dashed #334155;border-radius:7px;color:#94a3b8;padding:14px;text-align:center;font-size:12px;";
        locationsPanel.append(empty);
      }
      refs.locations.forEach((location, index) => {
        const card = document.createElement("div");
        card.style.cssText = rowCardStyle;
        const top = document.createElement("div");
        top.style.cssText = "display:grid;grid-template-columns:34px 76px minmax(180px,.85fr) minmax(320px,1.35fr) 100px;gap:10px;align-items:center;min-width:820px;";
        const number = document.createElement("div");
        number.textContent = String(index + 1);
        number.style.cssText = "font-size:18px;font-weight:900;color:#f8fafc;text-align:center;";
        const label = document.createElement("div");
        label.textContent = `Location ${index + 1}`;
        label.style.cssText = "display:none;";
        const remove = makeTinyButton("Remove");
        const name = makeInput(location.name || "");
        const imagePath = makeInput(location.image?.path || "");
        const imagePick = makeTinyButton("Pick");
        const imageField = makeEditField("Location image", imagePath, imagePick);
        const description = document.createElement("textarea");
        description.value = location.description || "";
        description.placeholder = "Location description...";
        description.style.cssText = "width:100%;box-sizing:border-box;min-height:70px;resize:vertical;border:1px solid #334155;border-radius:6px;background:#020617;color:#f8fafc;padding:8px;font-size:12px;";
        const grid = document.createElement("div");
        grid.style.cssText = fieldGridStyle;
        grid.append(makeField("Name", name), imageField);
        top.append(number, makeIdLoraThumb(location.image || {}, location.name || "Location"), makeField("Location label", name), makeField("Description", description), remove);
        card.append(top, imageField);
        locationsPanel.append(card);
        remove.onclick = () => {
          refs.locations.splice(index, 1);
          for (const entry of Object.values(refs.scene_map || {})) {
            if (entry.location_id === location.id) entry.location_id = "";
          }
          renderAll();
        };
        imagePick.onclick = async () => {
          const path = await pickPath("image", imagePath);
          if (path) {
            location.image = { ...(location.image || {}), path, data: "", name: "" };
            await saveRefsQuiet("ID-LoRA location image picked");
          }
        };
        const update = () => {
          location.name = name.value || "";
          location.description = description.value || "";
          location.image = { ...(location.image || {}), path: imagePath.value || "", data: "", name: location.image?.name || "" };
        };
        for (const control of [name, imagePath, description]) {
          control.addEventListener("input", update);
          control.addEventListener("change", () => {
            update();
            saveRefsQuiet("ID-LoRA location edited");
          });
        }
      });
    };
    const renderCasting = () => {
      castingPanel.innerHTML = "";
      if (!state.segments.length) {
        const empty = document.createElement("div");
        empty.textContent = "Create timeline scenes first.";
        empty.style.cssText = "border:1px dashed #334155;border-radius:7px;color:#94a3b8;padding:14px;text-align:center;font-size:12px;";
        castingPanel.append(empty);
        return;
      }
      const characterOptions = selectOptionsFromItems(refs.characters, "No character selected");
      const locationOptions = selectOptionsFromItems(refs.locations, "No location selected");
      const toolbar = document.createElement("div");
      toolbar.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:10px;";
      const summary = document.createElement("div");
      const autoCount = state.segments.reduce((count, segment) => count + (idLoraSceneEntry(refs, segment).auto_duration !== false ? 1 : 0), 0);
      summary.textContent = `${autoCount}/${state.segments.length} scenes use auto duration. Calculated durations update from each dialogue line.`;
      summary.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.35;";
      const recalcAll = makeTinyButton("Recalculate Auto Durations");
      toolbar.append(summary, recalcAll);
      castingPanel.append(toolbar);
      recalcAll.onclick = async () => {
        pushHistory();
        let updated = 0;
        for (const segment of state.segments) {
          const sceneId = String(segment.id || "");
          const entry = idLoraSceneEntry(refs, segment);
          if (entry.auto_duration === false) {
            refs.scene_map[sceneId] = entry;
            continue;
          }
          entry.dialogue = String(entry.dialogue || segment.lyric_text || "").trim();
          entry.estimated_duration = estimateIdLoraDialogueDuration(entry.dialogue);
          entry.manual_duration = entry.estimated_duration;
          refs.scene_map[sceneId] = entry;
          segment.lyric_text = entry.dialogue;
          setBaseSegmentDurationRipple(segment, entry.estimated_duration);
          updated += 1;
        }
        state.idLoraReferenceBuilder = normalizeIdLoraReferenceBuilder(refs);
        syncInspector();
        render();
        await autoSaveSessionQuiet("ID-LoRA auto durations recalculated");
        toast(`Recalculated ${updated} ID-LoRA auto duration${updated === 1 ? "" : "s"}.`);
        renderAll();
      };
      const makeCastingPreview = (item, emptyText) => {
        const wrap = document.createElement("div");
        wrap.style.cssText = "display:flex;gap:7px;align-items:center;min-height:58px;overflow:hidden;padding:5px;border:1px solid #1e3a5f;border-radius:7px;background:#071422;";
        if (!item) {
          const empty = document.createElement("div");
          empty.textContent = emptyText;
          empty.style.cssText = "font-size:11px;color:#94a3b8;padding:0 6px;";
          wrap.append(empty);
          return wrap;
        }
        const thumb = makeIdLoraThumb(item.image || {}, item.name || "Reference");
        thumb.style.width = "54px";
        thumb.style.height = "54px";
        const label = document.createElement("div");
        label.textContent = item.name || "Reference";
        label.style.cssText = "font-size:11px;color:#cbd5e1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;";
        wrap.append(thumb, label);
        return wrap;
      };
      state.segments.forEach((segment, index) => {
        const sceneId = String(segment.id || "");
        const entry = idLoraSceneEntry(refs, segment);
        refs.scene_map[sceneId] = entry;
        const card = document.createElement("div");
        card.style.cssText = rowCardStyle;
        const heading = document.createElement("div");
        heading.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
        const titleText = document.createElement("div");
        titleText.innerHTML = `<div style="font-weight:900;color:#cffafe;font-size:12px;">${escapeHtml(sceneDisplayName(segment, index))}</div><div style="font-size:11px;color:#94a3b8;margin-top:2px;">${formatTime(segment.start)} - ${formatTime(segment.end)}</div>`;
        const estimate = document.createElement("div");
        estimate.style.cssText = "font-size:12px;color:#a5f3fc;font-weight:900;white-space:nowrap;";
        heading.append(titleText, estimate);
        const character = makeSelect([], "");
        const location = makeSelect([], "");
        syncSelectOptions(character, characterOptions, entry.character_id);
        syncSelectOptions(location, locationOptions, entry.location_id);
        const dialogue = document.createElement("textarea");
        dialogue.value = entry.dialogue || "";
        dialogue.placeholder = "Dialogue line...";
        dialogue.style.cssText = "width:100%;box-sizing:border-box;min-height:62px;resize:vertical;border:1px solid #334155;border-radius:6px;background:#020617;color:#f8fafc;padding:8px;font-size:12px;";
        const autoDuration = makeCheckbox("Auto duration", entry.auto_duration !== false);
        const manualDuration = makeInput(String(Number(entry.manual_duration || entry.estimated_duration || 2).toFixed(2)), "number");
        manualDuration.min = "0.25";
        manualDuration.step = "0.25";
        const manualDurationField = makeField("Duration", manualDuration);
        const grid = document.createElement("div");
        grid.style.cssText = "display:grid;grid-template-columns:minmax(150px,.7fr) minmax(180px,1fr) minmax(150px,.7fr) minmax(180px,1fr) minmax(120px,.45fr) minmax(100px,.4fr);gap:8px;align-items:stretch;min-width:1080px;";
        const selectedCharacter = refs.characters.find((item) => item.id === entry.character_id);
        const selectedLocation = refs.locations.find((item) => item.id === entry.location_id);
        grid.append(
          makeField("Character", character),
          makeCastingPreview(selectedCharacter, "No character"),
          makeField("Location", location),
          makeCastingPreview(selectedLocation, "No location"),
          autoDuration.wrapper,
          manualDurationField
        );
        card.append(heading, grid, makeField("Dialogue line", dialogue));
        castingPanel.append(card);
        const refreshDuration = (applyTiming = false) => {
          entry.dialogue = dialogue.value || "";
          entry.estimated_duration = estimateIdLoraDialogueDuration(entry.dialogue);
          entry.auto_duration = Boolean(autoDuration.input.checked);
          entry.manual_duration = Math.max(0.25, Number(manualDuration.value || entry.estimated_duration || 2));
          estimate.textContent = entry.auto_duration ? `Estimated duration: ${entry.estimated_duration.toFixed(2)}s` : `Manual duration: ${entry.manual_duration.toFixed(2)}s`;
          manualDurationField.style.display = entry.auto_duration ? "none" : "flex";
          if (entry.auto_duration) manualDuration.value = entry.estimated_duration.toFixed(2);
          if (applyTiming && entry.auto_duration) {
            segment.lyric_text = entry.dialogue;
            setBaseSegmentDurationRipple(segment, entry.estimated_duration);
            renderSegments();
            drawWaveform();
          }
        };
        const updateEntry = (applyTiming = false) => {
          entry.character_id = character.value || "";
          entry.location_id = location.value || "";
          refreshDuration(applyTiming);
          refs.scene_map[sceneId] = entry;
        };
        character.onchange = () => {
          updateEntry(false);
          renderCasting();
        };
        location.onchange = () => {
          updateEntry(false);
          renderCasting();
        };
        dialogue.addEventListener("input", () => updateEntry(true));
        dialogue.addEventListener("change", () => {
          updateEntry(true);
          saveRefsQuiet("ID-LoRA scene dialogue edited");
        });
        autoDuration.input.onchange = () => {
          updateEntry(true);
          saveRefsQuiet("ID-LoRA auto duration changed");
        };
        manualDuration.oninput = () => updateEntry(false);
        manualDuration.onchange = () => {
          updateEntry(false);
          saveRefsQuiet("ID-LoRA manual duration changed");
        };
        refreshDuration(false);
      });
    };
    const renderAll = () => {
      refs = normalizeIdLoraReferenceBuilder(refs);
      renderCharacters();
      renderLocations();
      renderCasting();
    };
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const save = makeButton("Save", "primary");
    const cancel = makeButton("Cancel");
    actions.append(save, cancel);
    box.append(header, tabs.wrapper, actions);
    backdrop.append(box);
    const closeModal = () => backdrop.remove();
    close.onclick = closeModal;
    cancel.onclick = closeModal;
    save.onclick = async () => {
      state.idLoraReferenceBuilder = normalizeIdLoraReferenceBuilder(refs);
      await autoSaveSessionQuiet("ID-LoRA ref builder saved");
      syncInspector();
      render();
      toast("ID-LoRA Ref Builder saved.");
      closeModal();
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
    document.body.append(backdrop);
    renderAll();
  }

  function openIdLoraReferenceBuilderModalSafely() {
    try {
      openIdLoraReferenceBuilderModal();
    } catch (error) {
      console.error(error);
      toast(`ID-LoRA Ref Builder failed to open: ${error?.message || error}`, true);
    }
  }

  function openReferenceBuilderTargetChooser() {
    const miniMaxProject = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(520px,calc(100vw - 32px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Reference Builder Target</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">${miniMaxProject ? "Choose how these references will be used by MiniMax H3." : "Choose text scene mapping or an image-reference LTX workflow."}</div>`;
    const choices = document.createElement("div");
    choices.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;";
    const choiceCard = (button, description) => {
      const card = document.createElement("div");
      card.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;padding:8px;display:flex;flex-direction:column;gap:7px;min-width:0;";
      const text = document.createElement("div");
      text.textContent = description;
      text.style.cssText = "font-size:11px;line-height:1.35;color:#cbd5e1;";
      card.append(button, text);
      return card;
    };
    const textMappingButton = makeButton("I2V / T2V Text Mapping", "primary");
    const imageRefsButton = makeButton("Flux / Nano Image References", "primary");
    const ltxRefsButton = makeButton(miniMaxProject ? "Reference to Video" : "LTX Reference to Video", "primary");
    const videoToVideoRefsButton = makeButton("Video to Video", "primary");
    const ingredientsRefsButton = makeButton("Ingredients to Video", "primary");
    const idLoraRefsButton = makeButton("ID-LoRA Ref Builder", "primary");
    const close = makeButton("Cancel", "neutral");
    choices.append(
      choiceCard(textMappingButton, "Text-only character/location mapping for LLM image planning, Image to Video prompt writing, and Text to Video prompt writing. No reference images are sent."),
      choiceCard(imageRefsButton, "Image reference setup for Flux/Klein and Nano B image generation. Use when those image modes should receive character/location images."),
      choiceCard(ltxRefsButton, miniMaxProject
        ? "Build and map ordered character, location, prop, style, or storyboard images for MiniMax Reference to Video."
        : "LTX Reference-to-Video setup for the MSR LoRA workflow. Uses reference images for the video render."),
    );
    if (miniMaxProject) {
      choices.append(choiceCard(
        videoToVideoRefsButton,
        "Use a source video together with character, background, location, prop, or style images for MiniMax video editing, including person and background replacement.",
      ));
    } else {
      choices.append(
        choiceCard(ingredientsRefsButton, "Ingredients-to-Video setup for the Ingredients LoRA workflow. Maps ingredients sheets/images to scenes."),
        choiceCard(idLoraRefsButton, "Characters, voice samples, locations, dialogue, and auto duration for ID-LoRA I2V short film scenes."),
      );
    }
    box.append(title, choices, close);
    backdrop.append(box);
    const openAndClose = (target) => {
      backdrop.remove();
      if (miniMaxProject && ["reference_to_video", "video_to_video"].includes(target)) {
        const segment = activeSegment();
        if (segment) setMiniMaxH3ModeForSegment(segment, target);
        syncMiniMaxH3Panel();
        renderSegments();
        openFluxReferenceBuilderModal({ miniMaxTargetMode: target });
        return;
      }
      if (target === "rtv") {
        state.videoModelMode = "rtv";
        syncVideoModePanel();
      }
      openFluxReferenceBuilderModal();
    };
    textMappingButton.onclick = () => {
      backdrop.remove();
      openFluxReferenceBuilderModal({ textOnlyMode: true });
    };
    imageRefsButton.onclick = () => openAndClose("image");
    ltxRefsButton.onclick = () => openAndClose(miniMaxProject ? "reference_to_video" : "rtv");
    videoToVideoRefsButton.onclick = () => openAndClose("video_to_video");
    ingredientsRefsButton.onclick = () => {
      backdrop.remove();
      state.videoModelMode = "ingredients";
      syncVideoModePanel();
      openIngredientsReferenceBuilderModal();
    };
    idLoraRefsButton.onclick = () => {
      backdrop.remove();
      state.videoModelMode = "id_lora";
      syncVideoModePanel();
      openIdLoraReferenceBuilderModalSafely();
    };
    close.onclick = () => backdrop.remove();
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
    document.body.append(backdrop);
  }

  return { openIdLoraReferenceBuilderModalSafely, openReferenceBuilderTargetChooser };
}
