import { makeEditorImageUrl } from "./comfy_api.mjs";
import { escapeHtml, makeButton, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { rtvReferenceImagePayload } from "./image_references.mjs";
import { hasReferenceImage } from "./llm_runner.mjs";
import { normalizeMiniMaxH3Mode, normalizeMiniMaxH3Pipeline } from "./minimax_h3.mjs";
import { assignLabels, clothingChoices, composeRefmodItems, sceneSubjectCards, tokenStatusText } from "./refmod_labels.mjs";
import { expandSubjectReferencesForRender, normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { mediaPathKey } from "./timeline_state.mjs";

export function referenceBuilderSubjectHasImage(item = {}) {
  const image = item?.image || item || {};
  return Boolean(String(image.path || "").trim() || String(image.data || "").trim());
}

export function createMiniMaxReferences({
  activeSegment, autoSaveSessionQuiet, currentVideoMode, firstLastFrameResolvedEndImageSource,
  logicalExtraSubjectsForScene, logicalReferenceSubjects, logicalSubjectIdsForScene,
  miniMaxH3ContinuityReferenceReserved, miniMaxH3ImageReferencePromptItems, miniMaxH3ModeForSegment,
  miniMaxH3SceneImageIsPromptInspiration, miniMaxH3StartFrameCharacterInfluenceForSegment,
  openReferenceBuilderTargetChooser, pushHistory, requireActiveSegment, rtvReferenceBehaviorForSegment,
  rtvSceneImageAnchorPayload, sceneDisplayName, sceneReferenceMapValue, segmentImageSource, segmentIndexInfo,
  state, syncMiniMaxReferenceButtons,
}) {
  function referenceBuilderSubjectItemsForSegment(refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder), segment = activeSegment()) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    const rtvAutoUse = currentVideoMode() === "rtv" && (
      referenceBuilderSubjectHasImage(normalizedRefs.subject)
      || (normalizedRefs.subjects || []).some(referenceBuilderSubjectHasImage)
    );
    if (!normalizedRefs.use_subject_reference && !rtvAutoUse) return [];
    if (segment?.no_character_present) {
      const imageBackedSubjects = normalizedRefs.subjects.filter(referenceBuilderSubjectHasImage);
      if (imageBackedSubjects.length) return [imageBackedSubjects[0]];
      if (referenceBuilderSubjectHasImage(normalizedRefs.subject)) {
        return [{
          ...normalizedRefs.subjects?.[0],
          description: normalizedRefs.subject?.description || normalizedRefs.subjects?.[0]?.description || "",
          image: normalizedRefs.subject?.image || {},
        }];
      }
      return [];
    }
    if (normalizedRefs.subject_count > 1 && segment) {
      const subjectIds = logicalSubjectIdsForScene(normalizedRefs, segment);
      const idSet = new Set(Array.isArray(subjectIds) ? subjectIds : [subjectIds].filter(Boolean));
      const mapped = normalizedRefs.subjects.filter((item) => idSet.has(item.id));
      if (mapped.length) return expandSubjectReferencesForRender(normalizedRefs, mapped);
      const imageBackedSubjects = logicalReferenceSubjects(normalizedRefs).filter(referenceBuilderSubjectHasImage);
      if (imageBackedSubjects.length === 1) return imageBackedSubjects;
      if (!imageBackedSubjects.length && referenceBuilderSubjectHasImage(normalizedRefs.subject)) {
        return [{
          ...normalizedRefs.subjects?.[0],
          description: normalizedRefs.subject?.description || normalizedRefs.subjects?.[0]?.description || "",
          image: normalizedRefs.subject?.image || {},
        }];
      }
      return [];
    }
    const subject = {
      ...normalizedRefs.subjects?.[0],
      description: normalizedRefs.subject?.description || normalizedRefs.subjects?.[0]?.description || "",
      image: normalizedRefs.subject?.image || normalizedRefs.subjects?.[0]?.image || {},
    };
    return referenceBuilderSubjectHasImage(subject) || String(subject.name || subject.description || "").trim() ? [subject] : [];
  }

  function miniMaxReferenceBuilderCatalog(refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    const catalog = [];
    const add = (kind, item, fallbackLabel) => {
      const id = String(item?.id || "").trim();
      const image = item?.image && typeof item.image === "object" ? item.image : {};
      if (!id || (!String(image.path || "").trim() && !String(image.data || "").trim())) return;
      catalog.push({
        key: `${kind}:${id}`,
        kind,
        source_id: id,
        label: String(item?.name || item?.title || fallbackLabel || "Reference").trim() || "Reference",
        description: String(item?.description || "").trim(),
        reference_image_type: String(item?.reference_image_type || "single"),
        image,
      });
    };
    for (const subject of normalizedRefs.subjects || []) add("subject", subject, "Character reference");
    for (const extra of normalizedRefs.extra_subjects || []) {
      if (extra.send_to_minimax) add("extra", extra, "Extra character reference");
    }
    for (const location of normalizedRefs.locations || []) add("location", location, "Location reference");
    for (const sheet of normalizedRefs.ingredients_sheets || []) add("ingredients", sheet, "Ingredients sheet");
    return catalog;
  }

  function miniMaxMappedReferenceKeysForSegment(segment, refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    if (!segment) return [];
    const normalizedRefs = normalizeFluxReferenceBuilder(refs);
    const catalogKeys = new Set(miniMaxReferenceBuilderCatalog(normalizedRefs).map((item) => item.key));
    const keys = [];
    const add = (key) => {
      const clean = String(key || "").trim();
      if (clean && catalogKeys.has(clean) && !keys.includes(clean)) keys.push(clean);
    };
    if (!segment.no_character_present) {
      for (const subject of referenceBuilderSubjectItemsForSegment(normalizedRefs, segment)) {
        add(`subject:${String(subject?.id || "").trim()}`);
      }
      for (const { extra } of logicalExtraSubjectsForScene(normalizedRefs, segment)) {
        if (extra.send_to_minimax && hasReferenceImage(extra.image || {})) add(`extra:${String(extra.id || "").trim()}`);
      }
    }
    const locationId = String(sceneReferenceMapValue(normalizedRefs.scene_map, segment) || "").trim();
    if (locationId) add(`location:${locationId}`);
    const sheetId = String(normalizedRefs.ingredients_scene_map?.[segment.id] || "").trim();
    if (sheetId) add(`ingredients:${sheetId}`);
    return keys;
  }

  function miniMaxForcedExtraReferenceKeysForSegment(segment, refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    if (!segment || segment.no_character_present) return [];
    return logicalExtraSubjectsForScene(refs, segment)
      .filter(({ extra }) => extra.send_to_minimax && hasReferenceImage(extra.image || {}))
      .map(({ extra }) => `extra:${String(extra.id || "").trim()}`)
      .filter(Boolean);
  }

  function miniMaxDesiredReferenceKeysForSegment(segment, refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    if (!segment) return [];
    const catalogKeys = new Set(miniMaxReferenceBuilderCatalog(refs).map((item) => item.key));
    const selected = Array.isArray(segment.minimax_h3_reference_keys)
      ? segment.minimax_h3_reference_keys
      : miniMaxMappedReferenceKeysForSegment(segment, refs);
    const forcedExtraKeys = new Set(miniMaxForcedExtraReferenceKeysForSegment(segment, refs));
    const seen = new Set();
    return [...selected, ...forcedExtraKeys]
      .map((key) => String(key || "").trim())
      .filter((key) => key && catalogKeys.has(key) && (!key.startsWith("extra:") || forcedExtraKeys.has(key)) && !seen.has(key) && seen.add(key));
  }

  function miniMaxReferenceKeysForSegment(segment, refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder)) {
    if (!segment) return [];
    return miniMaxDesiredReferenceKeysForSegment(segment, refs).slice(0, 9);
  }

  function miniMaxReferenceBuilderImagePathsForSegment(segment, mode = miniMaxH3ModeForSegment(segment)) {
    return miniMaxOrderedImageReferenceItemsForSegment(segment, mode)
      .map((item) => String(item?.image?.path || "").trim())
      .filter(Boolean)
      .slice(0, 9);
  }

  function miniMaxPromptReferenceItemsForSegment(segment) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const byKey = new Map(miniMaxReferenceBuilderCatalog(refs).map((item) => [item.key, item]));
    return miniMaxReferenceKeysForSegment(segment, refs)
      .map((key) => byKey.get(key))
      .filter(Boolean)
      .slice(0, 9);
  }

  // The whole project switches pipeline at once, so the project setting decides.
  function isRefmodPipeline() {
    return normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3"
      && normalizeMiniMaxH3Pipeline(state.miniMaxH3Settings?.pipeline) === "refmod";
  }

  // The scene's RefMods in render order, labelled the way Text Encode with RefMods will number them. Items with
  // strength 0 are left out. Twin of refmod_items_for_scene in minimax/refmod_scene.py.
  function miniMaxRefmodItemsForSegment(segment) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const subjectCards = sceneSubjectCards(segment, referenceBuilderSubjectItemsForSegment(refs, segment));
    const locationId = String(sceneReferenceMapValue(refs.scene_map, segment) || "").trim();
    const location = locationId ? (refs.locations || []).find((item) => String(item?.id || "") === locationId) || null : null;
    const items = composeRefmodItems(subjectCards, [], location, refs.subjects || [], segment?.refmod_clothing_override);
    return assignLabels(items.filter((item) => item.strength > 0));
  }

  // RefMod items in the shape the prompt writer expects for reference images. The "image path" is a marker
  // (refmod://name) so the existing ordering, de-duplication and signature code keeps working without a picture.
  function miniMaxRefmodPromptItemsForSegment(segment) {
    return miniMaxRefmodItemsForSegment(segment).map((item) => ({
      key: item.key,
      kind: item.category === "background" ? "location" : item.category === "extra" ? "extra" : "subject",
      source_id: item.card_id,
      label: item.name,
      name: item.name,
      description: item.description,
      reference_image_type: "single",
      refmod: item,
      image: { path: `refmod://${item.mod_name}`, data: "", name: item.mod_name },
    }));
  }

  function miniMaxOrderedImageReferenceItemsForSegment(segment, mode = miniMaxH3ModeForSegment(segment)) {
    if (isRefmodPipeline()) return miniMaxRefmodPromptItemsForSegment(segment);
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    const ordered = [];
    if (["reference_to_video", "image_reference_to_video"].includes(normalizedMode) && segment?.minimax_h3_use_scene_image_as_start_frame) {
      const startFrame = segmentImageSource(segment);
      if (startFrame?.path || startFrame?.data) {
        const characterInfluence = miniMaxH3StartFrameCharacterInfluenceForSegment(segment);
        ordered.push({
          key: "scene:start_frame",
          kind: "start_frame",
          label: "Scene start frame",
          start_frame_character_influence: characterInfluence,
          description: characterInfluence === "face_hair_only"
            ? "Exact opening frame and authoritative source for pose, body proportions, wardrobe, accessories, composition, camera angle, environment, props, lighting, and every visible detail except face identity and hair."
            : "Exact opening frame, composition, pose, camera angle, environment, and lighting anchor.",
          image: {
            path: String(startFrame.path || "").trim(),
            data: String(startFrame.data || "").trim(),
            name: String(startFrame.name || "start_frame.png"),
          },
        });
      }
    }
    ordered.push(...miniMaxPromptReferenceItemsForSegment(segment));
    const seen = new Set();
    return ordered.filter((item) => {
      const path = String(item?.image?.path || "").trim();
      const data = String(item?.image?.data || "").trim();
      const fingerprint = path ? `path:${mediaPathKey(path)}` : data ? `data:${data}` : "";
      if (!fingerprint || seen.has(fingerprint)) return false;
      seen.add(fingerprint);
      return true;
    }).slice(0, 9);
  }

  function miniMaxReferencePurposeText(item, segment = null) {
    const faceHairOnly = Boolean(
      segment?.minimax_h3_use_scene_image_as_start_frame
      && miniMaxH3StartFrameCharacterInfluenceForSegment(segment) === "face_hair_only"
    );
    if (item?.kind === "start_frame") {
      return faceHairOnly
        ? "exact start frame and authority for every visible detail except face identity and hair"
        : "exact start frame, opening composition, pose, camera angle, environment, and lighting anchor";
    }
    if (item?.refmod?.category === "clothing") return "clothing reference: garments, fabrics, colours, trims and accessories, worn by the character it follows";
    if (item?.refmod?.category === "object") return "object reference: shape, material, colour, markings and scale";
    if (item?.refmod?.category === "style") return "visual style reference: medium, palette, line quality, lighting and texture, applied to the whole scene";
    if (item?.kind === "subject") {
      return faceHairOnly
        ? "face identity and hair reference only; do not copy clothing, body proportions, pose, accessories, framing, lighting, or background"
        : "character identity, face, hair, clothing, and body-proportion reference";
    }
    if (item?.kind === "extra") {
      return item?.reference_image_type === "multi_view"
        ? "character reference sheet depicting the same non-speaking extra throughout; if it contains multiple panels, treat every panel as the same person, not separate people or scenes, and preserve one consistent identity, face, hair, clothing, accessories, and body proportions"
        : "character identity, face, hair, clothing, and body-proportion reference for this non-speaking extra";
    }
    if (item?.kind === "location") return "environment, location, architecture, layout, and atmosphere reference";
    if (item?.kind === "ingredients") {
      const searchable = `${item?.label || ""} ${item?.description || ""}`.toLowerCase();
      return /storyboard|story board|grid|panel/.test(searchable)
        ? "storyboard-grid reference whose panels are ordered visual beats"
        : "ordered ingredients, props, identity, or visual-detail reference";
    }
    return "visual reference";
  }

  function miniMaxH3MissingReferenceDescriptions(segment, mode = miniMaxH3ModeForSegment(segment)) {
    const missing = [];
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    if (!["reference_to_video", "image_reference_to_video", "video_to_video"].includes(normalizedMode)) return missing;
    // A RefMod is shown to the encoder as <Video n> or <Picture n> and is named by its card, so a missing description
    // only means the prompt has less to say about it. It never blocks prompt creation.
    if (isRefmodPipeline()) return missing;
    const items = (normalizedMode === "image_reference_to_video"
      ? miniMaxH3ImageReferencePromptItems(segment)
      : miniMaxOrderedImageReferenceItemsForSegment(segment, mode))
      .filter((item) => item?.kind === "subject" || item?.kind === "extra" || item?.kind === "location");
    items.forEach((item, index) => {
      const label = String(item?.label || `${item?.kind === "location" ? "Location" : "Character"} ${index + 1}`).trim();
      if (!String(item?.description || "").trim()) missing.push(`${label} (${item?.kind === "location" ? "location" : "character"})`);
    });
    return missing;
  }

  function miniMaxH3ReferenceCapacityStatus(segment, mode = miniMaxH3ModeForSegment(segment)) {
    const normalizedMode = normalizeMiniMaxH3Mode(mode);
    if (isRefmodPipeline()) {
      // RefMods have no nine-image limit. The render allows 24 per scene.
      const items = miniMaxRefmodItemsForSegment(segment);
      const allSubjects = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder).subjects || [];
      return {
        count: items.length, overflow: Math.max(0, items.length - 24), labels: items.map((item) => item.name),
        tokenText: tokenStatusText(items), clothing: clothingChoices(items, allSubjects, segment?.refmod_clothing_override),
      };
    }
    if (!segment || !["reference_to_video", "image_reference_to_video", "video_to_video"].includes(normalizedMode)) return { count: 0, overflow: 0, labels: [] };
    if (normalizedMode === "image_reference_to_video") {
      const items = miniMaxH3ImageReferencePromptItems(segment, 99);
      return {
        count: items.length,
        overflow: Math.max(0, items.length - 9),
        labels: items.map((item, index) => String(item?.label || `Image ${index + 1}`)),
      };
    }
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const catalog = new Map(miniMaxReferenceBuilderCatalog(refs).map((item) => [item.key, item]));
    const seenImages = new Set();
    const imageFingerprint = (image = {}) => {
      const path = String(image?.path || "").trim();
      const data = String(image?.data || "").trim();
      return path ? `path:${mediaPathKey(path)}` : data ? `data:${data}` : "";
    };
    const desired = miniMaxDesiredReferenceKeysForSegment(segment, refs).map((key) => catalog.get(key)).filter((item) => {
      if (!item) return false;
      const fingerprint = imageFingerprint(item.image);
      if (!fingerprint || seenImages.has(fingerprint)) return false;
      seenImages.add(fingerprint);
      return true;
    });
    const labels = desired.map((item) => item.label);
    let count = desired.length;
    if (["reference_to_video", "image_reference_to_video"].includes(normalizedMode) && segment?.minimax_h3_use_scene_image_as_start_frame) {
      const startFrame = segmentImageSource(segment);
      if (startFrame?.path || startFrame?.data) {
        const fingerprint = imageFingerprint(startFrame);
        if (!fingerprint || !seenImages.has(fingerprint)) {
          if (fingerprint) seenImages.add(fingerprint);
          count += 1;
          labels.unshift("Scene start frame");
        }
      }
    }
    if (miniMaxH3ContinuityReferenceReserved(segment)) {
      count += 1;
      labels.push("Continuity frame");
    }
    return { count, overflow: Math.max(0, count - 9), labels };
  }

  function assertMiniMaxH3ReferenceCapacity(segment, mode = miniMaxH3ModeForSegment(segment)) {
    const status = miniMaxH3ReferenceCapacityStatus(segment, mode);
    if (!status.overflow) return status;
    throw new Error(`This scene requires ${status.count} MiniMax reference images, but H3 accepts at most 9 in this workflow. Uncheck or remove ${status.overflow} reference image${status.overflow === 1 ? "" : "s"}.\n\nCurrent references:\n- ${status.labels.join("\n- ")}`);
  }

  function assertMiniMaxH3ReferenceDescriptionsReady(segment, mode = miniMaxH3ModeForSegment(segment)) {
    assertMiniMaxH3ReferenceCapacity(segment, mode);
    const missing = miniMaxH3MissingReferenceDescriptions(segment, mode);
    if (!missing.length) return;
    if (isRefmodPipeline()) {
      throw new Error(`MiniMax prompt creation needs a description for every RefMod in the scene.\n\nMissing description for:\n- ${missing.join("\n- ")}\n\nOpen Reference Builder and type the description on that card (RefMods Studio can write one from the images).`);
    }
    throw new Error(
      `MiniMax prompt creation needs Reference Builder descriptions before it can assemble the fixed Image blocks.\n\nMissing description for:\n- ${missing.join("\n- ")}\n\nOpen Reference Builder and click Gemma Describe, or type the description manually, then create the prompt again.`,
    );
  }

  function openMiniMaxReferenceSelector() {
    if (normalizeProjectVideoEngine(state.projectVideoEngine) !== "minimax_h3") {
      toast("Set this project's Video Engine to MiniMax H3 before choosing MiniMax references.", true);
      return;
    }
    const segment = requireActiveSegment();
    if (!segment) return;
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const forcedExtraKeys = new Set(miniMaxForcedExtraReferenceKeysForSegment(segment, refs));
    const catalog = miniMaxReferenceBuilderCatalog(refs).filter((item) => item.kind !== "extra" || forcedExtraKeys.has(item.key));
    const startFrameReserved = miniMaxH3ModeForSegment(segment) === "reference_to_video"
      && Boolean(segment.minimax_h3_use_scene_image_as_start_frame)
      && Boolean(segmentImageSource(segment)?.path || segmentImageSource(segment)?.data);
    const promptInspiration = miniMaxH3ModeForSegment(segment) === "reference_to_video"
      && miniMaxH3SceneImageIsPromptInspiration(segment);
    const continuityReserved = miniMaxH3ContinuityReferenceReserved(segment);
    const maxLibraryReferences = Math.max(0, 9 - (startFrameReserved ? 1 : 0) - (continuityReserved ? 1 : 0));
    let selectedKeys = miniMaxReferenceKeysForSegment(segment, refs).slice(0, maxLibraryReferences);
    let useAutomaticMappings = !Array.isArray(segment.minimax_h3_reference_keys);

    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:22px;box-sizing:border-box;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(1100px,calc(100vw - 44px));max-height:calc(100vh - 48px);overflow:hidden;border:1px solid #155e75;border-radius:8px;background:#0b1220;color:#f8fafc;display:flex;flex-direction:column;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;padding:13px 15px;border-bottom:1px solid #155e75;background:#083344;";
    const heading = document.createElement("div");
    heading.textContent = `${sceneDisplayName(segment, segmentIndexInfo(segment).index)} — MiniMax H3 References`;
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Cancel");
    header.append(heading, close);
    const note = document.createElement("div");
    note.textContent = startFrameReserved
      ? `The scene image is reserved as Image 1 and the exact start frame. Choose up to ${maxLibraryReferences} additional Reference Builder image${maxLibraryReferences === 1 ? "" : "s"}.${continuityReserved ? " One final slot is reserved for the previous rendered scene's continuity frame." : ""}`
      : `Choose up to ${maxLibraryReferences} existing Reference Builder image${maxLibraryReferences === 1 ? "" : "s"}. The numbered order is sent into MiniMax H3 exactly as shown.${promptInspiration ? " The scene image is prompt-only LLM inspiration and does not consume a MiniMax reference slot." : ""}${continuityReserved ? " One final slot is reserved for the previous rendered scene's continuity frame." : ""} Scene character/location mappings are used automatically until you save a custom selection.`;
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;padding:11px 15px;border-bottom:1px solid #1e293b;";
    const body = document.createElement("div");
    body.style.cssText = "display:grid;grid-template-columns:minmax(0,1.25fr) minmax(320px,.75fr);gap:12px;padding:12px;overflow:auto;min-height:260px;";
    const availablePanel = document.createElement("div");
    const selectedPanel = document.createElement("div");
    for (const panel of [availablePanel, selectedPanel]) panel.style.cssText = "border:1px solid #334155;border-radius:7px;background:#111827;padding:10px;display:flex;flex-direction:column;gap:9px;min-width:0;";
    const availableTitle = document.createElement("div");
    availableTitle.textContent = "Reference Builder Images";
    availableTitle.style.cssText = "font-size:13px;font-weight:900;color:#a5f3fc;";
    const selectedTitle = document.createElement("div");
    selectedTitle.style.cssText = availableTitle.style.cssText;
    const availableList = document.createElement("div");
    availableList.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:8px;";
    const selectedList = document.createElement("div");
    selectedList.style.cssText = "display:flex;flex-direction:column;gap:7px;";
    availablePanel.append(availableTitle, availableList);
    selectedPanel.append(selectedTitle, selectedList);
    body.append(availablePanel, selectedPanel);

    const referenceThumb = (item) => {
      const wrap = document.createElement("div");
      wrap.style.cssText = "height:92px;border:1px solid #334155;border-radius:6px;background:#020617;overflow:hidden;display:flex;align-items:center;justify-content:center;";
      const img = document.createElement("img");
      img.alt = item.label || "Reference";
      img.src = String(item.image?.data || "").trim() || makeEditorImageUrl(item.image?.path || "");
      img.style.cssText = "width:100%;height:100%;object-fit:cover;";
      wrap.append(img);
      return wrap;
    };
    const kindLabel = (kind) => ({ subject: "Character", extra: "Mapped Extra", location: "Location", ingredients: "Ingredients" }[kind] || "Reference");
    const renderLists = () => {
      selectedKeys = selectedKeys.filter((key, index, list) => catalog.some((item) => item.key === key) && list.indexOf(key) === index).slice(0, maxLibraryReferences);
      selectedTitle.textContent = `Selected Order (${selectedKeys.length}/${maxLibraryReferences})${startFrameReserved ? " — follows Image 1 start frame" : ""}${useAutomaticMappings ? " — Automatic Scene Mappings" : ""}`;
      availableList.replaceChildren();
      selectedList.replaceChildren();
      if (!catalog.length) {
        const empty = document.createElement("div");
        empty.textContent = "No saved character, location, or ingredients-sheet images are available. Open Reference Builder and add images first.";
        empty.style.cssText = "grid-column:1/-1;border:1px dashed #475569;border-radius:6px;color:#94a3b8;padding:18px;line-height:1.45;";
        availableList.append(empty);
      }
      for (const item of catalog) {
        const chosen = selectedKeys.includes(item.key);
        const card = document.createElement("div");
        card.style.cssText = `border:2px solid ${chosen ? "#0891b2" : "#334155"};border-radius:7px;background:${chosen ? "#083344" : "#0f172a"};padding:7px;display:flex;flex-direction:column;gap:6px;min-width:0;`;
        card.append(referenceThumb(item));
        const label = document.createElement("div");
        label.textContent = item.label;
        label.title = item.description || item.label;
        label.style.cssText = "font-size:11px;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        const type = document.createElement("div");
        type.textContent = kindLabel(item.kind);
        type.style.cssText = "font-size:10px;color:#94a3b8;text-transform:uppercase;";
        const add = makeButton(chosen ? "Selected" : "Add");
        add.disabled = chosen || selectedKeys.length >= maxLibraryReferences;
        add.onclick = () => { useAutomaticMappings = false; selectedKeys.push(item.key); renderLists(); };
        card.append(label, type, add);
        availableList.append(card);
      }
      selectedKeys.forEach((key, index) => {
        const item = catalog.find((entry) => entry.key === key);
        if (!item) return;
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:34px 52px minmax(0,1fr) auto;gap:7px;align-items:center;border:1px solid #334155;border-radius:6px;background:#0f172a;padding:6px;";
        const number = document.createElement("div");
        number.textContent = String(index + 1 + (startFrameReserved ? 1 : 0));
        number.style.cssText = "font-size:18px;font-weight:900;text-align:center;color:#67e8f9;";
        const thumb = referenceThumb(item);
        thumb.style.height = "46px";
        const label = document.createElement("div");
        label.innerHTML = `<div style="font-size:11px;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(item.label)}</div><div style="font-size:9px;color:#94a3b8;text-transform:uppercase;margin-top:3px;">${escapeHtml(kindLabel(item.kind))}</div>`;
        const controls = document.createElement("div");
        controls.style.cssText = "display:flex;gap:4px;";
        const up = makeButton("↑");
        const down = makeButton("↓");
        const remove = makeButton("×");
        up.disabled = index === 0;
        down.disabled = index === selectedKeys.length - 1;
        up.onclick = () => { useAutomaticMappings = false; [selectedKeys[index - 1], selectedKeys[index]] = [selectedKeys[index], selectedKeys[index - 1]]; renderLists(); };
        down.onclick = () => { useAutomaticMappings = false; [selectedKeys[index + 1], selectedKeys[index]] = [selectedKeys[index], selectedKeys[index + 1]]; renderLists(); };
        remove.onclick = () => { useAutomaticMappings = false; selectedKeys.splice(index, 1); renderLists(); };
        controls.append(up, down, remove);
        row.append(number, thumb, label, controls);
        selectedList.append(row);
      });
      if (!selectedKeys.length) {
        const empty = document.createElement("div");
        empty.textContent = startFrameReserved
          ? "The scene start frame remains Image 1. No additional character, location, or storyboard references are selected."
          : "No Reference Builder images selected.";
        empty.style.cssText = "border:1px dashed #475569;border-radius:6px;color:#94a3b8;padding:14px;line-height:1.4;";
        selectedList.append(empty);
      }
    };

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;padding:11px 15px;border-top:1px solid #1e293b;";
    const utility = document.createElement("div");
    utility.style.cssText = "display:flex;gap:7px;flex-wrap:wrap;";
    const mappedDefaults = makeButton("Use Scene Mappings");
    const clearSelection = makeButton("Clear Selection");
    const openBuilder = makeButton("Open Reference Builder");
    mappedDefaults.onclick = () => { useAutomaticMappings = true; selectedKeys = miniMaxMappedReferenceKeysForSegment(segment, refs); renderLists(); };
    clearSelection.onclick = () => { useAutomaticMappings = false; selectedKeys = []; renderLists(); };
    openBuilder.onclick = () => { backdrop.remove(); openReferenceBuilderTargetChooser(); };
    utility.append(mappedDefaults, clearSelection, openBuilder);
    const save = makeButton("Save MiniMax Reference Order", "primary");
    save.onclick = async () => {
      pushHistory();
      segment.minimax_h3_reference_keys = useAutomaticMappings ? null : selectedKeys.slice(0, maxLibraryReferences);
      backdrop.remove();
      syncMiniMaxReferenceButtons();
      await autoSaveSessionQuiet("MiniMax H3 scene references");
      const savedCount = miniMaxReferenceKeysForSegment(segment).slice(0, maxLibraryReferences).length;
      const totalCount = savedCount + (startFrameReserved ? 1 : 0) + (continuityReserved ? 1 : 0);
      toast(useAutomaticMappings
        ? `MiniMax H3 will automatically follow this scene's Reference Builder mappings (${totalCount}/9 images${startFrameReserved ? ", including the start frame" : ""}${continuityReserved ? ", including the reserved continuity frame" : ""}).`
        : `Saved ${totalCount} ordered MiniMax H3 image${totalCount === 1 ? "" : "s"}${startFrameReserved ? " including Image 1 as the exact start frame" : ""}${continuityReserved ? " including the reserved continuity frame" : ""}.`);
    };
    footer.append(utility, save);
    box.append(header, note, body, footer);
    backdrop.append(box);
    document.body.append(backdrop);
    close.onclick = () => backdrop.remove();
    backdrop.addEventListener("pointerdown", (event) => { if (event.target === backdrop) backdrop.remove(); });
    renderLists();
  }

  function rtvReferencesForSegment(segment = activeSegment()) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const references = { subjects: [], background: {}, use_subject_placeholder: false };
    const referenceBehavior = rtvReferenceBehaviorForSegment(segment);
    references.reference_behavior = referenceBehavior;
    if (referenceBehavior === "first_last_frame") {
      const firstFrame = segmentImageSource(segment);
      const lastFrame = firstLastFrameResolvedEndImageSource(segment);
      if (firstFrame?.path || firstFrame?.data) {
        references.subjects.push({
          ...rtvReferenceImagePayload(firstFrame),
          label: "First frame",
          reference_type: "first_frame",
          first_last_frame_role: "first",
        });
      }
      if (lastFrame?.path || lastFrame?.data) {
        references.subjects.push({
          ...rtvReferenceImagePayload(lastFrame),
          label: "Last frame",
          reference_type: "last_frame",
          first_last_frame_role: "last",
        });
      }
      return references;
    }
    const addSubject = (item) => {
      const image = item?.image || {};
      if (!image.path && !image.data) return;
      references.subjects.push({
        ...rtvReferenceImagePayload(image),
        label: [item?.reference_type, item?.name || item?.description || ""].map((value) => String(value || "").trim()).filter(Boolean).join(": "),
        reference_type: String(item?.reference_type || "character"),
      });
    };
    referenceBuilderSubjectItemsForSegment(refs, segment).forEach(addSubject);
    const sceneImageAnchor = rtvSceneImageAnchorPayload(segment);
    references.subjects = references.subjects.slice(0, sceneImageAnchor ? 3 : 4);
    if (sceneImageAnchor) references.subjects.push(sceneImageAnchor);
    if (segment?.no_character_present && !references.subjects.length) {
      references.use_subject_placeholder = true;
      references.subjects.push({
        path: "",
        data: "",
        name: "",
        label: "Neutral placeholder used for MSR subject slot",
        reference_type: "placeholder",
        placeholder: true,
      });
    }
    if (refs.use_location_references && segment) {
      const locId = String(sceneReferenceMapValue(refs.scene_map, segment) || "");
      const location = refs.locations.find((item) => item.id === locId);
      const image = location?.image || {};
      if (image.path || image.data) {
        references.background = {
          ...rtvReferenceImagePayload(image),
          label: String(location?.name || location?.description || ""),
        };
      }
    }
    return references;
  }

  function nbReferenceContextForSegment(segment = activeSegment()) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const context = fluxReferenceContextForSegment(segment);
    if (!refs.use_subject_reference) {
      context.has_subject_reference = false;
      context.subject_reference_count = 0;
      context.subject_description = "";
    }
    if (!refs.use_location_references) {
      context.has_location_reference = false;
      context.location_name = "";
      context.location_description = "";
    }
    return context;
  }

  function fluxReferenceContextForSegment(segment = activeSegment()) {
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const context = {
      has_subject_reference: false,
      has_location_reference: false,
      subject_reference_count: 0,
      subject_description: "",
      location_name: "",
      location_description: "",
    };
    if (!segment?.no_character_present) {
      const mappedSubjects = referenceBuilderSubjectItemsForSegment(refs, segment);
      const described = mappedSubjects
        .filter((item) => item?.description || item?.name)
        .map((item, index) => {
          const type = String(item.reference_type || "character").trim();
          const label = item.name || `Reference ${index + 1}`;
          return `${type}: ${label}: ${item.description || ""}`.trim();
        });
      const hasImage = mappedSubjects.some((item) => item?.image?.path || item?.image?.data);
      if (hasImage || described.length) {
        context.has_subject_reference = hasImage;
        context.subject_reference_count = mappedSubjects.filter((item) => item?.image?.path || item?.image?.data).length || mappedSubjects.length;
        context.subject_description = described.join("\n");
      }
    }
    if (refs.use_location_references && segment) {
      const locId = String(sceneReferenceMapValue(refs.scene_map, segment) || "");
      const location = refs.locations.find((item) => item.id === locId);
      const image = location?.image || {};
      if (location && (image.path || image.data || location.name || location.description)) {
        context.has_location_reference = Boolean(image.path || image.data);
        context.location_name = location.name || "";
        context.location_description = location.description || "";
      }
    }
    return context;
  }

  return {
    assertMiniMaxH3ReferenceCapacity, assertMiniMaxH3ReferenceDescriptionsReady, isRefmodPipeline, miniMaxRefmodItemsForSegment,
    fluxReferenceContextForSegment, miniMaxDesiredReferenceKeysForSegment, miniMaxH3ReferenceCapacityStatus,
    miniMaxOrderedImageReferenceItemsForSegment, miniMaxReferenceBuilderImagePathsForSegment,
    miniMaxReferenceKeysForSegment, miniMaxReferencePurposeText, nbReferenceContextForSegment,
    openMiniMaxReferenceSelector, referenceBuilderSubjectItemsForSegment, rtvReferencesForSegment,
  };
}
