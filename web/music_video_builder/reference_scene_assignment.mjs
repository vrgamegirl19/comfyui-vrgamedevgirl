import { makeButton, makeCheckbox, makeField, makeInput, makeSelect, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { cloneMiniMaxH3Settings } from "./minimax_h3.mjs";
import { eligibleSharedLocationRuns } from "./scene_locations.mjs";

export function createSceneAssignment({
  allEditableSegments, backdrop, logicalReferenceSubjects, logicalSubjectIdsForScene, openLyricReviewModal,
  pushHistory, refs, renderAll, selectedSegmentsForBatch, state,
}) {
  function openSceneAssignmentDialog() {
    const scenes = allEditableSegments();
    const subjects = logicalReferenceSubjects(refs);
    const locations = refs.locations || [];
    if (!scenes.length) {
      toast("Add scenes before assigning character and location mappings.", true);
      return;
    }
    if (!subjects.length && !locations.length) {
      toast("Add at least one character or location reference first.", true);
      return;
    }
    const dialogBackdrop = document.createElement("div");
    dialogBackdrop.style.cssText = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;";
    const panel = document.createElement("div");
    panel.style.cssText = "width:min(820px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;border:1px solid #155e75;border-radius:9px;background:#111827;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.65);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const titleRow = document.createElement("div");
    titleRow.style.cssText = "display:flex;align-items:flex-start;justify-content:space-between;gap:12px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:17px;font-weight:900;color:#cffafe;">Character &amp; Location Assignment</div><div style="font-size:12px;color:#94a3b8;margin-top:4px;">Choose a distribution pattern, preview it, then apply it to the shared scene mapping.</div>`;
    const help = makeButton("?", "neutral");
    help.title = "Explain every Character & Location Assignment setting";
    help.style.cssText += "flex:0 0 34px;width:34px;height:34px;padding:0;border-radius:999px;font-size:16px;font-weight:900;";
    titleRow.append(title, help);
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;";
    const scope = makeSelect(["all", "selected", "range"], "all");
    scope.options[0].textContent = "All scenes";
    scope.options[1].textContent = "Multi-selected scenes";
    scope.options[2].textContent = "Scene range";
    scope.options[1].disabled = selectedSegmentsForBatch().length === 0;
    const rangeStart = makeInput("1", "number");
    const rangeEnd = makeInput(String(scenes.length), "number");
    rangeStart.min = rangeEnd.min = "1";
    rangeStart.max = rangeEnd.max = String(scenes.length);
    const characterMode = makeSelect(["random", "rotate", "blocks", "unchanged"], subjects.length ? "random" : "unchanged");
    characterMode.options[0].textContent = "Random character each scene";
    characterMode.options[1].textContent = "Rotate characters in order";
    characterMode.options[2].textContent = "Character blocks";
    characterMode.options[3].textContent = "Leave characters unchanged";
    const blockSize = makeInput("10", "number");
    blockSize.min = "1";
    const locationMode = makeSelect(["random", "rotate", "blocks", "unchanged"], locations.length ? "random" : "unchanged");
    locationMode.options[0].textContent = "Random location each scene";
    locationMode.options[1].textContent = "Rotate locations in order";
    locationMode.options[2].textContent = "Repeat each location for X scenes";
    locationMode.options[3].textContent = "Leave locations unchanged";
    const locationBlockSize = makeInput("4", "number");
    locationBlockSize.min = "1";
    const replaceExisting = makeCheckbox("Replace existing mappings", false);
    const avoidLocationRepeat = makeCheckbox("Avoid consecutive location repeats", true);
    const autoContinuous = makeCheckbox("Auto continuous shots for shared locations", false);
    autoContinuous.wrapper.title = "With Apply Mapping, use one shot across each consecutive location group. Later scenes continue from their predecessor when MiniMax H3 allows it.";
    autoContinuous.input.disabled = normalizeProjectVideoEngine(state.projectVideoEngine) !== "minimax_h3";
    grid.append(
      makeField("Scenes to assign", scope),
      makeField("Scene range start", rangeStart),
      makeField("Scene range end", rangeEnd),
      makeField("Character pattern", characterMode),
      makeField("Scenes per character block", blockSize),
      makeField("Location pattern", locationMode),
      makeField("Scenes per location block", locationBlockSize),
      replaceExisting.wrapper,
      avoidLocationRepeat.wrapper,
      autoContinuous.wrapper,
    );
    const preview = document.createElement("textarea");
    preview.readOnly = true;
    preview.style.cssText = "min-height:230px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#dbeafe;padding:10px;font:12px/1.5 monospace;";
    const note = document.createElement("div");
    note.style.cssText = "font-size:11px;color:#94a3b8;line-height:1.4;";
    note.textContent = "Scenes marked No character present keep their character mapping empty. Fill-empty mode preserves existing character and location assignments.";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:9px;";
    const cancelAssign = makeButton("Cancel");
    const shuffle = makeButton("Preview / Shuffle", "primary");
    const apply = makeButton("Apply Mapping", "primary");
    actions.append(cancelAssign, shuffle, apply);
    panel.append(titleRow, grid, note, preview, actions);
    dialogBackdrop.append(panel);
    document.body.append(dialogBackdrop);

    let proposal = [];
    help.onclick = () => {
      const helpBackdrop = document.createElement("div");
      helpBackdrop.style.cssText = "position:fixed;inset:0;z-index:100012;background:rgba(0,0,0,.76);display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;";
      const helpPanel = document.createElement("div");
      helpPanel.style.cssText = "width:min(760px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;border:1px solid #155e75;border-radius:9px;background:#111827;color:#e5e7eb;box-shadow:0 24px 80px rgba(0,0,0,.7);padding:18px;";
      helpPanel.innerHTML = `
          <div style="font-size:18px;font-weight:900;color:#cffafe;margin-bottom:5px;">Character &amp; Location Assignment Help</div>
          <div style="font-size:12px;color:#94a3b8;line-height:1.5;margin-bottom:16px;">This tool fills the Reference Builder's character and location mappings across multiple scenes. It changes mappings only; it does not generate prompts, images, or videos.</div>
          <div style="display:grid;gap:12px;font-size:12px;line-height:1.5;">
            <div><b style="color:#67e8f9;">Scenes to assign</b><br><b>All scenes</b> includes the entire timeline. <b>Multi-selected scenes</b> includes only scenes selected with Select Multi. <b>Scene range</b> includes the numbered scenes between Range Start and Range End.</div>
            <div><b style="color:#67e8f9;">Scene range start / end</b><br>These fields are used only when Scenes to assign is set to Scene range. Scene numbering starts at 1 and both endpoints are included.</div>
            <div><b style="color:#67e8f9;">Character pattern</b><br><b>Random</b> chooses any saved character for each target scene. <b>Rotate</b> cycles through saved characters in order. <b>Character blocks</b> keeps one character for a group of scenes, then moves to the next character. <b>Leave unchanged</b> does not edit character mappings.</div>
            <div><b style="color:#67e8f9;">Scenes per character block</b><br>Controls the size of each character block. For example, 10 assigns Character 1 to the first 10 target scenes, Character 2 to the next 10, and so on. It is used only with Character blocks.</div>
            <div><b style="color:#67e8f9;">Location pattern</b><br><b>Random</b> chooses any saved location for each target scene. <b>Rotate</b> changes to the next saved location every scene. <b>Repeat each location for X scenes</b> keeps one location for a block of target scenes, then advances to the next saved location. After the last saved location, it loops back to the first. <b>Leave unchanged</b> does not edit location mappings.</div>
            <div><b style="color:#67e8f9;">Scenes per location block</b><br>Controls how long each location is reused when Repeat each location for X scenes is selected. For example, a value of 4 assigns Location 1 to target scenes 1–4, Location 2 to target scenes 5–8, and so on. Once all saved locations are used, the same ordered cycle starts again. For a range or multi-selection, counting follows only the targeted scenes in timeline order.</div>
            <div><b style="color:#67e8f9;">Replace existing mappings</b><br>Off is the safe default: only empty character or location mappings are filled. Turn it on to overwrite mappings that are already assigned in the target scenes.</div>
            <div><b style="color:#67e8f9;">Avoid consecutive location repeats</b><br>When Random location is selected and at least two locations exist, this prevents two neighboring target scenes from receiving the same location.</div>
            <div><b style="color:#67e8f9;">Auto continuous shots for shared locations</b><br>With MiniMax H3, Apply Mapping sets consecutive scenes with the same mapped location to one shot. The first scene starts a new take; each later scene continues from its predecessor when its mode allows masked continuation. Location changes and timeline gaps start new takes.</div>
            <div><b style="color:#67e8f9;">No character present scenes</b><br>Scenes explicitly marked No character present always keep their character mapping empty. Their location can still be assigned.</div>
            <div><b style="color:#67e8f9;">Preview / Shuffle</b><br>Builds a preview without changing saved mappings. With random patterns, click it again to generate a different arrangement.</div>
            <div><b style="color:#67e8f9;">Apply Mapping</b><br>Applies the exact arrangement currently shown in the preview. One Undo history point is created before the mappings are changed.</div>
            <div><b style="color:#67e8f9;">Cancel</b><br>Closes the assignment tool without applying the preview.</div>
          </div>`;
      const closeHelp = makeButton("Close", "primary");
      closeHelp.style.cssText += "width:100%;margin-top:16px;";
      helpPanel.append(closeHelp);
      helpBackdrop.append(helpPanel);
      document.body.append(helpBackdrop);
      closeHelp.onclick = () => helpBackdrop.remove();
      helpBackdrop.addEventListener("pointerdown", (event) => {
        if (event.target === helpBackdrop) helpBackdrop.remove();
      });
    };
    const targetScenes = () => {
      if (scope.value === "selected") {
        const ids = new Set(selectedSegmentsForBatch().map((scene) => scene.id));
        return scenes.map((scene, index) => ({ scene, index })).filter(({ scene }) => ids.has(scene.id));
      }
      if (scope.value === "range") {
        const start = Math.max(1, Math.min(scenes.length, Number(rangeStart.value) || 1));
        const end = Math.max(start, Math.min(scenes.length, Number(rangeEnd.value) || scenes.length));
        return scenes.map((scene, index) => ({ scene, index })).filter(({ index }) => index + 1 >= start && index + 1 <= end);
      }
      return scenes.map((scene, index) => ({ scene, index }));
    };
    const randomItem = (items, previousId = "", avoidRepeat = false) => {
      const pool = avoidRepeat && items.length > 1 ? items.filter((item) => item.id !== previousId) : items;
      return pool[Math.floor(Math.random() * pool.length)] || null;
    };
    const createProposal = () => {
      let previousLocationId = "";
      proposal = targetScenes().map(({ scene, index }, targetIndex) => {
        const existingSubjects = logicalSubjectIdsForScene(refs, scene);
        const existingLocation = refs.scene_map?.[scene.id] || "";
        let subjectId = "";
        let locationId = "";
        if (!scene.no_character_present && characterMode.value !== "unchanged" && subjects.length) {
          if (characterMode.value === "random") subjectId = randomItem(subjects)?.id || "";
          else if (characterMode.value === "rotate") subjectId = subjects[targetIndex % subjects.length]?.id || "";
          else subjectId = subjects[Math.floor(targetIndex / Math.max(1, Number(blockSize.value) || 1)) % subjects.length]?.id || "";
        }
        if (locationMode.value !== "unchanged" && locations.length) {
          const chosen = locationMode.value === "rotate"
            ? locations[targetIndex % locations.length]
            : locationMode.value === "blocks"
              ? locations[Math.floor(targetIndex / Math.max(1, Number(locationBlockSize.value) || 1)) % locations.length]
              : randomItem(locations, previousLocationId, avoidLocationRepeat.input.checked);
          locationId = chosen?.id || "";
          previousLocationId = locationId || previousLocationId;
        }
        const finalSubjects = scene.no_character_present
          ? []
          : characterMode.value === "unchanged" || (!replaceExisting.input.checked && existingSubjects.length)
            ? existingSubjects
            : subjectId ? [subjectId] : [];
        const finalLocation = locationMode.value === "unchanged" || (!replaceExisting.input.checked && existingLocation)
          ? existingLocation
          : locationId;
        return { scene, index, subjectIds: finalSubjects, locationId: finalLocation };
      });
      preview.value = proposal.map((item) => {
        const characterNames = item.subjectIds.map((id) => subjects.find((subject) => subject.id === id)?.name || "Unknown").join(", ") || "Unassigned";
        const locationName = locations.find((location) => location.id === item.locationId)?.name || "Unassigned";
        return `Scene ${item.index + 1}: ${characterNames} — ${locationName}`;
      }).join("\n");
    };
    const syncOptions = () => {
      const isRange = scope.value === "range";
      rangeStart.disabled = rangeEnd.disabled = !isRange;
      blockSize.disabled = characterMode.value !== "blocks";
      locationBlockSize.disabled = locationMode.value !== "blocks";
      avoidLocationRepeat.input.disabled = locationMode.value !== "random";
      createProposal();
    };
    for (const control of [scope, rangeStart, rangeEnd, characterMode, blockSize, locationMode, locationBlockSize, replaceExisting.input, avoidLocationRepeat.input]) {
      control.addEventListener("change", syncOptions);
    }
    shuffle.onclick = createProposal;
    cancelAssign.onclick = () => dialogBackdrop.remove();
    dialogBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === dialogBackdrop) dialogBackdrop.remove();
    });
    apply.onclick = async () => {
      if (!proposal.length) createProposal();
      pushHistory();
      refs.subject_scene_map = refs.subject_scene_map && typeof refs.subject_scene_map === "object" ? refs.subject_scene_map : {};
      refs.scene_map = refs.scene_map && typeof refs.scene_map === "object" ? refs.scene_map : {};
      for (const item of proposal) {
        if (characterMode.value !== "unchanged") {
          if (item.subjectIds.length) refs.subject_scene_map[item.scene.id] = [...item.subjectIds];
          else delete refs.subject_scene_map[item.scene.id];
        }
        if (locationMode.value !== "unchanged") {
          if (item.locationId) refs.scene_map[item.scene.id] = item.locationId;
          else delete refs.scene_map[item.scene.id];
        }
      }
      refs.use_subject_reference = Boolean(subjects.length);
      refs.use_location_references = Boolean(locations.length);
      let continuityGroups = 0;
      if (autoContinuous.input.checked) {
        const eligible = eligibleSharedLocationRuns(refs, scenes, state.miniMaxH3Settings);
        for (const run of eligible) {
          for (const [index, scene] of run.entries()) {
            scene.location_continuous_shot = true;
            if (index === 0) continue;
            const base = cloneMiniMaxH3Settings(scene.use_scene_minimax_h3_settings && scene.minimax_h3_settings
              ? scene.minimax_h3_settings : state.miniMaxH3Settings);
            scene.use_scene_minimax_h3_settings = true;
            scene.minimax_h3_settings = cloneMiniMaxH3Settings({
              ...base,
              continuity_mode: "latent_continuation_masked",
              location_transition_preset: "masked",
            });
            scene.minimax_h3_mode = scene.minimax_h3_settings.video_mode;
          }
        }
        continuityGroups = eligible.length;
        syncInspector();
      }
      renderAll();
      dialogBackdrop.remove();
      if (autoContinuous.input.checked) await autoSaveSessionQuiet("scene mappings and shared-location continuity applied");
      toast(`Assigned mappings for ${proposal.length} scene${proposal.length === 1 ? "" : "s"}.${autoContinuous.input.checked ? ` Continuous shots applied to ${continuityGroups} location group${continuityGroups === 1 ? "" : "s"}.` : ""}`);
    };
    syncOptions();
  }

  function launchAdvancedLineMapping(segment = null) {
    state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
    backdrop.remove();
    openLyricReviewModal(segment ? { focusSceneId: segment.id } : {});
  }

  return { launchAdvancedLineMapping, openSceneAssignmentDialog };
}
