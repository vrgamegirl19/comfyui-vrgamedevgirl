import { postJson, saveStoryboardFile } from "./api.mjs";
import {
  copyTextToClipboard,
  createStoryboardProgressWindow,
  createToast,
  escapeHtml,
  makeButton,
  makeInput,
  makeSelect,
  makeTextarea,
} from "./controls.mjs";
import { normalizeReferenceBuilderCatalog } from "./references.mjs";
import {
  normalizeScene,
  normalizeStoryboardShortFilmPlanningMode,
  normalizeStoryLayer,
  slimSceneForRequest,
  slimStoryboardForRequest,
} from "./scenes.mjs";
import {
  normalizeStoryboardScriptImportState,
  parseStoryboardScriptImport,
  planStoryboardScriptScenes,
  storyboardScriptSpeakerMatchKey,
  suggestStoryboardScriptSpeakerMatch,
} from "./script_import.mjs";

export function createScriptMapper({
  applyDialoguePlanButton, idLoraDialogueSceneCount, isFullyCustomShortFilm, isIdLoraMode,
  isMiniMaxShortFilmMode, notifyStoryboardDefaultsChanged, promptRunnerName, refreshSetupPanelSummaries,
  renderTable, saveStoryboard, setMode, songStoryBriefInput, state, syncStoryLayerFromInputs,
  userStoryArcInput,
}) {
  function openMiniMaxScriptMapper() {
    if (!isMiniMaxShortFilmMode || state.miniMaxH3AudioMode !== "built_in_audio") {
      createToast("Script Mapper is available for MiniMax Short Film with Built-in MiniMax Audio.", true);
      return;
    }
    const mapperBackdrop = document.createElement("div");
    mapperBackdrop.style.cssText = "position:fixed;inset:0;z-index:100070;background:rgba(0,0,0,.78);display:flex;align-items:stretch;justify-content:center;padding:18px;box-sizing:border-box;";
    const mapperShell = document.createElement("div");
    mapperShell.style.cssText = "width:min(1320px,calc(100vw - 36px));height:calc(100vh - 36px);min-height:520px;border:1px solid #0e7490;border-radius:11px;background:#07111f;color:#e5e7eb;box-shadow:0 24px 90px rgba(0,0,0,.72);overflow:hidden;display:grid;grid-template-rows:auto minmax(0,1fr) auto;";
    const mapperHeader = document.createElement("div");
    mapperHeader.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;align-items:center;padding:15px 18px;background:#083344;border-bottom:1px solid #155e75;";
    const mapperHeading = document.createElement("div");
    mapperHeading.innerHTML = `<div style="font-size:20px;font-weight:900;color:#cffafe;">Import Script / Script Mapper</div><div style="font-size:12px;color:#bae6fd;line-height:1.4;margin-top:3px;">Import, map, and time exact dialogue, then activate it as the authoritative source for Guided Film Automation. Activation alone does not change the Video Builder timeline.</div>`;
    const mapperCloseTop = makeButton("Close");
    mapperHeader.append(mapperHeading, mapperCloseTop);

    const mapperBody = document.createElement("div");
    mapperBody.style.cssText = "min-height:0;overflow:auto;padding:16px 18px;display:grid;grid-template-columns:minmax(360px,.8fr) minmax(480px,1.2fr);gap:14px;align-items:stretch;";
    const sourcePanel = document.createElement("div");
    sourcePanel.style.cssText = "min-width:0;border:1px solid #334155;border-radius:9px;background:#0b1220;padding:12px;display:flex;flex-direction:column;gap:10px;";
    const sourceTitle = document.createElement("div");
    sourceTitle.innerHTML = `<div style="font-weight:900;color:#cffafe;">Script source</div><div style="font-size:12px;color:#94a3b8;line-height:1.4;margin-top:3px;">Plain text uses <strong style="color:#e2e8f0;">speaker: exact dialogue</strong>. JSON accepts cues with speaker/speaker_name and text/dialogue fields.</div>`;
    const existingScriptImport = normalizeStoryboardScriptImportState(state.scriptImport);
    const scriptInput = makeTextarea(existingScriptImport.raw_text, "woman: Have you tried the new MiniMax H3 model yet?\n\nman: I have, and honestly...", 22);
    scriptInput.style.flex = "1 1 auto";
    scriptInput.style.minHeight = "360px";
    const sourceActions = document.createElement("div");
    sourceActions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;";
    const loadScriptButton = makeButton("Load .txt / .json", "primary");
    const parseScriptButton = makeButton("Parse + Plan Preview", "purple");
    const clearScriptButton = makeButton("Clear");
    const scriptFileInput = document.createElement("input");
    scriptFileInput.type = "file";
    scriptFileInput.accept = ".txt,.json,text/plain,application/json";
    scriptFileInput.style.display = "none";
    sourceActions.append(loadScriptButton, parseScriptButton, clearScriptButton, scriptFileInput);
    const sceneLengthSettings = document.createElement("div");
    sceneLengthSettings.style.cssText = "border:1px solid #0e7490;border-radius:7px;background:#082f49;padding:10px;display:grid;grid-template-columns:minmax(150px,.8fr) minmax(180px,1.2fr);gap:9px;align-items:center;";
    const sceneLengthLabel = document.createElement("div");
    sceneLengthLabel.innerHTML = `<div style="font-weight:900;color:#cffafe;">Maximum scene length</div><div style="font-size:11px;color:#bae6fd;line-height:1.35;margin-top:3px;">Hard ceiling for every planned MiniMax clip. Shorter clips can reduce VRAM pressure and generation time.</div>`;
    const sceneLengthControls = document.createElement("div");
    sceneLengthControls.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 96px;gap:8px;align-items:center;";
    const maxSceneLengthSelect = makeSelect([
      { value: "5", label: "5 seconds — low VRAM" },
      { value: "8", label: "8 seconds — recommended" },
      { value: "10", label: "10 seconds" },
      { value: "12", label: "12 seconds" },
      { value: "15", label: "15 seconds — maximum" },
      { value: "custom", label: "Custom..." },
    ], "8");
    const customSceneLengthInput = makeInput("8", "3–15");
    customSceneLengthInput.type = "number";
    customSceneLengthInput.min = "3";
    customSceneLengthInput.max = "15";
    customSceneLengthInput.step = "0.5";
    customSceneLengthInput.title = "Custom maximum scene length from 3 to 15 seconds";
    customSceneLengthInput.style.display = "none";
    const presetSceneLengths = new Set([5, 8, 10, 12, 15]);
    if (existingScriptImport.enabled) {
      if (presetSceneLengths.has(existingScriptImport.maximum_scene_seconds)) {
        maxSceneLengthSelect.value = String(existingScriptImport.maximum_scene_seconds);
      } else {
        maxSceneLengthSelect.value = "custom";
        customSceneLengthInput.value = String(existingScriptImport.maximum_scene_seconds);
        customSceneLengthInput.style.display = "";
      }
    }
    const currentMaximumSceneSeconds = () => Math.max(3, Math.min(15, Number(maxSceneLengthSelect.value === "custom" ? customSceneLengthInput.value : maxSceneLengthSelect.value) || 8));
    sceneLengthControls.append(maxSceneLengthSelect, customSceneLengthInput);
    sceneLengthSettings.append(sceneLengthLabel, sceneLengthControls);
    const sourceStatus = document.createElement("div");
    sourceStatus.style.cssText = "min-height:18px;font-size:12px;color:#94a3b8;line-height:1.4;";
    sourcePanel.append(sourceTitle, scriptInput, sourceActions, sceneLengthSettings, sourceStatus);

    const previewPanel = document.createElement("div");
    previewPanel.style.cssText = "min-width:0;border:1px solid #155e75;border-radius:9px;background:#071827;padding:12px;overflow:auto;";
    const renderEmptyPreview = () => {
      previewPanel.innerHTML = `<div style="height:100%;min-height:360px;display:grid;place-items:center;border:1px dashed #334155;border-radius:8px;color:#94a3b8;text-align:center;padding:24px;box-sizing:border-box;"><div><strong style="display:block;color:#cffafe;font-size:15px;margin-bottom:6px;">No parsed script yet</strong>Paste dialogue or load a file, then click Parse Preview.</div></div>`;
    };
    let lastParsed = null;
    const speakerMatches = new Map();
    for (const match of existingScriptImport.speaker_matches || []) {
      const aliasKey = storyboardScriptSpeakerMatchKey(match?.speaker_alias);
      if (!aliasKey) continue;
      speakerMatches.set(aliasKey, {
        subject_id: String(match?.reference_subject_id || ""),
        method: String(match?.match_method || (match?.reference_subject_id ? "manual" : "unmatched")),
      });
    }
    const referenceCharacters = normalizeReferenceBuilderCatalog(state.referenceBuilder).subjects;
    const copyParsedButton = makeButton("Copy Parsed JSON");
    copyParsedButton.disabled = true;
    const useGuidedScriptButton = makeButton("Use This Script in Guided Film", "primary");
    useGuidedScriptButton.disabled = true;
    const removeGuidedScriptButton = makeButton("Remove Active Script");
    removeGuidedScriptButton.style.display = existingScriptImport.enabled ? "" : "none";
    const renderParsedPreview = (parsed) => {
      const characterById = new Map(referenceCharacters.map((character) => [String(character.id || ""), character]));
      for (const speaker of Array.isArray(parsed?.speakers) ? parsed.speakers : []) {
        const aliasKey = storyboardScriptSpeakerMatchKey(speaker?.name);
        if (!aliasKey || speakerMatches.has(aliasKey)) continue;
        const suggested = suggestStoryboardScriptSpeakerMatch(speaker.name, referenceCharacters);
        speakerMatches.set(aliasKey, suggested
          ? { subject_id: String(suggested.id || ""), method: "auto" }
          : { subject_id: "", method: "unmatched" });
      }
      const mappedSpeakers = (Array.isArray(parsed?.speakers) ? parsed.speakers : []).map((speaker) => {
        const aliasKey = storyboardScriptSpeakerMatchKey(speaker?.name);
        const match = speakerMatches.get(aliasKey) || { subject_id: "", method: "unmatched" };
        const character = characterById.get(String(match.subject_id || ""));
        const method = character ? String(match.method || "manual") : "unmatched";
        return {
          ...speaker,
          speaker_alias: String(speaker.name || ""),
          reference_subject_id: character ? String(character.id || "") : "",
          reference_subject_name: character ? String(character.name || "") : "",
          match_method: method,
        };
      });
      const mappedSpeakerByKey = new Map(mappedSpeakers.map((speaker) => [storyboardScriptSpeakerMatchKey(speaker.name), speaker]));
      parsed.speakers = mappedSpeakers;
      parsed.cues = (Array.isArray(parsed?.cues) ? parsed.cues : []).map((cue) => {
        const mappedSpeaker = mappedSpeakerByKey.get(storyboardScriptSpeakerMatchKey(cue?.speaker));
        return {
          ...cue,
          speaker_alias: String(cue?.speaker || ""),
          speaker_id: String(mappedSpeaker?.reference_subject_id || ""),
          speaker_name: String(mappedSpeaker?.reference_subject_name || cue?.speaker || ""),
          reference_subject_id: String(mappedSpeaker?.reference_subject_id || ""),
          reference_subject_name: String(mappedSpeaker?.reference_subject_name || ""),
          speaker_match_method: String(mappedSpeaker?.match_method || "unmatched"),
        };
      });
      parsed.speaker_matches = mappedSpeakers.map((speaker) => ({
        speaker_alias: String(speaker.speaker_alias || speaker.name || ""),
        reference_subject_id: String(speaker.reference_subject_id || ""),
        reference_subject_name: String(speaker.reference_subject_name || ""),
        match_method: String(speaker.match_method || "unmatched"),
      }));
      parsed.unmatched_speakers = mappedSpeakers.filter((speaker) => !speaker.reference_subject_id).map((speaker) => String(speaker.name || ""));
      parsed.scene_plan = planStoryboardScriptScenes(parsed.cues, { max_scene_seconds: currentMaximumSceneSeconds() });
      lastParsed = parsed;
      copyParsedButton.disabled = !parsed?.cues?.length;
      const cueCount = Array.isArray(parsed?.cues) ? parsed.cues.length : 0;
      const speakerCount = Array.isArray(parsed?.speakers) ? parsed.speakers.length : 0;
      const wordCount = Number(parsed?.word_count || 0);
      const speechSeconds = Number(parsed?.estimated_spoken_seconds || 0);
      const errorCount = Array.isArray(parsed?.errors) ? parsed.errors.length : 0;
      const scenePlan = parsed.scene_plan || { scenes: [], warnings: [] };
      const plannedSceneCount = Number(scenePlan.scene_count || 0);
      const speakerHtml = speakerCount
        ? parsed.speakers.map((speaker) => `<span style="display:inline-flex;gap:6px;align-items:center;border:1px solid #0e7490;border-radius:999px;background:#083344;color:#cffafe;padding:5px 9px;font-size:11px;font-weight:900;">${escapeHtml(speaker.name)} <span style="color:#67e8f9;">${Number(speaker.cue_count || 0)} cue${Number(speaker.cue_count || 0) === 1 ? "" : "s"}</span></span>`).join("")
        : `<span style="color:#fca5a5;">No speakers detected.</span>`;
      const matchedCount = parsed.speakers.filter((speaker) => speaker.reference_subject_id).length;
      useGuidedScriptButton.disabled = !cueCount || errorCount > 0 || matchedCount !== speakerCount;
      useGuidedScriptButton.title = errorCount
        ? "Resolve every parse issue before activating the script."
        : matchedCount !== speakerCount
          ? "Match every script speaker to a Reference Builder character first."
          : "Save this exact script and timed segment plan as the authoritative source for Guided Film Automation.";
      const speakerMappingHtml = speakerCount
        ? parsed.speakers.map((speaker, index) => {
          const status = speaker.reference_subject_id
            ? speaker.match_method === "auto" ? "Auto matched" : "Manually matched"
            : "Needs a character";
          const statusColor = speaker.reference_subject_id ? "#86efac" : "#fbbf24";
          const options = [
            `<option value="">Choose Reference Builder character...</option>`,
            ...referenceCharacters.map((character) => `<option value="${escapeHtml(character.id)}"${String(character.id) === String(speaker.reference_subject_id) ? " selected" : ""}>${escapeHtml(character.name)}</option>`),
          ].join("");
          return `<div style="display:grid;grid-template-columns:minmax(130px,.65fr) minmax(210px,1.35fr) auto;gap:9px;align-items:center;border-top:${index ? "1px solid #1e3a5f" : "0"};padding:${index ? "9px 0 0" : "0"};margin-top:${index ? "9px" : "0"};"><div style="min-width:0;"><div style="font-size:10px;color:#94a3b8;text-transform:uppercase;font-weight:900;">Script speaker</div><div title="${escapeHtml(speaker.name)}" style="color:#cffafe;font-weight:900;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:3px;">${escapeHtml(speaker.name)}</div></div><select data-script-speaker-index="${index}" style="width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:9px;"${referenceCharacters.length ? "" : " disabled"}>${options}</select><div style="color:${statusColor};font-size:11px;font-weight:900;white-space:nowrap;">${status}</div></div>`;
        }).join("")
        : `<div style="color:#fca5a5;">Parse at least one valid speaker before matching characters.</div>`;
      const rows = cueCount
        ? parsed.cues.map((cue) => `<tr style="border-top:1px solid #1e3a5f;"><td style="padding:8px;color:#67e8f9;font-weight:900;vertical-align:top;">${cue.index}</td><td style="padding:8px;color:#94a3b8;vertical-align:top;">${cue.line_number || "JSON"}</td><td style="padding:8px;color:#cffafe;font-weight:900;vertical-align:top;overflow-wrap:anywhere;">${escapeHtml(cue.speaker)}</td><td style="padding:8px;color:#e2e8f0;vertical-align:top;line-height:1.4;overflow-wrap:anywhere;">${escapeHtml(cue.text)}</td><td style="padding:8px;color:#a5f3fc;text-align:right;vertical-align:top;">${cue.word_count}</td></tr>`).join("")
        : `<tr><td colspan="5" style="padding:18px;color:#fca5a5;text-align:center;">No valid dialogue cues were parsed.</td></tr>`;
      const errorsHtml = errorCount
        ? `<div style="margin-top:12px;border:1px solid #991b1b;border-radius:7px;background:#3f0808;padding:10px;"><div style="font-weight:900;color:#fecaca;">${errorCount} issue${errorCount === 1 ? "" : "s"} found</div>${parsed.errors.map((error) => `<div style="margin-top:6px;color:#fecaca;font-size:12px;line-height:1.4;"><strong>${error.line_number ? `Line ${error.line_number}: ` : ""}</strong>${escapeHtml(error.message)}${error.source ? `<div style="color:#fca5a5;font-family:monospace;overflow-wrap:anywhere;">${escapeHtml(error.source)}</div>` : ""}</div>`).join("")}</div>`
        : `<div style="margin-top:12px;border:1px solid #166534;border-radius:7px;background:#052e16;color:#bbf7d0;padding:9px 10px;font-size:12px;font-weight:900;">All non-empty script lines parsed successfully. Exact dialogue was preserved.</div>`;
      const scenePlanHtml = plannedSceneCount
        ? scenePlan.scenes.map((scene) => {
          const participantHtml = Array.isArray(scene.participants) && scene.participants.length
            ? scene.participants.map((participant) => `<span style="display:inline-flex;border:1px solid #334155;border-radius:999px;background:#0f172a;color:#bae6fd;padding:3px 7px;font-size:10px;font-weight:900;">${escapeHtml(participant.name || participant.alias || "Unmatched speaker")}</span>`).join("")
            : `<span style="color:#fbbf24;font-size:11px;">No matched participants</span>`;
          const dialogueHtml = (Array.isArray(scene.speaker_assignments) ? scene.speaker_assignments : []).map((cue) => `<div style="display:grid;grid-template-columns:82px minmax(105px,.35fr) minmax(0,1fr);gap:8px;border-top:1px solid #1e3a5f;padding:7px 0;align-items:start;"><div style="color:#67e8f9;font:10px monospace;white-space:nowrap;">${Number(cue.planned_start_seconds || 0).toFixed(2)}–${Number(cue.planned_end_seconds || 0).toFixed(2)}s</div><div style="color:#cffafe;font-size:11px;font-weight:900;overflow-wrap:anywhere;">${escapeHtml(cue.speaker_name || cue.speaker_alias)}${Number(cue.part_count || 1) > 1 ? `<div style="color:#fbbf24;font-size:9px;margin-top:2px;">Split ${cue.part_index}/${cue.part_count}</div>` : ""}</div><div style="color:#e2e8f0;font-size:11px;line-height:1.4;overflow-wrap:anywhere;">${escapeHtml(cue.text)}</div></div>`).join("");
          return `<div style="border:1px solid #334155;border-radius:7px;background:#07111f;padding:9px;margin-top:8px;"><div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;"><div><div style="font-weight:900;color:#cffafe;">Segment ${scene.index}${scene.continuation_of_previous ? ` <span style="color:#fbbf24;font-size:10px;">CONTINUATION</span>` : ""}</div><div style="color:#94a3b8;font:10px monospace;margin-top:3px;">Timeline ${Number(scene.timeline_start_seconds || 0).toFixed(1)}–${Number(scene.timeline_end_seconds || 0).toFixed(1)}s</div><div style="display:flex;flex-wrap:wrap;gap:5px;margin-top:5px;">${participantHtml}</div></div><div style="text-align:right;"><div style="font-weight:900;color:#67e8f9;">${Number(scene.duration_seconds || 0).toFixed(1)}s</div><div style="color:#94a3b8;font-size:10px;">max ${Number(scene.maximum_scene_seconds || 0).toFixed(1)}s</div></div></div><div style="margin-top:7px;">${dialogueHtml}</div></div>`;
        }).join("")
        : `<div style="color:#fca5a5;padding:10px;text-align:center;">No scenes could be planned from the parsed dialogue.</div>`;
      const planWarningsHtml = Array.isArray(scenePlan.warnings) && scenePlan.warnings.length
        ? `<div style="margin-top:8px;color:#fde68a;font-size:11px;line-height:1.4;">${scenePlan.warnings.map((warning) => escapeHtml(warning)).join("<br>")}</div>`
        : "";
      previewPanel.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;">
          <div style="border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;"><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;">Format</div><div style="margin-top:3px;color:#cffafe;font-weight:900;">${escapeHtml(String(parsed.format || "text").toUpperCase())}</div></div>
          <div style="border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;"><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;">Speakers</div><div style="margin-top:3px;color:#cffafe;font-weight:900;">${speakerCount}</div></div>
          <div style="border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;"><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;">Dialogue cues</div><div style="margin-top:3px;color:#cffafe;font-weight:900;">${cueCount}</div></div>
          <div style="border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;"><div style="font-size:10px;color:#94a3b8;font-weight:900;text-transform:uppercase;">Words / raw speech</div><div style="margin-top:3px;color:#cffafe;font-weight:900;">${wordCount} / ${speechSeconds.toFixed(1)}s</div></div>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:11px;">${speakerHtml}</div>
        <div style="margin-top:12px;border:1px solid ${matchedCount === speakerCount && speakerCount ? "#166534" : "#92400e"};border-radius:7px;background:${matchedCount === speakerCount && speakerCount ? "#052e16" : "#291804"};padding:10px;">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:9px;"><div><div style="font-weight:900;color:${matchedCount === speakerCount && speakerCount ? "#bbf7d0" : "#fde68a"};">Speaker matching — ${matchedCount}/${speakerCount} matched</div><div style="font-size:11px;color:#cbd5e1;line-height:1.4;margin-top:3px;">Match each exact script name to the character that should speak it. Clear automatic matches can be changed manually.</div></div><div style="color:#94a3b8;font-size:11px;text-align:right;">${referenceCharacters.length} Reference Builder character${referenceCharacters.length === 1 ? "" : "s"}</div></div>
          ${referenceCharacters.length ? speakerMappingHtml : `<div style="border:1px solid #991b1b;border-radius:6px;background:#3f0808;color:#fecaca;padding:9px;font-size:12px;">No Reference Builder characters are available. Add or save the film characters in Reference Builder, then reopen Script Mapper.</div>`}
        </div>
        <div style="margin-top:12px;border:1px solid #0e7490;border-radius:7px;background:#06283d;padding:10px;">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;"><div><div style="font-weight:900;color:#cffafe;">Timed MiniMax scene plan</div><div style="font-size:11px;color:#bae6fd;line-height:1.4;margin-top:3px;">${plannedSceneCount} segment${plannedSceneCount === 1 ? "" : "s"}; ${Number(scenePlan.estimated_total_seconds || 0).toFixed(1)} estimated total seconds. Each clip stays at or below ${Number(scenePlan.maximum_scene_seconds || currentMaximumSceneSeconds()).toFixed(1)} seconds.</div></div><div style="color:#67e8f9;font-weight:900;white-space:nowrap;">${Number(scenePlan.split_cue_count || 0)} long cue${Number(scenePlan.split_cue_count || 0) === 1 ? "" : "s"} split</div></div>
          ${planWarningsHtml}
          <div style="margin-top:8px;max-height:620px;overflow:auto;padding-right:3px;">${scenePlanHtml}</div>
        </div>
        <div style="margin-top:12px;border:1px solid #334155;border-radius:7px;overflow:auto;max-height:520px;">
          <table style="width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px;">
            <thead><tr style="background:#0f172a;color:#bae6fd;text-align:left;"><th style="width:42px;padding:8px;">#</th><th style="width:58px;padding:8px;">Line</th><th style="width:150px;padding:8px;">Speaker</th><th style="padding:8px;">Exact dialogue</th><th style="width:54px;padding:8px;text-align:right;">Words</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        ${errorsHtml}`;
      previewPanel.querySelectorAll("select[data-script-speaker-index]").forEach((select) => {
        select.onchange = () => {
          const speaker = parsed.speakers[Number(select.dataset.scriptSpeakerIndex || 0)];
          if (!speaker) return;
          const aliasKey = storyboardScriptSpeakerMatchKey(speaker.name);
          speakerMatches.set(aliasKey, {
            subject_id: String(select.value || ""),
            method: select.value ? "manual" : "unmatched",
          });
          renderParsedPreview(parsed);
        };
      });
      sourceStatus.textContent = cueCount
        ? `Parsed ${cueCount} exact cue${cueCount === 1 ? "" : "s"} from ${speakerCount} speaker${speakerCount === 1 ? "" : "s"}; planned ${plannedSceneCount} MiniMax segment${plannedSceneCount === 1 ? "" : "s"} at a ${currentMaximumSceneSeconds()}s maximum. ${matchedCount}/${speakerCount} matched to Reference Builder.${errorCount ? ` Review ${errorCount} issue${errorCount === 1 ? "" : "s"}.` : ""}`
        : "No valid dialogue cues were found.";
      sourceStatus.style.color = cueCount && !errorCount && matchedCount === speakerCount ? "#67e8f9" : "#fbbf24";
    };
    renderEmptyPreview();
    mapperBody.append(sourcePanel, previewPanel);

    const mapperFooter = document.createElement("div");
    mapperFooter.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;padding:12px 18px;background:#0f172a;border-top:1px solid #334155;";
    const footerNote = document.createElement("div");
    footerNote.style.cssText = "color:#94a3b8;font-size:12px;line-height:1.4;";
    footerNote.textContent = "Use This Script makes the exact dialogue authoritative for Guided Film Automation. It still does not change the Video Builder timeline.";
    const footerActions = document.createElement("div");
    footerActions.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;";
    const mapperDone = makeButton("Done", "primary");
    footerActions.append(removeGuidedScriptButton, copyParsedButton, useGuidedScriptButton, mapperDone);
    mapperFooter.append(footerNote, footerActions);
    mapperShell.append(mapperHeader, mapperBody, mapperFooter);
    mapperBackdrop.append(mapperShell);
    document.body.append(mapperBackdrop);

    const closeMapper = () => {
      document.removeEventListener("keydown", onMapperKeyDown, true);
      mapperBackdrop.remove();
    };
    const onMapperKeyDown = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      closeMapper();
    };
    mapperCloseTop.onclick = closeMapper;
    mapperDone.onclick = closeMapper;
    mapperBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === mapperBackdrop) closeMapper();
    });
    document.addEventListener("keydown", onMapperKeyDown, true);
    loadScriptButton.onclick = () => scriptFileInput.click();
    maxSceneLengthSelect.onchange = () => {
      customSceneLengthInput.style.display = maxSceneLengthSelect.value === "custom" ? "" : "none";
      if (lastParsed?.cues?.length) renderParsedPreview(lastParsed);
    };
    customSceneLengthInput.onchange = () => {
      customSceneLengthInput.value = String(currentMaximumSceneSeconds());
      if (lastParsed?.cues?.length) renderParsedPreview(lastParsed);
    };
    scriptFileInput.onchange = async () => {
      const file = scriptFileInput.files?.[0];
      if (!file) return;
      try {
        scriptInput.value = await file.text();
        sourceStatus.textContent = `Loaded ${file.name}. Parsing preview...`;
        renderParsedPreview(parseStoryboardScriptImport(scriptInput.value));
      } catch (error) {
        sourceStatus.textContent = `Could not read ${file.name}: ${String(error?.message || error)}`;
        sourceStatus.style.color = "#fca5a5";
      } finally {
        scriptFileInput.value = "";
      }
    };
    parseScriptButton.onclick = () => renderParsedPreview(parseStoryboardScriptImport(scriptInput.value));
    clearScriptButton.onclick = () => {
      scriptInput.value = "";
      lastParsed = null;
      speakerMatches.clear();
      copyParsedButton.disabled = true;
      sourceStatus.textContent = "";
      renderEmptyPreview();
      scriptInput.focus();
    };
    copyParsedButton.onclick = async () => {
      if (!lastParsed?.cues?.length) return;
      await copyTextToClipboard(JSON.stringify(lastParsed, null, 2));
      createToast("Parsed script JSON copied. No timeline changes were made.");
    };
    useGuidedScriptButton.onclick = async () => {
      if (!lastParsed?.cues?.length || useGuidedScriptButton.disabled) return;
      useGuidedScriptButton.disabled = true;
      try {
        state.scriptImport = normalizeStoryboardScriptImportState({
          enabled: true,
          authoritative: true,
          format: lastParsed.format,
          raw_text: scriptInput.value,
          imported_at: new Date().toISOString(),
          maximum_scene_seconds: currentMaximumSceneSeconds(),
          cues: lastParsed.cues,
        });
        refreshSetupPanelSummaries();
        notifyStoryboardDefaultsChanged();
        if (state.projectFolder) {
          await saveStoryboardFile(state, slimStoryboardForRequest(state));
        }
        closeMapper();
        createToast(`Authoritative script activated: ${state.scriptImport.cues.length} exact cue${state.scriptImport.cues.length === 1 ? "" : "s"} across ${state.scriptImport.scene_plan.scene_count} planned MiniMax segment${state.scriptImport.scene_plan.scene_count === 1 ? "" : "s"}.`);
      } catch (error) {
        sourceStatus.textContent = `Could not activate the script: ${String(error?.message || error)}`;
        sourceStatus.style.color = "#fca5a5";
        useGuidedScriptButton.disabled = false;
      }
    };
    removeGuidedScriptButton.onclick = async () => {
      if (!window.confirm("Remove the authoritative imported script from Guided Film Automation?\n\nThis does not delete existing Storyboard or Video Builder scenes.")) return;
      removeGuidedScriptButton.disabled = true;
      try {
        state.scriptImport = normalizeStoryboardScriptImportState({});
        refreshSetupPanelSummaries();
        notifyStoryboardDefaultsChanged();
        if (state.projectFolder) {
          await saveStoryboardFile(state, slimStoryboardForRequest(state));
        }
        closeMapper();
        createToast("Authoritative script removed. Existing scenes were not changed.");
      } catch (error) {
        sourceStatus.textContent = `Could not remove the active script: ${String(error?.message || error)}`;
        sourceStatus.style.color = "#fca5a5";
        removeGuidedScriptButton.disabled = false;
      }
    };
    if (existingScriptImport.enabled && scriptInput.value.trim()) {
      renderParsedPreview(parseStoryboardScriptImport(scriptInput.value));
    } else {
      scriptInput.focus();
    }
  }

  async function planFilmDialogueScenesWithLlm() {
    if (!isIdLoraMode && !isMiniMaxShortFilmMode) return;
    if (isFullyCustomShortFilm()) {
      createToast("Fully Custom uses your manual scene cards. Switch to Guided Film Automation to ask the LLM to plan dialogue scenes.");
      return;
    }
    syncStoryLayerFromInputs();
    const authoritativeScript = normalizeStoryboardScriptImportState(state.scriptImport);
    const sceneCount = authoritativeScript.enabled
      ? Math.max(1, Math.min(80, Number(authoritativeScript.scene_plan.scene_count || 1)))
      : Math.max(1, Math.min(24, Number(idLoraDialogueSceneCount.value || 6)));
    idLoraDialogueSceneCount.value = String(sceneCount);
    const plannerLabel = isIdLoraMode ? "ID-LoRA Dialogue Scenes" : "MiniMax Short Film Scenes";
    const progress = createStoryboardProgressWindow(plannerLabel);
    try {
      progress.set(`Planning ${sceneCount} ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} dialogue scene${sceneCount === 1 ? "" : "s"} with ${promptRunnerName()}...`, 8);
      const data = await postJson(isIdLoraMode ? "/vrgdg/storyboard/id_lora_dialogue_scenes" : "/vrgdg/storyboard/minimax_dialogue_scenes", {
        ...(state.gemmaSettings || {}),
        story_source: authoritativeScript.enabled
          ? authoritativeScript.raw_text
          : [userStoryArcInput.value, songStoryBriefInput.value].map((item) => String(item || "").trim()).filter(Boolean).join("\n\n"),
        script_import: authoritativeScript,
        story_layer: normalizeStoryLayer(state.storyLayer),
        reference_builder: state.referenceBuilder || {},
        scenes: state.scenes.map((scene, index) => slimSceneForRequest(scene, index)),
        storyboard: slimStoryboardForRequest(state),
        scene_count: sceneCount,
        project_video_engine: state.projectVideoEngine,
        minimax_h3_mode: state.miniMaxH3Mode,
        video_prompt_type: isIdLoraMode ? "id_lora" : state.videoPromptType,
        short_film_planning_mode: "guided_film",
        performance_mode: "speaking",
        unload_after: true,
        max_new_tokens: Math.max(2200, sceneCount * 520),
        temperature: 0.55,
        top_p: 0.92,
      }, 240000);
      const generated = Array.isArray(data.scenes) ? data.scenes : [];
      if (!generated.length) throw new Error(`${promptRunnerName()} returned no dialogue scenes.`);
      state.scenes = generated.map((scene, index) => {
        const normalized = normalizeScene({
          ...scene,
          video_prompt_type: isIdLoraMode ? "id_lora" : state.videoPromptType,
          project_video_engine: isIdLoraMode ? "ltx" : "minimax_h3",
          minimax_h3_mode: isIdLoraMode ? "" : state.miniMaxH3Mode,
          performance_mode: "speaking",
        }, index);
        normalized.id_lora_character_id = scene.id_lora_character_id || scene.character_id || scene.subject_id || "";
        normalized.id_lora_location_id = scene.id_lora_location_id || scene.location_id || "";
        return normalized;
      });
      state.selected.clear();
      if (String(data.premise || "").trim() && !String(songStoryBriefInput.value || "").trim()) {
        state.storyLayer.song_story_brief = String(data.premise || "").trim();
        songStoryBriefInput.value = state.storyLayer.song_story_brief;
      }
      setMode("storyboard_prompts");
      renderTable();
      refreshSetupPanelSummaries();
      progress.set(`Storyboard scenes ready.\nCreated ${state.scenes.length} editable storyboard scene${state.scenes.length === 1 ? "" : "s"}. The Video Builder timeline has not been changed.`, 96);
      await saveStoryboard();
      progress.set(`Storyboard saved for review.\nNext: click Create ${state.scenes.length} Timeline Segment${state.scenes.length === 1 ? "" : "s"} to build the Video Builder timeline.`, 100);
      progress.close(1800);
      createToast(`Created ${state.scenes.length} ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} storyboard scene${state.scenes.length === 1 ? "" : "s"}. The timeline is unchanged until you click Create ${state.scenes.length} Timeline Segment${state.scenes.length === 1 ? "" : "s"}.`);
    } catch (error) {
      progress.set(`${isIdLoraMode ? "ID-LoRA" : "MiniMax"} dialogue planning failed:\n${String(error?.message || error)}`, 100);
      createToast(`${isIdLoraMode ? "ID-LoRA" : "MiniMax"} dialogue planning failed:\n${String(error?.message || error)}`, true);
    }
  }

  async function applyFilmDialoguePlanToVideoBuilder() {
    const applyCallback = isIdLoraMode ? state.onApplyIdLoraDialoguePlan : state.onApplyMiniMaxDialoguePlan;
    if (!applyCallback) return;
    const scenes = state.scenes
      .map((scene, index) => slimSceneForRequest(scene, index))
      .filter((scene) => String(scene.lyrics || scene.story_beat || scene.image_prompt || "").trim());
    if (!scenes.length) {
      createToast(`No reviewed ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} storyboard scenes were found to create timeline segments.`, true);
      return;
    }
    const confirmed = window.confirm(`Create ${scenes.length} Video Builder timeline segment${scenes.length === 1 ? "" : "s"} from the reviewed ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} storyboard scenes?\n\nThis is the step that creates the timeline. Export Prompt Files Only does not create timeline segments.\n\nThe blank starter scene will be replaced. If real scenes already exist, Video Builder will ask before replacing them.`);
    if (!confirmed) return;
    try {
      applyDialoguePlanButton.disabled = true;
      const result = await applyCallback({
        story_layer: normalizeStoryLayer(state.storyLayer),
        short_film_planning_mode: normalizeStoryboardShortFilmPlanningMode(state.shortFilmPlanningMode),
        scenes,
      });
      createToast(result?.message || `Created ${scenes.length} Video Builder timeline segment${scenes.length === 1 ? "" : "s"} from the reviewed ${isIdLoraMode ? "ID-LoRA" : "MiniMax"} storyboard.`);
    } catch (error) {
      createToast(`Create Timeline Segments failed:\n${String(error?.message || error)}`, true);
    } finally {
      applyDialoguePlanButton.disabled = false;
    }
  }

  return { applyFilmDialoguePlanToVideoBuilder, openMiniMaxScriptMapper, planFilmDialogueScenesWithLlm };
}
