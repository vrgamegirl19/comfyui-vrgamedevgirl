import { audioUrl } from "./comfy_api.mjs";
import { escapeHtml, makeButton, makeCheckbox, makeField, makeInput, toast } from "./controls.mjs";
import { formatDurationSeconds, formatTime } from "./format.mjs";
import { applyLyricSectionsFromReferenceText, showTimestampedLyricsHintModal } from "./lyric_transcription.mjs";
import { isNoLipSyncSingerChoice } from "./prompt_text.mjs";
import { normalizeFluxReferenceBuilder, normalizeLyricMapper, splitLyricsToMapperLines } from "./reference_data.mjs";
import { newSegment } from "./segments.mjs";

export function createLyricMapping({
  activeSegment, allEditableSegments, applyIngredientsReferenceMappings, applyLyricMapperToSegments,
  audioInput, autoSaveSessionQuiet, createScenesFromTimestampedLyrics, currentVideoMode, openLyricReviewModal,
  projectSrtFileInput, pushHistory, referenceBuilderSubjectChoices, render, saveSession, sceneDisplayName,
  state, syncIngredientsSceneMapFromSubjectMappings, syncInspector, syncLyricMapperFromSegments,
  syncLyricNoteControls, timelineDuration, transcribeLyricsForTimeline,
}) {
  function openLyricMapperModal() {
    state.lyricMapper = normalizeLyricMapper(state.lyricMapper);
    const mapper = state.lyricMapper;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(1180px,calc(100vw - 42px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Line Mapper</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Assign Reference Builder subjects to lyric/dialogue lines so Gemma knows who performs each scene.</div>`;
    const close = makeButton("Close");
    header.append(heading, close);

    const note = document.createElement("div");
    note.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:9px;";
    note.textContent = "Paste clean lyrics or dialogue, split them into lines, then choose one or more performers/speakers for each line. Instrumental and B-roll mean no one should lip-sync, but selected characters can still appear in the shot. After timeline transcription, Apply To Timeline matches scene line notes to these mapped lines.";

    const initialChoices = referenceBuilderSubjectChoices();
    const subjectWarning = document.createElement("div");
    subjectWarning.style.cssText = `font-size:12px;line-height:1.45;border:1px solid #92400e;border-radius:7px;background:#451a03;color:#fed7aa;padding:9px;${initialChoices.hasReferenceSubjects ? "display:none;" : ""}`;
    subjectWarning.textContent = "No Reference Builder subjects found. Add your characters in Reference Builder first for accurate performer mapping. Until then, Line Mapper uses fallback choices: Female, Male, Other performer, Group, and B-roll / no lip-sync.";

    const audioPanel = document.createElement("div");
    audioPanel.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:center;border:1px solid #334155;border-radius:7px;background:#0b1220;padding:9px;";
    const mapperAudio = document.createElement("audio");
    mapperAudio.controls = true;
    mapperAudio.preload = "metadata";
    mapperAudio.style.cssText = "width:100%;height:34px;";
    const audioPath = String(audioInput.value || state.audioPath || "").trim();
    if (audioPath) {
      mapperAudio.src = audioUrl(audioPath);
    } else {
      mapperAudio.style.display = "none";
      const noAudio = document.createElement("div");
      noAudio.textContent = "No project audio loaded yet.";
      noAudio.style.cssText = "font-size:12px;color:#94a3b8;";
      audioPanel.append(noAudio);
    }
    const jumpToScene = makeButton("Jump To Selected Scene");
    jumpToScene.style.whiteSpace = "nowrap";
    jumpToScene.onclick = () => {
      const segment = activeSegment();
      if (!segment || !mapperAudio.src) return;
      mapperAudio.currentTime = Math.max(0, Number(segment.start || 0));
      mapperAudio.play().catch(() => {});
    };
    if (audioPath) audioPanel.append(mapperAudio);
    audioPanel.append(jumpToScene);

    const sourceLyrics = document.createElement("textarea");
    sourceLyrics.value = mapper.source_text || "";
    sourceLyrics.placeholder = "Paste full lyrics or dialogue here. You can also prefix lines like [Sarah] line text or [Sarah + Daniel] line text.";
    sourceLyrics.style.cssText = "width:100%;box-sizing:border-box;min-height:170px;resize:vertical;border:1px solid #334155;border-radius:7px;background:#020617;color:#f8fafc;padding:10px;font-size:12px;line-height:1.45;font-family:monospace;";
    const sourceActions = document.createElement("div");
    sourceActions.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;";
    const splitButton = makeButton("Split Into Lines", "primary");
    const addInstrumental = makeButton("Add Instrumental Line");
    const addLine = makeButton("Add Line");
    sourceActions.append(splitButton, addInstrumental, addLine);

    const lineList = document.createElement("div");
    lineList.style.cssText = "display:flex;flex-direction:column;gap:8px;max-height:48vh;overflow:auto;padding-right:4px;";
    let lyricIngredientsRefs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const ingredientsMappingPanel = document.createElement("div");
    ingredientsMappingPanel.style.cssText = "display:none;border:1px solid #334155;border-radius:7px;background:#0b1220;padding:10px;gap:8px;flex-direction:column;";
    const ingredientsMappingTitle = document.createElement("div");
    ingredientsMappingTitle.innerHTML = `<div style="font-weight:900;color:#e0f2fe;">Ingredients Sheet Mapping</div><div style="font-size:12px;color:#94a3b8;margin-top:2px;">Optional scene image assignment for Ingredients to Video.</div>`;
    const ingredientsMappingList = document.createElement("div");
    ingredientsMappingList.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:7px;max-height:190px;overflow:auto;";
    ingredientsMappingPanel.append(ingredientsMappingTitle, ingredientsMappingList);

    const collectLyricIngredientsMappings = () => {
      lyricIngredientsRefs = normalizeFluxReferenceBuilder(lyricIngredientsRefs);
      lyricIngredientsRefs.ingredients_scene_map = {};
      for (const select of ingredientsMappingList.querySelectorAll("[data-lyric-ingredients-map='1']")) {
        const sceneId = select.dataset.sceneId || "";
        const sheetId = String(select.value || "").trim();
        if (sceneId && sheetId) lyricIngredientsRefs.ingredients_scene_map[sceneId] = sheetId;
      }
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder({
        ...state.fluxReferenceBuilder,
        ingredients_sheets: lyricIngredientsRefs.ingredients_sheets,
        ingredients_scene_map: lyricIngredientsRefs.ingredients_scene_map,
        ingredients_auto_map_sources: lyricIngredientsRefs.ingredients_auto_map_sources,
      });
    };

    const renderLyricIngredientsMappings = () => {
      lyricIngredientsRefs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const sheets = lyricIngredientsRefs.ingredients_sheets || [];
      const shouldShow = currentVideoMode() === "ingredients" || sheets.length > 0;
      ingredientsMappingPanel.style.display = shouldShow ? "flex" : "none";
      ingredientsMappingList.innerHTML = "";
      if (!shouldShow) return;
      if (!sheets.length) {
        const empty = document.createElement("div");
        empty.style.cssText = "grid-column:1/-1;border:1px dashed #334155;border-radius:7px;padding:12px;text-align:center;color:#94a3b8;font-size:12px;";
        empty.textContent = "Add Ingredients sheets in Reference Builder first.";
        ingredientsMappingList.append(empty);
        return;
      }
      allEditableSegments().forEach((segment, index) => {
        const row = document.createElement("div");
        row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr);gap:5px;border:1px solid #1f2937;border-radius:7px;background:#020617;padding:7px;";
        const label = document.createElement("div");
        label.style.cssText = "font-size:11px;font-weight:800;color:#cbd5e1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        label.textContent = sceneDisplayName(segment, index);
        const select = document.createElement("select");
        select.dataset.lyricIngredientsMap = "1";
        select.dataset.sceneId = segment.id || "";
        select.style.cssText = "width:100%;border:1px solid #334155;border-radius:6px;background:#111827;color:#f8fafc;padding:7px 8px;font-size:12px;";
        select.innerHTML = `<option value="">No Ingredients sheet</option>${sheets.map((sheet) => `<option value="${escapeHtml(sheet.id)}">${escapeHtml(sheet.name || "Ingredients Sheet")}</option>`).join("")}`;
        select.value = lyricIngredientsRefs.ingredients_scene_map?.[segment.id] || "";
        select.onchange = () => collectLyricIngredientsMappings();
        row.append(label, select);
        ingredientsMappingList.append(row);
      });
    };

    const subjectChoices = () => referenceBuilderSubjectChoices();
    const collectLines = () => {
      const rows = [...lineList.querySelectorAll("[data-lyric-map-row='1']")];
      return rows.map((row, index) => {
        const text = row.querySelector("[data-lyric-text]")?.value || "";
        const instrumental = Boolean(row.querySelector("[data-lyric-instrumental]")?.checked);
        const singerChecks = [...row.querySelectorAll("[data-lyric-singer-choice='1']")];
        const select = row.querySelector("[data-lyric-singers]");
        const pickedSingers = singerChecks.length
          ? singerChecks.filter((input) => input.checked).map((input) => input.value).filter(Boolean)
          : (select ? [...select.selectedOptions].map((option) => option.value).filter(Boolean) : []);
        const noLipSync = pickedSingers.some(isNoLipSyncSingerChoice);
        const singers = noLipSync ? [] : pickedSingers;
        return {
          id: row.dataset.lineId || `lyric_line_${Date.now()}_${index}`,
          text: instrumental ? "" : text.trim(),
          singers: instrumental ? [] : singers,
          instrumental,
          no_lip_sync: noLipSync,
        };
      });
    };

    const renderLines = (lines = mapper.lines) => {
      lineList.innerHTML = "";
      const choices = subjectChoices();
      const cleanLines = Array.isArray(lines) ? lines : [];
      if (!cleanLines.length) {
        const empty = document.createElement("div");
        empty.textContent = "No lines yet. Paste lyrics or dialogue above, then click Split Into Lines.";
        empty.style.cssText = "border:1px dashed #334155;border-radius:7px;padding:14px;text-align:center;color:#94a3b8;font-size:12px;";
        lineList.append(empty);
        return;
      }
      cleanLines.forEach((line, index) => {
        const row = document.createElement("div");
        row.dataset.lyricMapRow = "1";
        row.dataset.lineId = line.id || `lyric_line_${index + 1}`;
        row.style.cssText = "display:grid;grid-template-columns:44px minmax(220px,1fr) minmax(230px,300px) 120px 76px;gap:8px;align-items:start;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:8px;";
        const number = document.createElement("div");
        number.textContent = String(index + 1);
        number.style.cssText = "font-size:12px;font-weight:900;color:#67e8f9;padding-top:8px;text-align:center;";
        const text = document.createElement("textarea");
        text.dataset.lyricText = "1";
        text.value = line.instrumental ? "" : (line.text || "");
        text.placeholder = "Line text...";
        text.disabled = Boolean(line.instrumental);
        text.style.cssText = "width:100%;box-sizing:border-box;min-height:54px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.35;";
        const singers = document.createElement("div");
        singers.dataset.lyricSingers = "1";
        singers.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:7px;min-height:54px;box-sizing:border-box;";
        const selected = new Set(Array.isArray(line.singers) ? line.singers : []);
        const allChoices = [...choices];
        for (const singer of selected) {
          if (!allChoices.some((choice) => choice.label === singer || choice.id === singer)) allChoices.push({ id: singer, label: singer });
        }
        if (!allChoices.length) {
          const emptySinger = document.createElement("div");
          emptySinger.textContent = "Add subjects in Reference Builder first.";
          emptySinger.style.cssText = "font-size:11px;color:#94a3b8;";
          singers.append(emptySinger);
        }
        for (const choice of allChoices) {
          const label = document.createElement("label");
          label.style.cssText = "display:inline-flex;align-items:center;gap:5px;border:1px solid #334155;border-radius:999px;background:#0f172a;color:#e2e8f0;padding:5px 8px;font-size:11px;line-height:1;cursor:pointer;user-select:none;";
          const input = document.createElement("input");
          input.type = "checkbox";
          input.dataset.lyricSingerChoice = "1";
          input.value = choice.label;
          input.checked = selected.has(choice.label) || selected.has(choice.id);
          input.disabled = Boolean(line.instrumental);
          input.style.cssText = "margin:0;";
          const name = document.createElement("span");
          name.textContent = choice.label;
          label.append(input, name);
          singers.append(label);
        }
        const instrumental = makeCheckbox("Instrumental", Boolean(line.instrumental));
        instrumental.input.dataset.lyricInstrumental = "1";
        instrumental.wrapper.style.cssText += "padding-top:8px;";
        instrumental.input.onchange = () => {
          text.disabled = instrumental.input.checked;
          singers.disabled = instrumental.input.checked;
          if (instrumental.input.checked) {
            if (text.value.trim()) text.dataset.preInstrumentalText = text.value;
            text.value = "";
            for (const input of singers.querySelectorAll("[data-lyric-singer-choice='1']")) {
              input.checked = false;
              input.disabled = true;
            }
          } else {
            if (!text.value.trim() && text.dataset.preInstrumentalText) text.value = text.dataset.preInstrumentalText;
            for (const input of singers.querySelectorAll("[data-lyric-singer-choice='1']")) {
              input.disabled = false;
            }
          }
        };
        const remove = makeButton("Remove");
        remove.style.minWidth = "0";
        remove.onclick = () => {
          row.remove();
          [...lineList.querySelectorAll("[data-lyric-map-row='1']")].forEach((item, itemIndex) => {
            const label = item.firstChild;
            if (label) label.textContent = String(itemIndex + 1);
          });
        };
        row.append(number, text, singers, instrumental.wrapper, remove);
        lineList.append(row);
      });
    };

    splitButton.onclick = () => {
      const parsed = splitLyricsToMapperLines(sourceLyrics.value);
      mapper.source_text = sourceLyrics.value || "";
      mapper.lines = parsed;
      renderLines(parsed);
    };
    addInstrumental.onclick = () => {
      const lines = collectLines();
      lines.push({ id: `lyric_line_${Date.now()}_${Math.floor(Math.random() * 10000)}`, text: "", singers: [], instrumental: true });
      renderLines(lines);
    };
    addLine.onclick = () => {
      const lines = collectLines();
      lines.push({ id: `lyric_line_${Date.now()}_${Math.floor(Math.random() * 10000)}`, text: "", singers: [], instrumental: false });
      renderLines(lines);
    };

    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
    const save = makeButton("Save Mapper", "primary");
    const apply = makeButton("Apply To Timeline", "primary");
    const cancel = makeButton("Close");
    actions.append(cancel, save, apply);
    box.append(header, note, subjectWarning, audioPanel, makeField("Source text", sourceLyrics), sourceActions, lineList, ingredientsMappingPanel, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    renderLines(mapper.lines);
    renderLyricIngredientsMappings();

    const saveMapper = async () => {
      pushHistory();
      state.lyricMapper = normalizeLyricMapper({
        source_text: sourceLyrics.value || "",
        lines: collectLines(),
      });
      applyLyricMapperToSegments({ overwriteSingers: true });
      applyLyricSectionsFromReferenceText(state.segments, state.lyricMapper.source_text);
      collectLyricIngredientsMappings();
      if (currentVideoMode() === "ingredients") applyIngredientsReferenceMappings(state.fluxReferenceBuilder);
      await saveSession({ quiet: true, throwOnError: true });
      toast(`Saved ${state.lyricMapper.lines.length} mapped line${state.lyricMapper.lines.length === 1 ? "" : "s"}.`);
    };
    close.onclick = () => backdrop.remove();
    cancel.onclick = () => backdrop.remove();
    save.onclick = async () => {
      try {
        save.disabled = true;
        await saveMapper();
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        save.disabled = false;
      }
    };
    apply.onclick = async () => {
      try {
        apply.disabled = true;
        await saveMapper();
        pushHistory();
        const count = applyLyricMapperToSegments({ overwriteSingers: true });
        applyLyricSectionsFromReferenceText(state.segments, state.lyricMapper.source_text);
        syncLyricMapperFromSegments();
        const syncedIngredients = syncIngredientsSceneMapFromSubjectMappings(state.fluxReferenceBuilder);
        state.fluxReferenceBuilder = syncedIngredients.refs;
        if (currentVideoMode() === "ingredients") applyIngredientsReferenceMappings(state.fluxReferenceBuilder);
        syncInspector();
        render();
        await saveSession({ quiet: true, throwOnError: true });
        toast(`Applied Line Mapper to ${count} scene${count === 1 ? "" : "s"}.`);
      } catch (error) {
        toast(String(error?.message || error), true);
      } finally {
        apply.disabled = false;
      }
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) backdrop.remove();
    });
  }

  function openLyricMappingWorkflowModal(options = {}) {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100006;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(920px,calc(100vw - 42px));max-height:calc(100vh - 44px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;";
    const heading = document.createElement("div");
    heading.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">Line Mapping</div><div style="font-size:12px;color:#94a3b8;margin-top:3px;">Transcribe lyrics/dialogue, review scene timing, then assign performers for Gemma prompts.</div>`;
    const close = makeButton("Close");
    header.append(heading, close);

    const subjectChoices = referenceBuilderSubjectChoices();
    const warning = document.createElement("div");
    warning.style.cssText = `font-size:12px;line-height:1.45;border:1px solid #92400e;border-radius:7px;background:#451a03;color:#fed7aa;padding:9px;${subjectChoices.hasReferenceSubjects ? "display:none;" : ""}`;
    warning.textContent = "No Reference Builder subjects found. Set up Reference Builder first for best performer mapping. If you skip that, Review + Map Performers uses fallback performer choices.";

    const tabBar = document.createElement("div");
    tabBar.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;";
    const pane = document.createElement("div");
    pane.style.cssText = "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:14px;min-height:260px;";
    const tabs = [
      { id: "transcribe", label: "Step 1: Transcribe" },
      { id: "review", label: "Step 2: Review + Map Performers" },
      { id: "manual_timing", label: "Manual Timing" },
    ];
    const tabButtons = new Map();
    const makeStepButton = (label, kind = "") => {
      const button = makeButton(label, kind);
      button.style.width = "100%";
      button.style.justifyContent = "center";
      return button;
    };
    const statusLine = document.createElement("div");
    statusLine.style.cssText = "font-size:12px;color:#a5f3fc;line-height:1.4;";
    const setStatusText = () => {
      const scenes = allEditableSegments();
      const lyricCount = scenes.filter((segment) => String(segment.lyric_text || "").trim()).length;
      const singerCount = scenes.filter((segment) => (Array.isArray(segment.lyric_singers) && segment.lyric_singers.length) || segment.lyric_no_lip_sync || segment.no_character_present).length;
      statusLine.textContent = `${lyricCount}/${scenes.length} scenes have line notes. ${singerCount}/${scenes.length} scenes have performer/no-lip-sync/no-character mapping.`;
    };
    let activeLyricMappingTab = "transcribe";
    let manualTimingAudio = null;
    let manualTimingSplits = [];
    const manualTimingDuration = () => {
      const audioDuration = Number(manualTimingAudio?.duration || 0);
      if (Number.isFinite(audioDuration) && audioDuration > 0) return audioDuration;
      const timeline = timelineDuration();
      return timeline > 0 ? timeline : Number(state.duration || 0);
    };
    const sortedManualTimingSplits = () => Array.from(new Set(manualTimingSplits
      .map((value) => Number(value || 0))
      .filter((value) => Number.isFinite(value) && value > 0.01)))
      .sort((a, b) => a - b);
    const addManualTimingSplit = (time, minSeconds = 0.25) => {
      const duration = manualTimingDuration();
      const value = Math.max(0, Math.min(duration > 0 ? duration : Number.MAX_SAFE_INTEGER, Number(time || 0)));
      if (!Number.isFinite(value) || value <= 0.01) {
        toast("Play the audio and add a split after the start.", true);
        return false;
      }
      if (duration > 0 && value >= duration - 0.05) {
        toast("That split is too close to the end of the song.", true);
        return false;
      }
      const existing = sortedManualTimingSplits();
      if (existing.some((split) => Math.abs(split - value) < minSeconds)) {
        toast("That split is too close to an existing split.", true);
        return false;
      }
      manualTimingSplits = [...existing, value].sort((a, b) => a - b);
      return true;
    };
    const createManualTimingSegments = (minSceneSeconds = 0.5) => {
      const duration = manualTimingDuration();
      if (!Number.isFinite(duration) || duration <= 0) throw new Error("Could not read the audio duration yet. Play or load the audio first.");
      const points = [0, ...sortedManualTimingSplits(), duration];
      const created = [];
      for (let index = 0; index < points.length - 1; index += 1) {
        const start = points[index];
        const end = points[index + 1];
        if (end - start < minSceneSeconds) continue;
        const segment = newSegment(start, end);
        segment.label = `Scene ${created.length + 1}`;
        segment.source = "manual_timing";
        created.push(segment);
      }
      if (!created.length) throw new Error("No usable scenes were created. Add at least one split or lower the minimum scene length.");
      return created;
    };
    const onLyricMappingKeydown = (event) => {
      if (activeLyricMappingTab !== "manual_timing" || !backdrop.contains(event.target)) return;
      if (event.key !== "ArrowDown" && event.key !== " " && event.code !== "Space") return;
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      const tag = String(event.target?.tagName || "").toLowerCase();
      if (["input", "textarea", "select"].includes(tag) || event.target?.isContentEditable) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      if (event.repeat) return;
      const button = pane.querySelector("[data-manual-timing-add-split]");
      button?.click();
    };
    const setActiveTab = (id) => {
      activeLyricMappingTab = id;
      for (const [tabId, button] of tabButtons.entries()) {
        const active = tabId === id;
        button.style.background = active ? "#0891b2" : "#27272a";
        button.style.borderColor = active ? "#22d3ee" : "#3f3f46";
        button.style.color = active ? "#001018" : "#f4f4f5";
      }
      setStatusText();
      pane.textContent = "";
      if (id === "transcribe") {
        const title = document.createElement("div");
        title.style.cssText = "font-size:15px;font-weight:900;color:#cffafe;margin-bottom:8px;";
        title.textContent = "Step 1: Transcribe lines or create scenes";
        const copy = document.createElement("div");
        copy.style.cssText = "font-size:13px;color:#cbd5e1;line-height:1.5;margin-bottom:12px;";
        copy.textContent = "Choose the workflow that matches where you are. If scenes already exist, transcribe lines into those scene windows. If the project has no scenes yet, create scenes from timestamped lyrics or dialogue first.";
        const actions = document.createElement("div");
        actions.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;";
        const existingCard = document.createElement("div");
        existingCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#111827;padding:12px;display:flex;flex-direction:column;gap:8px;";
        const existingTitle = document.createElement("div");
        existingTitle.style.cssText = "font-weight:900;color:#cffafe;";
        existingTitle.textContent = "Option 1: Existing scenes";
        const existingText = document.createElement("div");
        existingText.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
        existingText.textContent = "Use this when your timeline already has scenes. It keeps current timing and fills the Line / lyric / dialogue field for each scene.";
        const run = makeStepButton("Transcribe Existing Scenes", "primary");
        existingCard.append(existingTitle, existingText, run);
        const timestampCard = document.createElement("div");
        timestampCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#111827;padding:12px;display:flex;flex-direction:column;gap:8px;";
        const timestampTitle = document.createElement("div");
        timestampTitle.style.cssText = "font-weight:900;color:#cffafe;";
        timestampTitle.textContent = "Option 2: No scenes yet";
        const timestampText = document.createElement("div");
        timestampText.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
        timestampText.textContent = "Use this when the project is blank. It uses stable-ts timestamps to create timeline scenes and line notes from the audio.";
        const createScenes = makeStepButton("Create Scenes From Lines", "primary");
        timestampCard.append(timestampTitle, timestampText, createScenes);
        const srtCard = document.createElement("div");
        srtCard.style.cssText = "border:1px solid #334155;border-radius:8px;background:#111827;padding:12px;display:flex;flex-direction:column;gap:8px;";
        const srtTitle = document.createElement("div");
        srtTitle.style.cssText = "font-weight:900;color:#cffafe;";
        srtTitle.textContent = "Option 3: Import SRT file";
        const srtText = document.createElement("div");
        srtText.style.cssText = "font-size:12px;color:#cbd5e1;line-height:1.45;";
        srtText.textContent = "Use this when you already have an SRT. It creates timeline scenes from the SRT timestamps and fills the Line Notes lane with each subtitle line.";
        const importSrt = makeStepButton("Import SRT File", "primary");
        srtCard.append(srtTitle, srtText, importSrt);
        actions.append(existingCard, timestampCard, srtCard);
        const utilityActions = document.createElement("div");
        utilityActions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;";
        const showNotes = makeStepButton(state.showTimelineLyricNotes ? "Hide Timeline Line Notes" : "Show Timeline Line Notes");
        const hint = makeStepButton("Timestamp Settings Hint");
        run.onclick = async () => {
          await transcribeLyricsForTimeline();
          setStatusText();
        };
        createScenes.onclick = async () => {
          await createScenesFromTimestampedLyrics();
          setStatusText();
        };
        importSrt.onclick = () => {
          projectSrtFileInput.click();
        };
        hint.onclick = async () => {
          await showTimestampedLyricsHintModal();
        };
        showNotes.onclick = () => {
          state.showTimelineLyricNotes = !state.showTimelineLyricNotes;
          syncLyricNoteControls();
          render();
          showNotes.textContent = state.showTimelineLyricNotes ? "Hide Timeline Line Notes" : "Show Timeline Line Notes";
          autoSaveSessionQuiet(state.showTimelineLyricNotes ? "timeline line notes shown" : "timeline line notes hidden");
        };
        utilityActions.append(showNotes, hint);
        pane.append(title, copy, actions, utilityActions);
      } else if (id === "review") {
        const title = document.createElement("div");
        title.style.cssText = "font-size:15px;font-weight:900;color:#cffafe;margin-bottom:8px;";
        title.textContent = "Step 2: Review lines and map performers";
        const copy = document.createElement("div");
        copy.style.cssText = "font-size:13px;color:#cbd5e1;line-height:1.5;margin-bottom:12px;";
        copy.textContent = "Open the scene-based editor to listen by scene, correct transcription mistakes, assign performers/speakers, mark instrumental/B-roll sections, and save the exact line notes Gemma will read.";
        const actions = document.createElement("div");
        actions.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;";
        const review = makeStepButton("Open Review + Performer Mapping", "primary");
        const showNotes = makeStepButton(state.showTimelineLyricNotes ? "Hide Timeline Line Notes" : "Show Timeline Line Notes");
        const autoMap = makeStepButton("Optional: Full Text Auto-Mapper");
        review.onclick = () => openLyricReviewModal();
        autoMap.onclick = () => openLyricMapperModal();
        showNotes.onclick = () => {
          state.showTimelineLyricNotes = !state.showTimelineLyricNotes;
          syncLyricNoteControls();
          render();
          showNotes.textContent = state.showTimelineLyricNotes ? "Hide Timeline Line Notes" : "Show Timeline Line Notes";
          autoSaveSessionQuiet(state.showTimelineLyricNotes ? "timeline line notes shown" : "timeline line notes hidden");
        };
        actions.append(review, showNotes, autoMap);
        pane.append(title, copy, actions);
      } else if (id === "manual_timing") {
        const title = document.createElement("div");
        title.style.cssText = "font-size:15px;font-weight:900;color:#cffafe;margin-bottom:8px;";
        title.textContent = "Manual Timing: tap scene splits while listening";
        const copy = document.createElement("div");
        copy.style.cssText = "font-size:13px;color:#cbd5e1;line-height:1.5;margin-bottom:12px;";
        copy.textContent = "Use this when you want to create scene timing by ear. Press Space or Down Arrow, or click Add Split At Playhead while the song plays. Nothing changes on the timeline until you click Create Scenes From Splits.";
        const audioPath = String(audioInput.value || state.audioPath || "").trim();
        const audioPanel = document.createElement("div");
        audioPanel.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:center;border:1px solid #334155;border-radius:7px;background:#0b1220;padding:9px;margin-bottom:10px;";
        manualTimingAudio = document.createElement("audio");
        manualTimingAudio.controls = false;
        manualTimingAudio.preload = "metadata";
        manualTimingAudio.hidden = true;
        if (audioPath) manualTimingAudio.src = audioUrl(audioPath);
        const addSplit = makeStepButton("Add Split At Playhead", "primary");
        addSplit.dataset.manualTimingAddSplit = "1";
        const undoSplit = makeStepButton("Undo Last Split");
        pane.tabIndex = -1;
        const playback = document.createElement("div");
        playback.style.cssText = "display:flex;gap:10px;align-items:center;min-width:0;";
        const playPause = makeButton("Play");
        const seek = document.createElement("input");
        seek.type = "range"; seek.min = "0"; seek.max = "0"; seek.step = "0.01"; seek.value = "0";
        seek.setAttribute("aria-label", "Manual timing audio position");
        seek.style.cssText = "flex:1;min-width:60px;";
        const playbackTime = document.createElement("span");
        playbackTime.style.cssText = "font-size:12px;white-space:nowrap;";
        const updatePlayback = () => {
          const duration = Number(manualTimingAudio.duration);
          const current = Number(manualTimingAudio.currentTime || 0);
          seek.max = String(Number.isFinite(duration) ? duration : 0);
          seek.value = String(current);
          playbackTime.textContent = `${formatTime(current)} / ${formatTime(Number.isFinite(duration) ? duration : 0)}`;
          playPause.textContent = manualTimingAudio.paused ? "Play" : "Pause";
        };
        playPause.onclick = async () => {
          pane.focus();
          try {
            if (manualTimingAudio.paused) await manualTimingAudio.play();
            else manualTimingAudio.pause();
          } catch (error) { toast(`Could not play audio: ${error.message || error}`, true); }
          updatePlayback();
        };
        seek.oninput = () => { manualTimingAudio.currentTime = Number(seek.value); updatePlayback(); };
        seek.onchange = () => pane.focus();
        for (const event of ["timeupdate", "loadedmetadata", "play", "pause", "ended"]) manualTimingAudio.addEventListener(event, updatePlayback);
        updatePlayback();
        playback.append(playPause, seek, playbackTime, manualTimingAudio);
        audioPanel.append(playback, addSplit, undoSplit);
        const controls = document.createElement("div");
        controls.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:10px;";
        const minScene = makeInput("0.5");
        minScene.type = "number";
        minScene.step = "0.1";
        minScene.min = "0.05";
        const clearSplits = makeStepButton("Clear Splits");
        const createScenes = makeStepButton("Create Scenes From Splits", "primary");
        controls.append(makeField("Minimum scene length", minScene), clearSplits, createScenes);
        const splitList = document.createElement("div");
        splitList.style.cssText = "border:1px solid #334155;border-radius:7px;background:#111827;padding:10px;display:flex;flex-direction:column;gap:6px;max-height:260px;overflow:auto;";
        const renderSplitList = () => {
          splitList.textContent = "";
          const duration = manualTimingDuration();
          const splits = sortedManualTimingSplits();
          const points = [0, ...splits, duration].filter((value, index, arr) => index === 0 || value > arr[index - 1]);
          const header = document.createElement("div");
          header.style.cssText = "font-size:12px;color:#a5f3fc;font-weight:900;";
          header.textContent = splits.length
            ? `${splits.length} split marker${splits.length === 1 ? "" : "s"} | ${Math.max(1, points.length - 1)} scene${points.length - 1 === 1 ? "" : "s"} preview`
            : "No split markers yet. Press Space, Down Arrow, or Add Split At Playhead while the audio plays.";
          splitList.append(header);
          if (!duration) {
            const wait = document.createElement("div");
            wait.style.cssText = "font-size:12px;color:#fbbf24;line-height:1.4;";
            wait.textContent = "Audio duration is not available yet. Load/play the audio first.";
            splitList.append(wait);
            return;
          }
          for (let index = 0; index < points.length - 1; index += 1) {
            const row = document.createElement("div");
            row.style.cssText = "display:grid;grid-template-columns:72px 1fr;gap:8px;border:1px solid #1f2937;border-radius:6px;background:#0f172a;padding:7px;font-size:12px;color:#e5e7eb;";
            const label = document.createElement("div");
            label.style.cssText = "font-weight:900;color:#cffafe;";
            label.textContent = `Scene ${index + 1}`;
            const time = document.createElement("div");
            time.textContent = `${formatTime(points[index])} - ${formatTime(points[index + 1])} | ${formatDurationSeconds(points[index], points[index + 1])}s`;
            row.append(label, time);
            splitList.append(row);
          }
        };
        addSplit.onclick = () => {
          pane.focus();
          if (!audioPath) {
            toast("Load an audio file first.", true);
            return;
          }
          const minSeconds = Math.max(0.05, Number(minScene.value || 0.5));
          if (addManualTimingSplit(Number(manualTimingAudio.currentTime || 0), minSeconds)) renderSplitList();
        };
        undoSplit.onclick = () => {
          const splits = sortedManualTimingSplits();
          splits.pop();
          manualTimingSplits = splits;
          renderSplitList();
        };
        clearSplits.onclick = () => {
          manualTimingSplits = [];
          renderSplitList();
        };
        manualTimingAudio.onloadedmetadata = renderSplitList;
        createScenes.onclick = async () => {
          try {
            if (!audioPath) throw new Error("Load an audio file first.");
            const minSeconds = Math.max(0.05, Number(minScene.value || 0.5));
            const created = createManualTimingSegments(minSeconds);
            pushHistory();
            state.segments = created;
            state.overlaySegments = [];
            state.activeTrack = "base";
            state.activeId = created[0]?.id || "";
            state.duration = Math.max(manualTimingDuration(), ...created.map((segment) => Number(segment.end || 0)));
            syncInspector();
            render();
            await saveSession({ quiet: true, throwOnError: true });
            setStatusText();
            toast(`Created ${created.length} manual timing scene${created.length === 1 ? "" : "s"}.`);
          } catch (error) {
            toast(String(error?.message || error), true);
          }
        };
        const hint = document.createElement("div");
        hint.style.cssText = "font-size:12px;color:#94a3b8;line-height:1.45;margin-top:8px;";
        hint.textContent = "Space or Down Arrow adds a split at the playhead. Hold-to-repeat is ignored. Shortcuts do not run while typing; use the audio player’s play/pause button to control playback.";
        pane.append(title, copy, audioPanel, controls, splitList, hint);
        renderSplitList();
      }
      pane.append(statusLine);
    };
    for (const tab of tabs) {
      const button = makeButton(tab.label);
      button.onclick = () => setActiveTab(tab.id);
      tabButtons.set(tab.id, button);
      tabBar.append(button);
    }
    window.addEventListener("keydown", onLyricMappingKeydown, true);
    const closeModal = () => {
      window.removeEventListener("keydown", onLyricMappingKeydown, true);
      manualTimingAudio?.pause?.();
      backdrop.remove();
      options.onClose?.();
    };
    close.onclick = closeModal;
    if (options.manualOnly) {
      heading.textContent = "Manual Timing";
      box.append(header, pane);
    } else box.append(header, warning, tabBar, pane);
    backdrop.append(box);
    document.body.append(backdrop);
    setActiveTab(options.manualOnly ? "manual_timing" : "transcribe");
    if (options.manualOnly) { pane.tabIndex = -1; pane.focus(); }
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) closeModal();
    });
  }

  return { openLyricMappingWorkflowModal };
}
