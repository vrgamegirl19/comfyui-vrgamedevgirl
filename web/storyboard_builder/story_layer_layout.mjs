import { makeButton, makeCollapsiblePanel, makeInput, makeSelect, makeTextarea, storyField } from "./controls.mjs";
import { normalizeStoryLayer } from "./scenes.mjs";

export function buildStoryLayerPanel({
  focusedSection, isIdLoraMode, isMiniMaxShortFilmMode, promptRunnerName, state, usesFilmPlanningProfile,
}) {
  const storyLayerBar = document.createElement("div");
  storyLayerBar.className = "vrgdg-storyboard-story-grid";
  storyLayerBar.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;min-width:0;max-width:100%;color:#cbd5e1;font-size:12px;";
  const storyLayerHeader = document.createElement("div");
  storyLayerHeader.style.cssText = "grid-column:1/-1;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;min-width:0;max-width:100%;";
  const storyLayerTitle = document.createElement("div");
  storyLayerTitle.style.cssText = "flex:1 1 420px;min-width:0;max-width:100%;overflow-wrap:anywhere;";
  storyLayerTitle.innerHTML = usesFilmPlanningProfile
    ? `<div style="font-weight:900;color:#cffafe;font-size:15px;">Short Film Story Layer</div><div style="color:#94a3b8;margin-top:2px;">Dialogue-first planning for ${isIdLoraMode ? "ID-LoRA" : "MiniMax H3"} scenes, characters, and locations.</div>`
    : `<div style="font-weight:900;color:#cffafe;font-size:15px;">Story Layer</div><div style="color:#94a3b8;margin-top:2px;">Optional narrative context for connecting lyrics, sections, subjects, and locations across scenes.</div>`;
  const storyLayerEnabledLabel = document.createElement("label");
  storyLayerEnabledLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;white-space:normal;max-width:100%;";
  const storyLayerEnabledInput = document.createElement("input");
  storyLayerEnabledInput.type = "checkbox";
  storyLayerEnabledInput.checked = state.storyLayer.enabled !== false;
  storyLayerEnabledLabel.append(storyLayerEnabledInput, document.createTextNode(`Use in ${promptRunnerName()} prompts`));
  storyLayerHeader.append(storyLayerTitle, storyLayerEnabledLabel);
  const shortFilmPlanningModeWrap = document.createElement("div");
  shortFilmPlanningModeWrap.style.cssText = "grid-column:1/-1;display:none;grid-template-columns:minmax(180px,260px) minmax(0,1fr);gap:12px;align-items:start;border:1px solid #155e75;border-radius:8px;background:#071a2b;padding:12px;";
  const shortFilmPlanningModeSelect = makeSelect([
    { value: "guided_film", label: "Guided Film Automation" },
    { value: "fully_custom", label: "Fully Custom" },
  ], state.shortFilmPlanningMode);
  const shortFilmPlanningModeField = document.createElement("label");
  shortFilmPlanningModeField.style.cssText = "display:flex;flex-direction:column;gap:6px;font-size:12px;font-weight:900;color:#cbd5e1;";
  shortFilmPlanningModeField.textContent = "Short Film Authoring Mode";
  shortFilmPlanningModeField.append(shortFilmPlanningModeSelect);
  const shortFilmPlanningModeInfo = document.createElement("div");
  shortFilmPlanningModeInfo.style.cssText = "color:#bae6fd;line-height:1.45;min-width:0;overflow-wrap:anywhere;";
  shortFilmPlanningModeWrap.append(shortFilmPlanningModeField, shortFilmPlanningModeInfo);
  const overallStoryIdeaInput = makeTextarea(
    state.storyLayer.overall_story_idea || "",
    "Optional short premise, e.g. A woman navigates a surreal dream world.",
    3,
  );
  overallStoryIdeaInput.title = `Optional. Sets the overall premise, world, or theme that ${promptRunnerName()} develops through the real lyric sections.`;
  const userStoryArcInput = makeTextarea(
    state.storyLayer.user_story_arc || "",
    usesFilmPlanningProfile ? "Short film premise, conflict, tone, character goal, or pasted script..." : "Optional user story arc, e.g. Verse 1: she feels trapped. Chorus: she breaks free...",
    5,
  );
  const songStoryBriefInput = makeTextarea(
    state.storyLayer.song_story_brief || "",
    usesFilmPlanningProfile ? "LLM-created short film story brief..." : `${promptRunnerName()}-created song story brief...`,
    5,
  );
  const lyricStoryStrengthInput = makeInput(String(normalizeStoryLayer(state.storyLayer).lyric_story_strength));
  lyricStoryStrengthInput.type = "range";
  lyricStoryStrengthInput.min = "0";
  lyricStoryStrengthInput.max = "10";
  lyricStoryStrengthInput.step = "1";
  lyricStoryStrengthInput.style.accentColor = "#22d3ee";
  const lyricStoryStrengthValue = document.createElement("div");
  lyricStoryStrengthValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:105px;text-align:right;";
  const lyricStoryStrengthHintButton = makeButton("Hint");
  lyricStoryStrengthHintButton.title = "Explain Lyric Story Strength.";
  const overallStoryIdeaField = storyField("Overall Story Idea (optional)", overallStoryIdeaInput);
  overallStoryIdeaField.style.gridColumn = "1/-1";
  const overallStoryIdeaHint = document.createElement("div");
  overallStoryIdeaHint.style.cssText = "font-size:11px;font-weight:500;color:#94a3b8;line-height:1.4;";
  overallStoryIdeaHint.textContent = `Sets the overall premise, world, or theme. ${promptRunnerName()} will develop it through the actual reference-lyric sections; leave blank for a lyric-led idea.`;
  overallStoryIdeaField.append(overallStoryIdeaHint);
  const lyricStoryStrengthRow = document.createElement("div");
  lyricStoryStrengthRow.style.cssText = "grid-column:1/-1;display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:8px;align-items:end;";
  lyricStoryStrengthRow.append(storyField("Lyric Story Strength", lyricStoryStrengthInput), lyricStoryStrengthValue, lyricStoryStrengthHintButton);
  lyricStoryStrengthRow.style.display = usesFilmPlanningProfile ? "none" : "grid";
  const adjacentLyricContextLabel = document.createElement("label");
  adjacentLyricContextLabel.style.cssText = "grid-column:1/-1;display:flex;align-items:center;gap:7px;color:#cbd5e1;font-size:12px;font-weight:800;";
  const adjacentLyricContextInput = document.createElement("input");
  adjacentLyricContextInput.type = "checkbox";
  adjacentLyricContextInput.checked = Boolean(state.sendAdjacentLyricContext);
  adjacentLyricContextLabel.append(adjacentLyricContextInput, document.createTextNode("Send last and next lyric line for context"));
  adjacentLyricContextLabel.title = "When enabled, each scene story beat receives the previous scene's last lyric line and the next scene's first lyric line when available.";
  adjacentLyricContextLabel.style.display = usesFilmPlanningProfile ? "none" : "flex";
  const idLoraDialoguePlanner = document.createElement("div");
  idLoraDialoguePlanner.style.cssText = "grid-column:1/-1;display:none;border:1px solid #155e75;border-radius:8px;background:#082f49;padding:12px;gap:10px;align-items:center;grid-template-columns:minmax(0,1fr) auto;";
  const idLoraDialoguePlannerText = document.createElement("div");
  idLoraDialoguePlannerText.innerHTML = `<div style="font-weight:900;color:#cffafe;">Plan Dialogue Scenes</div><div style="color:#bae6fd;line-height:1.35;margin-top:3px;">Enter a story idea, outline, or pasted script above. If left blank, the selected LLM invents a short-film dialogue scene plan from your ${isIdLoraMode ? "ID-LoRA" : "MiniMax H3"} characters and locations.</div>`;
  const idLoraDialogueControls = document.createElement("div");
  idLoraDialogueControls.style.cssText = "display:flex;gap:8px;align-items:end;flex-wrap:wrap;justify-content:flex-end;";
  const idLoraDialogueSceneCount = makeInput("6");
  idLoraDialogueSceneCount.type = "number";
  idLoraDialogueSceneCount.min = "1";
  idLoraDialogueSceneCount.max = "24";
  idLoraDialogueSceneCount.step = "1";
  idLoraDialogueSceneCount.style.width = "76px";
  const planDialogueScenesButton = makeButton("Plan Storyboard Scenes", "primary");
  planDialogueScenesButton.title = `${isIdLoraMode ? "ID-LoRA" : "MiniMax H3 Guided Film"}. Develop editable scene cards inside Storyboard Builder. This does not create Video Builder timeline segments.`;
  const applyDialoguePlanButton = makeButton("Create Timeline Segments", "primary");
  applyDialoguePlanButton.title = "Create real Video Builder timeline segments from the reviewed storyboard scenes.";
  applyDialoguePlanButton.style.display = "none";
  idLoraDialogueControls.append(storyField("Scenes", idLoraDialogueSceneCount), planDialogueScenesButton, applyDialoguePlanButton);
  idLoraDialoguePlanner.append(idLoraDialoguePlannerText, idLoraDialogueControls);
  const miniMaxGuidedWorkflowSteps = document.createElement("div");
  miniMaxGuidedWorkflowSteps.style.cssText = "grid-column:1/-1;display:none;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;border:1px solid #155e75;border-radius:8px;background:#041923;padding:10px;";
  miniMaxGuidedWorkflowSteps.innerHTML = `
    <div style="border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;"><strong style="color:#67e8f9;">STEP 1 — SCRIPT</strong><br><span style="color:#bae6fd;">Import or review the script, map speakers, then click <strong>Use This Script</strong>.</span></div>
    <div style="border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;"><strong style="color:#67e8f9;">STEP 2 — DEVELOP</strong><br><span style="color:#bae6fd;">Create the editable storyboard scene cards. The timeline is still unchanged.</span></div>
    <div style="border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;"><strong style="color:#67e8f9;">STEP 3 — REVIEW</strong><br><span style="color:#bae6fd;">Review and edit the scene cards, dialogue, references, shots, and continuity below.</span></div>
    <div style="border:1px solid #0891b2;border-radius:7px;background:#083344;padding:10px;line-height:1.35;"><strong style="color:#67e8f9;">STEP 4 — TIMELINE</strong><br><span style="color:#bae6fd;">Create the real Video Builder timeline segments only after reviewing the cards.</span></div>
  `;
  const miniMaxScriptImporter = document.createElement("div");
  miniMaxScriptImporter.style.cssText = "grid-column:1/-1;display:none;border:1px solid #0e7490;border-radius:8px;background:#06283d;padding:12px;gap:10px;align-items:center;grid-template-columns:minmax(0,1fr) auto;";
  const miniMaxScriptImporterText = document.createElement("div");
  miniMaxScriptImporterText.innerHTML = `<div style="font-weight:900;color:#cffafe;">Import Script / Script Mapper</div><div style="color:#bae6fd;line-height:1.4;margin-top:3px;">Paste a <strong>speaker: dialogue</strong> script or load a .txt/.json file. Validate exact cues, match speakers, and preview automatically timed MiniMax segments without changing the timeline.</div>`;
  const openMiniMaxScriptMapperButton = makeButton("Import Script / Script Mapper", "primary");
  openMiniMaxScriptMapperButton.title = "Import, map, time, and activate an exact dialogue script for MiniMax Guided Film Automation.";
  miniMaxScriptImporter.append(miniMaxScriptImporterText, openMiniMaxScriptMapperButton);
  const storyActions = document.createElement("div");
  storyActions.style.cssText = "grid-column:1/-1;display:flex;gap:8px;align-items:center;flex-wrap:wrap;";
  const storyActionsLabel = document.createElement("div");
  storyActionsLabel.style.cssText = "flex:0 0 100%;font-size:12px;font-weight:900;color:#fcd34d;border-top:1px solid #334155;padding-top:10px;";
  storyActionsLabel.textContent = usesFilmPlanningProfile
    ? "OPTIONAL STORY PLANNING TOOLS — not required when using an imported authoritative script"
    : "OPTIONAL STORY PLANNING TOOLS";
  const createStoryArcButton = makeButton("Create User Story Arc", "primary");
  const createStorySequenceButton = makeButton("Create Arc → Brief → Missing Beats", "primary");
  createStorySequenceButton.style.marginRight = "auto";
  const createStoryBriefButton = makeButton("Create Story Brief", "primary");
  const gptStoryButton = makeButton("GPT Story");
  gptStoryButton.title = "Copy all story, lyric, scene, reference, and preset details as JSON, then open the Storyboard GPT.";
  const importStoryJsonButton = makeButton("Import story from json");
  importStoryJsonButton.title = "Paste or load GPT story JSON and fill the overall idea, story arc, and story brief.";
  const createMissingBeatsButton = makeButton("Create Missing Scene Beats", "purple");
  const replaceBeatsButton = makeButton("Replace All Scene Beats");
  const detectSectionsButton = makeButton("Detect Lyric Sections");
  storyActions.append(storyActionsLabel, createStorySequenceButton, createStoryArcButton, createStoryBriefButton, createMissingBeatsButton, replaceBeatsButton, detectSectionsButton, gptStoryButton, importStoryJsonButton);
  storyLayerBar.append(
    storyLayerHeader,
    shortFilmPlanningModeWrap,
    lyricStoryStrengthRow,
    adjacentLyricContextLabel,
    overallStoryIdeaField,
    storyField("User Story Arc", userStoryArcInput),
    storyField("Song Story Brief", songStoryBriefInput),
    miniMaxGuidedWorkflowSteps,
    miniMaxScriptImporter,
    idLoraDialoguePlanner,
    storyActions,
  );

  const hasStoryLayerContent = Boolean(String(state.storyLayer.overall_story_idea || "").trim() || String(state.storyLayer.user_story_arc || "").trim() || String(state.storyLayer.song_story_brief || "").trim());
  const storyLayerPanel = makeCollapsiblePanel("Story Layer", "", storyLayerBar, { open: !focusedSection || focusedSection === "story" || hasStoryLayerContent || isMiniMaxShortFilmMode });
  storyLayerPanel.classList.add("vrgdg-storyboard-panel");

  return {
    adjacentLyricContextInput, applyDialoguePlanButton, createMissingBeatsButton, createStoryArcButton,
    createStorySequenceButton,
    createStoryBriefButton, detectSectionsButton, gptStoryButton, idLoraDialoguePlanner,
    idLoraDialoguePlannerText, idLoraDialogueSceneCount, importStoryJsonButton, lyricStoryStrengthHintButton,
    lyricStoryStrengthInput, lyricStoryStrengthValue, miniMaxGuidedWorkflowSteps, miniMaxScriptImporter,
    miniMaxScriptImporterText, openMiniMaxScriptMapperButton, overallStoryIdeaInput, planDialogueScenesButton,
    replaceBeatsButton, shortFilmPlanningModeInfo, shortFilmPlanningModeSelect, shortFilmPlanningModeWrap,
    songStoryBriefInput, storyActions, storyLayerEnabledInput, storyLayerPanel, userStoryArcInput,
  };
}
