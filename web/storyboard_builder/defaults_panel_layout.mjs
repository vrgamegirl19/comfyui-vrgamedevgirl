import { makeButton, makeCollapsiblePanel, makeInput, makeSelect, makeTextarea, sortOptionsAlphabetically } from "./controls.mjs";
import { normalizeStoryLayer, storyboardCutFrequencyValue, storyboardSpeedValue } from "./scenes.mjs";
import { STORYBOARD_CAMERA_FLOW_PRESETS } from "./shot_presets.mjs";
import {
  MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS,
  MINIMAX_VIDEO_STYLE_PRESETS,
  STORYBOARD_FX_PRESETS,
  storyboardTemporalIntensity,
} from "./video_style.mjs";

export function buildSceneDefaultsPanel({
  facialPerformancePresets, focusedSection, imageAestheticPresets, imageShotFlowPresets,
  performanceStylePresets, state, usesFilmPlanningProfile,
}) {
  const note = document.createElement("div");
  note.className = "vrgdg-storyboard-note";
  note.style.cssText = "margin:18px 24px 0;min-width:0;max-width:100%;box-sizing:border-box;border:1px solid #155e75;border-radius:8px;background:#0f172a;color:#cbd5e1;padding:12px 14px;font-size:13px;overflow-wrap:anywhere;";
  const middleContent = document.createElement("div");
  middleContent.style.cssText = "min-width:0;min-height:0;overflow-y:auto;overflow-x:hidden;padding-bottom:18px;scrollbar-width:thin;";

  const cameraFlowBar = document.createElement("div");
  cameraFlowBar.className = "vrgdg-storyboard-defaults-grid";
  cameraFlowBar.style.cssText = "display:grid;grid-template-columns:minmax(420px,700px) minmax(0,1fr);gap:8px 12px;align-items:center;width:100%;min-width:0;max-width:100%;box-sizing:border-box;color:#cbd5e1;font-size:12px;";
  const imageShotControls = document.createElement("div");
  imageShotControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const imageShotLabel = document.createElement("div");
  imageShotLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  imageShotLabel.textContent = "Still shot flow";
  const imageShotSelect = makeSelect(
    sortOptionsAlphabetically(Object.entries(imageShotFlowPresets).map(([value, preset]) => ({ value, label: preset.label }))),
    state.imageShotFlow,
  );
  imageShotSelect.style.width = "max-content";
  imageShotSelect.style.minWidth = "180px";
  const imageShotApply = makeButton("Fill Missing", "primary");
  imageShotApply.title = "Fill only blank shot/composition fields for Image Prep. Existing manual choices are kept.";
  const imageShotReplace = makeButton("Replace All");
  imageShotReplace.title = "Replace every scene's shot/composition field with the selected still shot flow.";
  imageShotControls.append(imageShotLabel, imageShotSelect, imageShotApply, imageShotReplace);
  const imageShotInfo = document.createElement("div");
  imageShotInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const imageAestheticControls = document.createElement("div");
  imageAestheticControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const imageAestheticLabel = document.createElement("div");
  imageAestheticLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  imageAestheticLabel.textContent = "Image aesthetic";
  const imageAestheticSelect = makeSelect(sortOptionsAlphabetically(imageAestheticPresets), state.imageAesthetic);
  imageAestheticSelect.style.width = "max-content";
  imageAestheticSelect.style.minWidth = "180px";
  const imageAestheticApply = makeButton("Fill Missing", "primary");
  imageAestheticApply.title = "Fill only scenes without a still camera style/aesthetic note.";
  const imageAestheticReplace = makeButton("Replace All");
  imageAestheticReplace.title = "Replace each scene's generated image aesthetic note.";
  imageAestheticControls.append(imageAestheticLabel, imageAestheticSelect, imageAestheticApply, imageAestheticReplace);
  const imageAestheticInfo = document.createElement("div");
  imageAestheticInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const videoStyleControls = document.createElement("div");
  videoStyleControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const videoStyleLabel = document.createElement("div");
  videoStyleLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  videoStyleLabel.textContent = "Video aesthetic";
  const videoStyleSelect = makeSelect(sortOptionsAlphabetically(MINIMAX_VIDEO_STYLE_PRESETS), state.videoStyle);
  videoStyleSelect.style.width = "max-content";
  videoStyleSelect.style.minWidth = "220px";
  const videoStyleApply = makeButton("Fill Missing", "primary");
  videoStyleApply.title = "Fill blank style fields on eligible video scenes.";
  const videoStyleReplace = makeButton("Replace All");
  videoStyleReplace.title = "Replace the style on all eligible video scenes.";
  videoStyleControls.append(videoStyleLabel, videoStyleSelect, videoStyleApply, videoStyleReplace);
  const videoStyleCustomControls = document.createElement("div");
  videoStyleCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;";
  const videoStyleCustomLabel = document.createElement("div");
  videoStyleCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:9px;";
  videoStyleCustomLabel.textContent = "Custom style wording";
  const videoStyleCustomInput = makeTextarea(
    state.videoStyleCustom,
    "Type the exact visual-style wording that must appear unchanged in every eligible prompt...",
    3,
  );
  videoStyleCustomInput.style.minWidth = "520px";
  videoStyleCustomControls.append(videoStyleCustomLabel, videoStyleCustomInput);
  const videoStyleInfo = document.createElement("div");
  videoStyleInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const temporalEffectControls = document.createElement("div");
  temporalEffectControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const temporalEffectLabel = document.createElement("div");
  temporalEffectLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  temporalEffectLabel.textContent = "Temporal / world effect";
  const temporalEffectSelect = makeSelect(sortOptionsAlphabetically(MINIMAX_TEMPORAL_WORLD_EFFECT_PRESETS), state.temporalWorldEffect);
  temporalEffectSelect.style.width = "max-content";
  temporalEffectSelect.style.minWidth = "300px";
  temporalEffectControls.append(temporalEffectLabel, temporalEffectSelect);
  const temporalEffectCustomControls = document.createElement("div");
  temporalEffectCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;";
  const temporalEffectCustomLabel = document.createElement("div");
  temporalEffectCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:9px;";
  temporalEffectCustomLabel.textContent = "Custom temporal wording";
  const temporalEffectCustomInput = makeTextarea(state.temporalWorldEffectCustom, "Describe the exact temporal separation or world behavior. Character protection and audio-safety rules will be added automatically...", 3);
  temporalEffectCustomInput.style.minWidth = "520px";
  temporalEffectCustomControls.append(temporalEffectCustomLabel, temporalEffectCustomInput);
  const fxControls = document.createElement("div");
  fxControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const fxLabel = document.createElement("div");
  fxLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  fxLabel.textContent = "Shot FX preset";
  const fxSelect = makeSelect(sortOptionsAlphabetically(STORYBOARD_FX_PRESETS), state.fxPreset);
  fxSelect.style.width = "max-content";
  fxSelect.style.minWidth = "260px";
  fxControls.append(fxLabel, fxSelect);
  const fxCustomControls = document.createElement("div");
  fxCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;";
  const fxCustomLabel = document.createElement("div");
  fxCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:9px;";
  fxCustomLabel.textContent = "Custom FX JSON";
  const fxCustomInput = makeTextarea(state.fxCustomJson, "{\n  \"label\": \"Custom FX\",\n  \"cues\": [\"A brief effect crosses the background on the beat.\"],\n  \"timing\": \"on the musical accent\",\n  \"intensity\": 6\n}", 5);
  fxCustomInput.style.minWidth = "520px";
  fxCustomControls.append(fxCustomLabel, fxCustomInput);
  const fxInfo = document.createElement("div");
  fxInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const temporalEffectOptions = document.createElement("div");
  temporalEffectOptions.style.cssText = "display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;padding:9px 10px;border:1px solid #1f3347;border-radius:7px;background:#07111f;";
  const temporalExtrasLabel = document.createElement("label");
  temporalExtrasLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;";
  const temporalExtrasInput = document.createElement("input");
  temporalExtrasInput.type = "checkbox";
  temporalExtrasInput.checked = state.temporalAllowBackgroundExtras !== false;
  temporalExtrasLabel.append(temporalExtrasInput, document.createTextNode("Allow location-appropriate anonymous extras"));
  const temporalEnvironmentLabel = document.createElement("label");
  temporalEnvironmentLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;";
  const temporalEnvironmentInput = document.createElement("input");
  temporalEnvironmentInput.type = "checkbox";
  temporalEnvironmentInput.checked = state.temporalEnvironmentTimePassage !== false;
  temporalEnvironmentLabel.append(temporalEnvironmentInput, document.createTextNode("Allow lighting / weather / time passage"));
  const temporalIntensityLabel = document.createElement("label");
  temporalIntensityLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;min-width:290px;";
  const temporalIntensityInput = makeInput(String(storyboardTemporalIntensity(state.temporalBackgroundIntensity)));
  temporalIntensityInput.type = "range";
  temporalIntensityInput.min = "0";
  temporalIntensityInput.max = "10";
  temporalIntensityInput.step = "1";
  temporalIntensityInput.style.width = "170px";
  temporalIntensityInput.style.accentColor = "#22d3ee";
  const temporalIntensityValue = document.createElement("span");
  temporalIntensityValue.style.cssText = "color:#cffafe;font-weight:900;min-width:38px;";
  temporalIntensityLabel.append(document.createTextNode("World intensity"), temporalIntensityInput, temporalIntensityValue);
  const temporalProtectedLabel = document.createElement("label");
  temporalProtectedLabel.style.cssText = "display:flex;align-items:center;gap:7px;font-weight:800;color:#cbd5e1;min-width:360px;";
  const temporalProtectedSelect = makeSelect([
    { value: "all_referenced", label: "Protect all referenced characters (recommended)" },
    { value: "lead_only", label: "Protect first referenced character only" },
    { value: "custom", label: "Protect named referenced characters" },
  ], state.temporalProtectedCharacters);
  temporalProtectedSelect.style.minWidth = "280px";
  temporalProtectedLabel.append(document.createTextNode("Real-time cast"), temporalProtectedSelect);
  temporalEffectOptions.append(temporalExtrasLabel, temporalEnvironmentLabel, temporalIntensityLabel, temporalProtectedLabel);
  const temporalProtectedCustomControls = document.createElement("div");
  temporalProtectedCustomControls.style.cssText = "display:flex;gap:8px;align-items:center;";
  const temporalProtectedCustomLabel = document.createElement("div");
  temporalProtectedCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  temporalProtectedCustomLabel.textContent = "Protected names";
  const temporalProtectedCustomInput = makeInput(state.temporalProtectedCustom, "Exact mapped character names, comma separated");
  temporalProtectedCustomInput.style.minWidth = "520px";
  temporalProtectedCustomControls.append(temporalProtectedCustomLabel, temporalProtectedCustomInput);
  const temporalEffectInfo = document.createElement("div");
  temporalEffectInfo.style.cssText = "color:#94a3b8;line-height:1.35;white-space:pre-wrap;";
  const consistencyControls = document.createElement("div");
  consistencyControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const consistencyLabel = document.createElement("div");
  consistencyLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  consistencyLabel.textContent = "Global consistency phrase";
  const consistencyInput = makeInput(state.globalConsistencyPhrase, "e.g. soft glittery eye makeup, wet-look hair, chrome jewelry");
  consistencyInput.style.minWidth = "520px";
  consistencyControls.append(consistencyLabel, consistencyInput);
  const consistencyInfo = document.createElement("div");
  consistencyInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const cameraFlowControls = document.createElement("div");
  cameraFlowControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const cameraFlowLabel = document.createElement("div");
  cameraFlowLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  cameraFlowLabel.textContent = "Auto camera flow";
  const cameraFlowSelect = makeSelect(
    sortOptionsAlphabetically(Object.entries(STORYBOARD_CAMERA_FLOW_PRESETS).map(([value, preset]) => ({ value, label: preset.label }))),
    state.cameraFlow,
  );
  cameraFlowSelect.style.width = "max-content";
  cameraFlowSelect.style.minWidth = "180px";
  const cameraFlowApply = makeButton("Fill Missing", "primary");
  cameraFlowApply.title = "Fill only blank shot type and camera motion fields. Existing manual choices are kept.";
  const cameraFlowReplace = makeButton("Replace All");
  cameraFlowReplace.title = "Replace every scene's shot type and camera motion with the selected auto camera flow.";
  cameraFlowControls.append(cameraFlowLabel, cameraFlowSelect, cameraFlowApply, cameraFlowReplace);
  const cameraFlowInfo = document.createElement("div");
  cameraFlowInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const cameraSpeedControls = document.createElement("div");
  cameraSpeedControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const cameraSpeedLabel = document.createElement("div");
  cameraSpeedLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  cameraSpeedLabel.textContent = "Camera motion speed";
  const cameraSpeedInput = makeInput(String(storyboardSpeedValue(state.cameraMotionSpeed, 4)));
  cameraSpeedInput.type = "range";
  cameraSpeedInput.min = "0";
  cameraSpeedInput.max = "10";
  cameraSpeedInput.step = "1";
  cameraSpeedInput.style.minWidth = "360px";
  cameraSpeedInput.style.accentColor = "#22d3ee";
  const cameraSpeedValue = document.createElement("div");
  cameraSpeedValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:120px;";
  const cameraSpeedHint = makeButton("Hint");
  cameraSpeedHint.title = "Explain camera motion speed.";
  cameraSpeedControls.append(cameraSpeedLabel, cameraSpeedInput, cameraSpeedValue, cameraSpeedHint);
  const cameraSpeedInfo = document.createElement("div");
  cameraSpeedInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const cutFrequencyControls = document.createElement("div");
  cutFrequencyControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const cutFrequencyLabel = document.createElement("div");
  cutFrequencyLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  cutFrequencyLabel.textContent = "Cut frequency";
  const cutFrequencyInput = makeInput(String(storyboardCutFrequencyValue(state.cutFrequency)));
  cutFrequencyInput.type = "range";
  cutFrequencyInput.min = "0";
  cutFrequencyInput.max = "10";
  cutFrequencyInput.step = "1";
  cutFrequencyInput.style.minWidth = "360px";
  cutFrequencyInput.style.accentColor = "#22d3ee";
  const cutFrequencyValue = document.createElement("div");
  cutFrequencyValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:150px;";
  const cutFrequencyHint = makeButton("Hint");
  cutFrequencyHint.title = "Explain MiniMax cut frequency.";
  cutFrequencyControls.append(cutFrequencyLabel, cutFrequencyInput, cutFrequencyValue, cutFrequencyHint);
  const cutFrequencyInfo = document.createElement("div");
  cutFrequencyInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const performanceControls = document.createElement("div");
  performanceControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const performanceLabel = document.createElement("div");
  performanceLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  performanceLabel.textContent = usesFilmPlanningProfile ? "Global acting style" : "Global performance style";
  const performanceSelect = makeSelect(sortOptionsAlphabetically(performanceStylePresets), state.performanceStyle);
  performanceSelect.style.width = "max-content";
  performanceSelect.style.minWidth = "180px";
  const performanceApply = makeButton("Fill Missing", "primary");
  performanceApply.title = usesFilmPlanningProfile ? "Fill only blank per-scene acting style fields. Existing scene choices are kept." : "Fill only blank per-scene performance/song style fields. Existing scene choices are kept.";
  const performanceReplace = makeButton("Replace All");
  performanceReplace.title = usesFilmPlanningProfile ? "Replace every scene's acting style with the selected global style." : "Replace every scene's performance/song style with the selected global style.";
  performanceControls.append(performanceLabel, performanceSelect, performanceApply, performanceReplace);
  const performanceInfo = document.createElement("div");
  performanceInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const characterSpeedControls = document.createElement("div");
  characterSpeedControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const characterSpeedLabel = document.createElement("div");
  characterSpeedLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  characterSpeedLabel.textContent = "Character motion speed";
  const characterSpeedInput = makeInput(String(storyboardSpeedValue(state.characterMotionSpeed, 4)));
  characterSpeedInput.type = "range";
  characterSpeedInput.min = "0";
  characterSpeedInput.max = "10";
  characterSpeedInput.step = "1";
  characterSpeedInput.style.minWidth = "360px";
  characterSpeedInput.style.accentColor = "#22d3ee";
  const characterSpeedValue = document.createElement("div");
  characterSpeedValue.style.cssText = "font-size:12px;color:#cffafe;font-weight:900;min-width:120px;";
  const characterSpeedHint = makeButton("Hint");
  characterSpeedHint.title = "Explain character motion speed.";
  characterSpeedControls.append(characterSpeedLabel, characterSpeedInput, characterSpeedValue, characterSpeedHint);
  const characterSpeedInfo = document.createElement("div");
  characterSpeedInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const storyArcDetailControls = document.createElement("div");
  storyArcDetailControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const storyArcDetailLabel = document.createElement("div");
  storyArcDetailLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  storyArcDetailLabel.textContent = "Story arc detail";
  const storyArcDetailSelect = makeSelect([
    { value: "compact", label: "Compact (~60 words per section)" },
    { value: "standard", label: "Standard (~100 words per section)" },
    { value: "detailed", label: "Detailed (~160 words per section)" },
    { value: "rich", label: "Rich (~240 words per section)" },
  ], String(state.storyArcDetail || "standard"));
  storyArcDetailSelect.style.minWidth = "260px";
  storyArcDetailControls.append(storyArcDetailLabel, storyArcDetailSelect);
  const storyArcDetailInfo = document.createElement("div");
  storyArcDetailInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const storyArcDetailGuidance = {
    compact: "Short story arc paragraphs. Total budget about 1000 words spread across all lyric sections.",
    standard: "Balanced story arc paragraphs. Total budget about 1500 words spread across all lyric sections.",
    detailed: "Fuller story arc paragraphs with more staging, lighting, and texture. Total budget about 2400 words. Uses more LLM output tokens.",
    rich: "Richest story arc paragraphs. Total budget about 3600 words, so a long song may need a large LLM context and slower generation.",
  };
  const refreshStoryArcDetailInfo = () => {
    storyArcDetailInfo.textContent = `${storyArcDetailGuidance[storyArcDetailSelect.value] || storyArcDetailGuidance.standard} Applies to Generate Story Arc. Per-section length shrinks automatically for songs with many sections.`;
  };
  storyArcDetailSelect.addEventListener("input", refreshStoryArcDetailInfo);
  refreshStoryArcDetailInfo();
  const facialControls = document.createElement("div");
  facialControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const facialLabel = document.createElement("div");
  facialLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  facialLabel.textContent = usesFilmPlanningProfile ? "Global screen face" : "Global facial performance";
  const facialSelect = makeSelect(sortOptionsAlphabetically(facialPerformancePresets), state.facialPerformance);
  facialSelect.style.width = "max-content";
  facialSelect.style.minWidth = "180px";
  const facialApply = makeButton("Fill Missing", "primary");
  facialApply.title = "Fill only blank per-scene facial performance fields.";
  const facialReplace = makeButton("Replace All");
  facialReplace.title = "Replace every scene's facial performance with the selected global facial preset.";
  facialControls.append(facialLabel, facialSelect, facialApply, facialReplace);
  const facialInfo = document.createElement("div");
  facialInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const facialCustomControls = document.createElement("div");
  facialCustomControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;white-space:nowrap;";
  const facialCustomLabel = document.createElement("div");
  facialCustomLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:8px;";
  facialCustomLabel.textContent = "Custom facial text";
  const facialCustomInput = makeTextarea(state.facialPerformanceCustom || "", "Optional custom facial performance text, e.g. expressive eyes, active brows, natural blinking...", 3);
  facialCustomInput.style.minWidth = "520px";
  facialCustomControls.append(facialCustomLabel, facialCustomInput);
  const facialCustomInfo = document.createElement("div");
  facialCustomInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const imageWorldStyleControls = document.createElement("div");
  imageWorldStyleControls.style.cssText = "display:flex;gap:8px;align-items:center;white-space:nowrap;";
  const imageWorldStyleLabel = document.createElement("div");
  imageWorldStyleLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;";
  imageWorldStyleLabel.textContent = "Image world style";
  const imageWorldStyleSelect = makeSelect([
    { value: "natural", label: "Natural / realistic world" },
    { value: "surreal_subject", label: "Realistic world + surreal subject" },
    { value: "balanced_surreal", label: "Balanced surrealism" },
    { value: "full_surreal", label: "Fully surreal world" },
    { value: "abstract", label: "Abstract / nonliteral world" },
    { value: "custom", label: "Fully custom" },
  ], normalizeStoryLayer(state.storyLayer).image_world_style);
  imageWorldStyleSelect.style.minWidth = "240px";
  imageWorldStyleControls.append(imageWorldStyleLabel, imageWorldStyleSelect);
  const imageWorldStyleInfo = document.createElement("div");
  imageWorldStyleInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const imageCustomStyleControls = document.createElement("div");
  imageCustomStyleControls.style.cssText = "display:flex;gap:8px;align-items:flex-start;";
  const imageCustomStyleLabel = document.createElement("div");
  imageCustomStyleLabel.style.cssText = "font-weight:900;color:#cffafe;white-space:nowrap;text-align:right;min-width:160px;padding-top:9px;";
  imageCustomStyleLabel.textContent = "Custom world direction";
  const imageCustomStyleInput = makeTextarea(normalizeStoryLayer(state.storyLayer).image_custom_style_direction, "Describe the whole visual world: environment, architecture, materials, lighting, color, perspective, subject styling, and anything to avoid...", 4);
  imageCustomStyleInput.style.minWidth = "520px";
  imageCustomStyleControls.append(imageCustomStyleLabel, imageCustomStyleInput);
  const imageCustomStyleInfo = document.createElement("div");
  imageCustomStyleInfo.style.cssText = "color:#94a3b8;line-height:1.35;";
  const responsiveDefaultRows = [
    imageShotControls,
    imageAestheticControls,
    videoStyleControls,
    videoStyleCustomControls,
    temporalEffectControls,
    temporalEffectCustomControls,
    fxControls,
    fxCustomControls,
    temporalEffectOptions,
    temporalProtectedCustomControls,
    imageWorldStyleControls,
    imageCustomStyleControls,
    consistencyControls,
    cameraFlowControls,
    cameraSpeedControls,
    cutFrequencyControls,
    performanceControls,
    characterSpeedControls,
    storyArcDetailControls,
    facialControls,
    facialCustomControls,
  ];
  for (const row of responsiveDefaultRows) {
    row.style.flexWrap = "wrap";
    row.style.whiteSpace = "normal";
    row.style.minWidth = "0";
    row.style.maxWidth = "100%";
  }
  const responsiveDefaultInputs = [
    imageShotSelect,
    imageAestheticSelect,
    videoStyleSelect,
    videoStyleCustomInput,
    temporalEffectSelect,
    temporalEffectCustomInput,
    fxSelect,
    fxCustomInput,
    temporalIntensityInput,
    temporalProtectedSelect,
    temporalProtectedCustomInput,
    imageWorldStyleSelect,
    imageCustomStyleInput,
    consistencyInput,
    cameraFlowSelect,
    cameraSpeedInput,
    cutFrequencyInput,
    performanceSelect,
    characterSpeedInput,
    storyArcDetailSelect,
    facialSelect,
    facialCustomInput,
  ];
  for (const control of responsiveDefaultInputs) {
    control.style.minWidth = "0";
    control.style.maxWidth = "100%";
  }
  for (const control of [videoStyleCustomInput, temporalEffectCustomInput, fxCustomInput, temporalProtectedCustomInput, imageCustomStyleInput, consistencyInput, cameraSpeedInput, cutFrequencyInput, characterSpeedInput, facialCustomInput]) {
    control.style.flex = "1 1 280px";
    control.style.width = "100%";
  }
  const responsiveDefaultInfo = [
    imageShotInfo,
    imageAestheticInfo,
    videoStyleInfo,
    temporalEffectInfo,
    fxInfo,
    imageWorldStyleInfo,
    imageCustomStyleInfo,
    consistencyInfo,
    cameraFlowInfo,
    cameraSpeedInfo,
    cutFrequencyInfo,
    performanceInfo,
    characterSpeedInfo,
    storyArcDetailInfo,
    facialInfo,
    facialCustomInfo,
  ];
  for (const info of responsiveDefaultInfo) {
    info.style.minWidth = "0";
    info.style.maxWidth = "100%";
    info.style.overflowWrap = "anywhere";
  }
  cameraFlowBar.append(imageShotControls, imageShotInfo, imageAestheticControls, imageAestheticInfo, videoStyleControls, videoStyleCustomControls, videoStyleInfo, temporalEffectControls, temporalEffectCustomControls, temporalEffectOptions, temporalProtectedCustomControls, temporalEffectInfo, fxControls, fxCustomControls, fxInfo, imageWorldStyleControls, imageWorldStyleInfo, imageCustomStyleControls, imageCustomStyleInfo, consistencyControls, consistencyInfo, cameraFlowControls, cameraFlowInfo, cameraSpeedControls, cameraSpeedInfo, cutFrequencyControls, cutFrequencyInfo, performanceControls, performanceInfo, characterSpeedControls, characterSpeedInfo, storyArcDetailControls, storyArcDetailInfo, facialControls, facialInfo, facialCustomControls, facialCustomInfo);
  const sceneDefaultsPanel = makeCollapsiblePanel("Scene Defaults", "", cameraFlowBar, { open: focusedSection === "defaults" });

  return {
    cameraFlowApply, cameraFlowControls, cameraFlowInfo, cameraFlowReplace, cameraFlowSelect,
    cameraSpeedControls, cameraSpeedHint, cameraSpeedInfo, cameraSpeedInput, cameraSpeedValue,
    characterSpeedControls, characterSpeedHint, characterSpeedInfo, characterSpeedInput, characterSpeedValue,
    consistencyInfo, consistencyInput, cutFrequencyControls, cutFrequencyHint, cutFrequencyInfo,
    cutFrequencyInput, cutFrequencyValue, facialApply, facialCustomInfo, facialCustomInput, facialInfo,
    facialReplace, facialSelect, fxControls, fxCustomControls, fxCustomInput, fxInfo, fxSelect,
    imageAestheticApply, imageAestheticControls, imageAestheticInfo, imageAestheticReplace,
    imageAestheticSelect, imageCustomStyleControls, imageCustomStyleInfo, imageCustomStyleInput,
    imageShotApply, imageShotControls, imageShotInfo, imageShotReplace, imageShotSelect,
    imageWorldStyleControls, imageWorldStyleInfo, imageWorldStyleSelect, middleContent, note,
    performanceApply, performanceInfo, performanceReplace, performanceSelect, sceneDefaultsPanel,
    storyArcDetailInfo, storyArcDetailSelect,
    temporalEffectControls, temporalEffectCustomControls, temporalEffectCustomInput, temporalEffectInfo,
    temporalEffectOptions, temporalEffectSelect, temporalEnvironmentInput, temporalExtrasInput,
    temporalIntensityInput, temporalIntensityValue, temporalProtectedCustomControls,
    temporalProtectedCustomInput, temporalProtectedSelect, videoStyleApply, videoStyleControls,
    videoStyleCustomControls, videoStyleCustomInput, videoStyleInfo, videoStyleReplace, videoStyleSelect,
  };
}
