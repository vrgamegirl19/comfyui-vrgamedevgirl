const FACE_FIX_ACCENT = "#22d3ee";

function button(label, primary = false) {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  element.style.cssText = `border:1px solid ${primary ? "#0891b2" : "#3f3f46"};border-radius:6px;background:${primary ? "#0e7490" : "#27272a"};color:#f4f4f5;padding:7px 10px;font-size:12px;font-weight:700;cursor:pointer;`;
  return element;
}

function field(label, control) {
  const wrap = document.createElement("label");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:4px;color:#a1a1aa;font-size:11px;font-weight:700;";
  const title = document.createElement("span");
  title.textContent = label;
  wrap.append(title, control);
  return wrap;
}

function describedField(label, control, description) {
  const wrap = field(label, control);
  const help = document.createElement("span");
  help.textContent = description;
  help.style.cssText = "color:#71717a;font-size:10px;font-weight:500;line-height:1.35;";
  control.title = description;
  wrap.append(help);
  return wrap;
}

function input(type, value, min = null, max = null, step = null) {
  const element = document.createElement("input");
  element.type = type;
  element.value = value;
  if (min !== null) element.min = String(min);
  if (max !== null) element.max = String(max);
  if (step !== null) element.step = String(step);
  element.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#f4f4f5;padding:7px 8px;font-size:12px;";
  return element;
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds || 0));
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function numericValue(control, fallback) {
  const value = Number(control?.value);
  return Number.isFinite(value) ? value : fallback;
}

export function createFaceFixTool(options = {}) {
  const launchButton = button("Face Fix (Experimental)");
  launchButton.style.width = "100%";
  let windowElement = null;
  const state = {
    inTime: null,
    outTime: null,
    referenceTime: null,
    referenceImage: "",
    videoPath: "",
    segmentId: "",
    sceneLabel: "",
    wholeScene: false,
  };

  function notify(message, error = false) {
    if (typeof options.toast === "function") options.toast(message, error);
  }

  function currentContext() {
    const context = options.getPlayheadContext?.() || {};
    const value = Number(context.time ?? options.getPlayheadSeconds?.());
    if (!Number.isFinite(value) || value < 0) throw new Error("Load a scene video and place its playhead first.");
    return {
      time: value,
      videoPath: String(context.videoPath || options.getVideoPath?.() || ""),
      segmentId: String(context.segmentId || ""),
      sceneLabel: String(context.sceneLabel || "scene"),
    };
  }

  function open() {
    if (windowElement?.isConnected) {
      windowElement.style.display = "flex";
      windowElement.style.zIndex = String(Date.now());
      return;
    }
    const panel = document.createElement("section");
    windowElement = panel;
    panel.style.cssText = "position:fixed;left:120px;top:90px;width:min(620px,calc(100vw - 40px));max-height:calc(100vh - 40px);z-index:10050;display:flex;flex-direction:column;border:1px solid #155e75;border-radius:10px;background:#111113;color:#f4f4f5;box-shadow:0 24px 80px rgba(0,0,0,.72);overflow:hidden;";

    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;gap:10px;padding:10px 12px;background:#164e63;cursor:move;user-select:none;";
    const title = document.createElement("div");
    title.textContent = "Face Fix (Experimental) v2";
    title.style.cssText = "font-size:13px;font-weight:900;flex:1;";
    const close = button("×");
    close.style.cssText += "padding:2px 9px;font-size:18px;";
    close.onclick = () => { panel.style.display = "none"; };
    header.append(title, close);

    let drag = null;
    header.addEventListener("pointerdown", (event) => {
      if (event.target === close) return;
      const rect = panel.getBoundingClientRect();
      drag = { x: event.clientX, y: event.clientY, left: rect.left, top: rect.top };
      header.setPointerCapture(event.pointerId);
    });
    header.addEventListener("pointermove", (event) => {
      if (!drag) return;
      panel.style.left = `${Math.max(0, Math.min(window.innerWidth - 120, drag.left + event.clientX - drag.x))}px`;
      panel.style.top = `${Math.max(0, Math.min(window.innerHeight - 50, drag.top + event.clientY - drag.y))}px`;
    });
    header.addEventListener("pointerup", () => { drag = null; });

    const body = document.createElement("div");
    body.style.cssText = "display:flex;flex-direction:column;gap:12px;padding:12px;overflow:auto;";
    const intro = document.createElement("div");
    intro.textContent = "Choose a bad-frame range with the scene playhead, then choose one clear frame for the editable face description.";
    intro.style.cssText = "font-size:12px;line-height:1.45;color:#a5f3fc;";

    const rangeGrid = document.createElement("div");
    rangeGrid.style.cssText = "display:grid;grid-template-columns:1fr auto 1fr auto;gap:8px;align-items:end;";
    const inDisplay = input("text", "Not set"); inDisplay.readOnly = true;
    const outDisplay = input("text", "Not set"); outDisplay.readOnly = true;
    const setIn = button("Set IN");
    const setOut = button("Set OUT");
    const lockedSourceDisplay = input("text", "Not set");
    lockedSourceDisplay.readOnly = true;
    setIn.onclick = () => {
      try {
        const context = currentContext();
        if (!context.videoPath) throw new Error("The playhead scene has no rendered video.");
        state.inTime = context.time;
        state.videoPath = context.videoPath;
        state.segmentId = context.segmentId;
        state.sceneLabel = context.sceneLabel;
        state.outTime = null;
        inDisplay.value = formatTime(state.inTime);
        outDisplay.value = "Not set";
        lockedSourceDisplay.value = `${context.sceneLabel}: ${context.videoPath}`;
      }
      catch (error) { notify(error.message, true); }
    };
    setOut.onclick = () => {
      try {
        const context = currentContext();
        if (!state.videoPath) throw new Error("Set Fix IN first so Face Fix can lock the repair scene.");
        if (state.segmentId && context.segmentId && state.segmentId !== context.segmentId) {
          throw new Error("Fix OUT must be inside the same scene as Fix IN.");
        }
        if (context.videoPath && context.videoPath !== state.videoPath) {
          throw new Error("Fix OUT must use the same rendered scene video as Fix IN.");
        }
        state.outTime = context.time;
        outDisplay.value = formatTime(state.outTime);
      }
      catch (error) { notify(error.message, true); }
    };
    rangeGrid.append(field("Fix IN", inDisplay), setIn, field("Fix OUT", outDisplay), setOut);

    const referenceRow = document.createElement("div");
    referenceRow.style.cssText = "display:grid;grid-template-columns:160px minmax(0,1fr);gap:10px;align-items:start;";
    const referencePreview = document.createElement("img");
    referencePreview.alt = "Face description frame";
    referencePreview.style.cssText = "width:160px;height:100px;object-fit:contain;border:1px solid #3f3f46;border-radius:7px;background:#050505;";
    const referenceActions = document.createElement("div");
    referenceActions.style.cssText = "display:flex;flex-direction:column;gap:7px;";
    const referenceDisplay = input("text", "Not set"); referenceDisplay.readOnly = true;
    const setReference = button("Use Playhead as Description Frame");
    setReference.onclick = async () => {
      try {
        state.referenceTime = currentContext().time;
        state.referenceImage = await options.captureCurrentFrame?.() || "";
        referenceDisplay.value = formatTime(state.referenceTime);
        if (state.referenceImage) referencePreview.src = state.referenceImage;
      } catch (error) { notify(String(error?.message || error), true); }
    };
    referenceActions.append(field("Description frame", referenceDisplay), setReference);
    referenceRow.append(referencePreview, referenceActions);

    const prompt = document.createElement("textarea");
    prompt.placeholder = "Generate a face description, then edit it here before processing.";
    prompt.style.cssText = "width:100%;min-height:110px;resize:vertical;box-sizing:border-box;border:1px solid #3f3f46;border-radius:7px;background:#18181b;color:#f4f4f5;padding:9px;font:12px/1.45 sans-serif;";
    const generatePrompt = button("Generate Face Description");
    generatePrompt.onclick = async () => {
      if (!state.referenceImage) return notify("Set a clear description frame first.", true);
      if (typeof options.generateFacePrompt !== "function") return notify("Face-description generation will be connected in the next Face Fix step.", true);
      try {
        generatePrompt.disabled = true;
        prompt.value = await options.generateFacePrompt({ ...state }) || prompt.value;
      } catch (error) { notify(String(error?.message || error), true); }
      finally { generatePrompt.disabled = false; }
    };

    const settings = document.createElement("div");
    settings.style.cssText = "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;";
    const confidence = input("number", "0.70", 0.1, 0.99, 0.01);
    const padding = input("number", "0.10", 0, 2, 0.01);
    const minimumFacePixels = input("number", "20", 4, 1024, 1);
    const rotationAssist = document.createElement("select");
    rotationAssist.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#f4f4f5;padding:7px 8px;font-size:12px;";
    [["off", "Off (fastest)"], ["light", "Light — ±15°"], ["strong", "Strong — ±15° and ±30°"]].forEach(([value, label]) => {
      const option = document.createElement("option"); option.value = value; option.textContent = label; rotationAssist.append(option);
    });
    rotationAssist.value = "light";
    const repairDistance = document.createElement("select");
    repairDistance.style.cssText = rotationAssist.style.cssText;
    [["all", "All detected faces"], ["very_far", "Very far faces only"], ["far", "Far faces (recommended)"], ["far_medium", "Far and medium faces"], ["custom", "Custom"]].forEach(([value, label]) => {
      const option = document.createElement("option"); option.value = value; option.textContent = label; repairDistance.append(option);
    });
    repairDistance.value = "far";
    const customDistanceThreshold = input("number", "9.0", 0.1, 50, 0.1);
    const feather = input("number", "18", 0, 256, 1);
    const colorMatch = input("number", "0.65", 0, 1, 0.05);
    const enhanceAmount = input("number", "8", 1, 20, 1);
    const anchorInterval = document.createElement("select");
    anchorInterval.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#f4f4f5;padding:7px 8px;font-size:12px;";
    [
      [8, "Maximum consistency — every 8 frames"],
      [16, "Balanced (recommended) — every 16 frames"],
      [24, "Faster — every 24 frames"],
      [32, "Long clip — every 32 frames"],
      [48, "Long clip faster — every 48 frames"],
      [64, "Very long clip — every 64 frames"],
      [96, "Very long clip faster — every 96 frames"],
      [120, "Minimum anchors — every 120 frames"],
    ].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = label;
      anchorInterval.append(option);
    });
    anchorInterval.value = "16";
    const customDistanceThresholdField = describedField(
      "Custom distance threshold (%)",
      customDistanceThreshold,
      "Repair is full two percentage points below this value and fades to unchanged at this value.",
    );
    const syncCustomDistanceThreshold = () => {
      customDistanceThresholdField.style.display = repairDistance.value === "custom" ? "flex" : "none";
    };
    repairDistance.addEventListener("change", syncCustomDistanceThreshold);
    syncCustomDistanceThreshold();
    settings.append(
      field("Detection confidence", confidence), field("Crop padding", padding),
      describedField("Minimum face pixels", minimumFacePixels, "Lower this for very distant faces. Raise it to reject tiny false detections. Recommended starting value: 20."),
      describedField("Rotation assist", rotationAssist, "Light scans ±15° for tilted faces. Strong also scans ±30° for difficult overhead angles but takes longer. Detection only; video frames are never rotated."),
      describedField("Repair distance", repairDistance, "Far fully repairs faces below 7% of frame width, fades from 7–9%, and leaves faces at 9% or larger unchanged. Close faces are excluded from anchor selection."),
      customDistanceThresholdField,
      field("Feather pixels", feather),
      field("Color match", colorMatch),
      describedField("Z-Image anchor enhance amount", enhanceAmount, "Lower values add more facial detail but can reduce character identity. Higher values preserve character identity more strongly but add less detail. Recommended starting range: 8–10."),
      describedField("Anchor interval", anchorInterval, "Smaller intervals improve consistency but require more Z-Image runs. Wider long-clip presets are faster but may carry less facial detail between anchors. First and final frames are always included.")
    );

    const advancedSummary = document.createElement("div");
    advancedSummary.textContent = "▶  Advanced LTX Settings";
    advancedSummary.role = "button";
    advancedSummary.tabIndex = 0;
    advancedSummary.setAttribute("aria-expanded", "false");
    advancedSummary.style.cssText = "display:block;border:1px solid #3f3f46;border-radius:8px;padding:9px 10px;color:#cffafe;font-size:12px;font-weight:900;background:#27272a;user-select:none;cursor:pointer;";
    const advancedGrid = document.createElement("div");
    advancedGrid.style.cssText = "display:none;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;border:1px solid #3f3f46;border-top:0;border-radius:0 0 8px 8px;background:#18181b;padding:10px;";
    const toggleAdvanced = () => {
      const expanded = advancedSummary.getAttribute("aria-expanded") !== "true";
      advancedSummary.setAttribute("aria-expanded", String(expanded));
      advancedSummary.textContent = `${expanded ? "▼" : "▶"}  Advanced LTX Settings`;
      advancedSummary.style.borderRadius = expanded ? "8px 8px 0 0" : "8px";
      advancedGrid.style.display = expanded ? "grid" : "none";
    };
    advancedSummary.addEventListener("click", toggleAdvanced);
    advancedSummary.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggleAdvanced();
    });
    const guidingStrength = input("number", "0.20", 0, 2, 0.05);
    const overlapCondStrength = input("number", "0.50", 0, 2, 0.05);
    const condImageStrength = input("number", "0.50", 0, 2, 0.05);
    const seed = input("number", "42", 0, Number.MAX_SAFE_INTEGER, 1);
    const sampler = document.createElement("select");
    sampler.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#f4f4f5;padding:7px 8px;font-size:12px;";
    ["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_sde", "dpmpp_2m_sde"].forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      sampler.append(option);
    });
    sampler.value = "euler_ancestral";
    const sigmas = input("text", "0.909375, 0.725, 0.421875, 0.0");
    advancedGrid.append(
      describedField("Original video guidance", guidingStrength, "How strongly LTX follows the original cropped video's motion and structure. Higher values preserve more of the source, including possible blur."),
      describedField("Temporal anchor blending", overlapCondStrength, "How strongly enhanced anchors carry across overlapping temporal tiles. Too high can cause pulsing or visible anchor transitions."),
      describedField("Enhanced anchor strength", condImageStrength, "How strongly the Z-Enhanced anchor images influence LTX. Lower this if anchors overpower expressions or motion."),
      describedField("Seed", seed, "Fixed LTX noise seed for repeatable results across the complete face-video run."),
      describedField("Sampler", sampler, "The LTX sampling algorithm. Euler ancestral is the tested default for this hidden workflow."),
      describedField("Sigma schedule", sigmas, "Advanced denoising schedule passed to ManualSigmas. Keep the tested default unless you understand the workflow's sampling schedule.")
    );
    const anchorEstimateRow = document.createElement("div");
    anchorEstimateRow.style.cssText = "display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px;align-items:center;";
    const calculateAnchors = button("Calculate Anchor Count");
    const anchorEstimate = input("text", "Choose a range, whole scene and preset, then calculate.");
    anchorEstimate.readOnly = true;
    anchorEstimateRow.append(calculateAnchors, anchorEstimate);
    const wholeSceneWrap = document.createElement("label");
    wholeSceneWrap.style.cssText = "display:flex;align-items:center;gap:8px;border:1px solid #155e75;border-radius:7px;background:#082f49;padding:9px 10px;color:#cffafe;font-size:12px;font-weight:800;cursor:pointer;";
    const wholeScene = document.createElement("input");
    wholeScene.type = "checkbox";
    const wholeSceneText = document.createElement("span");
    wholeSceneText.textContent = "Process the entire locked scene (ignore Fix IN and Fix OUT)";
    wholeSceneWrap.append(wholeScene, wholeSceneText);
    wholeScene.addEventListener("change", () => {
      state.wholeScene = Boolean(wholeScene.checked);
      if (!state.wholeScene) return;
      try {
        const context = currentContext();
        if (!context.videoPath) throw new Error("The visible preview scene has no rendered video.");
        state.videoPath = context.videoPath;
        state.segmentId = context.segmentId;
        state.sceneLabel = context.sceneLabel;
        lockedSourceDisplay.value = `${context.sceneLabel}: ${context.videoPath}`;
      } catch (error) {
        wholeScene.checked = false;
        state.wholeScene = false;
        notify(String(error?.message || error), true);
      }
    });

    const status = document.createElement("pre");
    status.textContent = "Ready to select a range.";
    status.style.cssText = "white-space:pre-wrap;margin:0;border:1px solid #27272a;border-radius:7px;background:#09090b;color:#a1a1aa;padding:9px;font:11px/1.4 monospace;";
    const comparison = document.createElement("div");
    comparison.style.cssText = "display:none;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;";
    const makeComparisonImage = (label, alt) => {
      const wrap = document.createElement("div");
      wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;align-items:center;";
      const title = document.createElement("div");
      title.textContent = label;
      title.style.cssText = "font-size:11px;font-weight:900;color:#a5f3fc;text-transform:uppercase;letter-spacing:.05em;";
      const image = document.createElement("img");
      image.alt = alt;
      image.style.cssText = "width:min(220px,100%);aspect-ratio:1/1;object-fit:contain;border:1px solid #155e75;border-radius:7px;background:#050505;";
      wrap.append(title, image);
      return { wrap, image };
    };
    const beforePreview = makeComparisonImage("Before", "Original detected face crop");
    const afterPreview = makeComparisonImage("After LTX", "LTX-enhanced face crop");
    comparison.append(beforePreview.wrap, afterPreview.wrap);
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:8px;";
    const preview = button("Preview Fix IN Frame");
    const previewSelected = button("Preview Selected Frame");
    const start = button("Start Face Fix", true);
    const jobPayload = () => ({
      video_path: state.videoPath || options.getVideoPath?.() || "",
      segment_id: state.segmentId,
      project_folder: options.getProjectFolder?.() || "",
      in_time: state.inTime,
      out_time: state.outTime,
      reference_time: state.referenceTime,
      prompt: prompt.value.trim(),
      seed: numericValue(seed, 42),
      confidence: Number(confidence.value || 0.7),
      crop_padding_factor: Number(padding.value || 0.1),
      minimum_face_pixels: numericValue(minimumFacePixels, 20),
      rotation_assist: String(rotationAssist.value || "light"),
      repair_distance: String(repairDistance.value || "far"),
      custom_distance_threshold: numericValue(customDistanceThreshold, 9.0),
      feather: Number(feather.value || 18),
      color_match: Number(colorMatch.value || 0.65),
      enhance_amount: numericValue(enhanceAmount, 8),
      anchor_interval: numericValue(anchorInterval, 16),
      ltx_guiding_strength: numericValue(guidingStrength, 0.2),
      ltx_temporal_overlap_cond_strength: numericValue(overlapCondStrength, 0.5),
      ltx_cond_image_strength: numericValue(condImageStrength, 0.5),
      ltx_sampler: String(sampler.value || "euler_ancestral"),
      ltx_sigmas: String(sigmas.value || "0.909375, 0.725, 0.421875, 0.0").trim(),
      enhance_size: 512,
      whole_scene: Boolean(state.wholeScene),
    });
    calculateAnchors.onclick = async () => {
      const payload = jobPayload();
      if (!payload.video_path) return notify("Set Fix IN or enable the whole-scene option to lock a video first.", true);
      if (!payload.whole_scene && (payload.in_time === null || payload.out_time === null || payload.out_time < payload.in_time)) {
        return notify("Set a valid Fix IN and Fix OUT range before calculating.", true);
      }
      if (typeof options.calculateAnchors !== "function") return notify("Anchor calculation is unavailable.", true);
      try {
        calculateAnchors.disabled = true;
        anchorEstimate.value = "Calculating from the locked video...";
        const result = await options.calculateAnchors(payload);
        anchorEstimate.value = `Up to ${result.anchor_count} anchors for ${result.frame_count} frames at ${Number(result.fps).toFixed(3)} FPS`;
        anchorEstimate.title = `Indices: ${result.anchor_indices_text}`;
      } catch (error) {
        anchorEstimate.value = "Could not calculate anchors.";
        notify(String(error?.message || error), true);
      } finally {
        calculateAnchors.disabled = false;
      }
    };
    const run = async (mode) => {
      const payload = jobPayload();
      if (mode === "frame") {
        if (state.inTime === null || !state.videoPath) return notify("Set Fix IN first, or use Preview Selected Frame.", true);
        payload.video_path = state.videoPath;
        payload.in_time = state.inTime;
        payload.out_time = state.inTime;
      } else if (mode === "selected_frame") {
        try {
          const context = currentContext();
          if (!context.videoPath) throw new Error("The visible preview scene has no rendered video.");
          payload.video_path = context.videoPath;
          payload.segment_id = context.segmentId;
          payload.in_time = context.time;
          payload.out_time = context.time;
          payload.whole_scene = false;
        } catch (error) {
          return notify(String(error?.message || error), true);
        }
      }
      if (!payload.video_path) return notify("Load or select a rendered scene video first.", true);
      if (!payload.whole_scene && (payload.in_time === null || payload.out_time === null || payload.out_time < payload.in_time)) return notify("Set a valid Fix IN and Fix OUT range, or enable Process the entire locked scene.", true);
      if (!payload.prompt) return notify("Generate or enter a face description first.", true);
      if (typeof options.startJob !== "function") return notify("Face processing will be connected in the next Face Fix step.", true);
      try {
        preview.disabled = previewSelected.disabled = start.disabled = true;
        comparison.style.display = "none";
        beforePreview.image.removeAttribute("src");
        afterPreview.image.removeAttribute("src");
        status.textContent = mode === "frame" || mode === "selected_frame" ? "Preparing one-frame enhancement preview..." : "Preparing face-fix range...";
        const executionMode = mode === "selected_frame" ? "frame" : mode;
        const result = await options.startJob(payload, executionMode, (message) => { status.textContent = message; });
        if (result?.crop_preview_data) {
          beforePreview.image.src = result.crop_preview_data;
        }
        if (result?.enhanced_preview_data) {
          afterPreview.image.src = result.enhanced_preview_data;
        }
        if (result?.crop_preview_data || result?.enhanced_preview_data) {
          comparison.style.display = "grid";
        }
        const enhancedLine = Number.isFinite(Number(result?.enhanced_count))
          ? `\nEnhanced ${result.enhanced_count}/${result?.anchor_count || 0} anchor(s) through Z-Enhance.`
          : "";
        const stageLine = result?.execution_stage === "complete"
          ? `\nRepaired ${result?.frames_repaired || 0} visible frame(s); faded ${result?.frames_faded || 0} short-gap frame(s); left ${result?.frames_skipped || 0} no-face frame(s) unchanged.`
          : (result?.execution_stage === "ltx_frames_ready"
            ? `\nLTX returned and validated ${result?.ltx_frame_count || 0} face-video frame(s).`
            : "");
        const outputLine = result?.output_video_path ? `\nRepaired scene video:\n${result.output_video_path}` : "";
        const anchorLine = Number.isFinite(Number(result?.anchor_count))
          ? `\nPrepared ${result.anchor_count} detected-face anchor(s) across ${result?.face_run_count || 0} visible run(s).`
          : "";
        const closeLine = Number(result?.close_skipped_frames || 0) > 0
          ? `\nLeft ${result.close_skipped_frames} close-face frame(s) unchanged by the distance preset.`
          : "";
        status.textContent = `Prepared ${result?.frame_count || 0} selected frame(s).${anchorLine}${enhancedLine}${stageLine}\nFrames ${result?.start_frame ?? "?"}–${result?.end_frame ?? "?"}.\nTracker faded across ${result?.carried_frames || 0} short-gap frame(s).\nLeft ${result?.skipped_frames || 0} no-face/unused frame(s) unchanged.${closeLine}${outputLine}\n\nJob: ${result?.job_id || "unknown"}`;
      } catch (error) {
        status.textContent = `Error: ${String(error?.message || error)}`;
        notify(status.textContent, true);
      } finally { preview.disabled = previewSelected.disabled = start.disabled = false; }
    };
    preview.onclick = () => run("frame");
    previewSelected.onclick = () => run("selected_frame");
    start.onclick = () => run("range");
    actions.append(preview, previewSelected, start);
    body.append(intro, field("Locked repair scene/video", lockedSourceDisplay), rangeGrid, wholeSceneWrap, referenceRow, generatePrompt, field("Editable face description", prompt), settings, anchorEstimateRow, advancedSummary, advancedGrid, comparison, status, actions);
    panel.append(header, body);
    document.body.append(panel);
  }

  function reset() {
    state.inTime = null;
    state.outTime = null;
    state.referenceTime = null;
    state.referenceImage = "";
    state.videoPath = "";
    state.segmentId = "";
    state.sceneLabel = "";
    state.wholeScene = false;
    if (windowElement) {
      windowElement.remove();
      windowElement = null;
    }
  }

  launchButton.onclick = open;
  return { button: launchButton, open, reset, state };
}
