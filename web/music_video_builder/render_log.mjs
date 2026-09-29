import { estimateRenderETA, formatRenderETA, renderETAProfile } from "../VRGDG_RenderETA.js";
import { postJson } from "./comfy_api.mjs";
import { escapeHtml, makeButton, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { cloneI2VVideoSettings } from "./model_settings.mjs";

function normalizeRenderLog(raw) {
  const value = raw && typeof raw === "object" ? raw : {};
  return {
    ...value,
    id: String(value.id || ""),
    status: String(value.status || "unknown"),
    scenes: Array.isArray(value.scenes)
      ? value.scenes.filter((scene) => scene && typeof scene === "object").map((scene) => ({ ...scene }))
      : [],
    summary: value.summary && typeof value.summary === "object" ? { ...value.summary } : {},
  };
}

export function normalizeRenderLogs(items) {
  const byId = new Map();
  for (const raw of (Array.isArray(items) ? items : [])) {
    const log = normalizeRenderLog(raw);
    if (log.id) byId.set(log.id, log);
  }
  return [...byId.values()]
    .sort((left, right) => String(left.started_at || "").localeCompare(String(right.started_at || "")))
    .slice(-20);
}

export function renderLogDuration(milliseconds) {
  const totalSeconds = Math.max(0, Math.round(Number(milliseconds || 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  if (minutes) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  return `${seconds}s`;
}

export function updateRenderLogSummary(log, now = Date.now()) {
  if (!log) return {};
  const scenes = Array.isArray(log.scenes) ? log.scenes : [];
  const startedMs = Date.parse(log.started_at || "") || now;
  const endedMs = Date.parse(log.ended_at || "") || (log.status === "running" ? now : startedMs);
  const totalMs = log.status === "running"
    ? Math.max(0, now - startedMs)
    : Math.max(0, Number(log.total_ms || endedMs - startedMs));
  const completed = scenes.filter((scene) => scene.status === "complete");
  const renderMs = completed.reduce((sum, scene) => sum + Math.max(0, Number(scene.render_ms || 0)), 0);
  const betweenRenderMs = scenes.reduce((sum, scene) => sum + Math.max(0, Number(scene.gap_before_render_ms || 0)), 0);
  const setupMs = Math.max(0, Number(log.setup_ms || 0));
  const stitchMs = Math.max(0, Number(log.stitch_ms || 0));
  const averageRenderMs = completed.length ? renderMs / completed.length : 0;
  const completedTotalMs = completed.reduce((sum, scene) => sum + Math.max(0, Number(scene.total_ms || 0)), 0);
  const averageSceneStepMs = completed.length ? completedTotalMs / completed.length : 0;
  const targetScenes = Math.max(0, Number(log.target_scene_count || scenes.length || 0));
  const remainingScenes = Math.max(0, targetScenes - completed.length);
  const etaMs = log.status === "running" && completed.length
    ? remainingScenes * averageSceneStepMs
    : 0;
  const longest = completed.reduce((best, scene) =>
    Number(scene.render_ms || 0) > Number(best?.render_ms || 0) ? scene : best, null);
  log.total_ms = totalMs;
  log.summary = {
    total_ms: totalMs,
    render_ms: renderMs,
    between_render_ms: betweenRenderMs,
    setup_ms: setupMs,
    stitch_ms: stitchMs,
    overhead_ms: Math.max(0, totalMs - renderMs - stitchMs),
    completed_scenes: completed.length,
    failed_scenes: scenes.filter((scene) => scene.status === "failed").length,
    target_scenes: targetScenes,
    skipped_existing_scenes: Math.max(0, Number(log.skipped_existing_count || 0)),
    average_render_ms: averageRenderMs,
    average_scene_step_ms: averageSceneStepMs,
    eta_ms: etaMs,
    longest_scene_label: longest?.label || "",
    longest_scene_ms: Math.max(0, Number(longest?.render_ms || 0)),
  };
  return log.summary;
}

export function createRenderLog({
  builderETA, builderETAState, builderFullETA, builderSceneETA, currentVideoMode, miniMaxH3ModeForSegment,
  miniMaxH3SettingsForSegment, overlay, positionBuilderETA, projectInput, state,
}) {
  function renderETAScene(segment) {
    const engine = normalizeProjectVideoEngine(state.projectVideoEngine);
    const miniMax = engine === "minimax_h3";
    const mode = miniMax ? miniMaxH3ModeForSegment(segment) : currentVideoMode();
    const settings = miniMax ? miniMaxH3SettingsForSegment(segment)
      : cloneI2VVideoSettings(segment.use_scene_i2v_video_settings ? segment.i2v_video_settings : state.i2vVideoSettings);
    return {
      scene_id: String(segment.id), video_mode: mode,
      eta_duration: Math.max(0.05, Number(segment.end) - Number(segment.start)),
      eta_profile: renderETAProfile(engine, mode, settings),
    };
  }

  function startSingleSceneETA(segment) {
    const plan = renderETAScene(segment);
    const started = new Date().toISOString();
    const log = {
      id: `render_single_${Date.now()}`, status: "running", scene_scope: "single", mode_label: "Render Scene",
      video_engine: normalizeProjectVideoEngine(state.projectVideoEngine), video_mode: plan.video_mode,
      started_at: started, skip_final_stitch: true, target_scene_count: 1, eta_plan: [plan],
      scenes: [{ ...plan, label: segment.label || "Scene", status: "running", started_at: started }],
    };
    startBuilderETA(log);
    return log;
  }

  async function persistRenderLog(log) {
    if (!log?.id) return null;
    upsertRenderLog(log);
    const projectFolder = String(state.projectFolder || projectInput.value || "").trim();
    if (!projectFolder) return null;
    try {
      const data = await postJson("/vrgdg/music_builder/save_render_log", {
        project_folder: projectFolder,
        log,
      }, 60000);
      if (data.log) Object.assign(log, data.log);
      if (data.report_json_path) log.report_json_path = data.report_json_path;
      if (data.report_text_path) log.report_text_path = data.report_text_path;
      log.persistence_error = "";
      upsertRenderLog(log);
      return data;
    } catch (error) {
      log.persistence_error = String(error?.message || error);
      console.warn("[VRGDG Music Builder] Could not persist render log:", error);
      upsertRenderLog(log);
      return null;
    }
  }

  function renderLogAsText(log) {
    const summary = updateRenderLogSummary(log);
    const lines = [
      "VRGDG Video Builder Render Log",
      "================================",
      `Session: ${log.id || ""}`,
      `Status: ${String(log.status || "unknown").toUpperCase()}`,
      `Project: ${log.project_folder || state.projectFolder || ""}`,
      `Mode: ${log.mode_label || log.scene_scope || "Render All"}`,
      `Started: ${log.started_at || ""}`,
      `Finished: ${log.ended_at || ""}`,
      "",
      "Summary",
      "--------------------------------",
      `Total wall time: ${renderLogDuration(summary.total_ms)}`,
      `Active scene rendering: ${renderLogDuration(summary.render_ms)}`,
      `Between-render time: ${renderLogDuration(summary.between_render_ms)}`,
      `Setup time: ${renderLogDuration(summary.setup_ms)}`,
      `Final stitching: ${renderLogDuration(summary.stitch_ms)}`,
      `Other overhead: ${renderLogDuration(summary.overhead_ms)}`,
      `Scenes completed: ${summary.completed_scenes}/${summary.target_scenes}`,
      `Existing scenes skipped: ${summary.skipped_existing_scenes}`,
      `Average render per scene: ${renderLogDuration(summary.average_render_ms)}`,
    ];
    if (summary.longest_scene_label) {
      lines.push(`Longest scene: ${summary.longest_scene_label} — ${renderLogDuration(summary.longest_scene_ms)}`);
    }
    if (log.final_video_path) lines.push(`Final video: ${log.final_video_path}`);
    if (log.error) lines.push("", `Error: ${log.error}`);
    lines.push("", "Scene Details", "--------------------------------");
    for (const scene of (log.scenes || [])) {
      lines.push(
        `${scene.label || `Scene ${scene.scene_number || "?"}`} [${String(scene.status || "pending").toUpperCase()}]`,
        `  Total scene step: ${renderLogDuration(scene.total_ms)}`,
        `  Preparation: ${renderLogDuration(scene.preparation_ms)}`,
        `  Video render: ${renderLogDuration(scene.render_ms)}`,
        `  Post-processing/cleanup: ${renderLogDuration(scene.post_ms)}`,
        `  Time since previous render: ${renderLogDuration(scene.gap_before_render_ms)}`,
      );
      if (scene.video_path) lines.push(`  Video: ${scene.video_path}`);
      if (scene.error) lines.push(`  Error: ${scene.error}`);
    }
    return lines.join("\n");
  }

  function downloadRenderLog(log, kind = "json") {
    if (!log) return;
    updateRenderLogSummary(log);
    const isJson = kind === "json";
    const content = isJson ? `${JSON.stringify(log, null, 2)}\n` : `${renderLogAsText(log)}\n`;
    const blob = new Blob([content], { type: isJson ? "application/json" : "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${log.id || "VRGDG_Render_Log"}.${isJson ? "json" : "txt"}`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function openRenderLogModal() {
    document.querySelector(".vrgdg-render-log-modal")?.remove();
    let selectedId = state.activeRenderLogId || state.renderLogs[state.renderLogs.length - 1]?.id || "";
    const backdrop = document.createElement("div");
    backdrop.className = "vrgdg-render-log-modal";
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:18px;box-sizing:border-box;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(1180px,calc(100vw - 36px));height:min(900px,calc(100vh - 36px));border:1px solid #155e75;border-radius:10px;background:#0b1220;color:#e2e8f0;box-shadow:0 28px 90px rgba(0,0,0,.72);display:flex;flex-direction:column;overflow:hidden;font-family:Arial,sans-serif;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid #334155;background:#0f172a;";
    const title = document.createElement("div");
    title.textContent = "Render Log";
    title.style.cssText = "font-size:17px;font-weight:900;color:#cffafe;";
    const close = makeButton("Close");
    header.append(title, close);
    const toolbar = document.createElement("div");
    toolbar.style.cssText = "display:grid;grid-template-columns:minmax(220px,1fr) auto auto auto auto;gap:8px;padding:10px 14px;border-bottom:1px solid #334155;align-items:center;";
    const history = document.createElement("select");
    history.style.cssText = "width:100%;border:1px solid #475569;border-radius:6px;background:#020617;color:#f8fafc;padding:8px;";
    const openSaved = makeButton("Open Saved Report");
    const copy = makeButton("Copy Summary");
    const downloadJson = makeButton("Download JSON");
    const downloadText = makeButton("Download Text");
    toolbar.append(history, openSaved, copy, downloadJson, downloadText);
    const content = document.createElement("div");
    content.style.cssText = "flex:1 1 auto;min-height:0;overflow:auto;padding:14px;";
    box.append(header, toolbar, content);
    backdrop.append(box);
    document.body.append(backdrop);

    const selectedLog = () => state.renderLogs.find((item) => item.id === selectedId) || state.renderLogs[state.renderLogs.length - 1] || null;
    const refresh = () => {
      const logs = normalizeRenderLogs(state.renderLogs).slice().reverse();
      const previous = history.value || selectedId;
      history.textContent = "";
      for (const log of logs) {
        const option = document.createElement("option");
        option.value = log.id;
        option.textContent = `${log.status === "running" ? "● LIVE — " : ""}${log.started_at ? new Date(log.started_at).toLocaleString() : log.id} — ${String(log.status || "").toUpperCase()}`;
        history.append(option);
      }
      if (logs.some((log) => log.id === previous)) selectedId = previous;
      else selectedId = logs[0]?.id || "";
      history.value = selectedId;
      const log = selectedLog();
      if (!log) {
        content.innerHTML = '<div style="border:1px dashed #475569;border-radius:8px;padding:30px;text-align:center;color:#94a3b8;">No Render All sessions have been recorded for this project yet.</div>';
        openSaved.disabled = true;
        copy.disabled = true;
        downloadJson.disabled = true;
        downloadText.disabled = true;
        return;
      }
      const summary = updateRenderLogSummary(log);
      openSaved.disabled = !log.report_text_path;
      copy.disabled = false;
      downloadJson.disabled = false;
      downloadText.disabled = false;
      const statusColor = log.status === "complete" ? "#86efac" : log.status === "running" ? "#67e8f9" : "#fca5a5";
      const card = (label, value) => `<div style="border:1px solid #334155;border-radius:8px;background:#111827;padding:10px;"><div style="color:#64748b;font-size:10px;font-weight:900;text-transform:uppercase;">${escapeHtml(label)}</div><div style="margin-top:4px;color:#f8fafc;font-size:16px;font-weight:900;">${escapeHtml(value)}</div></div>`;
      const rows = (log.scenes || []).map((scene) => `
        <tr>
          <td>${escapeHtml(scene.label || `Scene ${scene.scene_number || "?"}`)}</td>
          <td style="color:${scene.status === "complete" ? "#86efac" : scene.status === "failed" ? "#fca5a5" : "#67e8f9"};font-weight:900;">${escapeHtml(String(scene.status || "pending").toUpperCase())}</td>
          <td>${escapeHtml(renderLogDuration(scene.render_ms))}</td>
          <td>${escapeHtml(renderLogDuration(scene.preparation_ms))}</td>
          <td>${escapeHtml(renderLogDuration(scene.post_ms))}</td>
          <td>${escapeHtml(renderLogDuration(scene.gap_before_render_ms))}</td>
          <td>${escapeHtml(renderLogDuration(scene.total_ms))}</td>
        </tr>`).join("");
      content.innerHTML = `
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px;">
          <div><div style="font-size:18px;font-weight:900;color:${statusColor};">${escapeHtml(String(log.status || "unknown").toUpperCase())}</div><div style="color:#94a3b8;font-size:11px;margin-top:4px;">${escapeHtml(log.mode_label || "Render All")} · Started ${escapeHtml(log.started_at ? new Date(log.started_at).toLocaleString() : "")}</div></div>
          ${log.status === "running" && summary.eta_ms ? `<div style="border:1px solid #0e7490;border-radius:7px;background:#083344;padding:8px 10px;color:#a5f3fc;font-weight:900;">Estimated remaining: ${escapeHtml(renderLogDuration(summary.eta_ms))}</div>` : ""}
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:9px;">
          ${card("Total wall time", renderLogDuration(summary.total_ms))}
          ${card("Active rendering", renderLogDuration(summary.render_ms))}
          ${card("Between renders", renderLogDuration(summary.between_render_ms))}
          ${card("Final stitching", renderLogDuration(summary.stitch_ms))}
          ${card("Setup", renderLogDuration(summary.setup_ms))}
          ${card("Other overhead", renderLogDuration(summary.overhead_ms))}
          ${card("Scenes completed", `${summary.completed_scenes}/${summary.target_scenes}`)}
          ${card("Average render", renderLogDuration(summary.average_render_ms))}
        </div>
        ${summary.longest_scene_label ? `<div style="margin-top:10px;color:#cbd5e1;font-size:11px;">Longest scene: <strong>${escapeHtml(summary.longest_scene_label)}</strong> — ${escapeHtml(renderLogDuration(summary.longest_scene_ms))}</div>` : ""}
        ${log.error ? `<div style="margin-top:12px;border:1px solid #7f1d1d;border-radius:7px;background:#2a0b0b;color:#fecaca;padding:10px;white-space:pre-wrap;">${escapeHtml(log.error)}</div>` : ""}
        <div style="margin-top:14px;border:1px solid #334155;border-radius:8px;overflow:auto;">
          <table style="width:100%;border-collapse:collapse;font-size:11px;">
            <thead style="position:sticky;top:0;background:#083344;color:#cffafe;"><tr><th>Scene</th><th>Status</th><th>Render</th><th>Prep</th><th>Post/Cleanup</th><th>Between</th><th>Total Step</th></tr></thead>
            <tbody>${rows || '<tr><td colspan="7" style="padding:18px;text-align:center;color:#64748b;">Waiting for the first scene...</td></tr>'}</tbody>
          </table>
        </div>
        <style>.vrgdg-render-log-modal th,.vrgdg-render-log-modal td{padding:8px 9px;border-bottom:1px solid #1e293b;text-align:left;white-space:nowrap}.vrgdg-render-log-modal tbody tr:hover{background:#111827}</style>
        <div style="margin-top:12px;border:1px solid #334155;border-radius:7px;background:#020617;padding:9px;color:#94a3b8;font-size:10px;line-height:1.45;white-space:pre-wrap;">JSON: ${escapeHtml(log.report_json_path || "Not saved yet")}\nText: ${escapeHtml(log.report_text_path || "Not saved yet")}${log.persistence_error ? `\nSave warning: ${escapeHtml(log.persistence_error)}` : ""}</div>`;
    };
    const finish = () => {
      clearInterval(timer);
      if (state.renderLogModalRefresh === refresh) state.renderLogModalRefresh = null;
      backdrop.remove();
    };
    state.renderLogModalRefresh = refresh;
    const timer = setInterval(refresh, 1000);
    history.onchange = () => { selectedId = history.value; refresh(); };
    close.onclick = finish;
    backdrop.onpointerdown = (event) => { if (event.target === backdrop) finish(); };
    openSaved.onclick = async () => {
      const log = selectedLog();
      if (log?.report_text_path) await postJson("/vrgdg/music_builder/open_local_file", { path: log.report_text_path }, 30000);
    };
    copy.onclick = async () => {
      const log = selectedLog();
      if (!log) return;
      try {
        await navigator.clipboard.writeText(renderLogAsText(log));
        toast("Render Log summary copied.");
      } catch (error) {
        toast(`Could not copy Render Log:\n${String(error?.message || error)}`, true);
      }
    };
    downloadJson.onclick = () => downloadRenderLog(selectedLog(), "json");
    downloadText.onclick = () => downloadRenderLog(selectedLog(), "text");
    refresh();
  }

  function refreshBuilderETA() {
    if (!builderETAState.log || !overlay.isConnected) return;
    builderETA.style.display = "block";
    const log = builderETAState.log;
    const fullLabel = log.scene_scope === "selected" ? "Selected Scenes" : log.scene_scope === "single" ? "This Render" : "Full Video";
    if (log.status !== "running") {
      const status = log.status === "complete" ? "Done" : log.status === "canceled" ? "Stopped" : "Finished with errors";
      builderSceneETA.textContent = `Current Scene: ${status}`;
      builderFullETA.textContent = `${fullLabel}: ${status}`;
      clearInterval(builderETAState.timer);
    } else {
      const eta = estimateRenderETA(log, state.renderLogs);
      const active = log.scenes.find((scene) => scene.status === "running");
      builderSceneETA.textContent = `Current Scene: ${state.batchCancelled ? "Stopping…" : eta.stitching ? "Done" : active ? formatRenderETA(eta.sceneMs) : "Preparing…"}`;
      builderFullETA.textContent = `${fullLabel}: ${state.batchCancelled ? "Stopping…" : eta.stitching && eta.totalMs == null ? "Stitching…" : formatRenderETA(eta.totalMs)}${eta.stitchUnknown && eta.totalMs != null ? " + stitch" : ""}`;
      builderETA.title = "Approximate remaining time based on completed scene jobs, including preparation, rendering and cleanup. Updates every second. Different hardware and workload can change the estimate."
        + (eta.stitchUnknown ? " Final stitching is not timed yet; + stitch excludes that step." : "")
        + (log.scene_scope === "selected" || log.scene_scope === "single" ? " Only this render selection is included." : "");
    }
    positionBuilderETA();
  }

  function resetBuilderETA() {
    clearInterval(builderETAState.timer);
    builderETAState.log = null;
    builderETA.style.display = "none";
  }

  function startBuilderETA(log) {
    clearInterval(builderETAState.timer);
    builderETAState.log = log;
    refreshBuilderETA();
    builderETAState.timer = setInterval(refreshBuilderETA, 1000);
  }

  async function finishSingleSceneETA(log, status) {
    const now = Date.now();
    log.status = status;
    log.ended_at = new Date(now).toISOString();
    Object.assign(log.scenes[0], { status, ended_at: log.ended_at, total_ms: now - Date.parse(log.started_at) });
    await persistRenderLog(log);
    if (builderETAState.log === log) refreshBuilderETA();
  }

  function upsertRenderLog(log) {
    if (!log?.id) return;
    updateRenderLogSummary(log);
    const logs = normalizeRenderLogs(state.renderLogs);
    const index = logs.findIndex((item) => item.id === log.id);
    if (index >= 0) logs[index] = log;
    else logs.push(log);
    state.renderLogs = logs.slice(-20);
    state.activeRenderLogId = log.id;
    state.renderLogModalRefresh?.();
    if (builderETAState.log?.id === log.id) { builderETAState.log = log; refreshBuilderETA(); }
  }

  return {
    finishSingleSceneETA, openRenderLogModal, persistRenderLog, renderETAScene, resetBuilderETA,
    startBuilderETA, startSingleSceneETA, upsertRenderLog,
  };
}
