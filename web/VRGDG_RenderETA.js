// Estimate complete scene jobs, including preparation, decoding and cleanup.
const median = (values) => {
  const sorted = values.filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

export function renderETAProfile(engine, mode, settings) {
  const timingSettings = Object.keys(settings || {}).sort()
    .filter((key) => /steps|sampler|scheduler|width|height|megapixels|resolution|model|vae|turbo|cache|offload|precision|dtype|attention|upscale|chunk|tile|warmup|cooldown|continuity_mode|latent_context|audio_mode|ltx_version/.test(key))
    .map((key) => [key, settings[key]]);
  return JSON.stringify([engine, mode, timingSettings]);
}

export function formatRenderETA(milliseconds) {
  if (milliseconds == null || !Number.isFinite(milliseconds)) return "Estimating…";
  if (milliseconds < 0) return "Re-estimating…";
  const seconds = Math.ceil(milliseconds / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60) % 60;
  return `${hours ? `${String(hours).padStart(2, "0")}:` : ""}${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

export function estimateRenderETA(log, history = [], now = Date.now()) {
  const result = { sceneMs: null, totalMs: null, stitching: false, stitchUnknown: false };
  if (!log || log.status !== "running") return result;
  if (!Array.isArray(log.eta_plan)) return result;
  const scenes = log.scenes || [];
  const active = scenes.find((scene) => scene.status === "running");
  const finishedIds = new Set(scenes.filter((scene) => scene.status !== "running").map((scene) => scene.scene_id));
  const pending = (log.eta_plan || []).filter((scene) => !finishedIds.has(scene.scene_id));
  const compatible = history.filter((item) => item.id !== log.id && item.video_engine === log.video_engine);
  const samples = [...compatible.flatMap((item) => item.scenes || []), ...scenes]
    .filter((scene) => scene.status === "complete" && Number(scene.total_ms) > 0);
  function expected(scene) {
    const matches = samples.filter((sample) => sample.eta_profile === scene.eta_profile && sample.eta_duration > 0).slice(-12);
    if (matches.length) {
      return median(matches.map((sample) => Number(sample.total_ms) / sample.eta_duration)) * scene.eta_duration;
    }
    // Older logs have no settings or duration; use their matching-mode average only as a rough fallback.
    return median(samples.filter((sample) => !sample.eta_profile && sample.video_mode === scene.video_mode).slice(-12).map((sample) => Number(sample.total_ms)));
  }
  let total = 0;
  for (const scene of pending) {
    let remaining = expected(scene);
    if (active?.scene_id === scene.scene_id && remaining != null) {
      remaining -= Math.max(0, now - Date.parse(active.started_at));
      if (remaining <= 0) remaining = -1;
    }
    if (active?.scene_id === scene.scene_id) result.sceneMs = remaining;
    if (remaining == null || remaining < 0) total = null;
    else if (total != null) total += remaining;
  }
  result.stitching = Boolean(log.stitch_started_at);
  if (!log.skip_final_stitch) {
    const rates = compatible.filter((item) => item.status === "complete" && item.stitch_ms > 0 && item.eta_video_duration > 0)
      .slice(-8).map((item) => item.stitch_ms / item.eta_video_duration);
    let stitch = median(rates);
    if (stitch != null) {
      stitch *= log.eta_video_duration;
    } else {
      stitch = median(compatible.filter((item) => item.status === "complete" && !item.eta_video_duration)
        .slice(-8).map((item) => Number(item.stitch_ms)));
    }
    if (stitch != null && result.stitching) stitch -= Math.max(0, now - Date.parse(log.stitch_started_at));
    if (stitch != null && stitch <= 0) stitch = -1;
    result.stitchUnknown = stitch == null;
    if ((stitch == null && result.stitching) || (stitch != null && stitch < 0)) total = null;
    else if (stitch != null && total != null) total += stitch;
  }
  result.totalMs = total;
  return result;
}
