import { api } from "../../../scripts/api.js";

let getContext = () => ({});
const recentErrors = [];
let pendingWrites = Promise.resolve();

export function secretValues(value) {
  if (!value || typeof value !== "object") return [];
  return Object.entries(value).flatMap(([key, item]) => {
    if (/api.?key|token|password|secret|authorization|credential/i.test(key) && typeof item === "string" && item) return [item];
    return item && typeof item === "object" ? secretValues(item) : [];
  });
}

export function redactErrorText(value, secrets = []) {
  let text = String(value || "");
  for (const secret of [...new Set(secrets)].sort((a, b) => b.length - a.length)) {
    if (secret) text = text.split(secret).join("[redacted]");
  }
  return text
    .replace(/(Bearer\s+)[^\s,'"<>]+/gi, "$1[redacted]")
    .replace(/((?:api[_-]?key|access[_-]?token|token|password|secret|authorization)["']?\s*[:=]\s*["']?)[^\s,"'&<>]+/gi, "$1[redacted]")
    .replace(/(https?:\/\/)[^/\s:@]+:[^/\s@]+@/g, "$1[redacted]@");
}

export function errorLogContext(payload = {}) {
  const current = getContext();
  return {
    ...current,
    project_folder: payload.project_folder || current.project_folder || "",
    scene: payload.scene_number ?? payload.scene_id ?? current.scene,
    mode: payload.video_prompt_type || payload.mode || current.mode,
    secrets: [...(current.secrets || []), ...secretValues(payload)],
  };
}

export function setErrorLogContext(callback) {
  getContext = callback;
  const reportUnhandled = event => {
    const error = event.error || event.reason;
    if (/comfyui-vrgamedevgirl|music_video_builder|storyboard_builder|VRGDG_/i.test(`${event.filename || ""}\n${error?.stack || ""}`)) {
      reportProjectError(error || event.message, { component: "Browser", stage: "Builder UI" }, callback());
    }
  };
  window.addEventListener("error", reportUnhandled);
  window.addEventListener("unhandledrejection", reportUnhandled);
  return () => {
    window.removeEventListener("error", reportUnhandled);
    window.removeEventListener("unhandledrejection", reportUnhandled);
    if (getContext === callback) getContext = () => ({});
  };
}

export function errorLogHeaders(context) {
  return context.project_folder ? {
    "X-VRGDG-Project": encodeURIComponent(context.project_folder),
    "X-VRGDG-Error-Context": encodeURIComponent(JSON.stringify({ scene: context.scene, mode: context.mode })),
  } : {};
}

async function logRequest(action, payload) {
  const response = await api.fetchApi(`/vrgdg/music_builder/error_log${action}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || `Error log request failed (${response.status})`);
  return data;
}

export function reportProjectError(error, metadata = {}, context = errorLogContext()) {
  const message = redactErrorText(error?.message || error, context.secrets);
  const entry = {
    scene: context.scene, mode: context.mode, source: "Frontend", component: "Video Builder",
    ...metadata, message,
    details: redactErrorText(metadata.details || error?.message || error, context.secrets),
    stack: redactErrorText(metadata.stack || error?.stack || "", context.secrets),
  };
  for (const key of ["stage", "mode", "source", "component"]) {
    if (entry[key]) entry[key] = redactErrorText(entry[key], context.secrets);
  }
  console.error("[VRGDG Builder]", message, entry.stack || "");
  if (!context.project_folder) return;
  const now = Date.now();
  while (recentErrors.length && (now - recentErrors[0].time > 5000 || recentErrors.length >= 50)) recentErrors.shift();
  if (recentErrors.some(item => item.project === context.project_folder
      && (item.message === message || (item.message && message.includes(item.message))))) return;
  recentErrors.push({ project: context.project_folder, message, time: now });
  if (metadata.logged) return;
  // Capture the project now; a queued append must never follow a later project switch.
  pendingWrites = pendingWrites.then(() => logRequest("/append", { project_folder: context.project_folder, entry }))
    .catch(() => console.warn("[VRGDG Builder] Could not write the project error log."));
}

export async function openProjectErrorLog(projectFolder, { makeButton, copyTextToClipboard }) {
  if (!projectFolder) {
    window.alert("Save or open a project before viewing its error log.");
    return;
  }
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100040;background:#0009;display:grid;place-items:center;";
  const panel = document.createElement("div");
  panel.style.cssText = "width:min(1000px,94vw);background:#18181b;color:#f4f4f5;border:1px solid #52525b;border-radius:10px;padding:18px;";
  const title = document.createElement("h3");
  title.textContent = "Project Error Log";
  const location = document.createElement("div");
  location.style.cssText = "font-size:12px;overflow-wrap:anywhere;margin-bottom:10px;";
  const text = document.createElement("textarea");
  text.readOnly = true;
  text.style.cssText = "box-sizing:border-box;width:100%;height:55vh;background:#09090b;color:#e4e4e7;padding:12px;font-family:monospace;white-space:pre;";
  const actions = document.createElement("div");
  actions.style.cssText = "display:flex;gap:8px;margin-top:12px;";
  const refresh = makeButton("Refresh");
  const copy = makeButton("Copy Log");
  const clear = makeButton("Clear Log", "danger");
  const close = makeButton("Close");
  const load = async (action = "") => {
    try {
      await pendingWrites;
      const data = await logRequest(action, { project_folder: projectFolder });
      location.textContent = data.path;
      text.value = data.text;
      text.scrollTop = text.scrollHeight;
    } catch (error) {
      location.textContent = `Could not access the error log: ${redactErrorText(error.message, errorLogContext().secrets)}`;
    }
  };
  refresh.onclick = () => load();
  copy.onclick = async () => {
    try {
      const copied = await copyTextToClipboard(text.value);
      if (!copied) throw new Error("Clipboard access failed.");
      copy.textContent = "Copied";
    } catch {
      location.textContent = "Could not copy the log. Select its text and copy it manually.";
    }
  };
  clear.onclick = () => {
    if (window.confirm("Clear this project's error log? Previous entries will be deleted.")) load("/clear");
  };
  close.onclick = () => backdrop.remove();
  actions.append(refresh, copy, clear, close);
  panel.append(title, location, text, actions);
  backdrop.append(panel);
  document.body.append(backdrop);
  await load();
}
