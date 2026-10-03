import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

let builderAutomaticMemoryCleanupEnabled = false;
const EMBEDDED_MEMORY_CLEANUP_NODE_TYPES = new Set([
  "ramcleanup",
  "vramcleanup",
  "vrgdg_unloadgemmamodels",
]);

export function setBuilderAutomaticMemoryCleanupEnabled(value) {
  builderAutomaticMemoryCleanupEnabled = Boolean(value);
  return builderAutomaticMemoryCleanupEnabled;
}

function withoutEmbeddedMemoryCleanupNodes(prompt) {
  if (!prompt || typeof prompt !== "object" || Array.isArray(prompt)) {
    return { prompt, removed: [] };
  }
  const cloned = Object.fromEntries(Object.entries(prompt).map(([nodeId, node]) => [
    nodeId,
    node && typeof node === "object"
      ? { ...node, inputs: node.inputs && typeof node.inputs === "object" ? { ...node.inputs } : node.inputs }
      : node,
  ]));
  const cleanupIds = new Set(Object.entries(cloned)
    .filter(([, node]) => EMBEDDED_MEMORY_CLEANUP_NODE_TYPES.has(String(node?.class_type || "").trim().toLowerCase()))
    .map(([nodeId]) => String(nodeId)));
  if (!cleanupIds.size) return { prompt: cloned, removed: [] };

  const resolvePassthrough = (value, visited = new Set()) => {
    if (!Array.isArray(value) || value.length < 2) return value;
    const sourceId = String(value[0]);
    if (!cleanupIds.has(sourceId)) return value;
    if (visited.has(sourceId)) return null;
    visited.add(sourceId);
    const passthrough = cloned[sourceId]?.inputs?.anything;
    return Array.isArray(passthrough) ? resolvePassthrough(passthrough, visited) : null;
  };

  Object.entries(cloned).forEach(([nodeId, node]) => {
    if (cleanupIds.has(String(nodeId)) || !node?.inputs || typeof node.inputs !== "object") return;
    Object.entries(node.inputs).forEach(([inputName, value]) => {
      if (!Array.isArray(value) || value.length < 2 || !cleanupIds.has(String(value[0]))) return;
      const passthrough = resolvePassthrough(value);
      if (passthrough) node.inputs[inputName] = passthrough;
      else delete node.inputs[inputName];
    });
  });
  cleanupIds.forEach((nodeId) => delete cloned[nodeId]);
  return { prompt: cloned, removed: [...cleanupIds] };
}

export function normalizeOwnServerTimeoutMinutes(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 6;
  return Math.max(1, Math.min(6, Math.round(parsed)));
}

function ownServerPayloadTimeoutMs(payload) {
  if (!payload || typeof payload !== "object") return 0;
  const runner = String(payload.text_runner || payload.text_gemma_runner || "").trim().toLowerCase();
  const timeoutSec = Number(payload.own_server_timeout);
  const hasTimeout = Number.isFinite(timeoutSec) && timeoutSec > 0;
  if (runner !== "own_server" && runner !== "own-server" && runner !== "custom_openai" && runner !== "openai_compatible" && !hasTimeout) return 0;
  const seconds = hasTimeout ? timeoutSec : 360;
  return Math.round(Math.max(15, Math.min(600, seconds)) * 1000) + 15000;
}

export async function postJson(url, payload, timeoutMs = 120000) {
  const ownTimeoutMs = ownServerPayloadTimeoutMs(payload);
  if (ownTimeoutMs) timeoutMs = Math.max(timeoutMs, ownTimeoutMs);
  const controller = new AbortController();
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await api.fetchApi(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data?.ok) {
      const failure = new Error(String(data?.error || `Request failed (${response.status})`));
      // Callers that need more than the message (for example a 409 "already exists") read these.
      failure.status = response.status;
      failure.data = data;
      throw failure;
    }
    return data;
  } catch (error) {
    if (timedOut || controller.signal.aborted || error?.name === "AbortError") {
      const timeoutSeconds = Math.max(1, Math.round(timeoutMs / 1000));
      const timeoutAmount = timeoutSeconds >= 60 ? Math.round(timeoutSeconds / 60) : timeoutSeconds;
      const timeoutUnit = timeoutSeconds >= 60 ? "minute" : "second";
      throw new Error(`Request timed out after ${timeoutAmount} ${timeoutUnit}${timeoutAmount === 1 ? "" : "s"}. The backend may still be processing it.`);
    }
    const message = String(error?.message || error || "");
    if (/NetworkError|Failed to fetch|fetch resource|Load failed/i.test(message)) {
      throw new Error("Connection to the ComfyUI backend was lost. Check that ComfyUI is still running and inspect its console. If this happened while loading a local LLM, lower its GPU layers or context limit and try again.");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

// Session snapshots must commit in the same order they were created. Lyric
// edits and performer assignments can trigger autosave while Quick Save is
// also in flight; without this queue an older snapshot can finish last and
// restore stale lyrics/cast assignments.
let builderSessionSaveQueue = Promise.resolve();
let builderSessionSaveRevision = 0;

export function saveBuilderSessionJson(payload, timeoutMs = 60000) {
  const revision = ++builderSessionSaveRevision;
  payload.session = {
    ...(payload.session || {}),
    builder_save_revision: revision,
  };
  const save = () => postJson("/vrgdg/music_builder/save_session", payload, timeoutMs);
  const result = builderSessionSaveQueue.then(save, save);
  builderSessionSaveQueue = result.catch(() => null);
  return result;
}

export function syncBuilderSessionSaveRevision(revision) {
  builderSessionSaveRevision = Math.max(builderSessionSaveRevision, Number(revision || 0) || 0);
}

export const GEMMA_VIDEO_PROMPT_TIMEOUT_MS = 600000;
export const GEMMA_VIDEO_ENHANCE_TIMEOUT_MS = 300000;

export async function getJson(url) {
  const response = await api.fetchApi(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) throw new Error(String(data?.error || `Request failed (${response.status})`));
  return data;
}

export function makeImageViewUrl(image) {
  const params = new URLSearchParams();
  params.set("filename", image.filename || "");
  params.set("type", image.type || "output");
  if (image.subfolder) params.set("subfolder", image.subfolder);
  params.set("rand", String(Date.now()));
  return `/view?${params.toString()}`;
}

export function makeEditorImageUrl(path) {
  return `/vrgdg/video_editor/image?path=${encodeURIComponent(path)}&rand=${Date.now()}`;
}

const EDITOR_THUMBNAIL_SESSION_VERSION = Date.now();
let editorThumbnailRefreshCounter = 0;
const editorThumbnailVersionByPath = new Map();

function editorThumbnailPathKey(path) {
  return String(path || "").trim().replace(/\//g, "\\").toLowerCase();
}

export function refreshEditorThumbnailUrl(path) {
  const key = editorThumbnailPathKey(path);
  if (!key) return;
  editorThumbnailRefreshCounter += 1;
  editorThumbnailVersionByPath.set(key, `${Date.now()}_${editorThumbnailRefreshCounter}`);
}

export function makeEditorThumbnailUrl(path) {
  const key = editorThumbnailPathKey(path);
  const version = editorThumbnailVersionByPath.get(key) || EDITOR_THUMBNAIL_SESSION_VERSION;
  return `/vrgdg/video_editor/image?path=${encodeURIComponent(path)}&thumbv=${encodeURIComponent(version)}`;
}

export function makeEditorVideoUrl(path, bust) {
  // A stable bust value (e.g. segment.video_cache_bust, which only changes when the
  // scene is re-rendered) lets the browser reuse a cached fetch of the same clip
  // across preload/playback instead of re-downloading it from disk on every cut.
  const version = bust === undefined || bust === null || bust === "" ? Date.now() : bust;
  return `/vrgdg/video_editor/video?path=${encodeURIComponent(path)}&rand=${version}`;
}

function extractImagesFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const images = [];
  for (const output of Object.values(outputs)) {
    if (Array.isArray(output?.images)) {
      for (const image of output.images) images.push(image);
    }
  }
  return images;
}

function extractPromptErrorFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const messages = [];
  const status = root?.status || {};
  if (status.status_str && !/success|completed/i.test(String(status.status_str))) {
    messages.push(`status: ${status.status_str}`);
  }
  const candidates = [
    root?.error,
    status?.error,
    status?.exception_message,
    status?.message,
    ...(Array.isArray(status?.messages) ? status.messages : []),
  ];
  const visit = (value) => {
    if (value == null) return;
    if (typeof value === "string") {
      if (value.trim() && !/execution_(start|cached|success)/i.test(value)) messages.push(value.trim());
      return;
    }
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    if (typeof value === "object") {
      for (const key of ["exception_message", "error", "message", "node_id", "node_type", "class_type"]) {
        if (value[key] != null) visit(value[key]);
      }
    }
  };
  candidates.forEach(visit);
  return [...new Set(messages)].join("\n");
}

function promptHistoryFinished(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  if (!root || !Object.keys(root || {}).length) return false;
  const status = String(root?.status?.status_str || "").toLowerCase();
  if (status) return /success|completed|error|failed/i.test(status);
  return Boolean(root?.outputs);
}

function extractTextFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const values = [];
  for (const output of Object.values(outputs)) {
    const text = output?.text ?? output?.ui?.text;
    if (Array.isArray(text)) values.push(...text);
    else if (text != null) values.push(text);
  }
  return values.flat(Infinity).map((value) => String(value ?? "")).filter((value) => value.trim());
}

export function resolveComfyVideoPath(video) {
  const params = video?.params || video || {};
  if (params.fullpath) return String(params.fullpath);
  const filename = params.filename || video?.filename || "";
  const subfolder = params.subfolder || video?.subfolder || "";
  if (filename && subfolder && /^[A-Za-z]:[\\/]/.test(String(subfolder))) {
    return `${String(subfolder).replace(/[\\/]+$/, "")}\\${filename}`;
  }
  return "";
}

function extractVideosFromHistory(historyPayload, promptId) {
  const root = historyPayload?.[promptId] || historyPayload;
  const outputs = root?.outputs || {};
  const videos = [];
  for (const output of Object.values(outputs)) {
    for (const key of ["gifs", "videos", "animated"]) {
      if (Array.isArray(output?.[key])) {
        for (const video of output[key]) videos.push(video);
      }
    }
  }
  return videos;
}

export const DEFAULT_SCENE_RENDER_WAIT_HOURS = 2;
export const MAX_SCENE_RENDER_WAIT_HOURS = 24;

export function normalizeSceneRenderWaitHours(value) {
  const hours = Number(value);
  if (!Number.isFinite(hours)) return DEFAULT_SCENE_RENDER_WAIT_HOURS;
  return Math.max(1, Math.min(MAX_SCENE_RENDER_WAIT_HOURS, Math.round(hours)));
}

function sceneRenderWaitLabel(value) {
  const hours = normalizeSceneRenderWaitHours(value);
  return `${hours}-hour`;
}

export function sceneVideoTimeoutMessage({
  promptId = "",
  sceneLabel = "Scene",
  modeLabel = "video",
  outputFolder = "",
  projectFolder = "",
  finalFolder = "",
  waitHours = DEFAULT_SCENE_RENDER_WAIT_HOURS,
} = {}) {
  const lines = [
    `${sceneLabel}: ${modeLabel} did not finish before the ${sceneRenderWaitLabel(waitHours)} wait limit.`,
    "",
    "What this means:",
    "- ComfyUI may still be rendering, or the workflow may have stopped without returning a video to the builder.",
    "",
    "What to check next:",
    "- Look at the ComfyUI console for a red error near this prompt ID, especially out-of-memory, missing model, missing file, or ffmpeg errors.",
    "- Check whether the ComfyUI queue is still running. If it is, wait for it to finish, then use Recover Scene Videos.",
    "- Check the temporary output folder for a finished .mp4. If it exists, the render finished but the builder could not collect it.",
    "- Try a shorter scene, lower resolution, fewer frames, or a lighter video model if the console shows VRAM/RAM pressure.",
  ];
  if (promptId) lines.push("", `Prompt ID: ${promptId}`);
  if (outputFolder) lines.push(`Temporary output folder: ${outputFolder}`);
  if (finalFolder) lines.push(`Builder scene-video folder: ${finalFolder}`);
  if (projectFolder) lines.push(`Project folder: ${projectFolder}`);
  return lines.join("\n");
}

export async function waitForVideos(promptId, onStatus, shouldCancel, findOutputFallback = null, options = {}) {
  const started = Date.now();
  const timeoutHours = normalizeSceneRenderWaitHours(options.timeoutHours);
  const timeoutMs = timeoutHours * 60 * 60 * 1000;
  let lastFallbackCheck = 0;
  while (Date.now() - started < timeoutMs) {
    if (shouldCancel?.()) throw new Error("Stopped by user.");
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`History request failed (${response.status})`);
    const promptError = extractPromptErrorFromHistory(data, promptId);
    if (promptError) throw new Error(`Scene video workflow failed:\n${promptError}`);
    const videos = extractVideosFromHistory(data, promptId);
    if (videos.length) return videos;
    if (typeof findOutputFallback === "function" && Date.now() - lastFallbackCheck > 10000) {
      lastFallbackCheck = Date.now();
      const fallbackPath = await findOutputFallback().catch(() => "");
      if (fallbackPath) return [{ params: { fullpath: fallbackPath }, fullpath: fallbackPath }];
    }
    if (promptHistoryFinished(data, promptId)) {
      if (typeof findOutputFallback === "function") {
        const fallbackPath = await findOutputFallback().catch(() => "");
        if (fallbackPath) return [{ params: { fullpath: fallbackPath }, fullpath: fallbackPath }];
      }
      throw new Error("Scene video workflow finished, but no video output was found in history.");
    }
    onStatus?.("Waiting for scene video...");
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  if (typeof options.timeoutMessage === "function") {
    throw new Error(options.timeoutMessage({ promptId }));
  }
  throw new Error(options.timeoutMessage || "Timed out waiting for the scene video.");
}

export async function waitForImages(promptId, onStatus, shouldCancel) {
  const started = Date.now();
  while (Date.now() - started < 20 * 60 * 1000) {
    if (shouldCancel?.()) throw new Error("Stopped by user.");
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`History request failed (${response.status})`);
    const promptError = extractPromptErrorFromHistory(data, promptId);
    if (promptError) throw new Error(`Image workflow failed:\n${promptError}`);
    const images = extractImagesFromHistory(data, promptId);
    if (images.length) return images;
    if (promptHistoryFinished(data, promptId)) {
      throw new Error("Image workflow finished, but no image output was found in history.");
    }
    onStatus?.("Waiting for image output...");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error("Timed out waiting for the ZImage preview.");
}

export async function waitForText(promptId, onStatus, shouldCancel, timeoutMs = 5 * 60 * 1000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (shouldCancel?.()) throw new Error("Stopped by user.");
    const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`History request failed (${response.status})`);
    const promptError = extractPromptErrorFromHistory(data, promptId);
    if (promptError) throw new Error(`Text workflow failed:\n${promptError}`);
    const text = extractTextFromHistory(data, promptId);
    if (text.length) return text;
    if (promptHistoryFinished(data, promptId)) {
      throw new Error("Text workflow finished, but no text output was found in history.");
    }
    onStatus?.("Waiting for cleanup result...");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Timed out waiting for text output.");
}

function queueListsFromStatus(data) {
  return {
    running: Array.isArray(data?.queue_running) ? data.queue_running : [],
    pending: Array.isArray(data?.queue_pending) ? data.queue_pending : [],
  };
}

async function getComfyQueueStatus() {
  const response = await api.fetchApi("/queue");
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(`Queue status request failed (${response.status})`);
  return queueListsFromStatus(data);
}

async function waitForComfyQueueIdle(onStatus, options = {}) {
  const timeoutMs = Number(options.timeoutMs || 10 * 60 * 1000);
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (options.shouldCancel?.()) throw new Error("Stopped by user.");
    const status = await getComfyQueueStatus();
    if (!status.running.length && !status.pending.length) return status;
    onStatus?.(`Waiting for ComfyUI queue to become idle...\nRunning: ${status.running.length}\nPending: ${status.pending.length}`);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Timed out waiting for ComfyUI queue to become idle. Nothing new was queued.");
}

async function clearPendingComfyQueue() {
  const response = await api.fetchApi("/queue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clear: true }),
  }).catch(() => null);
  if (!response) return;
  if (!response.ok) {
    console.warn("[VRGDG Music Builder] Could not clear pending ComfyUI queue:", response.status);
  }
}

export async function cancelComfyExecutionAndWaitIdle(onStatus, options = {}) {
  onStatus?.("Interrupting ComfyUI and clearing pending queue...");
  await api.fetchApi("/interrupt", { method: "POST" }).catch(() => null);
  await clearPendingComfyQueue();
  return await waitForComfyQueueIdle(onStatus, {
    timeoutMs: options.timeoutMs || 5 * 60 * 1000,
    shouldCancel: options.shouldCancel,
  });
}

export async function queueWorkflowPrompt(prompt, options = {}) {
  await waitForComfyQueueIdle(options.onStatus, {
    timeoutMs: options.idleTimeoutMs,
    shouldCancel: options.shouldCancel,
  });
  let queuedPrompt = prompt;
  if (!options.allowMemoryCleanup && !builderAutomaticMemoryCleanupEnabled) {
    const sanitized = withoutEmbeddedMemoryCleanupNodes(prompt);
    queuedPrompt = sanitized.prompt;
    if (sanitized.removed.length) {
      options.onStatus?.(`Bypassed ${sanitized.removed.length} embedded RAM/VRAM cleanup node(s) because automatic memory cleanup is disabled.`);
      console.info(`[VRGDG Music Builder] Bypassed embedded memory cleanup nodes: ${sanitized.removed.join(", ")}`);
    }
  }
  const clientId = api.clientId || app?.clientId || crypto.randomUUID();
  const response = await api.fetchApi("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: queuedPrompt, client_id: clientId }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data?.error) {
    throw new Error(data?.error?.message || data?.error || `Queue failed (${response.status})`);
  }
  if (data?.node_errors && Object.keys(data.node_errors).length) {
    const details = Object.entries(data.node_errors).map(([nodeId, error]) => {
      const messages = [error?.class_type, error?.exception_message, error?.message]
        .filter(Boolean)
        .map((value) => String(value));
      const inputErrors = Array.isArray(error?.errors)
        ? error.errors.map((item) => item?.message || item).filter(Boolean).map((value) => String(value))
        : [];
      return `Node ${nodeId}: ${[...messages, ...inputErrors].join(" | ") || JSON.stringify(error)}`;
    }).join("\n");
    throw new Error(`Workflow validation failed:\n${details}`);
  }
  return data;
}

export function audioUrl(path) {
  return `/vrgdg/music_builder/audio?path=${encodeURIComponent(path)}&v=${Date.now()}`;
}
