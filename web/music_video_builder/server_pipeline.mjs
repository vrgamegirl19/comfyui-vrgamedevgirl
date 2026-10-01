import { api } from "../../../scripts/api.js";

// Client for running a Video Builder pipeline as a background job on the server
// (Agent API /vrgdg/api/v1/projects/{pid}/pipelines/...). The job keeps running if the
// browser tab is closed or reloaded, and the page can pick it up again from /jobs.

const API_V1 = "/vrgdg/api/v1";
const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled", "interrupted"]);

export function projectIdFromFolder(folder) {
  return String(folder || "").trim().split(/[\\/]/).filter(Boolean).pop() || "";
}

// Browser build options -> server pipeline body (names differ for scope and seed mode).
export function serverBuildBody(options = {}) {
  const scope = String(options.sceneScope || "all");
  return {
    build_mode: options.buildMode || "resume_missing",
    scope: ["all", "selected", "from_selected"].includes(scope) ? scope : "all",
    video_seed_mode: options.videoSeedMode === "random" ? "randomize" : "keep",
    max_auto_retries: Math.max(0, Math.min(5, Number(options.maxAutoRetries ?? 3))),
    // Selected-scenes runs never stitch, matching the browser loop.
    stitch: scope !== "selected",
  };
}

function apiErrorMessage(data, status) {
  const error = data?.error;
  if (error && typeof error === "object") return String(error.message || error.code || `Request failed (${status})`);
  return String(error || `Request failed (${status})`);
}

export async function apiV1(method, path, body = undefined) {
  const response = await api.fetchApi(`${API_V1}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data?.ok) throw new Error(apiErrorMessage(data, response.status));
  return data;
}

// Start a job and follow it until it ends. Injected functions keep this testable:
//   start()            -> job object ({ id, status, ... })
//   getJob(id)         -> job object
//   cancelJob(id)      -> void, called once when shouldCancel() turns true
//   onUpdate(job)      -> called after every poll
// Resolves with the finished job when it succeeded, otherwise throws with the server's message.
export async function followServerJob({
  start,
  getJob,
  cancelJob,
  onUpdate = () => {},
  shouldCancel = () => false,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  pollMs = 2000,
  maxPollErrors = 5,
}) {
  let job = await start();
  let cancelSent = false;
  let pollErrors = 0;
  while (!TERMINAL_STATUSES.has(String(job?.status || ""))) {
    onUpdate(job);
    await sleep(pollMs);
    if (!cancelSent && shouldCancel()) {
      cancelSent = true;
      await cancelJob(job.id);
    }
    try {
      job = await getJob(job.id);
      pollErrors = 0;
    } catch (error) {
      // A restart or a brief network drop must not abandon a job that is still running.
      pollErrors += 1;
      if (pollErrors >= maxPollErrors) throw error;
    }
  }
  onUpdate(job);
  if (job.status === "succeeded") return job;
  if (job.status === "cancelled") throw new Error("Server build was cancelled.");
  const reason = String(job?.error?.message || job?.error || "").trim();
  throw new Error(reason || `Server build ${job.status}.`);
}

export function runServerPipeline({ path, body, onUpdate, shouldCancel }) {
  return followServerJob({
    start: async () => (await apiV1("POST", path, body)).data.job,
    getJob: async (id) => (await apiV1("GET", `/jobs/${encodeURIComponent(id)}`)).data,
    cancelJob: (id) => apiV1("POST", `/jobs/${encodeURIComponent(id)}/cancel`),
    onUpdate,
    shouldCancel,
  });
}

export function describeJobProgress(job) {
  const progress = job?.progress || {};
  const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
  const stage = String(progress.stage || job?.status || "working").replace(/_/g, " ");
  const message = String(progress.message || "").trim();
  return { percent, text: message ? `${stage}: ${message}` : stage };
}
