import { api } from "../../../scripts/api.js";

export const STORYBOARD_GEMMA_TIMEOUT_MS = 600000;

// Save (or export) the Storyboard file. It sends the revision this window loaded, so an Agent API / MCP edit
// saved in the meantime is never overwritten silently: the user is asked first.
export async function saveStoryboardFile(state, storyboard, url = "/vrgdg/storyboard/save") {
  const body = { project_folder: state.projectFolder, storyboard };
  if (Number.isFinite(state.storyboardRevision)) body.expected_revision = state.storyboardRevision;
  let data;
  try {
    data = await postJson(url, body);
  } catch (error) {
    if (!error?.conflict) throw error;
    const overwrite = window.confirm(
      "The Storyboard file was changed by the Agent API / MCP after this window loaded it.\n\n"
      + "OK: save this window's cards anyway (the API's newer Storyboard edits are replaced).\n"
      + "Cancel: keep the file; reopen the Storyboard to load the API changes.",
    );
    if (!overwrite) throw new Error("Storyboard not saved: it was changed elsewhere. Reopen the Storyboard to load the latest version.");
    delete body.expected_revision;
    data = await postJson(url, body);
  }
  const revision = Number(data?.storyboard?.revision ?? data?.storyboard_revision);
  if (Number.isFinite(revision)) state.storyboardRevision = revision;
  state.onStoryboardFileSaved?.();
  return data;
}

export async function postJson(url, payload = {}, timeoutMs = 120000) {
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
    if (!response.ok || data?.ok === false) {
      const requestError = new Error(data?.error || `Request failed (${response.status})`);
      if (data?.diagnostics && typeof data.diagnostics === "object") requestError.diagnostics = data.diagnostics;
      requestError.status = response.status;
      requestError.conflict = Boolean(data?.conflict);
      throw requestError;
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
