// The Restart ComfyUI button of the top bar. POST /vrgdg/video_builder/comfyui/restart asks ComfyUI-Manager to restart ComfyUI (the node pack
// serves the route inside ComfyUI, H3 Studio serves it itself). The restart usually cuts the answer short, so a dropped connection is not an
// error. The button then waits until ComfyUI stops answering and until it answers again.

import { BUILDER_FONT_STACK } from "./constants.mjs";
import { makeButton } from "./controls.mjs";

export const COMFY_RESTART_URL = "/vrgdg/video_builder/comfyui/restart";
const ALIVE_URL = "/system_stats";
const POLL_MS = 1000;
const GONE_WITHIN_MS = 20000;
const BACK_WITHIN_MS = 180000;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Does ComfyUI answer right now? Both the pack's ComfyUI and H3 Studio pass the route on to ComfyUI.
async function comfyAnswers(api) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2500);
  try {
    const response = await api.fetchApi(ALIVE_URL, { signal: controller.signal, cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

// How many jobs ComfyUI runs or has waiting. A restart loses them. 0 when ComfyUI does not say.
export async function runningJobCount(api) {
  try {
    const response = await api.fetchApi("/queue", { cache: "no-store" });
    if (!response.ok) return 0;
    const queue = await response.json();
    return (queue?.queue_running?.length || 0) + (queue?.queue_pending?.length || 0);
  } catch {
    return 0;
  }
}

async function waitFor(api, answering, limitMs, { wait, now }) {
  const start = now();
  while (now() - start < limitMs) {
    if ((await comfyAnswers(api)) === answering) return true;
    await wait(POLL_MS);
  }
  return false;
}

// Restart ComfyUI and wait until it answers again. Throws an Error with the text to show when it cannot be restarted or does not come back.
export async function restartComfyUi(api, { wait = sleep, now = Date.now } = {}) {
  let response = null;
  try {
    response = await api.fetchApi(COMFY_RESTART_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  } catch {
    // The restart closed the connection before it answered. That is how it goes when it works. The waiting below finds out.
  }
  if (response && !response.ok) {
    const reply = await response.json().catch(() => ({}));
    throw new Error(reply?.error || `ComfyUI could not be restarted (HTTP ${response.status}).`);
  }
  const clock = { wait, now };
  if (!(await waitFor(api, false, GONE_WITHIN_MS, clock))) {
    throw new Error("ComfyUI did not restart. ComfyUI-Manager may have refused. Close ComfyUI and start it again.");
  }
  if (!(await waitFor(api, true, BACK_WITHIN_MS, clock))) {
    throw new Error("ComfyUI is not answering after the restart. Look at its console window.");
  }
}

// Ask before restarting. Gives true for "Restart ComfyUI" and false for Cancel.
export function confirmComfyRestart(jobs = 0) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:20px;box-sizing:border-box;";
    const box = document.createElement("div");
    box.style.cssText = `width:min(520px,calc(100vw - 40px));border:1px solid #155e75;border-radius:10px;background:#0f172a;color:#f8fafc;box-shadow:0 24px 80px rgba(0,0,0,.65);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`;
    const title = document.createElement("div");
    title.textContent = "Restart ComfyUI?";
    title.style.cssText = "font-size:17px;font-weight:600;color:#cffafe;";
    const note = document.createElement("div");
    note.textContent = jobs > 0
      ? `ComfyUI has ${jobs} running or queued job${jobs === 1 ? "" : "s"}. They stop and are lost. ComfyUI is back in about a minute.`
      : "ComfyUI is back in about a minute. Nothing can render until then.";
    note.style.cssText = "font-size:13px;line-height:1.45;color:#cbd5e1;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const restart = makeButton("Restart ComfyUI", "primary");
    const cancel = makeButton("Cancel");
    const finish = (answer) => {
      backdrop.remove();
      resolve(answer);
    };
    restart.onclick = () => finish(true);
    cancel.onclick = () => finish(false);
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) finish(false);
    });
    actions.append(restart, cancel);
    box.append(title, note, actions);
    backdrop.append(box);
    document.body.append(backdrop);
  });
}

// Make `button` ask, restart and wait. `confirm` and `clock` are for tests.
export function wireRestartComfyButton(button, { api, toast, confirm = confirmComfyRestart, clock = {} }) {
  const idleTitle = button.title;
  button.onclick = async () => {
    if (button.disabled) return;
    if (!(await confirm(await runningJobCount(api)))) return;
    button.disabled = true;
    button.style.opacity = "0.6";
    button.setAttribute("aria-busy", "true");
    button.title = "Restarting ComfyUI...";
    toast("Restarting ComfyUI. This takes about a minute.");
    try {
      await restartComfyUi(api, clock);
      toast("ComfyUI restarted and is ready.");
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      button.disabled = false;
      button.style.opacity = "";
      button.removeAttribute?.("aria-busy");
      button.title = idleTitle;
    }
  };
}
