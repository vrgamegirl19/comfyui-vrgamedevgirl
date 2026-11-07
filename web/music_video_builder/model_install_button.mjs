// A download button that installs the file itself instead of opening the model page, so it is saved under the name the Builder looks for
// (the Edit LoRA is called adapter_model.safetensors on Hugging Face). The install goes through the Builder's own routes
// (core/model_downloads.py), the same ones as the Dmad Models tab, and the button follows the download until it is done.

import { getJson, postJson } from "./comfy_api.mjs";
import { toast } from "./controls.mjs";

const STATUS_URL = "/vrgdg/video_builder/models/status";
const INSTALL_URL = "/vrgdg/video_builder/models/install";
const CANCEL_URL = "/vrgdg/video_builder/models/cancel";

// The button's text for one model's state (`model` is the server's entry for the item, or null when it is not known yet).
export function installButtonText(item, model) {
  const job = model?.job || {};
  if (job.state === "downloading") {
    const total = Number(job.total) || 0;
    const done = Number(job.downloaded) || 0;
    const progress = total ? ` ${Math.min(100, Math.floor((done / total) * 100))}%` : "";
    return `Installing ${item.label}…${progress} (click to cancel)`;
  }
  if (model?.installed) return `${item.label} · installed`;
  if (job.state === "error") return `${item.label} · try again`;
  return item.label;
}

// Makes `button` install `item.install` (an id from DOWNLOADABLE_MODELS) as `item.filename`. Returns a function that stops its polling:
// call it when the button is removed. `deps` is only given by tests.
export function wireInstallButton(button, item, deps = {}) {
  const {
    get = getJson,
    post = postJson,
    notify = toast,
    open = (url) => window.open(url, "_blank", "noopener,noreferrer"),
    announce = (ids) => window.dispatchEvent(new CustomEvent("vrgdg-models-changed", { detail: { ids } })),
    setTimer = (fn, ms) => window.setTimeout(fn, ms),
    clearTimer = (id) => window.clearTimeout(id),
  } = deps;
  let model = null;
  let timer = 0;
  let stopped = false;
  const busy = () => model?.job?.state === "downloading";

  const paint = () => {
    button.textContent = installButtonText(item, model);
    if (model?.job?.state === "error") button.title = model.job.error || "The install failed.";
    else if (model?.installed && model.path) button.title = `In: ${model.path}`;
    else button.title = `Installs into ComfyUI's models/loras as ${item.filename}`;
  };

  // Take the server's list. A download that was running and is now installed or failed is announced once.
  const apply = (models) => {
    const before = model;
    model = (Array.isArray(models) ? models : []).find((entry) => entry.id === item.install) || null;
    if (before?.job?.state === "downloading" && model) {
      if (model.installed) {
        notify(`${item.label} installed in ${model.destination || "models/loras"} as ${item.filename}.`);
        announce([item.install]);
      } else if (model.job?.state === "error") {
        notify(`${item.label}: ${model.job.error}`, true);
      }
    }
    paint();
  };

  const load = async () => {
    timer = 0;
    if (stopped) return;
    try {
      const result = await get(STATUS_URL);
      if (stopped) return;
      apply(result?.models);
    } catch (_error) {
      // The label stays as it is. A click then falls back to the browser download.
      return;
    }
    if (!stopped && busy()) timer = setTimer(load, 1000);
  };

  button.addEventListener("click", async (event) => {
    event.preventDefault?.();
    event.stopPropagation?.();
    if (model?.installed) {
      notify(`${item.filename} is already installed${model.path ? ` (${model.path})` : ""}.`);
      return;
    }
    try {
      if (busy()) {
        await post(CANCEL_URL, { ids: [item.install] }, 30000);
      } else {
        apply((await post(INSTALL_URL, { ids: [item.install] }, 60000))?.models);
        // The installer can fail at once (ComfyUI has no loras folder): that is not a download that was running, so say it here.
        if (model?.job?.state === "error") notify(`${item.label}: ${model.job.error}`, true);
      }
    } catch (error) {
      // A build without the installer: the browser download still works, under the name Hugging Face gives the file.
      open(item.url);
      notify(`The Builder's model installer could not be reached (${String(error?.message || error)}). Opened the browser download instead: save the file as ${item.filename}.`, true);
      return;
    }
    if (timer) clearTimer(timer);
    timer = setTimer(load, 400);
  });

  const stop = () => {
    stopped = true;
    if (timer) clearTimer(timer);
    timer = 0;
  };
  paint();
  load();
  return stop;
}
