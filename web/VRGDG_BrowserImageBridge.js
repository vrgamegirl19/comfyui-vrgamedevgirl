import { api } from "../../scripts/api.js";

const FLOW_GPT_BUILD_ENDPOINT = "/vrgdg/workflow_runner/build_flow_gpt_image_prompt";
const BROWSER_IMAGE_STATUS_ENDPOINT = "/vrgdg/browser_image/status";
const BROWSER_IMAGE_SETUP_ENDPOINT = "/vrgdg/browser_image/setup";
const BROWSER_IMAGE_LOGIN_ENDPOINT = "/vrgdg/browser_image/open_login";
const BROWSER_IMAGE_MANUAL_OPEN_ENDPOINT = "/vrgdg/browser_image/manual_open";
const BROWSER_IMAGE_MANUAL_UPLOAD_ENDPOINT = "/vrgdg/browser_image/manual_upload";
const BROWSER_IMAGE_MANUAL_WAIT_DOWNLOAD_ENDPOINT = "/vrgdg/browser_image/manual_wait_download";
const BROWSER_IMAGE_MANUAL_IMPORT_LATEST_ENDPOINT = "/vrgdg/browser_image/manual_import_latest";

const PROVIDERS = Object.freeze({
  FLOW_NANO_BANANA: "flow_nano_banana",
  GPT_IMAGE: "gpt_image",
});

async function readJsonResponse(response) {
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok || data?.ok === false) {
    throw new Error(data?.error || response.statusText || "Browser image request failed.");
  }
  return data || {};
}

async function postJson(url, payload = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await api.fetchApi(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
      signal: controller.signal,
    });
    return await readJsonResponse(response);
  } finally {
    clearTimeout(timer);
  }
}

async function getJson(url, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await api.fetchApi(url, { signal: controller.signal });
    return await readJsonResponse(response);
  } finally {
    clearTimeout(timer);
  }
}

function normalizeBrowserImageProvider(provider) {
  const value = String(provider || "").trim().toLowerCase().replace(/[-\s]+/g, "_");
  if (["flow", "flow_browser", "flow_nano", "flow_nanobanana", "flow_nano_banana"].includes(value)) {
    return PROVIDERS.FLOW_NANO_BANANA;
  }
  if (["chatgpt", "chatgpt_image", "chatgpt_images", "gpt", "gpt_image", "gpt_image_2", "gpt_images"].includes(value)) {
    return PROVIDERS.GPT_IMAGE;
  }
  return value || PROVIDERS.FLOW_NANO_BANANA;
}

function promptForBrowserImageProvider(prompt, provider, settings = {}) {
  const normalizedProvider = normalizeBrowserImageProvider(provider);
  const text = String(prompt || "").trim();
  if (normalizedProvider !== PROVIDERS.GPT_IMAGE) return text;
  const aspectRatio = String(settings.aspect_ratio || settings.aspectRatio || "").trim();
  if (!aspectRatio) return text;
  if (text.toLowerCase().includes("aspect ratio") && text.includes(aspectRatio)) return text;
  return `${text}\n\nAspect ratio: ${aspectRatio}.`.trim();
}

async function getBrowserImageStatus() {
  return await getJson(BROWSER_IMAGE_STATUS_ENDPOINT);
}

async function setupBrowserImageAutomation(options = {}) {
  return await postJson(BROWSER_IMAGE_SETUP_ENDPOINT, options, Number(options.timeoutMs || 900000));
}

async function openBrowserImageLogin(provider, options = {}) {
  return await postJson(BROWSER_IMAGE_LOGIN_ENDPOINT, {
    ...options,
    provider: normalizeBrowserImageProvider(provider),
  }, Number(options.timeoutMs || 60000));
}

async function openManualBrowserImageProvider(provider, options = {}) {
  return await postJson(BROWSER_IMAGE_MANUAL_OPEN_ENDPOINT, {
    ...options,
    provider: normalizeBrowserImageProvider(provider),
  }, Number(options.timeoutMs || 60000));
}

async function uploadManualBrowserImageRefs(provider, options = {}) {
  return await postJson(BROWSER_IMAGE_MANUAL_UPLOAD_ENDPOINT, {
    ...options,
    provider: normalizeBrowserImageProvider(provider),
  }, Number(options.timeoutMs || 300000));
}

async function waitForManualBrowserImageDownload(provider, options = {}) {
  return await postJson(BROWSER_IMAGE_MANUAL_WAIT_DOWNLOAD_ENDPOINT, {
    ...options,
    provider: normalizeBrowserImageProvider(provider),
  }, Number(options.timeoutMs || 900000));
}

async function importLatestManualBrowserImageDownload(provider, options = {}) {
  return await postJson(BROWSER_IMAGE_MANUAL_IMPORT_LATEST_ENDPOINT, {
    ...options,
    provider: normalizeBrowserImageProvider(provider),
  }, Number(options.timeoutMs || 120000));
}

async function buildBrowserImagePrompt(payload = {}) {
  return await postJson(FLOW_GPT_BUILD_ENDPOINT, {
    ...payload,
    provider: normalizeBrowserImageProvider(payload.provider),
  }, Number(payload.timeoutMs || 120000));
}

export {
  PROVIDERS as BROWSER_IMAGE_PROVIDERS,
  buildBrowserImagePrompt,
  getBrowserImageStatus,
  importLatestManualBrowserImageDownload,
  normalizeBrowserImageProvider,
  openBrowserImageLogin,
  openManualBrowserImageProvider,
  promptForBrowserImageProvider,
  setupBrowserImageAutomation,
  uploadManualBrowserImageRefs,
  waitForManualBrowserImageDownload,
};

window.VRGDGBrowserImageBridge = {
  PROVIDERS,
  buildBrowserImagePrompt,
  getBrowserImageStatus,
  importLatestManualBrowserImageDownload,
  normalizeBrowserImageProvider,
  openBrowserImageLogin,
  openManualBrowserImageProvider,
  promptForBrowserImageProvider,
  setupBrowserImageAutomation,
  uploadManualBrowserImageRefs,
  waitForManualBrowserImageDownload,
};
