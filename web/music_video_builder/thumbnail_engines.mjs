// How each way of making the thumbnail is asked to do it from the chosen frame. The dialog hands in the Builder's own settings for the
// workflow (the models and the key the user already set for it) and gets back the route and its payload. No browser code in here, so
// tests/thumbnail_engines.cjs can run it.
//
// NanoBanana, Qwen Image Edit and the three Browser AI engines (Flow, GPT Image, Meta AI) are ComfyUI workflows: they take the frame as a
// reference picture and follow an instruction. Overlay and API image are not: each has its own route that saves the result in the project itself.

// The build route of each ComfyUI workflow, written out in full so a search for a route finds this module.
const ENGINE_ROUTES = {
  qwen_image_edit: "/vrgdg/workflow_runner/build_qwen_image_edit_prompt",
  nano_banana: "/vrgdg/workflow_runner/build_nb_image_prompt",
  browser_flow: "/vrgdg/workflow_runner/build_flow_gpt_image_prompt",
  browser_gpt: "/vrgdg/workflow_runner/build_flow_gpt_image_prompt",
  browser_meta: "/vrgdg/workflow_runner/build_flow_gpt_image_prompt",
};
// The Browser AI engines are the ones that share that route: the site is `settings.provider`, which the dialog sets from the engine.
const isBrowserEngine = (engine) => ENGINE_ROUTES[engine] === ENGINE_ROUTES.browser_flow;

export const THUMBNAIL_ENGINE_LABELS = {
  overlay: "Overlay: your frame exactly, no AI (grade, glow, title)",
  qwen_image_edit: "Qwen Image Edit (local, edit)",
  nano_banana: "NanoBanana (API, edit)",
  api_image: "API image: Gemini, Grok or OpenAI of the LLM Runner (edit)",
  browser_flow: "Flow: Nano Banana in your browser (edit)",
  browser_gpt: "GPT Image: ChatGPT Images in your browser (edit)",
  browser_meta: "Meta AI: meta.ai in your browser (edit)",
};

// The LLM Runner's provider names and the image provider each one draws with (the pack's `llm/api_image.py` has the same table). Any
// other provider (Anthropic, DeepSeek, OpenRouter ...) has no image API here.
const API_IMAGE_PROVIDERS = {
  google: { name: "gemini", label: "Gemini" },
  gemini: { name: "gemini", label: "Gemini" },
  grok: { name: "grok", label: "Grok" },
  xai: { name: "grok", label: "Grok" },
  openai: { name: "openai", label: "OpenAI" },
};

// `{ name, label }` of the image provider behind the LLM Runner's API provider, or null when that provider cannot draw pictures.
export function thumbnailApiImageProvider(runner) {
  const key = String((runner && typeof runner === "object" ? runner.llm_api_provider : "") || "").trim().toLowerCase();
  return API_IMAGE_PROVIDERS[key] || null;
}

// The Overlay engine is not an image workflow: it has its own route, which keeps the frame and puts the grade and the title on top,
// and saves the result in the project itself (so there is no ComfyUI queue and no separate save). `{ url, payload, label }`.
export function thumbnailOverlayRequest({ sourcePath, title, overlay, projectFolder }) {
  const path = String(sourcePath || "").trim();
  if (!path) throw new Error("Choose the frame first: scrub the video to the moment you want and click Use this frame.");
  const folder = String(projectFolder || "").trim();
  if (!folder) throw new Error("Create or load a project before making a thumbnail.");
  const look = overlay && typeof overlay === "object" ? overlay : {};
  return {
    url: "/vrgdg/music_builder/thumbnail/overlay",
    payload: { project_folder: folder, source_path: path, title: String(title || ""), color: look.color, position: look.position, strength: look.strength },
    label: THUMBNAIL_ENGINE_LABELS.overlay,
  };
}

// The API image engine: the pack sends the frame and the prompt to the image API of the LLM Runner's provider with that provider's
// key, and saves the result in the project itself. `runner` is the LLM Runner payload, `model` the image model ("" is the provider's
// first one). Throws a message the user can act on when something is missing. `{ url, payload, label }`.
export function thumbnailApiImageRequest({ prompt, sourcePath, projectFolder, runner, model }) {
  const text = String(prompt || "").trim();
  if (!text) throw new Error("Write the prompt first: click Write prompt with LLM or type one.");
  const path = String(sourcePath || "").trim();
  if (!path) throw new Error("Choose the frame first: scrub the video to the moment you want and click Use this frame.");
  const folder = String(projectFolder || "").trim();
  if (!folder) throw new Error("Create or load a project before making a thumbnail.");
  const provider = thumbnailApiImageProvider(runner);
  if (!provider) throw new Error("The LLM Runner's provider cannot draw pictures. Pick Gemini (google), Grok or OpenAI in the LLM Runner and add its key.");
  const apiKey = String(runner.llm_api_key || "").trim();
  if (!apiKey) throw new Error(`Add your ${provider.label} API key in the LLM Runner first.`);
  return {
    url: "/vrgdg/music_builder/thumbnail/api_image",
    payload: { project_folder: folder, source_path: path, prompt: text, provider: provider.name, api_key: apiKey, model: String(model || "").trim() },
    label: THUMBNAIL_ENGINE_LABELS.api_image,
  };
}

// A Browser AI engine when the build drives Chrome itself (`browser_ai_direct` in its capabilities, as H3 Studio does): the build opens the
// site of that engine, uploads the frame, types the prompt and keeps the picture in the project, so there is no ComfyUI workflow and no
// separate save. `engine` is the dialog's Browser AI engine (it names the label), `settings` the Image panel's Browser AI settings with the
// site of that engine (`provider`, `timeout_seconds`). `{ url, payload, label }`.
export function thumbnailBrowserAiDirectRequest({ engine, prompt, sourcePath, projectFolder, settings }) {
  const text = String(prompt || "").trim();
  if (!text) throw new Error("Write the prompt first: click Write prompt with LLM or type one.");
  const path = String(sourcePath || "").trim();
  if (!path) throw new Error("Choose the frame first: scrub the video to the moment you want and click Use this frame.");
  const folder = String(projectFolder || "").trim();
  if (!folder) throw new Error("Create or load a project before making a thumbnail.");
  const s = settings && typeof settings === "object" ? settings : {};
  return {
    url: "/vrgdg/music_builder/thumbnail/browser_ai",
    payload: { project_folder: folder, source_path: path, prompt: text, provider: s.provider, timeout_seconds: s.timeout_seconds, reuse_open_project: false },
    label: THUMBNAIL_ENGINE_LABELS[engine] || THUMBNAIL_ENGINE_LABELS.browser_flow,
  };
}

// Opens the site of a Browser AI engine in the build's own Chrome window (each site has its own), so the person can sign in once. `{ url, payload }`.
export function thumbnailBrowserAiOpenRequest(settings) {
  const s = settings && typeof settings === "object" ? settings : {};
  return { url: "/vrgdg/music_builder/thumbnail/browser_ai/open", payload: { provider: s.provider } };
}

// `{ url, payload, label }` for the build route of a ComfyUI workflow. Throws a message the user can act on when something is missing.
// `settings` is the Builder's settings object for that workflow.
export function thumbnailEngineRequest({ engine, prompt, sourcePath, settings, seed, defaultNbModel = "" }) {
  const route = ENGINE_ROUTES[engine];
  if (!route) throw new Error(`Unknown image workflow: ${engine}`);
  const text = String(prompt || "").trim();
  if (!text) throw new Error("Write the prompt first: click Write prompt with LLM or type one.");
  const path = String(sourcePath || "").trim();
  if (!path) throw new Error("Choose the frame first: scrub the video to the moment you want and click Use this frame.");
  const s = settings && typeof settings === "object" ? settings : {};
  const sourceFrame = [{ path }];
  let payload;

  if (engine === "nano_banana") {
    if (!String(s.api_key || "").trim()) {
      throw new Error("The NanoBanana API key is missing. Enter it in the NanoBanana image settings, or choose another image workflow.");
    }
    payload = { api_key: s.api_key, model: s.model || defaultNbModel, prompt: text, image_ingredients: sourceFrame };
  } else if (isBrowserEngine(engine)) {
    // The Image panel's Browser AI: `settings.provider` is the site of the chosen engine, the only one that runs, and `timeout_seconds`
    // how long to wait for it. A new project is opened in the site, not the open one.
    payload = {
      provider: s.provider, prompt: text, aspect_ratio: "16:9", image_ingredients: sourceFrame, timeout_seconds: s.timeout_seconds,
      reuse_open_project: false,
    };
  } else {
    // Qwen Image Edit: the builder picks the newest Qwen Image Edit model that is installed (2511 before 2509), the text encoder and
    // the VAE it needs, and the Lightning LoRA of the same version (4 steps) when there is one, so nothing about the models is sent.
    payload = { prompt: text, image_ingredients: sourceFrame, seed, seed_mode: "fixed" };
  }
  return { url: route, payload, label: THUMBNAIL_ENGINE_LABELS[engine] };
}
