import { postJson } from "./comfy_api.mjs";
import { makeMiniButton } from "./controls.mjs";

// The Hugging Face token box at the top of the Models dialog's install tabs. Hugging Face keeps some files (the Flux Klein 9B diffusion model)
// behind a login: with a token saved here the installer fetches them like any other file. The token is saved by the Builder's server
// (core/hf_token.py in the pack, the same routes in H3 Studio, which also has it in App Settings) and never comes back to this page. The box
// shows only when the server says whether a token is saved (`hf_token_saved` in the models status), so an older Builder has no box.

export const HF_TOKEN_URL = "/vrgdg/video_builder/models/hf_token";
export const HF_TOKEN_TEST_URL = "/vrgdg/video_builder/models/hf_token/test";
export const HF_TOKEN_PAGE = "https://huggingface.co/settings/tokens";

// What is wrong with the pasted text, or "" when it can be sent. The server checks the same thing.
export function hfTokenProblem(text) {
  const token = String(text ?? "").trim();
  if (!token) return "Paste your Hugging Face token first.";
  if (!token.startsWith("hf_") || /\s/.test(token) || token.length > 400) {
    return "That does not look like a Hugging Face token. It starts with hf_. Paste the whole token without spaces.";
  }
  return "";
}

// `onChange(saved)` is called after a token was saved or removed, so the files that waited for it can be offered. `setSaved(value)` tells the
// box what the server holds: a boolean shows the box, anything else hides it.
export function renderHfTokenBox({ onChange = () => {} } = {}) {
  const box = document.createElement("div");
  box.style.cssText = "display:none;flex-direction:column;gap:8px;border:1px solid #334155;border-radius:8px;background:#0b1220;padding:12px 14px;";
  const title = document.createElement("div");
  title.textContent = "Hugging Face token";
  title.style.cssText = "font-size:15px;font-weight:800;color:#f8fafc;";
  const note = document.createElement("div");
  note.textContent = "Needed only for files Hugging Face keeps behind a login, such as the Flux Klein 9B diffusion model: accept the model's licence on its page with the same account. A token with Read access is enough. It also lifts the limit Hugging Face puts on anonymous downloads. It is saved on this computer in a file of its own and is never shown again.";
  note.style.cssText = "font-size:12px;line-height:1.4;color:#cbd5e1;";
  const row = document.createElement("div");
  row.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;";
  const input = document.createElement("input");
  input.type = "password";
  input.placeholder = "hf_...";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.setAttribute("aria-label", "Hugging Face token");
  input.style.cssText = "flex:1;min-width:240px;padding:9px 10px;border:1px solid #475569;border-radius:6px;background:#020617;color:#e2e8f0;font-size:13px;";
  const save = makeMiniButton("Save Token");
  save.style.cssText += ";background:#15803d;border-color:#4ade80;color:#f0fdf4;font-weight:900;";
  const test = makeMiniButton("Test Token");
  const clear = makeMiniButton("Remove Token");
  const create = makeMiniButton("Create a Token");
  row.append(input, save, test, clear, create);
  const status = document.createElement("div");
  status.style.cssText = "font-size:12px;font-weight:700;color:#cbd5e1;";
  box.append(title, note, row, status);

  let saved = false;
  // Until the person does something, the line under the box says what the server holds.
  let idle = true;
  const say = (text, kind = "info") => {
    status.textContent = text;
    status.style.color = kind === "error" ? "#fca5a5" : kind === "ok" ? "#86efac" : "#cbd5e1";
  };
  const sayIdle = () => say(saved ? "A token is saved on this computer." : "No token is saved yet.");
  const draw = () => {
    save.disabled = false;
    test.disabled = false;
    clear.disabled = !saved;
  };
  const run = async (busyText, action) => {
    idle = false;
    save.disabled = true;
    test.disabled = true;
    clear.disabled = true;
    say(busyText);
    try {
      await action();
    } catch (error) {
      say(String(error?.message || error), "error");
    } finally {
      draw();
    }
  };
  const send = (url, payload, timeoutMs) => postJson(url, payload, timeoutMs, { skipLlmSetup: true });

  const saveToken = () => {
    const problem = hfTokenProblem(input.value);
    if (problem) {
      idle = false;
      say(problem, "error");
      return undefined;
    }
    return run("Saving...", async () => {
      await send(HF_TOKEN_URL, { api_key: input.value.trim() }, 30000);
      input.value = "";
      saved = true;
      say("Token saved. Files that need a login can be installed now.", "ok");
      onChange(true);
    });
  };
  const testToken = () => {
    const typed = input.value.trim();
    const problem = typed ? hfTokenProblem(typed) : saved ? "" : "Paste a token first, or save one.";
    if (problem) {
      idle = false;
      say(problem, "error");
      return undefined;
    }
    return run("Asking Hugging Face...", async () => {
      const result = await send(HF_TOKEN_TEST_URL, typed ? { api_key: typed } : {}, 40000);
      say(result.user ? `Hugging Face accepted the token. It belongs to ${result.user}.` : "Hugging Face accepted the token.", "ok");
    });
  };
  const removeToken = () =>
    run("Removing...", async () => {
      await send(HF_TOKEN_URL, { api_key: "" }, 30000);
      input.value = "";
      saved = false;
      say("Token removed.", "ok");
      onChange(false);
    });

  save.onclick = saveToken;
  test.onclick = testToken;
  clear.onclick = removeToken;
  create.onclick = () => window.open(HF_TOKEN_PAGE, "_blank", "noopener,noreferrer");
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") saveToken();
  });
  draw();

  return {
    element: box,
    setSaved(value) {
      box.style.display = typeof value === "boolean" ? "flex" : "none";
      saved = value === true;
      if (idle) sayIdle();
      draw();
    },
  };
}
