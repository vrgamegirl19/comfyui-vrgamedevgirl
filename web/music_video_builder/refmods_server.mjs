import { api } from "../../../scripts/api.js";
import { BUILDER_FONT_STACK } from "./constants.mjs";
import { makeButton, makeInput } from "./controls.mjs";
import { isWebAddress, providerLabel } from "./refmods_viewer_data.mjs";

// The RefMod server: a hosted library where people share RefMods (the H3 RefMod Server, not part of this repository). The Viewer
// reads it through the Builder's own routes under /vrgdg/refmod/server, so this page never talks to the server directly and
// the sign-in token stays on this computer. The server wants a sign-in with GitHub or Discord: the dialog below starts it,
// opens the sign-in page in the browser and waits until the person has finished there.

async function call(path, options) {
  let response;
  try {
    response = await api.fetchApi(path, options);
  } catch (error) {
    throw new Error("Could not reach the Video Builder.");
  }
  const result = await response.json().catch(() => ({}));
  if (!response.ok || !result?.ok) {
    const failure = new Error(result?.error || `The request failed (${response.status}).`);
    failure.code = result?.code || "";
    throw failure;
  }
  return result;
}

const post = (path, body = {}) => call(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const loadServerSettings = () => call("/vrgdg/refmod/server/settings");
export const saveServerSettings = ({ url }) => post("/vrgdg/refmod/server/settings", { url });
export const loadSignInOptions = () => call("/vrgdg/refmod/server/signin");
export const startSignIn = (provider) => post("/vrgdg/refmod/server/signin", { provider });
export const pollSignIn = () => post("/vrgdg/refmod/server/signin/poll");
export const signOut = () => post("/vrgdg/refmod/server/signout");
export const loadServerLibrary = () => call("/vrgdg/refmod/server/library");
export const shareRefmod = (name) => post("/vrgdg/refmod/server/upload", { name });
export const addServerRefmod = (id) => post("/vrgdg/refmod/server/download", { id });
export const removeServerRefmod = (id) => post("/vrgdg/refmod/server/delete", { id });
export const reportServerRefmod = (id, reason) => post("/vrgdg/refmod/server/report", { id, reason });
export const serverPreviewUrl = (id) => api.apiURL(`/vrgdg/refmod/server/preview?id=${encodeURIComponent(id)}`);

const FIELD_STYLE = `display:flex;flex-direction:column;gap:5px;font-family:${BUILDER_FONT_STACK};font-size:12px;color:#d4d4d8;font-weight:500;`;
const HINT_STYLE = "font-size:11px;color:#a1a1aa;line-height:1.4;";
const TEXT_STYLE = "font-size:13px;color:#d4d4d8;line-height:1.45;";
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const messageOf = (error) => String(error?.message || error);

const element = (tag, css, text) => {
  const node = document.createElement(tag);
  if (css) node.style.cssText = css;
  if (text !== undefined) node.textContent = text;
  return node;
};

// The dark box in the middle of the screen. role="alertdialog" makes the Viewer under it leave Escape to this dialog.
function modal(title, lines) {
  const backdrop = element("div", "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:20px;");
  backdrop.setAttribute("role", "alertdialog");
  backdrop.setAttribute("aria-modal", "true");
  const box = element("div", `width:min(520px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`);
  box.append(element("div", "font-size:16px;font-weight:900;color:#cffafe;", title));
  for (const line of [].concat(lines).filter(Boolean)) box.append(element("div", TEXT_STYLE, line));
  backdrop.append(box);
  return { backdrop, box };
}

// The server address and the account on it. Resolves { confirmed, changed }: `confirmed` when the confirm button was pressed
// (with requireSignIn it only works while signed in), `changed` when the person signed in or out or changed the address, so
// the Viewer knows to ask the server again. Used for "Server settings" and as the step before sharing.
export function openServerDialog({ title = "Server settings", message = [], confirmLabel = "Done", requireSignIn = false } = {}) {
  return new Promise((resolve) => {
    const { backdrop, box } = modal(title, message);
    const url = makeInput("");
    url.placeholder = "http://127.0.0.1:8787";
    const urlHint = element("span", HINT_STYLE, "Where the RefMod server runs.");
    const urlField = element("label", FIELD_STYLE);
    urlField.append(element("span", "", "Server address"), url, urlHint);
    const accountBox = element("div", "border:1px solid #27272a;border-radius:8px;background:#18181b;padding:10px;display:flex;flex-direction:column;gap:8px;");
    const problem = element("div", "font-size:12px;color:#fca5a5;");
    const actions = element("div", "display:grid;grid-template-columns:1fr 1fr;gap:8px;");
    const cancel = makeButton("Cancel");
    const confirm = makeButton(confirmLabel, "primary");
    actions.append(cancel, confirm);
    box.append(urlField, accountBox, problem, actions);

    let alive = true;
    let changed = false;
    let savedUrl = "";
    let drawTicket = 0;
    let signInTicket = 0;
    const setProblem = (text) => {
      problem.textContent = text || "";
    };
    const finish = (confirmed) => {
      if (!alive) return;
      alive = false;
      document.removeEventListener("keydown", onKeyDown, true);
      backdrop.remove();
      resolve({ confirmed, changed });
    };

    // Saves the address when it was edited. A different server needs a new sign-in, which the backend takes care of.
    const commitAddress = async () => {
      if (url.value.trim() === savedUrl) return;
      const settings = await saveServerSettings({ url: url.value });
      savedUrl = settings.url || "";
      url.value = savedUrl;
      changed = true;
    };

    const text = (line) => element("div", TEXT_STYLE, line);

    const draw = async () => {
      const ticket = ++drawTicket;
      accountBox.replaceChildren(text("Checking the sign-in..."));
      let settings;
      try {
        settings = await loadServerSettings();
      } catch (error) {
        if (alive && ticket === drawTicket) setProblem(messageOf(error));
        return;
      }
      if (!alive || ticket !== drawTicket) return;
      urlHint.textContent = `Where the RefMod server runs. Empty means ${settings.default_url}.`;
      if (!savedUrl) {
        savedUrl = settings.url || "";
        url.value = savedUrl;
      }
      confirm.disabled = requireSignIn && !settings.signed_in;
      if (settings.signed_in) {
        const row = element("div", "display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;");
        const out = makeButton("Sign out");
        out.onclick = async () => {
          out.disabled = true;
          try {
            await signOut();
            changed = true;
            setProblem("");
          } catch (error) {
            setProblem(messageOf(error));
          }
          draw();
        };
        row.append(text(`Signed in as ${settings.account || "your account"}.`), out);
        accountBox.replaceChildren(row);
        return;
      }
      let options;
      try {
        options = await loadSignInOptions();
      } catch (error) {
        accountBox.replaceChildren(text("Not signed in."));
        setProblem(messageOf(error));
        return;
      }
      if (!alive || ticket !== drawTicket) return;
      const providers = options.providers || [];
      if (!providers.length) {
        accountBox.replaceChildren(text("Not signed in, and this server has no way to sign in set up yet."));
        return;
      }
      const buttons = element("div", "display:flex;gap:8px;flex-wrap:wrap;");
      for (const provider of providers) {
        const button = makeButton(`Sign in with ${providerLabel(provider)}`, "primary");
        button.onclick = () => signIn(provider);
        buttons.append(button);
      }
      accountBox.replaceChildren(text("Not signed in. Sign in to share RefMods and to see what others shared. Your account name is shown with what you share."), buttons);
    };

    // Opens the sign-in page in the browser, then asks the server every couple of seconds whether the person has finished.
    const signIn = async (provider) => {
      const ticket = ++signInTicket;
      setProblem("");
      try {
        await commitAddress();
        const started = await startSignIn(provider);
        if (!isWebAddress(started.url)) throw new Error("The server sent a sign-in address that is not a web page.");
        window.open(started.url, "_blank", "noopener,noreferrer");
        const again = element("a", "color:#67e8f9;font-size:12px;", "Open the sign-in page again");
        again.href = started.url;
        again.target = "_blank";
        again.rel = "noopener noreferrer";
        accountBox.replaceChildren(text("Finish signing in in the browser tab that opened. This window carries on by itself."), again);
        const deadline = Date.now() + (Number(started.expires_in) || 600) * 1000;
        while (alive && ticket === signInTicket && Date.now() < deadline) {
          await sleep((Number(started.interval) || 2) * 1000);
          if (!alive || ticket !== signInTicket) return;
          const result = await pollSignIn();
          if (result.status === "signed_in") {
            changed = true;
            draw();
            return;
          }
        }
        if (alive && ticket === signInTicket) setProblem("The sign-in took too long. Try again.");
      } catch (error) {
        if (alive && ticket === signInTicket) setProblem(messageOf(error));
      }
      if (alive && ticket === signInTicket) draw();
    };

    const submit = async () => {
      confirm.disabled = true;
      setProblem("");
      try {
        await commitAddress();
        finish(true);
      } catch (error) {
        setProblem(messageOf(error));
        confirm.disabled = false;
      }
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        finish(false);
      } else if (event.key === "Enter" && event.target === url) {
        event.preventDefault();
        event.stopPropagation();
        url.blur();
      }
    };
    // Leaving the address field with a changed address saves it and shows the ways to sign in on that server.
    url.addEventListener("change", async () => {
      setProblem("");
      signInTicket += 1;
      try {
        await commitAddress();
      } catch (error) {
        setProblem(messageOf(error));
        return;
      }
      draw();
    });
    cancel.onclick = () => finish(false);
    confirm.onclick = submit;
    backdrop.onclick = (event) => {
      if (event.target === backdrop) finish(false);
    };
    document.addEventListener("keydown", onKeyDown, true);
    document.body.append(backdrop);
    confirm.disabled = requireSignIn;
    draw();
  });
}

// Asks why a shared RefMod is being reported. Resolves the reason (it may be empty), or null when cancelled.
export function askReportReason({ title = "Report this RefMod", message = [] } = {}) {
  return new Promise((resolve) => {
    const { backdrop, box } = modal(title, message);
    const reason = element(
      "textarea",
      `width:100%;box-sizing:border-box;min-height:76px;resize:vertical;background:#18181b;color:#fafafa;border:1px solid #3f3f46;border-radius:6px;padding:8px 10px;font-family:${BUILDER_FONT_STACK};font-size:13px;`,
    );
    reason.maxLength = 200;
    reason.placeholder = "Why? For example: this is not a RefMod I would expect here.";
    const actions = element("div", "display:grid;grid-template-columns:1fr 1fr;gap:8px;");
    const cancel = makeButton("Cancel");
    const send = makeButton("Send report", "primary");
    actions.append(cancel, send);
    box.append(reason, actions);
    let alive = true;
    const finish = (value) => {
      if (!alive) return;
      alive = false;
      document.removeEventListener("keydown", onKeyDown, true);
      backdrop.remove();
      resolve(value);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        finish(null);
      }
    };
    cancel.onclick = () => finish(null);
    send.onclick = () => finish(reason.value.trim());
    backdrop.onclick = (event) => {
      if (event.target === backdrop) finish(null);
    };
    document.addEventListener("keydown", onKeyDown, true);
    document.body.append(backdrop);
    reason.focus();
  });
}
