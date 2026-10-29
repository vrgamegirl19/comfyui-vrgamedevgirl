const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

function fixture() {
  const writes = [], errors = [], warnings = [], nodes = [], confirmations = [];
  const context = vm.createContext({
    console: { error: (...args) => errors.push(args), warn: (...args) => warnings.push(args) },
    Date, setTimeout: () => 1, clearTimeout() {}, AbortController, encodeURIComponent,
    api: { fetchApi: async (url, options) => {
      writes.push({ url, payload: JSON.parse(options.body) });
      return { ok: true, json: async () => ({ ok: true, path: "project/logs/error_log.txt", text: "Saved error details" }) };
    } },
    window: { alert() {}, confirm: () => confirmations.shift(), dispatchEvent() {}, addEventListener() {}, removeEventListener() {} },
    CustomEvent: class {},
    prepareLlmRequest: async (_url, payload) => payload,
    document: {
      createElement: tag => { const node = { tag, style: {}, children: [], append(...items) { this.children.push(...items); }, remove() {} }; nodes.push(node); return node; },
      body: { append() {}, appendChild() {} },
    },
  });
  for (const file of ["music_video_builder/error_log.mjs", "music_video_builder/controls.mjs"]) {
    const source = fs.readFileSync(path.join(__dirname, "../web", file), "utf8")
      .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
    vm.runInContext(source, context);
  }
  return { context, writes, errors, warnings, nodes, confirmations };
}

const settle = () => new Promise(resolve => setImmediate(resolve));

test("error toasts log once, redact secrets, retain stacks and follow the original project", async () => {
  const f = fixture();
  let project = "Project A";
  f.context.setErrorLogContext(() => ({ project_folder: project, scene: 44, secrets: ["private-value"] }));
  f.context.toast("Ordinary success");
  assert.equal(f.errors.length, 0);
  const error = new Error("Render failed private-value");
  f.context.reportProjectError(error, { stage: "Video Generation", component: "Prompt Generator" });
  f.context.toast("Scene 44: Render failed private-value", true);
  project = "Project B";
  await settle();
  assert.equal(f.writes.length, 1);
  assert.equal(f.writes[0].payload.project_folder, "Project A");
  assert.equal(f.writes[0].payload.entry.scene, 44);
  assert.match(f.writes[0].payload.entry.stack, /Render failed \[redacted\]/);
  assert.doesNotMatch(JSON.stringify([f.writes, f.errors]), /private-value/);
  f.context.toast("Render failed private-value", true);
  await settle();
  assert.equal(f.writes[1].payload.project_folder, "Project B");
});

test("backend-recorded failures and logging failures do not create duplicate reports or recursive toasts", async () => {
  const f = fixture();
  const dispose = f.context.setErrorLogContext(() => ({ project_folder: "Project A" }));
  f.context.reportProjectError(new Error("Backend failed"), { logged: true });
  f.context.toast("Operation: Backend failed", true);
  await settle();
  assert.equal(f.writes.length, 0);
  f.context.api.fetchApi = async () => { throw new Error("Log disk full"); };
  f.context.toast("Different failure", true);
  await settle();
  assert.equal(f.warnings.length, 1);
  dispose();
  f.context.toast("After close", true);
  await settle();
  assert.equal(f.warnings.length, 1);
});

test("viewer copies the displayed report and requires confirmation before clearing", async () => {
  const f = fixture();
  let copied = "";
  await f.context.openProjectErrorLog("Project A", {
    makeButton: label => { const button = { textContent: label }; f.nodes.push(button); return button; },
    copyTextToClipboard: async text => { copied = text; return true; },
  });
  const button = name => f.nodes.find(node => node.textContent === name);
  await button("Copy Log").onclick();
  assert.equal(copied, "Saved error details");
  f.confirmations.push(false);
  button("Clear Log").onclick();
  await settle();
  assert.equal(f.writes.length, 1);
  f.confirmations.push(true);
  button("Clear Log").onclick();
  await settle();
  assert.equal(f.writes[1].url, "/vrgdg/music_builder/error_log/clear");
  assert.equal(f.writes[1].payload.project_folder, "Project A");
});

test("API helpers preserve errors, capture stacks and avoid re-appending backend-recorded errors", async () => {
  const f = fixture();
  const source = fs.readFileSync(path.join(__dirname, "../web/music_video_builder/comfy_api.mjs"), "utf8")
    .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
  vm.runInContext(source, f.context);
  f.context.setErrorLogContext(() => ({ project_folder: "Project A", secrets: ["private-value"] }));
  const append = f.context.api.fetchApi;
  let recorded = true;
  f.context.api.fetchApi = async (url, options) => {
    if (url.includes("error_log")) return append(url, options);
    assert.equal(options.headers["X-VRGDG-Project"], "Project%20A");
    return { ok: false, status: 500, headers: { get: () => recorded ? "1" : null },
      json: async () => ({ ok: false, error: "Request private-value failed" }) };
  };
  await assert.rejects(f.context.postJson("/vrgdg/test/fail", {}), /Request private-value failed/);
  f.context.toast("Request private-value failed", true);
  await settle();
  assert.equal(f.writes.length, 0);
  recorded = false;
  await assert.rejects(f.context.getJson("/vrgdg/test/different"), /Request private-value failed/);
  // A different project must get its own failure even when the message is identical.
  f.context.setErrorLogContext(() => ({ project_folder: "Project B", secrets: ["private-value"] }));
  f.context.api.fetchApi = async (url, options) => url.includes("error_log") ? append(url, options) : Promise.reject(new Error("Network private-value failure"));
  await assert.rejects(f.context.postJson("/vrgdg/test/fail", {}), /Network private-value failure/);
  await settle();
  assert.equal(f.writes.length, 1);
  assert.equal(f.writes[0].payload.project_folder, "Project B");
  assert.match(f.writes[0].payload.entry.stack, /Network \[redacted\] failure/);
});

test("ComfyUI history failures retain the original execution traceback", async () => {
  const f = fixture();
  const source = fs.readFileSync(path.join(__dirname, "../web/music_video_builder/comfy_api.mjs"), "utf8")
    .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
  vm.runInContext(source, f.context);
  f.context.setErrorLogContext(() => ({ project_folder: "Project A", scene: 12 }));
  const append = f.context.api.fetchApi;
  f.context.api.fetchApi = async (url, options) => url.includes("error_log") ? append(url, options) : {
    ok: true, json: async () => ({ prompt_1: { status: { status_str: "error", messages: [["execution_error", {
      exception_message: "Model failed", traceback: ["Traceback: original model file, line 44", "Original kernel error"],
    }]] } } }),
  };
  await assert.rejects(f.context.waitForImages("prompt_1"), /Original kernel error/);
  await settle();
  assert.equal(f.writes.length, 1);
  assert.equal(f.writes[0].payload.entry.source, "ComfyUI");
  assert.match(f.writes[0].payload.entry.details, /original model file, line 44/);
});
