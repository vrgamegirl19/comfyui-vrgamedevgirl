const assert = require("assert");
const vm = require("vm");
const { readBuilderModule, functionSource } = require("./builder_source.cjs");

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = {}; this.isConnected = true; this.value = ""; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  removeAttribute(key) { delete this[key]; }
  pause() { this.paused = true; }
  load() {}
  querySelectorAll(tag) { return this.children.flatMap(child => child instanceof Element ? [...(child.tagName === tag ? [child] : []), ...child.querySelectorAll(tag)] : []); }
}
const document = { createElement: tag => new Element(tag) };
const notices = [];
let resolveFetch;
let payload;
const context = vm.createContext({ document, console,
  normalizeVideoType: value => value,
  makeInput: (value, type) => Object.assign(new Element("input"), { value, type }),
  makeButton: text => Object.assign(new Element("button"), { textContent: text }),
  makeCheckbox: (text, checked) => {
    const wrapper = new Element("label"), input = new Element("input");
    input.checked = checked; wrapper.append(input); return { wrapper, input };
  },
  makeSelect: () => new Element("select"),
  makeField: (label, input) => { const field = new Element("label"); field.append(input); return field; },
  makeSettingsSection: (title, children) => { const panel = new Element("section"); panel.append(...children); return panel; },
  toast: (message, error) => notices.push({ message, error }),
  postJson: async (url, body) => {
    assert.strictEqual(url, "/vrgdg/music_builder/elevenlabs_voices");
    payload = body;
    return new Promise(resolve => { resolveFetch = resolve; });
  },
});
vm.runInContext(readBuilderModule("elevenlabs_voice_design.mjs"), context);
vm.runInContext(readBuilderModule("elevenlabs.mjs"), context);

async function main() {
  const state = { videoType: "singing", projectFolder: "project", elevenLabsApiKey: "" };
  const subject = { id: "alice", reference_type: "character" };
  assert.strictEqual(context.makeElevenLabsVoicePicker({ state, subject }), null);
  assert.strictEqual(context.makeElevenLabsSettings({ state }), null);
  state.videoType = "speaking";
  let saved = [], failSave = false, staleSave = false;
  const settings = context.makeElevenLabsSettings({ state, projectInput: { value: "project" },
    saveSession: async opts => { assert.strictEqual(opts.throwOnError, true); if (failSave) throw Error("disk full"); if (staleSave) return { stale: true }; saved.push(state.elevenLabsApiKeyProject); },
  });
  const input = settings.children[0].children[0];
  const [test, save, clear] = settings.children[1].children;
  assert.strictEqual(input.type, "password");
  input.value = "test-key"; input.oninput();
  assert.strictEqual(state.elevenLabsApiKey, "test-key");
  assert.strictEqual(state.elevenLabsApiKeyProject, undefined);
  input.value = "autofilled-key"; // Autofill can change the field without an input event.
  const testRequest = test.onclick();
  assert.strictEqual(payload.api_key, "autofilled-key");
  resolveFetch({ voices: [] }); await testRequest;
  assert.deepStrictEqual(saved, []); // Test never saves a credential.
  input.value = "test-key"; input.oninput();
  await save.onclick();
  assert.deepStrictEqual(saved, ["test-key"]);
  input.value = "second-key"; input.oninput(); failSave = true;
  await save.onclick();
  assert.strictEqual(state.elevenLabsApiKeyProject, "test-key");
  failSave = false;
  staleSave = true;
  await save.onclick();
  assert.strictEqual(state.elevenLabsApiKeyProject, "test-key");
  staleSave = false;
  await clear.onclick();
  assert.strictEqual(state.elevenLabsApiKeyProject, "");
  assert.strictEqual(state.elevenLabsApiKey, "");

  state.elevenLabsApiKey = "account-key";
  assert.strictEqual(context.makeElevenLabsVoicePicker({ state, subject: { reference_type: "prop" } }), null);
  assert.strictEqual(context.makeElevenLabsVoicePicker({ state, subject: { reference_type: "character", extra_reference_for: "alice" } }), null);
  const picker = context.makeElevenLabsVoicePicker({ state, subject });
  const enabled = picker.children[0].children[0];
  const fields = picker.children[1];
  const select = fields.children[0].children[0];
  const [refresh, more] = fields.children[1].children;
  const player = fields.children[2];
  assert.strictEqual(fields.style.display, "none");
  enabled.checked = true; enabled.onchange();
  assert.strictEqual(fields.style.display, "flex");
  let request = refresh.onclick();
  resolveFetch({ voices: [{ voice_id: "v1", name: "Alice Voice", category: "generated", preview_url: "https://example.com/a.mp3" }], has_more: true, next_page_token: "p2" });
  await request;
  assert.strictEqual(more.hidden, false);
  select.value = "v1"; select.onchange();
  assert.strictEqual(subject.elevenlabs_voice.voice_id, "v1");
  assert.strictEqual(subject.elevenlabs_voice.name, "Alice Voice");
  assert.strictEqual(player.src, "https://example.com/a.mp3");
  request = more.onclick(); assert.strictEqual(payload.next_page_token, "p2");
  resolveFetch({ voices: [{ voice_id: "v2", name: "Bob" }], has_more: false }); await request;
  assert.strictEqual(select.children.length, 3);
  assert.strictEqual(more.hidden, true);
  enabled.checked = false; enabled.onchange();
  assert.strictEqual(player.src, undefined);
  assert.strictEqual(subject.elevenlabs_voice.voice_id, "v1");

  request = refresh.onclick();
  state.projectFolder = "another-project";
  resolveFetch({ voices: [{ voice_id: "other", name: "Wrong account" }], has_more: false }); await request;
  assert.strictEqual(select.children.length, 3); // Stale response cannot overwrite a different project.
  state.projectFolder = "project";
  request = refresh.onclick();
  state.elevenLabsApiKey = "different-key";
  resolveFetch({ voices: [], has_more: false }); await request;
  assert.strictEqual(select.children.length, 3);

  // Exercise the real reference normalizer's explicit field lists, legacy path and round trip.
  context.normalizeMiniMaxH3Voice = value => value || {};
  vm.runInContext(functionSource(readBuilderModule("prompt_text.mjs"), "defaultFluxReferenceBuilder"), context);
  vm.runInContext(readBuilderModule("reference_data.mjs"), context);
  const voice = { enabled: true, voice_id: "v1", name: "Alice Voice" };
  const draft = { user_input: "warm storyteller", voice_description: "A warm, mature voice with measured cadence.", model_id: "eleven_ttv_v3", audio_base_64: "transient", generated_voice_id: "temporary1" };
  const refs = context.normalizeFluxReferenceBuilder({ subjects: [{ id: "alice", name: "Alice", elevenlabs_voice: voice, elevenlabs_voice_design: draft }] });
  const restored = context.normalizeFluxReferenceBuilder(JSON.parse(JSON.stringify(refs)));
  assert.strictEqual(restored.subjects[0].elevenlabs_voice.voice_id, "v1");
  assert.strictEqual(restored.subject.elevenlabs_voice.name, "Alice Voice");
  assert.strictEqual(restored.subjects[0].elevenlabs_voice_design.user_input, draft.user_input);
  assert.strictEqual(restored.subject.elevenlabs_voice_design.voice_description, draft.voice_description);
  assert.strictEqual(restored.subjects[0].elevenlabs_voice_design.audio_base_64, undefined);
  assert.strictEqual(restored.subjects[0].elevenlabs_voice_design.generated_voice_id, undefined);
  const legacy = context.normalizeFluxReferenceBuilder({ subject_count: 1, subject: { description: "Alice", elevenlabs_voice: voice } });
  assert.strictEqual(legacy.subjects[0].elevenlabs_voice.voice_id, "v1");
  state.videoType = "singing";
  context.refreshElevenLabsUI(state);
  assert.strictEqual(picker.style.display, "none");
  assert.strictEqual(settings.style.display, "none");
  assert.strictEqual(player.src, undefined);
  console.log("ElevenLabs UI checks passed.");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
