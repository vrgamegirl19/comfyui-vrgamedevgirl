const assert = require("assert");
const vm = require("vm");
const { readBuilderModule } = require("./builder_source.cjs");
class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = {}; this.dataset = {}; this.isConnected = true; this.value = ""; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  remove() { this.isConnected = false; }
  removeAttribute(key) { delete this[key]; }
  pause() { this.paused = true; }
  load() {}
  querySelectorAll(tag) { return this.children.flatMap(child => child instanceof Element ? [...(child.tagName === tag ? [child] : []), ...child.querySelectorAll(tag)] : []); }
}
let resolveRequest, body, route, assignments = [];
const context = vm.createContext({ document: { createElement: tag => new Element(tag) },
  normalizeVideoType: x => x, toast: () => {},
  makeButton: text => Object.assign(new Element("button"), { textContent: text }),
  makeInput: value => Object.assign(new Element("input"), { value }),
  makeSelect: () => new Element("select"),
  makeCheckbox: (_label, checked) => { const wrapper = new Element("label"), input = new Element("input"); input.checked = checked; wrapper.append(input); return { wrapper, input }; },
  makeField: (label, input) => { const field = new Element("label"); field.append(input); return field; },
  postJson: async (url, payload) => { route = url; body = payload; return new Promise(resolve => { resolveRequest = resolve; }); },
});
vm.runInContext(readBuilderModule("elevenlabs_voice_design.mjs"), context);
vm.runInContext(readBuilderModule("elevenlabs.mjs"), context);
async function main() {
  const state = { videoType: "speaking", elevenLabsApiKey: "test-key" };
  const subject = { name: "Alice", description: "A storyteller" }, host = new Element("div");
  const open = () => context.openElevenLabsVoiceDesign({ state, subject, host,
    isCurrent: () => state.videoType === "speaking", getLlmPayload: () => ({ text_runner: "own_server", own_server_model: "selected" }),
    onAssigned: voice => assignments.push(voice) });
  const panel = open();
  const [brief, description, model, text, name] = [panel.children[1].children[0], panel.children[3].children[0], panel.children[4].children[0], panel.children[5].children[0], panel.children[6].children[0]];
  const help = panel.children[2], previews = panel.children[9], [generate, save, close] = panel.children[10].children;
  assert(save.disabled);
  brief.value = "Warm and calm"; brief.oninput();
  let request = help.onclick();
  assert.strictEqual(route, "/vrgdg/music_builder/elevenlabs_voice_description");
  assert.strictEqual(body.text_runner, "own_server");
  assert.strictEqual(body.character_description, "A storyteller");
  const reviewed = "A warm, calm voice with deliberate pacing and clear articulation.";
  resolveRequest({ voice_description: reviewed }); await request;
  assert.strictEqual(description.value, reviewed);
  assert.strictEqual(subject.elevenlabs_voice_design.voice_description, reviewed);
  assert.strictEqual(assignments.length, 0);
  const candidate = { generated_voice_id: "temporary1", audio_base_64: "YXVkaW8=" };
  request = generate.onclick(); assert.strictEqual(body.voice_description, reviewed);
  resolveRequest({ previews: [candidate] }); await request;
  assert.strictEqual(previews.children.length, 1);
  assert(save.disabled);
  previews.children[0].children[0].onclick(); assert(!save.disabled);
  description.value += " Gentle rasp."; description.oninput();
  assert(save.disabled); assert.strictEqual(previews.children.length, 0);
  request = generate.onclick(); resolveRequest({ previews: [candidate] }); await request;
  previews.children[0].children[0].onclick();
  request = save.onclick();
  assert.strictEqual(route, "/vrgdg/music_builder/elevenlabs_voice_create");
  assert.strictEqual(body.generated_voice_id, "temporary1");
  assert.strictEqual(body.voice_name, "Alice");
  resolveRequest({ voice: { voice_id: "permanent1", name: "Alice" } }); await request;
  assert.strictEqual(assignments.length, 1); assert(save.disabled);
  assert(!JSON.stringify(subject).includes("temporary1"));
  assert(!JSON.stringify(subject).includes("YXVkaW8="));
  request = generate.onclick(); state.videoType = "singing";
  resolveRequest({ previews: [candidate] }); await request;
  assert.strictEqual(previews.children.length, 0);
  assert.strictEqual(open(), undefined);
  state.videoType = "speaking";
  request = generate.onclick(); close.onclick(); resolveRequest({ previews: [candidate] }); await request;
  assert.strictEqual(panel.isConnected, false); assert.strictEqual(previews.children.length, 0);
  // Exercise the real picker callback, replacing an already assigned account voice.
  state.projectFolder = "project";
  const character = { id: "alice", name: "Alice", reference_type: "character", elevenlabs_voice: { enabled: true, voice_id: "old-voice", name: "Previous voice" } };
  const picker = context.makeElevenLabsVoicePicker({ state, subject: character });
  const dropdown = picker.children[1].children[0].children[0];
  assert.strictEqual(dropdown.value, "old-voice");
  const editor = picker.children[2].onclick();
  const prompt = editor.children[3].children[0];
  prompt.value = reviewed; prompt.oninput();
  const [generateVoice, saveVoice] = editor.children[10].children;
  request = generateVoice.onclick(); resolveRequest({ previews: [candidate] }); await request;
  editor.children[9].children[0].children[0].onclick();
  request = saveVoice.onclick(); resolveRequest({ voice: { voice_id: "new-voice", name: "Designed Alice" } }); await request;
  assert.strictEqual(character.elevenlabs_voice.voice_id, "new-voice");
  assert.strictEqual(dropdown.value, "new-voice");
  assert.strictEqual(picker.children[0].children[0].checked, true);
  console.log("Voice Design review, invalidation, save/assignment and lifecycle checks passed.");
}
main().catch(error => { console.error(error); process.exitCode = 1; });
