const { readBuilderModule, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const source = readBuilderSource();

function fakeDocument() {
  const listeners = {};
  const makeNode = (tag) => ({
    tag, style: {}, dataset: {}, attrs: {}, children: [], listeners: {}, textContent: '', value: '', checked: false, removed: false, focused: false,
    append(...kids) { this.children.push(...kids); },
    setAttribute(name, value) { this.attrs[name] = value; },
    addEventListener(type, fn) { this.listeners[type] = fn; },
    remove() { this.removed = true; },
    focus() { this.focused = true; },
    select() { this.selected = true; },
  });
  return {
    body: makeNode('body'), createElement: makeNode, createTextNode: (text) => ({ text }),
    addEventListener: (type, fn) => { listeners[type] = fn; }, removeEventListener: (type) => { delete listeners[type]; }, listeners,
  };
}

function profileContext(extra = {}) {
  const document = fakeDocument();
  const context = vm.createContext({ document, console, ...extra });
  vm.runInContext(`${readBuilderModule('minimax_h3.mjs')}\n${readBuilderModule('video_profiles.mjs')}\n;Object.assign(globalThis,{applyVideoProfileSettings,videoProfileLabel,createVideoProfileActions});`, context);
  return context;
}

// ---- what a profile applies ---------------------------------------------------------------------------------
test('applying a profile sets the video selection and keeps audio, continuity and version markers as they are', () => {
  const c = profileContext();
  const current = c.cloneMiniMaxH3Settings({
    video_mode: 'text_to_video', render_pass: 'single', steps: 10,
    audio_mode: 'built_in_audio', continuity_mode: 'latent_continuation', latent_context_frames: 39,
    continuity_prompt_from_last_frame: true, location_transition_preset: 'surreal',
  });
  // The server never sends the excluded keys, so this is what a profile carries.
  const profile = {
    video_mode: 'reference_to_video', render_pass: 'three_pass', resolution_preset: '4k', steps: 25,
    advanced_two_pass_vram_preset: '12gb', pass1_use_te_speed: true, two_pass_lora_strength: 0.8,
  };
  const merged = c.applyVideoProfileSettings(current, profile);
  assert.equal(merged.video_mode, 'reference_to_video');
  assert.equal(merged.render_pass, 'three_pass');
  assert.equal(merged.resolution_preset, '4k');
  assert.equal(merged.megapixels, 7.9688);
  assert.equal(merged.steps, 25);
  assert.equal(merged.advanced_two_pass_vram_preset, '12gb');
  assert.equal(merged.pass1_use_te_speed, true);
  assert.equal(merged.two_pass_lora_strength, 0.8);
  assert.equal(merged.audio_mode, 'built_in_audio');
  assert.equal(merged.continuity_mode, current.continuity_mode);
  assert.equal(merged.latent_context_frames, 39);
  assert.equal(merged.continuity_prompt_from_last_frame, true);
  assert.equal(merged.location_transition_preset, 'surreal');
  assert.equal(merged.advanced_two_pass_defaults_version, c.cloneMiniMaxH3Settings({}).advanced_two_pass_defaults_version);
});

test('the profile list labels name the video type and, for reference modes, the render pass', () => {
  const c = profileContext();
  assert.equal(c.videoProfileLabel({ name: 'A', video_mode: 'reference_to_video', render_pass: 'three_pass' }), 'A (Reference to Video, 2 pass advanced)');
  assert.equal(c.videoProfileLabel({ name: 'B', video_mode: 'reference_to_video', render_pass: 'single' }), 'B (Reference to Video, Single pass)');
  assert.equal(c.videoProfileLabel({ name: 'C', video_mode: 'text_to_video', render_pass: 'two_pass' }), 'C (Text to Video)');
  assert.equal(c.videoProfileLabel({ name: 'D' }), 'D');
});

// ---- the actions -----------------------------------------------------------------------------------------------
function harness({ answers = {}, segment = { id: 's1' }, global = false, saved = null } = {}) {
  const log = [];
  const select = {
    value: '', options: [], listeners: {},
    set textContent(_) { this.options.length = 0; },
    append(...nodes) { this.options.push(...nodes); },
    addEventListener(type, fn) { this.listeners[type] = fn; },
  };
  const addButton = {}, removeButton = {};
  const state = { miniMaxH3Settings: null };
  let profilesOnServer = [{ name: 'Cinema', video_mode: 'reference_to_video', render_pass: 'two_pass' }];
  const postJson = async (url, body) => {
    log.push(['post', url, body]);
    if (url.endsWith('list_video_profiles')) return { ok: true, profiles: profilesOnServer };
    if (url.endsWith('load_video_profile')) {
      return { ok: true, profile: { name: body.name, settings: { video_mode: 'reference_to_video', render_pass: 'three_pass', steps: 31 } } };
    }
    if (url.endsWith('save_video_profile')) {
      if (answers.exists && !body.overwrite) {
        const error = new Error('exists');
        error.data = { exists: true, name: 'Cinema' };
        throw error;
      }
      profilesOnServer = [...profilesOnServer.filter((item) => item.name !== body.name), { name: body.name, video_mode: 'reference_to_video', render_pass: 'three_pass' }];
      return { ok: true, profile: { name: body.name } };
    }
    if (url.endsWith('delete_video_profile')) {
      profilesOnServer = profilesOnServer.filter((item) => item.name !== body.name);
      return { ok: true, name: body.name };
    }
    throw new Error(`unexpected ${url}`);
  };
  const c = profileContext();
  const actions = c.createVideoProfileActions({
    controls: { select, addButton, removeButton }, state, postJson,
    toast: (message, isError) => log.push(['toast', message, Boolean(isError)]),
    confirmDestructiveAction: async (options) => { log.push(['confirm', options.title]); return { confirmed: answers.confirm !== false, optionChecked: false }; },
    promptForText: async (options) => { log.push(['prompt', options.title]); return 'name' in answers ? answers.name : 'My profile'; },
    requireActiveSegment: () => segment, wizardVideoSettings: { global },
    pushHistory: () => log.push(['pushHistory']),
    saveMiniMaxH3SettingsFromPanel: () => saved || c.cloneMiniMaxH3Settings({ video_mode: 'text_to_video', render_pass: 'single', audio_mode: 'built_in_audio' }),
    clearMiniMaxImageReferenceStartFrameOnModeSwitch: (seg, mode) => log.push(['clearStart', mode]),
    setMiniMaxH3RenderPassForSegment: (seg, pass) => log.push(['setPass', pass]),
    setMiniMaxH3ModeForSegment: (seg, mode) => log.push(['setMode', mode]),
    syncMiniMaxH3Panel: () => log.push(['sync']),
    autoSaveSessionQuiet: async (reason) => log.push(['autosave', reason]),
  });
  return { actions, select, addButton, removeButton, state, log };
}

test('refresh lists the saved profiles after a "No profile" choice and keeps the selection', async () => {
  const h = harness();
  await h.actions.refresh('cinema');
  assert.equal(h.select.options[0].textContent, 'No profile');
  assert.equal(h.select.options[1].textContent, 'Cinema (Reference to Video, 2 pass)');
  assert.equal(h.select.value, 'Cinema');
  assert.equal(h.removeButton.disabled, false);
  await h.actions.refresh('missing');
  assert.equal(h.select.value, '');
  assert.equal(h.removeButton.disabled, true, '- is disabled when no profile is selected');
});

test('+ asks for a name, saves the current settings and selects the new profile', async () => {
  const h = harness();
  await h.actions.saveCurrentAsProfile();
  const prompt = h.log.find((entry) => entry[0] === 'prompt');
  assert.match(prompt[1], /Save video profile/);
  const save = h.log.find((entry) => entry[0] === 'post' && entry[1].endsWith('save_video_profile'));
  assert.equal(save[2].name, 'My profile');
  assert.equal(save[2].overwrite, false);
  assert.equal(save[2].settings.audio_mode, 'built_in_audio', 'the panel settings are sent; the server drops the excluded keys');
  assert.equal(h.select.value, 'My profile');
  assert.ok(h.log.some((entry) => entry[0] === 'toast' && /Saved video profile "My profile"/.test(entry[1])));
});

test('cancelling the name prompt saves nothing', async () => {
  const h = harness({ answers: { name: null } });
  await h.actions.saveCurrentAsProfile();
  assert.equal(h.log.some((entry) => entry[0] === 'post'), false);
});

test('saving over an existing name asks before replacing it', async () => {
  const replaced = harness({ answers: { exists: true, name: 'Cinema', confirm: true } });
  await replaced.actions.saveCurrentAsProfile();
  assert.ok(replaced.log.some((entry) => entry[0] === 'confirm' && /Replace profile "Cinema"/.test(entry[1])));
  const saves = replaced.log.filter((entry) => entry[0] === 'post' && entry[1].endsWith('save_video_profile'));
  assert.deepEqual(saves.map((entry) => entry[2].overwrite), [false, true]);

  const kept = harness({ answers: { exists: true, name: 'Cinema', confirm: false } });
  await kept.actions.saveCurrentAsProfile();
  assert.equal(kept.log.filter((entry) => entry[0] === 'post' && entry[1].endsWith('save_video_profile')).length, 1);
});

test('choosing a profile applies it the same way the video type and pass buttons do', async () => {
  const h = harness();
  await h.actions.refresh('');
  h.select.value = 'Cinema';
  await h.actions.applySelectedProfile();
  assert.equal(h.state.miniMaxH3Settings.render_pass, 'three_pass');
  assert.equal(h.state.miniMaxH3Settings.steps, 31);
  assert.equal(h.state.miniMaxH3Settings.audio_mode, 'built_in_audio', 'audio is not changed by a profile');
  const order = h.log.filter((entry) => ['pushHistory', 'clearStart', 'setPass', 'setMode', 'sync', 'autosave'].includes(entry[0])).map((entry) => entry[0]);
  assert.deepEqual(order, ['pushHistory', 'clearStart', 'setPass', 'setMode', 'sync', 'autosave']);
  assert.ok(h.log.some((entry) => entry[0] === 'setPass' && entry[1] === 'three_pass'));
  assert.ok(h.log.some((entry) => entry[0] === 'setMode' && entry[1] === 'reference_to_video'));
  assert.ok(h.log.some((entry) => entry[0] === 'toast' && /Applied video profile "Cinema"/.test(entry[1])));
});

test('a locked scene gets the profile, the project settings stay untouched', async () => {
  const segment = { id: 's1', use_scene_minimax_h3_settings: true };
  const h = harness({ segment });
  await h.actions.refresh('');
  h.select.value = 'Cinema';
  await h.actions.applySelectedProfile();
  assert.equal(segment.minimax_h3_settings.render_pass, 'three_pass');
  assert.equal(h.state.miniMaxH3Settings, null);
});

test('choosing "No profile" or having no active scene changes nothing', async () => {
  const none = harness();
  await none.actions.refresh('');
  none.select.value = '';
  await none.actions.applySelectedProfile();
  assert.equal(none.log.filter((entry) => entry[0] !== 'post').length, 0);

  const noScene = harness({ segment: null });
  await noScene.actions.refresh('');
  noScene.select.value = 'Cinema';
  await noScene.actions.applySelectedProfile();
  assert.equal(noScene.log.some((entry) => entry[1] && String(entry[1]).endsWith('load_video_profile')), false);
  assert.equal(noScene.state.miniMaxH3Settings, null);
});

test('the wizard (global) mode applies without an active scene', async () => {
  const h = harness({ segment: null, global: true });
  await h.actions.refresh('');
  h.select.value = 'Cinema';
  await h.actions.applySelectedProfile();
  assert.equal(h.state.miniMaxH3Settings.render_pass, 'three_pass');
});

test('- asks first and only then deletes the profile for every project', async () => {
  const cancelled = harness({ answers: { confirm: false } });
  await cancelled.actions.refresh('Cinema');
  await cancelled.actions.deleteSelectedProfile();
  assert.ok(cancelled.log.some((entry) => entry[0] === 'confirm' && /Delete profile "Cinema"/.test(entry[1])));
  assert.equal(cancelled.log.some((entry) => entry[1] && String(entry[1]).endsWith('delete_video_profile')), false);

  const confirmed = harness();
  await confirmed.actions.refresh('Cinema');
  await confirmed.actions.deleteSelectedProfile();
  assert.ok(confirmed.log.some((entry) => entry[0] === 'post' && entry[1].endsWith('delete_video_profile') && entry[2].name === 'Cinema'));
  assert.equal(confirmed.select.value, '');
  assert.equal(confirmed.select.options.length, 1);

  const nothing = harness();
  await nothing.actions.refresh('');
  await nothing.actions.deleteSelectedProfile();
  assert.equal(nothing.log.some((entry) => entry[0] === 'confirm'), false);
});

test('server errors are reported and never leave the buttons disabled', async () => {
  const h = harness();
  await h.actions.refresh('');
  const original = h.log.length;
  const failing = profileContext().createVideoProfileActions({
    controls: { select: h.select, addButton: h.addButton, removeButton: h.removeButton }, state: {},
    postJson: async () => { throw new Error('disk full'); }, toast: (message, isError) => h.log.push(['toast', message, Boolean(isError)]),
    confirmDestructiveAction: async () => ({ confirmed: true }), promptForText: async () => 'x',
    requireActiveSegment: () => ({}), wizardVideoSettings: { global: false }, pushHistory() {},
    saveMiniMaxH3SettingsFromPanel: () => ({}), clearMiniMaxImageReferenceStartFrameOnModeSwitch() {},
    setMiniMaxH3RenderPassForSegment() {}, setMiniMaxH3ModeForSegment() {}, syncMiniMaxH3Panel() {}, autoSaveSessionQuiet: async () => {},
  });
  await failing.saveCurrentAsProfile();
  assert.ok(h.log.slice(original).some((entry) => entry[0] === 'toast' && entry[2] === true && /disk full/.test(entry[1])));
  assert.notEqual(h.addButton.disabled, true);
});

// ---- the name prompt ---------------------------------------------------------------------------------------------
function openPrompt(options) {
  const document = fakeDocument();
  const context = vm.createContext({
    document,
    makeButton: (label) => ({ ...document.createElement('button'), label }),
    makeCheckbox: () => ({ wrapper: document.createElement('label'), input: document.createElement('input') }),
  });
  vm.runInContext(`${readBuilderModule('confirm_dialog.mjs')};globalThis.promptForText = promptForText;`, context);
  const result = context.promptForText(options);
  const backdrop = document.body.children[0];
  const box = backdrop.children[0];
  const input = box.children.find((child) => child.tag === 'input');
  const actions = box.children[box.children.length - 1];
  const [cancel, confirm] = actions.children;
  return { result, backdrop, box, input, cancel, confirm, document };
}

test('the name prompt returns the trimmed text on Enter or OK and null on Cancel or Escape', async () => {
  const typed = openPrompt({ title: 'Save video profile', confirmLabel: 'Save' });
  assert.equal(typed.input.focused, true);
  assert.equal(typed.confirm.label, 'Save');
  typed.input.value = '  24 GB advanced  ';
  typed.document.listeners.keydown({ key: 'Enter', target: typed.input, preventDefault() {}, stopPropagation() {} });
  assert.equal(await typed.result, '24 GB advanced');

  const ok = openPrompt({ title: 't' });
  ok.input.value = 'Name';
  ok.confirm.onclick();
  assert.equal(await ok.result, 'Name');

  const cancelled = openPrompt({ title: 't' });
  cancelled.input.value = 'Name';
  cancelled.cancel.onclick();
  assert.equal(await cancelled.result, null);

  const escaped = openPrompt({ title: 't' });
  escaped.document.listeners.keydown({ key: 'Escape', preventDefault() {}, stopPropagation() {} });
  assert.equal(await escaped.result, null);
});

test('the name prompt will not accept an empty name', async () => {
  const p = openPrompt({ title: 't', initialValue: '' });
  p.input.value = '   ';
  p.confirm.onclick();
  assert.equal(p.backdrop.removed, false, 'the dialog stays open');
  p.input.value = 'Real name';
  p.confirm.onclick();
  assert.equal(await p.result, 'Real name');
});

// ---- the panel ---------------------------------------------------------------------------------------------------
test('the profile row sits above the video type buttons with + and - buttons', () => {
  assert.ok(source.includes('makeButton("+")'));
  assert.ok(source.includes('Save current video settings as a profile'));
  assert.ok(source.includes('Delete selected video profile'));
  assert.ok(source.includes('miniMaxEnginePanel.append(miniMaxBanner, miniMaxVideoProfileRow, miniMaxPipelineChooser, miniMaxRefmodNote, miniMaxRefmodClothing, miniMaxModeChooser'));
  assert.ok(source.includes('videoProfiles.wire();'));
  assert.ok(source.includes('"/vrgdg/music_builder/save_video_profile"'));
});
