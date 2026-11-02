const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');
const { readBuilderModule, functionSource } = require('./builder_source.cjs');

function context(extra = {}) {
  const c = vm.createContext({ ...extra });
  vm.runInContext(readBuilderModule('api_video.mjs'), c);
  return c;
}

test('reference collection keeps ordering, deduplicates images, and respects mapped-reference switch', () => {
  const c = context();
  const scene = { start: 0, end: 4, api_video_prompt: 'Image 1 waves', api_video_image_paths: ['one.png', 'two.png'] };
  const refs = c.apiVideoReferences(scene, [{ path: 'one.png', label: 'Character' }]);
  assert.deepEqual(Array.from(refs.reference_images, item => item.path), ['one.png', 'two.png']);
  assert.equal(c.validateApiVideoScene(scene, refs).length, 0);
  scene.api_video_settings = { use_mapped_references: false };
  assert.equal(c.apiVideoReferences(scene, [{ path: 'mapped.png' }]).reference_images[0].path, 'one.png');
});

test('scene preflight accepts audio-only references and rejects oversized scenes and missing prompts', () => {
  const c = context();
  const scene = { start: 0, end: 1.5, api_video_prompt: 'Follow Audio 1', api_video_audio_paths: ['voice.wav'] };
  assert.equal(c.validateApiVideoScene(scene, c.apiVideoReferences(scene)).length, 0);
  scene.end = 31;
  scene.api_video_prompt = '';
  assert.equal(c.validateApiVideoScene(scene, c.apiVideoReferences(scene)).length, 2);
});

test('official SaveVideo output is collected without mistaking input previews or animated flags for video', () => {
  const c = vm.createContext({});
  vm.runInContext(functionSource(readBuilderModule('comfy_api.mjs'), 'extractVideosFromHistory'), c);
  const history = { p: { outputs: {
    load: { images: [{ filename: 'reference.mp4' }], animated: [true] },
    save_video: { images: [{ filename: 'generated.mp4', subfolder: 'vrgdg_api_video' }], animated: [true] },
  } } };
  const result = c.extractVideosFromHistory(history, 'p', 'save_video');
  assert.equal(result.length, 1);
  assert.equal(result[0].filename, 'generated.mp4');
});

test('partner queue passes current ComfyUI credentials only in extra_data and refuses unauthenticated submissions', async () => {
  const calls = [];
  const c = vm.createContext({
    errorLogContext: () => ({}), waitForComfyQueueIdle: async () => {},
    builderAutomaticMemoryCleanupEnabled: true, crypto: { randomUUID: () => 'client' }, app: {},
    api: { authToken: 'test-token', apiKey: undefined, fetchApi: async (url, options) => {
      calls.push(JSON.parse(options.body));
      return { ok: true, json: async () => ({ prompt_id: 'p' }) };
    } },
  });
  const source = readBuilderModule('comfy_api.mjs');
  vm.runInContext(functionSource(source, 'comfyOrgCredentials'), c);
  vm.runInContext(functionSource(source, 'queueWorkflowPrompt'), c);
  await c.queueWorkflowPrompt({ seedance: { inputs: {} } }, { partnerApi: true });
  assert.equal(calls[0].extra_data.auth_token_comfy_org, 'test-token');
  assert.equal(JSON.stringify(calls[0].prompt).includes('test-token'), false);
  c.api.authToken = undefined;
  await assert.rejects(c.queueWorkflowPrompt({}, { partnerApi: true }), /Sign into your ComfyUI account/);
  assert.equal(calls.length, 1);
  await c.queueWorkflowPrompt({}, {});
  assert.equal('extra_data' in calls[1], false);
});

test('partner queue reads the signed-in account from the frontend auth stores outside a normal Queue', async () => {
  const calls = [];
  const stores = new Map([
    ['auth', { getWorkspaceAuthToken: async () => 'store-token' }],
    ['apiKeyAuth', { getApiKey: () => 'store-key' }],
  ]);
  const vueApp = { config: { globalProperties: { $pinia: { _s: stores } } } };
  const c = vm.createContext({
    errorLogContext: () => ({}), waitForComfyQueueIdle: async () => {},
    builderAutomaticMemoryCleanupEnabled: true, crypto: { randomUUID: () => 'client' }, app: {}, console,
    document: { getElementById: (id) => (id === 'vue-app' ? { __vue_app__: vueApp } : null) },
    api: { authToken: undefined, apiKey: undefined, fetchApi: async (url, options) => {
      calls.push(JSON.parse(options.body));
      return { ok: true, json: async () => ({ prompt_id: 'p' }) };
    } },
  });
  const source = readBuilderModule('comfy_api.mjs');
  vm.runInContext(functionSource(source, 'comfyOrgCredentials'), c);
  vm.runInContext(functionSource(source, 'queueWorkflowPrompt'), c);
  await c.queueWorkflowPrompt({}, { partnerApi: true });
  assert.equal(calls[0].extra_data.auth_token_comfy_org, 'store-token');
  assert.equal(calls[0].extra_data.api_key_comfy_org, 'store-key');
});

test('API mode preserves the local mode and survives engine normalization', () => {
  const c = vm.createContext({ state: { projectVideoEngine: 'api_nodes', videoModelMode: 'flf' } });
  vm.runInContext(functionSource(readBuilderModule('controls.mjs'), 'normalizeProjectVideoEngine'), c);
  vm.runInContext(functionSource(readBuilderModule('video_settings_panel.mjs'), 'currentVideoMode'), c);
  vm.runInContext(functionSource(readBuilderModule('video_settings_panel.mjs'), 'syncVideoModePanel'), c);
  c.syncVideoModePanel();
  assert.equal(c.currentVideoMode(), 'api_reference_to_video');
  assert.equal(c.state.videoModelMode, 'flf');
  assert.equal(c.normalizeProjectVideoEngine('api_nodes'), 'api_nodes');
  c.state.projectVideoEngine = 'ltx';
  assert.equal(c.currentVideoMode(), 'flf');
});

test('API scene renderer uses the partner queue, exact scene trim, collection, and saves the selected take', async () => {
  const scene = { id: 'one', start: 3, end: 5.5, api_video_prompt: 'Image 1 waves' };
  const posts = [], queues = [], saves = [];
  const c = context({
    activeSegment: () => scene, apiVideoPanel: { save() {}, references: () => ({ reference_images: [{ path: 'ref.png' }], reference_videos: [], reference_audios: [] }) },
    projectInput: { value: 'project' }, state: {}, sceneSlotNumber: () => 1, timelineSegmentDuration: s => s.end - s.start,
    renderList() {}, saveSessionForSceneVideo: async () => 'scene.srt', segmentUsesNoLipSyncPerformance: () => false,
    postJson: async (url, body) => {
      posts.push([url, body]);
      if (url.includes('build_seedance')) return { prompt: { seedance: {} }, output_root: 'output' };
      return { video_path: url.includes('trim') ? 'trimmed.mp4' : 'collected.mp4', thumbnail_path: 'thumb.png', video_folder: 'scenes' };
    },
    queueWorkflowPrompt: async (graph, options) => { queues.push(options); return { prompt_id: 'p' }; },
    waitForVideos: async (id, status, cancel, fallback, options) => {
      assert.equal(id, 'p'); assert.equal(options.outputNodeId, 'save_video');
      return [{ filename: 'generated.mp4', subfolder: 'seedance' }];
    }, resolveComfyVideoPath: () => '', autoSaveSessionQuiet: async reason => saves.push(reason),
    pushHistory() {}, applySceneLutToRenderedVideo: async () => ({}), applySceneAdjustToRenderedVideo: async () => ({}), applySceneFilmGrainToRenderedVideo: async () => ({}),
    activateSegmentVideoPath: (s, path) => { s.video_path = path; }, syncPreview() {}, render() {},
  });
  vm.runInContext(functionSource(readBuilderModule('video_render.mjs'), 'renderApiSceneVideoWithProgress'), c);
  assert.equal(await c.renderApiSceneVideoWithProgress(scene, 0, null), 'collected.mp4');
  assert.equal(queues.length, 1);
  assert.equal(queues[0].partnerApi, true);
  assert.equal(posts[0][1].duration, 2.5);
  assert.equal(posts[1][1].duration, 2.5);
  assert.equal(posts[1][1].source_path, 'output/seedance/generated.mp4');
  assert.equal(scene.video_status, 'done');
  assert.equal(scene.api_video_prompt_id, 'p');
  assert.equal(saves.length, 2);
});

test('API panel saves edits to the correct scene and restores them after scene changes and JSON reload', async () => {
  const elements = [];
  const element = (tag) => {
    const result = { tag, value: '', style: {}, listeners: {}, children: [], append(...items) { this.children.push(...items); }, addEventListener(name, callback) { this.listeners[name] = callback; } };
    elements.push(result);
    return result;
  };
  const first = { id: 'first' }, second = { id: 'second' };
  let active = first;
  const c = context({
    document: { createElement: element },
    makeSelect: (options, value) => Object.assign(element('select'), { value }),
    makeCheckbox: (_, checked) => ({ input: Object.assign(element('checkbox'), { checked }), wrapper: element('label') }),
    makeButton: () => element('button'), makeField: (_, input) => input, makeSettingsSection: () => element('details'),
    toast() {},
  });
  let converted = 0;
  const panel = c.createApiVideoPanel({ activeSegment: () => active, state: { projectVideoEngine: 'api_nodes' }, getMappedImages: () => [], autoSave: async () => {}, onRender: async () => {},
    onConvertScene: async () => { converted++; active.api_video_prompt = 'Converted selected scene'; } });
  panel.sync();
  assert.equal(panel.panel.style.display, 'block');
  const prompt = elements.find(item => item.tag === 'textarea');
  prompt.value = 'Image 1 on the stage';
  prompt.listeners.input();
  active = second;
  panel.sync();
  assert.equal(prompt.value, '');
  prompt.value = 'Image 1 outside';
  prompt.listeners.input();
  active = JSON.parse(JSON.stringify(first));
  panel.sync();
  assert.equal(prompt.value, 'Image 1 on the stage');
  assert.equal(second.api_video_prompt, 'Image 1 outside');
  assert.equal(active.api_video_settings.resolution, '720p');
  await panel.convertButton.onclick();
  assert.equal(converted, 1);
  assert.equal(prompt.value, 'Converted selected scene');
  assert.equal(panel.convertButton.disabled, false);
  active = null;
  panel.sync();
  assert.equal(panel.convertButton.disabled, true);
});

test('API batch preflight bypasses local model, scene-image, SRT and project-audio requirements', () => {
  const scene = { id: 'one', start: 0, end: 5, api_video_prompt: 'Image 1 waves' };
  const c = context({ state: { projectVideoEngine: 'api_nodes' },
    sceneDisplayName: () => 'Scene 1', apiVideoReferences: () => ({ reference_images: [{ path: 'one.png' }], reference_videos: [], reference_audios: [] }),
    normalizeBatchScope: () => 'all', batchTargetItems: () => [{ segment: scene }],
    normalizeProjectVideoEngine: value => value, segmentIndexInfo: () => ({ index: 0 }),
    selectedSegmentVideoPath: () => '', projectInput: { value: 'project' },
  });
  const source = readBuilderModule('scene_render_prep.mjs');
  c.apiVideoReferences = () => ({ reference_images: [{ path: 'one.png' }], reference_videos: [], reference_audios: [] });
  vm.runInContext(functionSource(source, 'validateSceneReadyForVideo'), c);
  vm.runInContext(functionSource(source, 'validateRenderAllReady'), c);
  assert.equal(c.validateRenderAllReady().length, 0);
  scene.api_video_prompt = '';
  assert.match(c.validateRenderAllReady()[0], /Seedance.*prompt is missing/);
});

for (const custom of [false, true]) {
  test(`scene-audio rendering trims the ${custom ? 'custom clip offset' : 'project timeline position'}, attaches Audio 1 and restores supplied audio`, async () => {
    const scene = { id: 'one', start: 12.5, end: 18, api_video_prompt: 'Image 1 sings',
      api_video_settings: { audio_mode: 'scene_audio', lip_sync_image: 1 },
      ...(custom ? { custom_audio_path: 'custom.wav', custom_audio_source_start: 2.25 } : {}) };
    const posts = [];
    let queued = 0;
    const c = context({
      activeSegment: () => scene, apiVideoPanel: { save() {}, references: () => ({ reference_images: [{ path: 'singer.png' }], reference_videos: [], reference_audios: [] }) },
      projectInput: { value: 'project' }, state: {}, audioInput: { value: 'song.wav' }, currentProjectAudioPath: () => 'song.wav',
      sceneSlotNumber: () => 1, timelineSegmentDuration: s => s.end - s.start,
      audioSourceStart: s => s.custom_audio_source_start || 0, prepareEditedAudio: async () => null,
      segmentUsesNoLipSyncPerformance: () => false, renderList() {}, saveSessionForSceneVideo: async () => 'scene.srt',
      postJson: async (url, body) => {
        posts.push([url, body]);
        if (url.includes('prepare_seedance')) return { audio_path: 'prepared.wav' };
        if (url.includes('build_seedance')) return { prompt: { seedance: {} }, output_root: 'output' };
        return { video_path: 'collected.mp4', thumbnail_path: 'thumb.png', video_folder: 'scenes' };
      },
      queueWorkflowPrompt: async () => { queued++; return { prompt_id: 'p' }; },
      waitForVideos: async () => [{ filename: 'generated.mp4' }], resolveComfyVideoPath: () => '',
      autoSaveSessionQuiet: async () => {}, pushHistory() {},
      applySceneLutToRenderedVideo: async () => ({}), applySceneAdjustToRenderedVideo: async () => ({}), applySceneFilmGrainToRenderedVideo: async () => ({}),
      activateSegmentVideoPath: (s, path) => { s.video_path = path; }, syncPreview() {}, render() {},
    });
    vm.runInContext(functionSource(readBuilderModule('video_render.mjs'), 'renderApiSceneVideoWithProgress'), c);
    await c.renderApiSceneVideoWithProgress(scene, 0, null);
    assert.equal(posts[0][1].source_path, custom ? 'custom.wav' : 'song.wav');
    assert.equal(posts[0][1].start, custom ? 2.25 : 12.5);
    assert.equal(posts[0][1].duration, 5.5);
    assert.equal(posts[1][1].reference_audios[0], 'prepared.wav');
    assert.equal(posts[1][1].lip_sync_image, 1);
    assert.equal(posts[2][1].restore_audio_path, 'prepared.wav');
    assert.equal(posts[2][1].restore_audio_start_seconds, 0);
    assert.equal(queued, 1);
  });
}

test('scene audio reserves Audio 1 and blocks conflicting manual audio files or a missing performer', () => {
  const c = context();
  const scene = { start: 0, end: 5, api_video_prompt: 'Sing', api_video_settings: { audio_mode: 'scene_audio', lip_sync_image: 2 } };
  const errors = c.validateApiVideoScene(scene, { reference_images: [{ path: 'one.png' }], reference_videos: [], reference_audios: ['extra.wav'] });
  assert.match(errors.join(' '), /Clear the audio-reference file list/);
  assert.match(errors.join(' '), /performer image is missing/);
});

test('custom audio placement preserves leading silence, selected source length, and already elapsed audio', () => {
  const c = context();
  const scene = { start: 10, end: 16, custom_audio_path: 'custom.wav', custom_audio_source_start: 2,
    custom_audio_duration: 3, custom_audio_timeline_start: 11 };
  const late = c.seedanceSceneAudioInput(scene, 'song.wav');
  assert.equal(late.start, 2);
  assert.equal(late.leading_silence, 1);
  assert.equal(late.source_duration, 3);
  scene.custom_audio_timeline_start = 8;
  const early = c.seedanceSceneAudioInput(scene, 'song.wav');
  assert.equal(early.start, 4);
  assert.equal(early.leading_silence, 0);
  assert.equal(early.source_duration, 1);
  scene.custom_audio_timeline_start = 20;
  assert.throws(() => c.seedanceSceneAudioInput(scene, 'song.wav'), /does not overlap/);
});

test('final soundtrack honours custom scene audio alongside project-track scenes', () => {
  const c = context();
  const scenes = [{ start: 0, end: 5 }, { start: 5, end: 10, custom_audio_path: 'voice.wav', api_video_audio_reference_path: 'prepared_voice.wav' }];
  const items = c.seedanceStitchAudioItems(scenes, 'song.wav');
  assert.equal(items[0].path, 'song.wav');
  assert.equal(items[0].start, 0);
  assert.equal(items[1].path, 'prepared_voice.wav');
  assert.equal(items[1].start, 0);
  assert.equal(items[1].duration, 5);
});
