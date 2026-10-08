const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const {readBuilderModule, functionSource} = require('./builder_source.cjs');

function fixture() {
  const c = vm.createContext({});
  vm.runInContext(readBuilderModule('minimax_keyframe_state.mjs'), c);
  return c;
}

test('Normal ignores an existing last frame, while FLF requires one', () => {
  const c = fixture();
  const scene = {minimax_h3_i2v_frame_mode:'normal', first_last_frame_end_image_path:'end.png'};
  assert.equal(c.miniMaxI2VFramePaths(scene, 'first.png').last, '');
  scene.minimax_h3_i2v_frame_mode = 'flf';
  assert.equal(c.miniMaxI2VFramePaths(scene, 'first.png').last, 'end.png');
  scene.first_last_frame_end_image_path = '';
  assert.throws(() => c.miniMaxI2VFramePaths(scene, 'first.png'), /last-frame image/);
  assert.throws(() => c.miniMaxI2VFramePaths(scene, ''), /first-frame image/);
});

test('legacy end images migrate to FLF and the mode survives JSON save/reload', () => {
  const c = fixture();
  assert.equal(c.miniMaxI2VFrameMode({}), 'normal');
  assert.equal(c.miniMaxI2VFrameMode({first_last_frame_end_image_path:'old.png'}), 'flf');
  const scene = JSON.parse(JSON.stringify({minimax_h3_i2v_frame_mode:'normal', first_last_frame_end_image_path:'old.png'}));
  assert.equal(c.miniMaxI2VFrameMode(scene), 'normal');
});

test('timeline eligibility uses MiniMax scene mode and does not inherit a stale LTX FLF setting', () => {
  const source = readBuilderModule('timeline_view.mjs');
  const start = source.indexOf('const videoMode = currentVideoMode();');
  const end = source.indexOf('const previewThumbPath', start);
  const code = source.slice(start, end) + '\nshowFirstLastFrameThumb;';
  const c = fixture();
  const scene = {minimax_h3_i2v_frame_mode:'flf'};
  Object.assign(c, {state:{projectVideoEngine:'minimax_h3'}, segment:scene, isOverlay:false,
    currentVideoMode:()=> 'flf', normalizeProjectVideoEngine:v=>v, miniMaxH3ModeForSegment:()=>c.mode,
    rtvReferenceBehaviorForSegment:()=> 'off', mode:'image_to_video'});
  const run = () => vm.runInContext(`{${code}}`, c);
  assert.equal(run(), true);
  scene.minimax_h3_i2v_frame_mode = 'normal';
  assert.equal(run(), false);
  scene.minimax_h3_i2v_frame_mode = 'flf';
  c.mode = 'reference_to_video';
  assert.equal(run(), false);
  c.state.projectVideoEngine = 'ltx';
  assert.equal(run(), true);
});

test('MiniMax timeline draws its own first and last frames without using LTX chaining', () => {
  const c = fixture();
  const element = () => ({style:{}, children:[], append(...items){this.children.push(...items);}});
  Object.assign(c, {state:{projectVideoEngine:'minimax_h3'},normalizeProjectVideoEngine:v=>v,
    miniMaxH3ModeForSegment:()=> 'image_to_video', segmentImageSource:()=>({path:'own_first.png'}),
    timelineImageSourceUrl:s=>s.path || s.data || '', document:{createElement:()=>element()},
    firstLastFramePromptReferences:()=>assert.fail('MiniMax must not use LTX frame resolution'),
    firstLastFrameStartImageSource:()=>assert.fail('MiniMax must not chain a previous frame'),
    firstLastFrameResolvedEndImageSource:()=>assert.fail('MiniMax must not resolve a chained end')});
  vm.runInContext(functionSource(readBuilderModule('selection_preview.mjs'), 'appendTimelineFirstLastFrameThumbnail'), c);
  const scene = {minimax_h3_i2v_frame_mode:'flf',first_last_frame_end_image_path:'own_last.png'};
  const block = element();
  c.appendTimelineFirstLastFrameThumbnail(block, scene);
  const slots = block.children[0].children;
  assert.equal(slots[0].children[0].src, 'own_first.png');
  assert.equal(slots[2].children[0].src, 'own_last.png');
  scene.first_last_frame_end_image_path = '';
  const missing = element();
  c.appendTimelineFirstLastFrameThumbnail(missing, scene);
  assert.equal(missing.children[0].children[2].children[0].textContent, 'END?');
});

test('prompt writing sees both endpoints in FLF and only the opening image in Normal', () => {
  const c = fixture();
  Object.assign(c, {segmentImageSource:s=>({path:s.approved_image_path || ''}), selectedSegmentImagePath:()=>'', normalizeMiniMaxH3Mode:v=>v});
  const source = readBuilderModule('minimax_prompt.mjs');
  vm.runInContext(functionSource(source, 'miniMaxH3PromptVisionImages'), c);
  vm.runInContext(functionSource(source, 'miniMaxH3ReferenceAssignmentLines'), c);
  const scene = {approved_image_path:'first.png', first_last_frame_end_image_path:'last.png', minimax_h3_i2v_frame_mode:'flf'};
  assert.deepEqual(Array.from(c.miniMaxH3PromptVisionImages(scene,'image_to_video'), item=>item.path), ['first.png','last.png']);
  assert.match(c.miniMaxH3ReferenceAssignmentLines(scene,'image_to_video')[1], /Picture 2.*exact last frame/);
  scene.minimax_h3_i2v_frame_mode = 'normal';
  assert.deepEqual(Array.from(c.miniMaxH3PromptVisionImages(scene,'image_to_video'), item=>item.path), ['first.png']);
  assert.equal(c.miniMaxH3ReferenceAssignmentLines(scene,'image_to_video').length, 1);
  scene.minimax_h3_i2v_frame_mode = 'flf';
  scene.first_last_frame_end_image_path = '';
  assert.throws(()=>c.miniMaxH3PromptVisionImages(scene,'image_to_video'), /last-frame image/);
});

test('frame controls save per scene, hide last in Normal, and keep first/last assignments independent', async () => {
  const c = fixture();
  let archived = 0;
  let repaints = 0;
  const saved = [];
  const element = () => ({style:{}, removeAttribute(){}, replaceChildren(){}});
  const controls = {mode:element(), cards:{}};
  for (const role of ['first','last']) controls.cards[role] = Object.fromEntries(
    ['card','preview','label','existing','upload','clear','file'].map(key => [key, element()]));
  Object.assign(c, {Option: function(text,value){this.text=text;this.value=value;},
    postJson: async (url,payload) => {assert.equal(url, '/vrgdg/music_builder/archive_scene_image'); assert.equal(payload.project_folder,'project'); return {saved_path:`archived_${++archived}.png`};},
    toast: () => assert.fail('unexpected error')});
  vm.runInContext(readBuilderModule('minimax_keyframes.mjs'), c);
  const scene = {id:'s1',label:'Scene 1', approved_image_path:'first.png', image_history:['first.png','generated_end.png']};
  c.wireMiniMaxKeyframes({controls,activeSegment:()=>scene,segmentImageSource:s=>s?.custom_image_path ? {path:s.custom_image_path} : s?.approved_image_path ? {path:s.approved_image_path} : null,
    allEditableSegments:()=>[scene], projectFolder:()=> 'project',sceneSlotNumber:()=>1,pushHistory:()=>{},autoSaveSessionQuiet:async r=>saved.push(r),syncMiniMaxH3Panel:()=>{},renderList:()=>{},renderTimeline:()=>repaints++});
  controls.sync(scene,'image_to_video');
  assert.equal(controls.cards.last.card.style.display,'none');
  controls.mode.value = 'flf';
  await controls.mode.onchange();
  controls.sync(scene,'image_to_video');
  assert.equal(controls.cards.last.card.style.display,'flex');
  controls.cards.last.existing.value = '1';
  await controls.cards.last.existing.onchange();
  assert.equal(scene.first_last_frame_end_image_path,'archived_1.png');
  assert.equal(scene.approved_image_path,'first.png');
  controls.cards.first.existing.value = '1';
  await controls.cards.first.existing.onchange();
  assert.equal(scene.custom_image_path,'archived_2.png');
  assert.equal(scene.first_last_frame_end_image_path,'archived_1.png');
  await controls.cards.last.clear.onclick();
  assert.equal(scene.first_last_frame_end_image_path,'');
  assert.equal(scene.custom_image_path,'archived_2.png');
  assert.equal(saved.length,4);
  assert.equal(repaints,4);
  controls.sync(null,'image_to_video');
  assert.equal(controls.mode.disabled,true);
});
