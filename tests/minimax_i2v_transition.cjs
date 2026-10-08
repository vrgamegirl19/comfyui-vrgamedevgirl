const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const {readBuilderModule, functionSource} = require('./builder_source.cjs');
function fixture() {
  const c = vm.createContext({});
  for (const name of ['minimax_h3.mjs', 'minimax_keyframe_state.mjs', 'minimax_i2v_transition.mjs']) vm.runInContext(readBuilderModule(name), c);
  return c;
}
test('all presets produce FLF guidance while Normal and other modes ignore it', () => {
  const c = fixture();
  const scene = {minimax_h3_i2v_frame_mode:'flf'};
  for (const style of ['natural','surreal_morph','dreamlike_dissolve','environment_transformation','camera_reveal','custom']) {
    const settings = c.cloneMiniMaxH3Settings({i2v_transition_style:style,i2v_transition_direction:'  petals become stars  '});
    const prompt = c.miniMaxI2VTransitionPrompt(scene,'image_to_video',settings);
    assert.match(prompt,/petals become stars/);
    assert.match(prompt,/story beat, lyrics\/audio timing, performance requirements and scene defaults/);
    assert.equal(c.miniMaxI2VTransitionPrompt(scene,'reference_to_video',settings),'');
    assert.equal(c.miniMaxI2VTransitionPrompt({...scene,minimax_h3_i2v_frame_mode:'normal'},'image_to_video',settings),'');
  }
  assert.match(c.miniMaxI2VTransitionPrompt(scene,'image_to_video',{i2v_transition_style:'surreal_morph'}),/Creatively morph/);
  assert.match(c.miniMaxI2VTransitionPrompt(scene,'image_to_video',{}),/without morphing/);
});
test('transition survives pass switching and project reload independently of sampler profiles', () => {
  const c = fixture();
  let s = c.cloneMiniMaxH3Settings({video_mode:'image_to_video',render_pass:'single'});
  s = c.selectMiniMaxH3PassSettings(s,'two_pass');
  s.i2v_transition_style = 'surreal_morph';
  s.i2v_transition_direction = 'flowers into an ocean';
  s = c.selectMiniMaxH3PassSettings(s,'single');
  s = c.cloneMiniMaxH3Settings(JSON.parse(JSON.stringify(s)));
  assert.equal(s.i2v_transition_style,'surreal_morph');
  assert.equal(s.i2v_transition_direction,'flowers into an ocean');
  assert.equal(c.cloneMiniMaxH3Settings({i2v_transition_style:'invalid'}).i2v_transition_style,'natural');
});
test('effective settings inherit global transitions unless the existing scene lock is checked', () => {
  const c = fixture();
  Object.assign(c,{state:{miniMaxH3Settings:{i2v_transition_style:'surreal_morph',i2v_transition_direction:'global'}},videoSettingsSegment:()=>null});
  vm.runInContext(functionSource(readBuilderModule('minimax_panel.mjs'),'miniMaxH3SettingsForSegment'),c);
  const scene = {minimax_h3_settings:{i2v_transition_style:'camera_reveal',i2v_transition_direction:'local'}};
  assert.equal(c.miniMaxH3SettingsForSegment(scene).i2v_transition_direction,'global');
  scene.use_scene_minimax_h3_settings = true;
  assert.equal(c.miniMaxH3SettingsForSegment(scene).i2v_transition_style,'camera_reveal');
  assert.equal(c.miniMaxH3SettingsForSegment(scene).i2v_transition_direction,'local');
  c.state.miniMaxH3Settings.i2v_transition_direction = 'changed global';
  assert.equal(c.miniMaxH3SettingsForSegment(scene).i2v_transition_direction,'local');
  scene.use_scene_minimax_h3_settings = false;
  assert.equal(c.miniMaxH3SettingsForSegment(scene).i2v_transition_direction,'changed global');
});
test('transition controls display globally and for FLF scenes with correct lock scope', () => {
  const c = fixture();
  const element = () => ({style:{},removeAttribute(){},replaceChildren(){}});
  const card = () => Object.fromEntries(['card','preview','label','existing','upload','clear','file'].map(k=>[k,element()]));
  const controls = {mode:element(),cards:{first:card(),last:card()},transitionSection:element(),transitionStyle:element(),transitionDirection:element(),transitionScope:element()};
  Object.assign(c,{Option:function(){}});
  vm.runInContext(readBuilderModule('minimax_keyframes.mjs'),c);
  c.wireMiniMaxKeyframes({controls,activeSegment:()=>null,segmentImageSource:()=>({}),allEditableSegments:()=>[],syncMiniMaxH3Panel:()=>{}});
  const settings = {i2v_transition_style:'camera_reveal',i2v_transition_direction:'orbit'};
  controls.sync(null,'image_to_video',settings);
  assert.equal(controls.transitionSection.style.display,'flex');
  assert.match(controls.transitionScope.textContent,/Global/);
  assert.equal(controls.transitionDirection.value,'orbit');
  controls.sync({minimax_h3_i2v_frame_mode:'normal'},'image_to_video',settings);
  assert.equal(controls.transitionSection.style.display,'none');
  controls.sync({minimax_h3_i2v_frame_mode:'flf',use_scene_minimax_h3_settings:true},'image_to_video',settings);
  assert.match(controls.transitionScope.textContent,/Locked scene/);
  assert.equal(controls.transitionSection.style.display,'flex');
  controls.sync(null,'text_to_video',settings);
  assert.equal(controls.transitionSection.style.display,'none');
});
