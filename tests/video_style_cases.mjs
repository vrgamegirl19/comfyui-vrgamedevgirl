import assert from 'node:assert/strict';
import fs from 'node:fs';
import { register } from 'node:module';
import { test } from 'node:test';

// The Python twin (agent_api/video_style.py) is checked against the same cases.
// Run with --write-cases to regenerate the expected values from the JS.
register('./stub_comfy_loader.mjs', import.meta.url);
const { storyboardSceneVideoStyle, storyboardMiniMaxVideoStyleVerbiage, storyboardFxContract } = await import(new URL('../web/storyboard_builder/video_style.mjs', import.meta.url));
const file = new URL('./video_style_cases.json', import.meta.url);

const styleOf = (item) => {
  const d = item.defaults;
  const { value, custom } = storyboardSceneVideoStyle(item.card, { videoStyle: d.video_style, videoStyleCustom: d.video_style_custom });
  return storyboardMiniMaxVideoStyleVerbiage(value, custom);
};
const fxOf = (item) => {
  const c = storyboardFxContract(item.defaults.fx_preset, item.defaults.fx_custom_json, item.shot_index);
  return c ? `FX accent: ${c.cue} Timing: ${c.timing}. Keep the mapped subject stable and readable.` : '';
};

if (process.argv.includes('--write-cases')) {
  const style = [
    { name: 'project default', card: {}, defaults: { video_style: 'film_noir' } },
    { name: 'scene overrides the default', card: { video_style: 'cyberpunk' }, defaults: { video_style: 'film_noir' } },
    { name: 'no style', card: {}, defaults: {} },
    { name: 'project custom', card: {}, defaults: { video_style: 'custom', video_style_custom: ' Teal haze and grain ' } },
    { name: 'scene custom', card: { video_style: 'custom', video_style_custom: 'Scene wording' }, defaults: { video_style: 'custom', video_style_custom: 'Project wording' } },
    { name: 'empty custom', card: {}, defaults: { video_style: 'custom' } },
    { name: 'unknown key', card: {}, defaults: { video_style: 'nope' } },
  ].map((item) => ({ ...item, expected: styleOf(item) }));
  const fx = [
    { name: 'off', defaults: {}, shot_index: 0 },
    { name: 'lighting shot 0', defaults: { fx_preset: 'lighting' }, shot_index: 0 },
    { name: 'lighting shot 4 wraps', defaults: { fx_preset: 'lighting' }, shot_index: 4 },
    { name: 'music video shot 2', defaults: { fx_preset: 'music_video' }, shot_index: 2 },
    { name: 'unknown preset', defaults: { fx_preset: 'nope' }, shot_index: 0 },
    { name: 'custom json', defaults: { fx_preset: 'custom', fx_custom_json: '{"primary":"A red flash.","cues":["Dust drifts.",{"text":"Sparks."}],"secondary":"A hum.","timing":"on the drop"}' }, shot_index: 1 },
    { name: 'custom list', defaults: { fx_preset: 'custom', fx_custom_json: '["One.","Two.","One."]' }, shot_index: 3 },
    { name: 'custom bad json', defaults: { fx_preset: 'custom', fx_custom_json: '{oops' }, shot_index: 0 },
  ].map((item) => ({ ...item, expected: fxOf(item) }));
  fs.writeFileSync(file, `${JSON.stringify({ style, fx }, null, 2)}\n`);
} else {
  const { style, fx } = JSON.parse(fs.readFileSync(file, 'utf8'));
  for (const item of style) test(`video style verbiage: ${item.name}`, () => assert.equal(styleOf(item), item.expected));
  for (const item of fx) test(`FX accent: ${item.name}`, () => assert.equal(fxOf(item), item.expected));
}
