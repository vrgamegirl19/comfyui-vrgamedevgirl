import assert from 'node:assert/strict';
import fs from 'node:fs';
import { register } from 'node:module';
import { test } from 'node:test';

// The Python twin (agent_api/temporal_effect.py) is checked against the same cases.
register('./stub_comfy_loader.mjs', import.meta.url);
const { storyboardTemporalWorldEffectForScene } = await import(new URL('../web/storyboard_builder/video_style.mjs', import.meta.url));
const { cases } = JSON.parse(fs.readFileSync(new URL('./temporal_effect_cases.json', import.meta.url), 'utf8'));

for (const item of cases) {
  test(`temporal / world effect verbiage: ${item.name}`, () => {
    const d = item.defaults;
    const state = { temporalWorldEffect: d.temporal_world_effect, temporalWorldEffectCustom: d.temporal_world_effect_custom,
      temporalProtectedCharacters: d.temporal_protected_characters, temporalProtectedCustom: d.temporal_protected_custom };
    assert.equal(storyboardTemporalWorldEffectForScene(item.card, state)?.exact_verbiage || '', item.expected);
  });
}
