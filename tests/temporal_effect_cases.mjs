import assert from 'node:assert/strict';
import fs from 'node:fs';
import { register } from 'node:module';
import { test } from 'node:test';

// The Python twin (agent_api/temporal_effect.py) writes the exact "Temporal / World Effect — Label — Mandatory:" block for these cases.
// The Builder no longer appends that block: it hands the same direction and the same protection of characters to the LLM as
// writing_guidance, which the LLM works into the prompt. So the Builder is checked on what it still shares with the block: which
// cases give an effect at all, its label, its direction and who is protected.
register('./stub_comfy_loader.mjs', import.meta.url);
const { storyboardTemporalWorldEffectForScene } = await import(new URL('../web/storyboard_builder/video_style.mjs', import.meta.url));
const { cases } = JSON.parse(fs.readFileSync(new URL('./temporal_effect_cases.json', import.meta.url), 'utf8'));

for (const item of cases) {
  test(`temporal / world effect: ${item.name}`, () => {
    const d = item.defaults;
    const state = { temporalWorldEffect: d.temporal_world_effect, temporalWorldEffectCustom: d.temporal_world_effect_custom,
      temporalProtectedCharacters: d.temporal_protected_characters, temporalProtectedCustom: d.temporal_protected_custom };
    const effect = storyboardTemporalWorldEffectForScene(item.card, state);
    if (!item.expected) {
      assert.equal(effect, null);
      return;
    }
    assert.equal(effect.enabled, true);
    // The block is: header, direction, who is protected, then the rest. The guidance starts with the same direction and protection.
    const [header, ...paragraphs] = item.expected.split('\n\n');
    assert.equal(header, `Temporal / World Effect — ${effect.label} — Mandatory:`);
    const guidance = effect.writing_guidance.split('\n\n');
    const protectionAt = paragraphs.findIndex((paragraph) => paragraph.startsWith('Protect '));
    assert.ok(protectionAt >= 0, 'the block names who is protected');
    assert.deepEqual(guidance.slice(0, protectionAt + 1), paragraphs.slice(0, protectionAt + 1));
  });
}
