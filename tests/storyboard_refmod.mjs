import assert from 'node:assert/strict';
import { register } from 'node:module';
import { test } from 'node:test';

// The storyboard modules import ComfyUI's scripts/api.js, which only exists in the browser. Stub it for Node.
register('./stub_comfy_loader.mjs', import.meta.url);

const root = new URL('../web/storyboard_builder/', import.meta.url);
const { normalizeReferenceBuilderCatalog, storyboardRefmodLabels, referenceChipHtml } = await import(new URL('references.mjs', root));
const { storyboardGptPayload } = await import(new URL('gpt_payload.mjs', root));

const card = (id, name, extra = {}) => ({
  id, name, description: `${name} description`, reference_type: 'character', source: 'refmod',
  refmod: { name: `identity/${name}`, kind: 'video', tokens: 1000, frames: 4, strength: 1, type: 'identity' }, ...extra,
});

test('the storyboard catalog keeps RefMod fields', () => {
  const catalog = normalizeReferenceBuilderCatalog({ subjects: [card('a', 'Brad')], locations: [{ id: 'l', name: 'Cabin', source: 'refmod', refmod: { name: 'background/cabin', kind: 'image', tokens: 500 } }] });
  assert.equal(catalog.subjects[0].source, 'refmod');
  assert.equal(catalog.subjects[0].refmod.name, 'identity/Brad');
  assert.equal(catalog.subjects[0].reference_type, 'character');
  assert.equal(catalog.locations[0].refmod.kind, 'image');
});

test('labels for a scene follow the Video Builder rules', () => {
  const man = card('a', 'The man', { refmod: { name: 'identity/man', kind: 'image', tokens: 500, strength: 1 } });
  const brad = card('b', 'Brad');
  const labels = storyboardRefmodLabels({ subject_refs: [brad, man] });
  assert.equal(labels.get('a'), '<Picture 1>');
  assert.equal(labels.get('b'), '<Video 1>');
});

test('a RefMod chip shows its badge and label', () => {
  assert.match(referenceChipHtml(card('a', 'Brad'), 'Subject', '<Video 1>'), /◈ &lt;Video 1&gt;|◈ <Video 1>/);
  assert.doesNotMatch(referenceChipHtml({ id: 'x', name: 'Plain' }, 'Subject'), /◈/);
});

test('the GPT payload names each RefMod label and how to write it', () => {
  const brad = card('b', 'Brad');
  const scene = { id: 's1', scene_number: 1, subject_refs: [brad], subjects: ['Brad'], exact_duration: 6, project_video_engine: 'minimax_h3', lyrics: '', video_prompt_type: 'rtv', minimax_h3_mode: 'reference_to_video' };
  const base = { mode: 'image_to_video_prep', projectVideoEngine: 'minimax_h3', scenes: [scene], referenceBuilder: { subjects: [brad], locations: [] }, storyLayer: {} };
  const on = JSON.stringify(storyboardGptPayload({ ...base, refmodPipeline: true }));
  assert.match(on, /"refmod_label":"<Video 1>"/);
  assert.match(on, /refmod_pipeline/);
  const off = JSON.stringify(storyboardGptPayload({ ...base, refmodPipeline: false }));
  assert.doesNotMatch(off, /refmod_label|refmod_pipeline/);
});
