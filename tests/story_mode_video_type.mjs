import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { test } from 'node:test';

// controls.mjs imports ComfyUI's app, so the video type helpers are taken from its source text.
const BUILDER = new URL('../web/music_video_builder/', import.meta.url);
const controls = readFileSync(new URL('controls.mjs', BUILDER), 'utf8').replace(/\r\n/g, '\n');

function exported(name) {
  const start = controls.indexOf(`export function ${name}(`);
  assert.ok(start >= 0, `${name} is missing from controls.mjs`);
  const end = controls.indexOf('\n}\n', start);
  return controls.slice(start, end + 2).replace('export ', '');
}

const optionsStart = controls.indexOf('export const VIDEO_TYPE_OPTIONS');
const optionsSource = controls.slice(optionsStart, controls.indexOf('];', optionsStart) + 2).replace('export ', '');
const { normalizeVideoType, normalizeProjectVideoType, VIDEO_TYPE_OPTIONS } = new Function(
  `${optionsSource}\n${exported('normalizeVideoType')}\n${exported('normalizeProjectVideoType')}\n`
  + 'return { normalizeVideoType, normalizeProjectVideoType, VIDEO_TYPE_OPTIONS };',
)();

test('Story Mode is offered as a video type', () => {
  assert.ok(VIDEO_TYPE_OPTIONS.some((item) => item.value === 'story'));
});

test('the project type keeps Story Mode and the other types as before', () => {
  assert.equal(normalizeProjectVideoType('story'), 'story');
  assert.equal(normalizeProjectVideoType('Story Mode'), 'story');
  assert.equal(normalizeProjectVideoType('speaking'), 'speaking');
  assert.equal(normalizeProjectVideoType('short film'), 'speaking');
  assert.equal(normalizeProjectVideoType('no_lip_sync'), 'no_lip_sync');
  assert.equal(normalizeProjectVideoType(''), 'singing');
  assert.equal(normalizeProjectVideoType('unknown'), 'singing');
});

test('Story Mode renders as singing, and a scene performance still wins', () => {
  // Builder code reads a scene's performance as normalizeVideoType(segment.performance_mode || state.videoType).
  const scenePerformance = (sceneMode) => normalizeVideoType(sceneMode || 'story');
  assert.equal(scenePerformance(''), 'singing');
  assert.equal(scenePerformance('speaking'), 'speaking');
  assert.equal(scenePerformance('no_lip_sync'), 'no_lip_sync');
});

test('every place that stores the project type keeps Story Mode', () => {
  const offenders = [];
  for (const name of readdirSync(BUILDER).filter((file) => file.endsWith('.mjs'))) {
    const source = readFileSync(new URL(name, BUILDER), 'utf8');
    source.split('\n').forEach((line, index) => {
      const stores = /state\.videoType\s*=\s*normalizeVideoType\(|video_type:\s*normalizeVideoType\(/.test(line);
      if (stores) offenders.push(`${name}:${index + 1}`);
    });
  }
  assert.deepEqual(offenders, [], 'use normalizeProjectVideoType where the project type is stored');
});
