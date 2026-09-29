const { functionSource, readBuilderSource } = require('./builder_source.cjs');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const { test } = require('node:test');

const source = readBuilderSource();

function element() {
  return { style: {}, children: [], append(...items) { this.children.push(...items); }, remove() { this.removed = true; } };
}

function fixture(retryHandler) {
  const toasts = [];
  const body = element();
  const c = {
    window: {},
    toasts,
    body,
    document: { createElement: element, body },
    makeButton: (label) => ({ ...element(), textContent: label }),
    toast: (message, isError) => toasts.push([message, isError]),
    retryHandler,
  };
  vm.createContext(c);
  vm.runInContext(['gemmaBatchFailureStore', 'recordGemmaBatchFailure', 'showGemmaBatchFailures'].map((name) => functionSource(source, name)).join('\n'), c);
  return c;
}

function buttons(c) {
  const backdrop = c.body.children[0];
  const actions = backdrop.children[0].children[2];
  const [close, retry] = actions.children;
  return { backdrop, close, retry };
}

test('recording a failure uses the caller scene label without builder state', () => {
  const c = fixture();
  const failure = c.recordGemmaBatchFailure('t2i:zimage:a', { id: 'a' }, '2. Chorus', new Error('bad JSON'), 'debug.txt');
  assert.equal(failure.sceneLabel, '2. Chorus');
  assert.equal(failure.segmentId, 'a');
  assert.equal(failure.error, 'bad JSON');
  assert.equal(c.window.__vrgdgGemmaBatchFailures['t2i:zimage:a'], failure);
});

test('retry runs the caller handler with the failed items and closes the dialog', async () => {
  const calls = [];
  const c = fixture(async (items) => calls.push(items.map((item) => item.segmentId)));
  const failure = c.recordGemmaBatchFailure('i2v:t2v:b', { id: 'b' }, '3. Bridge', new Error('empty'));
  c.showGemmaBatchFailures([failure], { retryHandler: c.retryHandler });
  const { backdrop, retry } = buttons(c);
  await retry.onclick();
  assert.deepEqual(calls, [['b']]);
  assert.equal(backdrop.removed, true);
});

test('a failed retry keeps the dialog open and reports the error', async () => {
  const c = fixture(async () => { throw new Error('Select 3. Bridge to retry it.'); });
  const failure = c.recordGemmaBatchFailure('minimax:t2v:b', { id: 'b' }, '3. Bridge', new Error('empty'));
  c.showGemmaBatchFailures([failure], { retryHandler: c.retryHandler });
  const { backdrop, retry } = buttons(c);
  await retry.onclick();
  assert.notEqual(backdrop.removed, true);
  assert.equal(retry.disabled, false);
  assert.deepEqual(c.toasts, [['Select 3. Bridge to retry it.', true]]);
});

test('every failure dialog is given a retry handler', () => {
  const calls = [...source.matchAll(/showGemmaBatchFailures\(/g)].map((match) => match.index)
    .filter((index) => !source.slice(index - 9, index).endsWith('function '));
  assert.ok(calls.length >= 9);
  for (const index of calls) {
    assert.match(source.slice(index, index + 260), /retryHandler/, source.slice(index, index + 120));
  }
});
