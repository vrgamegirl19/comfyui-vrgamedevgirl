const assert = require("node:assert/strict");
const vm = require("node:vm");
const { test } = require("node:test");
const { functionSource, readBuilderModule } = require("./builder_source.cjs");

const source = functionSource(readBuilderModule("timeline_view.mjs"), "drawWaveform");

function fixture(audioDuration, zoom, timelineDuration = audioDuration) {
  const strokes = [];
  const context2d = { clearRect() { strokes.length = 0; }, fillRect() {}, beginPath() {},
    moveTo(x, y) { strokes.push({ x, y }); }, lineTo() {}, stroke() {}, fillText() {} };
  const canvas = { style: {}, getContext: () => context2d };
  const state = { pxPerSecond: zoom, waveformMode: "medium", peaks: [0.1, 0.2, 0.3, 0.4] };
  const context = vm.createContext({ state, timelineCanvas: canvas, segmentLayer: { style: {} },
    stemLayer: { style: {} }, playhead: { style: {} }, timelineHeight: () => 100,
    timelineWaveTop: () => 50, timelineDuration: () => timelineDuration,
    loadedGlobalAudioDuration: () => audioDuration, currentProjectAudioPath: () => "song.mp3",
    WAVEFORM_MODES: { medium: { gain: 1 } }, formatTime: value => String(value) });
  vm.runInContext("let drawnWaveformKey = ''; let drawnWaveformPeaks = null;\n" + source, context);
  return { context, state, canvas, strokes, draw: () => context.drawWaveform() };
}

test("short audio ends at its timestamp instead of the padded canvas edge", () => {
  const f = fixture(24.92, 28.75);
  f.draw();
  assert.equal(f.canvas.width, 900);
  assert.equal(f.strokes.length, Math.ceil(24.92 * 28.75));
  assert.equal(f.strokes.at(-1).x, Math.ceil(24.92 * 28.75) - 1);
  assert.ok(f.strokes.every(point => point.x < 24.92 * 28.75));
});

test("known peaks retain the same time scale as scene cards at every zoom", () => {
  for (const zoom of [10, 100]) {
    const f = fixture(20, zoom);
    f.draw();
    assert.equal(f.strokes.length, 20 * zoom);
    assert.equal(f.strokes[5 * zoom].y, 70 - 0.2 * 20);
    assert.equal(f.strokes[10 * zoom].y, 70 - 0.3 * 20);
  }
});

test("timeline space beyond the audio stays empty", () => {
  const f = fixture(5, 100, 20);
  f.draw();
  assert.equal(f.canvas.width, 2000);
  assert.equal(f.strokes.length, 500);
});

test("duration changes invalidate the waveform cache even with the same peak array", () => {
  const f = fixture(5, 20, 20);
  f.draw();
  assert.equal(f.strokes.length, 100);
  f.context.loadedGlobalAudioDuration = () => 10;
  f.draw();
  assert.equal(f.strokes.length, 200);
});

test("clipping the canvas does not compress later audio into an earlier timestamp", () => {
  const f = fixture(40, 100, 20);
  f.draw();
  assert.equal(f.strokes.length, 2000);
  assert.equal(f.strokes[1000].y, 70 - 0.2 * 20);
});

test("a timeline without audio does not draw a fake waveform", () => {
  const f = fixture(0, 20, 20);
  f.context.currentProjectAudioPath = () => "";
  f.draw();
  assert.equal(f.strokes.length, 0);
});
test("edited audio draws its silent gaps instead of the original waveform", () => {
  const f = fixture(20, 20);
  f.state.audioClips = [];
  f.state.audioClipMixPeaks = [0, 0.5, 0, 0.8];
  f.draw();
  assert.equal(f.strokes[0].y, 69.6);
  assert.equal(f.strokes[100].y, 60);
  assert.equal(f.strokes[200].y, 69.6);
  assert.equal(f.strokes[300].y, 54);
});
