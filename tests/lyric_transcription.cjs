const assert = require("node:assert/strict");
const vm = require("node:vm");
const { test } = require("node:test");
const { functionSource, readBuilderModule } = require("./builder_source.cjs");

const references = readBuilderModule("reference_data.mjs");
const transcription = readBuilderModule("lyric_transcription.mjs");

function fixture() {
  const state = {
    audioPath: "song.wav",
    segments: [
      { id: "a", start: 0, end: 4, lyric_text: "", lyric_mapper_line_id: "line-a" },
      { id: "b", start: 4, end: 8, lyric_text: "", lyric_mapper_line_id: "line-b" },
      { id: "c", start: 8, end: 12, lyric_text: "", lyric_mapper_line_id: "line-c" },
    ],
    lyricMapper: { source_text: "", lines: [
      { id: "line-a", text: "", singers: ["Singer"], instrumental: false },
      { id: "line-b", text: "Old lyrics", singers: ["Singer"], instrumental: true },
      { id: "line-c", text: "Old ending", singers: ["Singer"], instrumental: false },
    ] },
  };
  const payload = { segments: [
    { type: "vocal", start: 0, end: 4, text: "First line" },
    { type: "vocal", start: 4, end: 8, text: "Second line" },
  ] };
  const saved = [];
  const context = vm.createContext({
    state, audioInput: { value: "song.wav" }, saved,
    cleanTimestampedLyricText: text => String(text).trim(),
    isInstrumentalLyricText: text => String(text).trim() === "[instrumental]",
    isNoLipSyncSingerChoice: () => false,
    postJson: async () => ({ prompt: {} }),
    queueWorkflowPrompt: async () => ({ prompt_id: "transcription" }),
    waitForText: async () => [JSON.stringify(payload)],
    autoSaveSessionQuiet: async () => {}, pushHistory() {},
    applyLyricSectionsFromReferenceText: () => 0,
    syncLyricNoteControls() {}, projectLyricNotesPath: () => "notes.json",
    syncLyricAndSubjectNoteFiles: async () => {
      saved.push(state.segments.map(scene => scene.lyric_text));
    },
    syncInspector() {}, render() {}, activeProjectFolderForSave: () => "project",
    saveSession: async () => {
      saved.push(state.segments.map(scene => scene.lyric_text));
    },
  });
  for (const name of ["defaultLyricMapper", "normalizeLyricMapper", "normalizedLyricMatchText",
    "lyricMatchScore", "applyLyricMapperToSegments", "syncLyricMapperFromSegments"]) {
    vm.runInContext(functionSource(references, name), context);
  }
  for (const name of ["parseTimestampedLyricsOutput", "referenceVocalLines",
    "mapTimestampedReferenceLyricsToExistingScenes", "transcribeExistingScenesWithOptions"]) {
    vm.runInContext(functionSource(transcription, name), context);
  }
  return { state, saved, run: code => vm.runInContext(code, context) };
}

test("existing-scene transcription survives blank and stale mapper rows and is saved", async () => {
  const f = fixture();
  const result = await f.run('transcribeExistingScenesWithOptions({ referenceLyrics: "First line\\nSecond line", replaceAll: true })');
  assert.equal(result.applied, 3);
  assert.deepEqual(f.state.segments.map(scene => scene.lyric_text), ["First line", "Second line", "[instrumental]"]);
  assert.deepEqual(f.state.segments.map(scene => [scene.start, scene.end]), [[0, 4], [4, 8], [8, 12]]);
  assert.deepEqual(Array.from(f.state.segments[0].lyric_singers), ["Singer"]);
  assert.deepEqual(Array.from(f.state.segments[1].lyric_singers), ["Singer"]);
  assert.equal(f.state.segments[1].lyric_no_lip_sync, false);
  assert.equal(f.state.segments[2].lyric_no_lip_sync, true);
  assert.deepEqual(Array.from(f.state.segments[2].lyric_singers), []);
  assert.deepEqual(f.saved, [
    ["First line", "Second line", "[instrumental]"],
    ["First line", "Second line", "[instrumental]"],
  ]);
  assert.deepEqual(Array.from(f.state.lyricMapper.lines, line => line.text), ["First line", "Second line", ""]);
  f.run('applyLyricMapperToSegments()');
  assert.deepEqual(f.state.segments.map(scene => scene.lyric_text), ["First line", "Second line", "[instrumental]"]);
});

test("fill-missing transcription preserves existing line notes", async () => {
  const f = fixture();
  f.state.segments[0].lyric_text = "User correction";
  const result = await f.run('transcribeExistingScenesWithOptions({ referenceLyrics: "First line\\nSecond line", replaceAll: false })');
  assert.equal(result.applied, 2);
  assert.deepEqual(f.state.segments.map(scene => scene.lyric_text), ["User correction", "Second line", "[instrumental]"]);
});

test("explicit mapping edits still replace scene lyrics by default", () => {
  const f = fixture();
  f.state.lyricMapper.lines[0].text = "Edited in mapping";
  f.run('applyLyricMapperToSegments()');
  assert.equal(f.state.segments[0].lyric_text, "Edited in mapping");
  assert.equal(f.state.segments[1].lyric_text, "[instrumental]");
});

function lengthContext() {
  let nextId = 0;
  const context = vm.createContext({
    DEFAULT_SCENE_VARIANCE_SECONDS: 2,
    cleanTimestampedLyricText: text => String(text || "").trim(),
    isInstrumentalLyricText: text => String(text || "").trim() === "[instrumental]",
    newSegment: (start, end) => ({ id: `n${nextId += 1}`, start, end, label: "New scene" }),
    sortSegments: list => list.sort((a, b) => a.start - b.start),
  });
  for (const name of ["timestampedSceneVarianceSeconds", "splitTimestampedLyricText", "mergeTimestampedLyricText",
    "enforceTimestampedSceneLengths", "createSegmentsFromTimestampedLyricsPayload"]) {
    vm.runInContext(functionSource(transcription, name), context);
  }
  return code => vm.runInContext(code, context);
}

const lengthsOf = scenes => Array.from(scenes, scene => Math.round((scene.end - scene.start) * 100) / 100);

test("scene lengths stay between min and max + variance, and short lines merge", () => {
  const run = lengthContext();
  const scenes = run(`enforceTimestampedSceneLengths([
    { start: 0, end: 9.5, lyric_text: "[instrumental]" },
    { start: 9.5, end: 11, lyric_text: "short line" },
    { start: 11, end: 23, lyric_text: "one two three four five six" },
    { start: 23, end: 24.2, lyric_text: "tail" },
  ], { minSceneSeconds: 3, hardMaxSceneSeconds: 10 })`);
  assert.ok(lengthsOf(scenes).every(value => value >= 3 && value <= 10), lengthsOf(scenes).join(", "));
  assert.equal(scenes[0].start, 0);
  assert.equal(scenes.at(-1).end, 24.2);
  const words = scenes.map(scene => scene.lyric_text).join(" ").split(/\s+/).filter(word => word !== "[instrumental]");
  assert.deepEqual(words, ["short", "line", "one", "two", "three", "four", "five", "six", "tail"]);
  assert.deepEqual(scenes.map(scene => scene.label), scenes.map((_, index) => `SCENE ${index + 1}`));
});

test("a scene within max + variance is left alone", () => {
  const run = lengthContext();
  const scenes = run(`enforceTimestampedSceneLengths([
    { start: 0, end: 9.23, lyric_text: "a b" },
    { start: 9.23, end: 13, lyric_text: "c d" },
  ], { minSceneSeconds: 3, hardMaxSceneSeconds: 10 })`);
  assert.deepEqual(lengthsOf(scenes), [9.23, 3.77]);
});

test("a long lyric line is cut at a word boundary past max + variance", () => {
  const run = lengthContext();
  const words = Array.from({ length: 14 }, (_, i) => ({ text: `w${i}`, start: i, end: i + 0.9 }));
  const payload = { duration: 14, segments: [{ type: "vocal", start: 0, end: 14, text: "x", words }] };
  const scenes = run(`createSegmentsFromTimestampedLyricsPayload(${JSON.stringify(payload)},
    { segmentMode: "reference_lines", minSceneSeconds: 3, maxSceneSeconds: 8, sceneVarianceSeconds: 2 })`);
  assert.equal(scenes.length, 2);
  assert.ok(lengthsOf(scenes).every(value => value >= 3 && value <= 10), lengthsOf(scenes).join(", "));
  const exact = run(`createSegmentsFromTimestampedLyricsPayload(${JSON.stringify(payload)},
    { segmentMode: "exact_reference_lines", minSceneSeconds: 3, maxSceneSeconds: 8, sceneVarianceSeconds: 2 })`);
  assert.equal(exact.length, 1);
});

test("scene variance defaults to 2 and accepts 0", () => {
  const run = lengthContext();
  assert.equal(run("timestampedSceneVarianceSeconds({})"), 2);
  assert.equal(run("timestampedSceneVarianceSeconds({ sceneVarianceSeconds: 0 })"), 0);
});
