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
