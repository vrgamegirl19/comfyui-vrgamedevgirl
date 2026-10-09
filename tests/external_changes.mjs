// Agent API / MCP edits reach the open Video Builder and Storyboard without losing unsaved edits.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import vm from "node:vm";

const here = path.dirname(fileURLToPath(import.meta.url));
const web = path.join(here, "..", "web");
const builder = await import(pathToFileURL(path.join(web, "music_video_builder", "external_changes.mjs")).href);
const storyboard = await import(pathToFileURL(path.join(web, "storyboard_builder", "external_changes.mjs")).href);

const savedFrom = (segments, markers = [], refs = {}) => builder.savedStateFromSession({
  segments, timeline_markers: markers, flux_reference_builder: refs,
});

test("scene fields: API values apply, unsaved local edits win and are reported", () => {
  const saved = savedFrom([{ id: "s1", timeline_note: "old", notes: "plan", shot_type: "Wide" }]);
  const live = [{ id: "s1", timeline_note: "old", notes: "MY UNSAVED PLAN", shot_type: "Wide", _latentDirty: true }];
  const fresh = [{ id: "s1", timeline_note: "API note", notes: "API plan", shot_type: "" }];
  const result = builder.mergeSceneFields({ liveSegments: live, freshSegments: fresh, saved,
    scenes: { s1: { segment: ["timeline_note", "notes", "shot_type"] } } });
  assert.equal(live[0].timeline_note, "API note");
  assert.equal(live[0].shot_type, "", "an API clear is applied, not skipped");
  assert.equal(live[0].notes, "MY UNSAVED PLAN");
  assert.deepEqual(result.conflicts, [{ sceneId: "s1", key: "notes" }]);
  assert.equal(saved.segments.s1.notes, "API plan", "the saved state now holds the server value");
});

test("scene fields are matched by id only; an unknown scene asks for a reload", () => {
  const saved = savedFrom([{ id: "s1", label: "A" }]);
  const live = [{ id: "s1", label: "A" }];
  const result = builder.mergeSceneFields({ liveSegments: live, freshSegments: [{ id: "s2", label: "B" }], saved,
    scenes: { s2: { segment: ["label"] } } });
  assert.deepEqual(result.missing, ["s2"]);
  assert.equal(live[0].label, "A");
});

test("false and zero values from the API are applied", () => {
  const saved = savedFrom([{ id: "s1", include_microphone: true, minimax_h3_continuation_start_seconds: 0.5 }]);
  const live = [{ id: "s1", include_microphone: true, minimax_h3_continuation_start_seconds: 0.5 }];
  builder.mergeSceneFields({ liveSegments: live, freshSegments: [{ id: "s1", include_microphone: false, minimax_h3_continuation_start_seconds: 0 }],
    saved, scenes: { s1: { segment: ["include_microphone", "minimax_h3_continuation_start_seconds"] } } });
  assert.equal(live[0].include_microphone, false);
  assert.equal(live[0].minimax_h3_continuation_start_seconds, 0);
});

test("timeline notes: create, change and delete by id; a locally edited note is kept", () => {
  const saved = savedFrom([], [{ id: "m1", start: 1, end: 3, note: "a" }, { id: "m2", start: 5, end: null, note: "b" }]);
  const live = [{ id: "m1", start: 1, end: 3, note: "a" }, { id: "m2", start: 5, end: null, note: "LOCAL" }];
  const fresh = [{ id: "m2", start: 5, end: null, note: "API" }, { id: "m3", start: 7, end: 9, note: "new" }];
  const result = builder.mergeTimelineMarkers({ liveMarkers: live, freshMarkers: fresh, saved, markerIds: ["m1", "m2", "m3"] });
  assert.deepEqual(result.markers.map((marker) => marker.id), ["m2", "m3"]);
  assert.equal(result.markers[0].note, "LOCAL");
  assert.deepEqual(result.conflicts, ["m2"]);
  assert.equal(result.markers[1].end, 9);
});

test("reference maps follow the API's scene mapping", () => {
  const refs = { subject_scene_map: { s1: ["ben"] }, scene_map: {} };
  const saved = savedFrom([], [], refs);
  const liveRefs = JSON.parse(JSON.stringify(refs));
  builder.mergeReferenceMaps({ liveRefs, freshRefs: { subject_scene_map: { s1: [] }, scene_map: { s1: "porch" }, use_location_references: true },
    saved, scenes: { s1: { references: ["subjects", "locations"] } } });
  assert.deepEqual(liveRefs.subject_scene_map.s1, []);
  assert.equal(liveRefs.scene_map.s1, "porch");
  assert.equal(liveRefs.use_location_references, true);
});

test("only a contiguous run of scene and note edits is merged; anything else reloads", () => {
  const sceneEvent = (revision) => ({ revision, change: { kind: "scene_fields", scenes: {} } });
  assert.equal(builder.planExternalChanges([sceneEvent(5), sceneEvent(6)], 4, 6).mergeable, true);
  assert.equal(builder.planExternalChanges([sceneEvent(6)], 4, 6).contiguous, false, "revision 5 was missed");
  assert.equal(builder.planExternalChanges([{ revision: 5, change: { kind: "project" } }], 4, 5).mergeable, false);
});

test("unsaved-edit detection ignores runtime-only keys", () => {
  const session = { segments: [{ id: "s1", label: "A" }] };
  const key = builder.unsavedKey(session);
  assert.equal(builder.unsavedKey({ segments: [{ id: "s1", label: "A", _latentDirty: true }] }), key);
  assert.notEqual(builder.unsavedKey({ segments: [{ id: "s1", label: "B" }] }), key);
  assert.equal(builder.projectFolderKey("C:/Out/Song/"), builder.projectFolderKey("c:\\out\\song"));
});

test("storyboard cards: values come from the file when the API wrote it, else from the timeline", () => {
  const byScene = storyboard.changedCardKeys([{ kind: "scene_fields", scenes: {
    s1: { segment: ["i2v_notes", "timeline_note"], card: ["timeline_note", "motion_summary"], references: [] },
    s2: { segment: ["notes"], card: [], references: ["subjects"] },
  } }]);
  assert.deepEqual([...byScene.s1.keys].sort(), ["motion_summary", "timeline_note"]);
  assert.deepEqual([...byScene.s2.keys].sort(), ["location_ref", "notes", "setting", "subject_refs", "subjects"]);
  const scenes = [
    { id: "s1", timeline_note: "old", motion_summary: "MY EDIT" },
    { id: "s2", notes: "plan", subject_refs: [], subjects: [], location_ref: null, setting: "" },
    { id: "s3", notes: "untouched" },
  ];
  const saved = new Map([["s1", { timeline_note: "old", motion_summary: "old motion" }],
    ["s2", { notes: "plan", subject_refs: [], subjects: [], location_ref: null, setting: "" }]]);
  const fileCards = new Map([["s1", { timeline_note: "API note", motion_summary: "API motion" }]]);
  const liveCards = new Map([["s2", { notes: "API plan", subject_refs: [{ id: "ana" }], subjects: ["Ana"], location_ref: null, setting: "" }]]);
  const result = storyboard.mergeCardChanges({ scenes, saved, byScene, fileCards, liveCards });
  assert.equal(scenes[0].timeline_note, "API note");
  assert.equal(scenes[0].motion_summary, "MY EDIT", "an unsaved card edit wins");
  assert.deepEqual(result.conflicts, [{ sceneId: "s1", key: "motion_summary" }]);
  assert.equal(scenes[1].notes, "API plan");
  assert.deepEqual(scenes[1].subjects, ["Ana"]);
  assert.equal(scenes[2].notes, "untouched");
});

test("storyboard cards: the card open in the editor is left alone", () => {
  const byScene = storyboard.changedCardKeys([{ kind: "scene_fields", scenes: { s1: { segment: ["timeline_note"], card: ["timeline_note"] } } }]);
  const scenes = [{ id: "s1", timeline_note: "old" }];
  const result = storyboard.mergeCardChanges({ scenes, saved: new Map([["s1", { timeline_note: "old" }]]), byScene,
    fileCards: new Map([["s1", { timeline_note: "API" }]]), editingSceneId: "s1" });
  assert.equal(scenes[0].timeline_note, "old");
  assert.deepEqual(result.deferred, [{ sceneId: "s1", key: "timeline_note" }]);
});

function loadSaveHelper(responses, confirmAnswer) {
  const requests = [];
  const context = vm.createContext({
    console, AbortController, setTimeout, clearTimeout,
    window: { confirm: () => confirmAnswer },
    api: {
      fetchApi: async (url, options) => {
        const body = JSON.parse(options.body);
        requests.push({ url, body });
        const reply = responses.shift();
        return { ok: reply.status === 200, status: reply.status, json: async () => reply.data };
      },
    },
  });
  const source = fs.readFileSync(path.join(web, "storyboard_builder", "api.mjs"), "utf8")
    .replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
  vm.runInContext(source, context, { filename: "api.mjs" });
  return { save: vm.runInContext("saveStoryboardFile", context), requests };
}

test("storyboard saves send the loaded revision and never overwrite newer edits silently", async () => {
  const conflict = { status: 409, data: { ok: false, conflict: true, error: "changed elsewhere" } };
  const kept = loadSaveHelper([conflict], false);
  const state = { projectFolder: "P", storyboardRevision: 4 };
  await assert.rejects(kept.save(state, { scenes: [] }), /not saved/);
  assert.equal(kept.requests[0].body.expected_revision, 4);
  assert.equal(kept.requests.length, 1, "nothing is resent when the user keeps the file");

  const replaced = loadSaveHelper([conflict, { status: 200, data: { ok: true, storyboard: { revision: 9 } } }], true);
  let marked = 0;
  const state2 = { projectFolder: "P", storyboardRevision: 4, onStoryboardFileSaved: () => { marked += 1; } };
  await replaced.save(state2, { scenes: [] });
  assert.equal(replaced.requests[1].body.expected_revision, undefined, "the user chose to replace the file");
  assert.equal(state2.storyboardRevision, 9);
  assert.equal(marked, 1);
});
