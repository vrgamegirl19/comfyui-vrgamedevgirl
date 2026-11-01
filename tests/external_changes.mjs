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

const block = (id, extra = {}) => ({ id, transition: "BlockDissolve", start: 1, seconds: 1, ...extra });
const lanesFrom = (lists) => builder.savedStateFromSession({ segments: [], ...lists });

test("lane lists: a list the window has not edited takes the agent's blocks, an empty one included", () => {
  const saved = lanesFrom({ scene_transitions: [block("t1")] });
  const live = { scene_transitions: [block("t1")], video_effects: [], video_filters: [] };
  const fresh = { scene_transitions: [block("t1"), block("t2", { start: 3 })], video_effects: [{ id: "e1", look: "BeatBounce", start: 0, seconds: 2 }] };
  const result = builder.mergeLaneLists({ live, fresh, saved, keys: ["scene_transitions", "video_effects"] });
  assert.deepEqual(result.applied, ["scene_transitions", "video_effects"]);
  assert.deepEqual(result.take.scene_transitions.map((item) => item.id), ["t1", "t2"]);
  assert.equal(result.take.video_effects[0].look, "BeatBounce");
  assert.deepEqual(result.conflicts, []);
  assert.deepEqual(saved.lists.scene_transitions.map((item) => item.id), ["t1", "t2"], "the saved state now holds the server's list");
  // The agent clears a lane.
  const cleared = builder.mergeLaneLists({ live: { scene_transitions: result.take.scene_transitions }, fresh: { scene_transitions: [] }, saved, keys: ["scene_transitions"] });
  assert.deepEqual(cleared.take, { scene_transitions: [] });
});

test("lane lists: blocks the person added and has not saved are kept, and reported", () => {
  const saved = lanesFrom({ video_filters: [] });
  const live = { video_filters: [{ id: "f9", look: "Adjust", start: 0, seconds: 1 }] };
  const result = builder.mergeLaneLists({ live, fresh: { video_filters: [{ id: "f1", look: "Adjust", start: 2, seconds: 1 }] }, saved, keys: ["video_filters"] });
  assert.deepEqual(result.take, {});
  assert.deepEqual(result.conflicts, ["video_filters"]);
  assert.equal(saved.lists.video_filters[0].id, "f1", "the window now counts as holding an unsaved edit against the server's list");
});

test("lane lists: only the lists named are touched, and what is already the same is not a change", () => {
  const saved = lanesFrom({ scene_transitions: [block("t1")], video_effects: [{ id: "e1" }] });
  const live = { scene_transitions: [block("t1")], video_effects: [{ id: "e1" }] };
  const result = builder.mergeLaneLists({ live, fresh: { scene_transitions: [block("t1")], video_effects: [] }, saved, keys: ["scene_transitions", "not_a_lane"] });
  assert.deepEqual(result, { take: {}, applied: [], conflicts: [] });
  assert.deepEqual(saved.lists.video_effects, [{ id: "e1" }], "a list that was not named keeps its saved state");
  assert.deepEqual(Object.keys(builder.LANE_LISTS), ["scene_transitions", "video_effects", "video_filters"]);
});

test("a saved state and a drift absorb both carry the lane lists, a missing list as an empty one", () => {
  assert.deepEqual(builder.savedStateFromSession({ segments: [] }).lists, { scene_transitions: [], video_effects: [], video_filters: [] });
  assert.deepEqual(lanesFrom({ video_effects: [{ id: "e1" }] }).lists.video_effects, [{ id: "e1" }]);
});

test("only a contiguous run of scene and note edits is merged; anything else reloads", () => {
  const sceneEvent = (revision) => ({ revision, change: { kind: "scene_fields", scenes: {} } });
  assert.equal(builder.planExternalChanges([sceneEvent(5), sceneEvent(6)], 4, 6).mergeable, true);
  assert.equal(builder.planExternalChanges([sceneEvent(6)], 4, 6).contiguous, false, "revision 5 was missed");
  assert.equal(builder.planExternalChanges([{ revision: 5, change: { kind: "project" } }], 4, 5).mergeable, false);
  const laneEvent = { revision: 5, change: { kind: "scene_fields", scenes: {}, lists: ["scene_transitions"] } };
  assert.equal(builder.planExternalChanges([laneEvent], 4, 5).mergeable, true, "a lane change needs no reload");
});

test("unsaved-edit detection ignores runtime-only keys", () => {
  const session = { segments: [{ id: "s1", label: "A" }] };
  const key = builder.unsavedKey(session);
  assert.equal(builder.unsavedKey({ segments: [{ id: "s1", label: "A", _latentDirty: true }] }), key);
  assert.notEqual(builder.unsavedKey({ segments: [{ id: "s1", label: "B" }] }), key);
  assert.equal(builder.projectFolderKey("C:/Out/Song/"), builder.projectFolderKey("c:\\out\\song"));
});

test("defaults the Builder fills in are not unsaved edits, real edits are", () => {
  const saved = savedFrom([{ id: "s1", label: "A", notes: "plan" }, { id: "s2", label: "B" }]);
  // Selecting scene 2 writes empty defaults into it.
  const selected = [{ id: "s1", label: "A", notes: "plan" }, { id: "s2", label: "B", notes: "", t2i_prompt: "", use_vision_reference: false, tags: [] }];
  assert.deepEqual(builder.localSegmentEdits(selected, saved), []);
  assert.equal(builder.unsavedKey({ segments: selected }), builder.unsavedKey({ segments: [{ id: "s1", label: "A", notes: "plan" }, { id: "s2", label: "B" }] }));
  assert.equal(builder.isBlank(0), false, "zero is a value");
  // A video history the Builder derived from the backups is not an edit.
  const rendered = savedFrom([{ id: "s1", label: "A", video_history: ["a"], video_history_index: 0 }]);
  assert.deepEqual(builder.localSegmentEdits([{ id: "s1", label: "A", video_history: ["x", "a"], video_history_index: 1 }], rendered), []);
  // A real edit, a cleared value, a switch turned on and a moved scene all count.
  assert.deepEqual(builder.localSegmentEdits([{ id: "s1", label: "A", notes: "" }, { id: "s2", label: "B" }], saved), [{ sceneId: "s1", key: "notes" }]);
  assert.deepEqual(builder.localSegmentEdits([{ id: "s1", label: "A", notes: "plan" }, { id: "s2", label: "B", use_vision_reference: true }], saved), [{ sceneId: "s2", key: "use_vision_reference" }]);
  assert.deepEqual(builder.localSegmentEdits([{ id: "s2", label: "B" }, { id: "s1", label: "A", notes: "plan" }], saved), [{ sceneId: "", key: "" }]);
  assert.deepEqual(builder.localSegmentEdits([{ id: "s1", label: "A", notes: "plan" }], saved), [{ sceneId: "", key: "" }], "a scene removed here");
});

test("rebasing on the server's copy: unannounced fields arrive, local edits win, agreement is not unsaved", () => {
  const saved = savedFrom([{ id: "s1", label: "A", notes: "plan", video_history: ["a"], video_cache_bust: 1 }]);
  // The window derived the new history itself; the server sent the same history and a new cache key. The user edited the notes.
  const live = [{ id: "s1", label: "A", notes: "MY PLAN", video_history: ["a", "b"], video_cache_bust: 1 }];
  const fresh = [{ id: "s1", label: "A", notes: "API plan", video_history: ["a", "b"], video_cache_bust: 2 }];
  const result = builder.rebaseSegments({ liveSegments: live, freshSegments: fresh, saved });
  assert.equal(live[0].video_cache_bust, 2, "a field the server changed and the user did not touch is taken, announced or not");
  assert.equal(live[0].notes, "MY PLAN", "the user's unsaved edit wins");
  assert.deepEqual(result.applied, [{ sceneId: "s1", key: "video_cache_bust", was: 1 }], "what the field held is kept, so a value put back by the redraw can be told from a change");
  assert.deepEqual(result.conflicts, [{ sceneId: "s1", key: "notes" }]);
  // Only the user's edit is still unsaved. The history both sides agree on is not.
  assert.deepEqual(builder.localSegmentEdits(live, saved), [{ sceneId: "s1", key: "notes" }]);
});

test("rebasing keeps a missing value and an empty one apart from a real change", () => {
  const saved = savedFrom([{ id: "s1", label: "A" }]);
  const live = [{ id: "s1", label: "A", notes: "" }];
  const result = builder.rebaseSegments({ liveSegments: live, freshSegments: [{ id: "s1", label: "A" }], saved });
  assert.deepEqual(result, { applied: [], conflicts: [] });
  const cleared = [{ id: "s1", label: "A", notes: "kept" }];
  const again = builder.rebaseSegments({ liveSegments: cleared, freshSegments: [{ id: "s1", label: "A", notes: "" }], saved: savedFrom([{ id: "s1", label: "A", notes: "kept" }]) });
  assert.equal(cleared[0].notes, "", "an API clear is applied to a field the user did not edit");
  assert.equal(again.applied.length, 1);
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
    errorLogContext: payload => payload, errorLogHeaders: () => ({}), reportProjectError() {},
    window: { confirm: () => confirmAnswer },
    api: {
      fetchApi: async (url, options) => {
        const body = JSON.parse(options.body);
        requests.push({ url, body });
        const reply = responses.shift();
        return { ok: reply.status === 200, status: reply.status, headers: { get: () => null }, json: async () => reply.data };
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
