// Shows Agent API / MCP edits in the open Video Builder.
//
// agent_api/project_events.py broadcasts "vrgdg.project_changed" after every API save. Scene-card and
// Timeline Note edits name the scenes, fields and notes they changed; those are merged into the open
// project field by field against the last saved state, so unsaved local edits are kept (a field edited on
// both sides keeps the local value and is reported). Any other change reloads the project. A reload waits
// while a text field is focused or a render runs, then happens by itself, and the scene fields the user
// edited and has not saved are put back on the reloaded project. The server keeps rejecting this window's
// older snapshot until it has caught up, so nothing the agent saved is overwritten.
//
// "Unsaved" means a scene field that differs from the last saved state in a way that matters. The Builder
// fills empty defaults into a scene when it is selected or drawn (an empty note, a switch that is off); those
// are not edits and never block a reload.

export const PROJECT_CHANGED_EVENT = "vrgdg.project_changed";
export const STORYBOARD_EXTERNAL_CHANGE_EVENT = "vrgdg:storyboard-external-change";
const MERGEABLE_KINDS = new Set(["scene_fields", "timeline_markers"]);
const REFERENCE_MAP_KEYS = { subjects: "subject_scene_map", locations: "scene_map" };
const REFERENCE_SWITCHES = { subjects: "use_subject_reference", locations: "use_location_references" };
// Scene fields the Builder works out from the scene's video, its backups and thumbnails. The server writes a shorter history
// than the Builder derives, so they differ after every render without anyone having edited anything.
const DERIVED_KEYS = new Set(["video_history", "video_thumbnail_history", "video_history_index"]);

export function projectFolderKey(folder) {
  return String(folder || "").trim().replace(/\//g, "\\").replace(/\\+$/, "").toLowerCase();
}

function clone(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

function sameValue(a, b) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

function durableSegment(segment) {
  // Keys starting with "_" are runtime-only (latent badges, drag state) and are never edited by the API.
  return Object.fromEntries(Object.entries(segment || {}).filter(([key]) => !key.startsWith("_")));
}

// A value that says nothing: missing, null, "", false, an empty list or an empty object. Zero is a value.
export function isBlank(value) {
  if (value === undefined || value === null || value === "" || value === false) return true;
  if (Array.isArray(value)) return value.length === 0;
  return typeof value === "object" && Object.keys(value).length === 0;
}

// Two values the user could not tell apart: equal, or both saying nothing (a missing note and an empty one).
function sameish(a, b) {
  return sameValue(a, b) || (isBlank(a) && isBlank(b));
}

function setOrDelete(target, key, value) {
  if (value === undefined) delete target[key];
  else target[key] = clone(value);
}

// A scene without the keys that say nothing, for comparing what was saved with what the window holds.
export function comparableSegment(segment) {
  return Object.fromEntries(Object.entries(durableSegment(segment)).filter(([, value]) => !isBlank(value)));
}

// The parts of the session that are not scenes. An API edit to one of them reloads the project.
function sectionParts(session = {}) {
  return [
    (session.overlay_segments || []).map(comparableSegment),
    session.timeline_markers || [],
    session.flux_reference_builder || null,
    session.minimax_h3_settings || null,
    session.builder_story_layer || null,
    session.builder_storyboard_defaults || null,
    session.lyric_mapper || null,
  ];
}

export function sectionsKey(session = {}) {
  return JSON.stringify(sectionParts(session));
}

// The saved state the merge compares against, from a session payload (currentSessionData() or a save).
export function savedStateFromSession(session = {}) {
  const refs = session.flux_reference_builder && typeof session.flux_reference_builder === "object" ? session.flux_reference_builder : {};
  const scenes = (Array.isArray(session.segments) ? session.segments : []).filter((segment) => segment?.id);
  return {
    key: unsavedKey(session),
    sections: sectionsKey(session),
    order: scenes.map((segment) => String(segment.id)),
    segments: Object.fromEntries(scenes.map((segment) => [String(segment.id), clone(durableSegment(segment))])),
    markers: Object.fromEntries((Array.isArray(session.timeline_markers) ? session.timeline_markers : [])
      .filter((marker) => marker?.id)
      .map((marker) => [String(marker.id), clone(marker)])),
    referenceMaps: Object.fromEntries(Object.entries(REFERENCE_MAP_KEYS)
      .map(([name, key]) => [name, clone(refs[key] && typeof refs[key] === "object" ? refs[key] : {})])),
  };
}

// Everything an API edit could conflict with. Equal keys mean nothing is unsaved.
export function unsavedKey(session = {}) {
  return JSON.stringify([(session.segments || []).map(comparableSegment), ...sectionParts(session)]);
}

// Merge the listed fields of external scene edits into the live segments (matched by id only).
// Returns { applied: [{sceneId, key}], conflicts: [{sceneId, key}], missing: [sceneId] }.
export function mergeSceneFields({ liveSegments = [], freshSegments = [], saved, scenes = {} }) {
  const result = { applied: [], conflicts: [], missing: [] };
  const liveById = new Map(liveSegments.filter((s) => s?.id).map((s) => [String(s.id), s]));
  const freshById = new Map(freshSegments.filter((s) => s?.id).map((s) => [String(s.id), s]));
  for (const [sceneId, change] of Object.entries(scenes || {})) {
    const live = liveById.get(String(sceneId));
    const fresh = freshById.get(String(sceneId));
    if (!live || !fresh) {
      result.missing.push(String(sceneId));
      continue;
    }
    const base = saved.segments[String(sceneId)] || (saved.segments[String(sceneId)] = {});
    for (const key of Array.isArray(change?.segment) ? change.segment : []) {
      if (sameValue(live[key], fresh[key])) {
        base[key] = clone(fresh[key]);
        continue;
      }
      if (!sameValue(live[key], base[key])) {
        result.conflicts.push({ sceneId: String(sceneId), key });
      } else {
        if (fresh[key] === undefined) delete live[key];
        else live[key] = clone(fresh[key]);
        result.applied.push({ sceneId: String(sceneId), key });
      }
      base[key] = clone(fresh[key]);
    }
  }
  return result;
}

// The scene fields this window holds that differ from the last saved state, as [{sceneId, key}]. A key of "" says the scenes
// themselves differ (one was added, removed or moved here), which cannot be put back on a reloaded project.
export function localSegmentEdits(liveSegments = [], saved) {
  const edits = [];
  const live = (Array.isArray(liveSegments) ? liveSegments : []).filter((segment) => segment?.id);
  if (live.map((segment) => String(segment.id)).join("\n") !== (saved.order || []).join("\n")) edits.push({ sceneId: "", key: "" });
  for (const segment of live) {
    const id = String(segment.id);
    const base = saved.segments[id];
    if (!base) continue;
    const held = durableSegment(segment);
    for (const key of new Set([...Object.keys(held), ...Object.keys(base)])) {
      if (DERIVED_KEYS.has(key)) continue;
      if (!sameish(held[key], base[key])) edits.push({ sceneId: id, key });
    }
  }
  return edits;
}

// Bring every scene the window holds up to the server's copy, field by field, against the last saved state. A field the user
// has not edited takes the server's value (including values an API save did not announce, such as the video history of a
// render). A field edited here keeps the local value, and is reported when the server changed it too. Whatever the server and
// the window now agree on stops counting as unsaved. Scenes are matched by id.
export function rebaseSegments({ liveSegments = [], freshSegments = [], saved }) {
  const result = { applied: [], conflicts: [] };
  const freshById = new Map(freshSegments.filter((segment) => segment?.id).map((segment) => [String(segment.id), durableSegment(segment)]));
  for (const live of liveSegments) {
    const id = String(live?.id ?? "");
    const fresh = freshById.get(id);
    if (!id || !fresh) continue;
    const base = saved.segments[id] || (saved.segments[id] = {});
    const held = durableSegment(live);
    for (const key of new Set([...Object.keys(held), ...Object.keys(fresh), ...Object.keys(base)])) {
      if (sameish(held[key], fresh[key])) {
        setOrDelete(base, key, fresh[key]);
        continue;
      }
      if (sameish(held[key], base[key])) {
        const was = clone(held[key]);
        setOrDelete(live, key, fresh[key]);
        result.applied.push({ sceneId: id, key, was });
      } else if (!sameish(base[key], fresh[key])) {
        result.conflicts.push({ sceneId: id, key });
      }
      setOrDelete(base, key, fresh[key]);
    }
  }
  return result;
}

// Merge the scene reference maps (characters, location) an external scene edit changed.
export function mergeReferenceMaps({ liveRefs, freshRefs = {}, saved, scenes = {} }) {
  const result = { applied: [], conflicts: [] };
  for (const [sceneId, change] of Object.entries(scenes || {})) {
    for (const name of Array.isArray(change?.references) ? change.references : []) {
      const mapKey = REFERENCE_MAP_KEYS[name];
      if (!mapKey) continue;
      const liveMap = liveRefs[mapKey] && typeof liveRefs[mapKey] === "object" ? liveRefs[mapKey] : (liveRefs[mapKey] = {});
      const freshMap = freshRefs[mapKey] && typeof freshRefs[mapKey] === "object" ? freshRefs[mapKey] : {};
      const baseMap = saved.referenceMaps[name] || (saved.referenceMaps[name] = {});
      const live = liveMap[sceneId];
      const fresh = freshMap[sceneId];
      if (!sameValue(live, fresh)) {
        if (!sameValue(live, baseMap[sceneId])) {
          result.conflicts.push({ sceneId, key: name });
        } else {
          if (fresh === undefined) delete liveMap[sceneId];
          else liveMap[sceneId] = clone(fresh);
          liveRefs[REFERENCE_SWITCHES[name]] = Boolean(freshRefs[REFERENCE_SWITCHES[name]]);
          result.applied.push({ sceneId, key: name });
        }
      }
      baseMap[sceneId] = clone(fresh);
    }
  }
  return result;
}

// Merge external Timeline Note edits (create, change, delete) by note id.
export function mergeTimelineMarkers({ liveMarkers = [], freshMarkers = [], saved, markerIds = [] }) {
  const result = { markers: liveMarkers.slice(), applied: [], conflicts: [] };
  for (const markerId of markerIds) {
    const id = String(markerId);
    const index = result.markers.findIndex((marker) => String(marker?.id) === id);
    const live = index >= 0 ? result.markers[index] : undefined;
    const fresh = freshMarkers.find((marker) => String(marker?.id) === id);
    const base = saved.markers[id];
    if (!sameValue(live, fresh)) {
      if (!sameValue(live, base)) {
        result.conflicts.push(id);
      } else {
        if (fresh === undefined) result.markers.splice(index, 1);
        else if (index >= 0) result.markers[index] = clone(fresh);
        else result.markers.push(clone(fresh));
        result.applied.push(id);
      }
    }
    if (fresh === undefined) delete saved.markers[id];
    else saved.markers[id] = clone(fresh);
  }
  return result;
}

// Which queued events the fresh session covers, and whether they can be merged field by field.
export function planExternalChanges(events = [], knownRevision = 0, freshRevision = 0) {
  const covered = events.filter((event) => Number(event.revision) > knownRevision && Number(event.revision) <= freshRevision);
  const revisions = new Set(covered.map((event) => Number(event.revision)));
  let contiguous = true;
  for (let revision = knownRevision + 1; revision <= freshRevision; revision += 1) {
    if (!revisions.has(revision)) contiguous = false;
  }
  const mergeable = contiguous && covered.every((event) => MERGEABLE_KINDS.has(String(event.change?.kind || "")));
  return { covered, contiguous, mergeable };
}

export function createExternalChangeSync({
  api, state, overlay, currentSessionData, loadSessionFromProject, postJson, syncBuilderSessionSaveRevision,
  whenBuilderSessionSavesIdle, normalizeTimelineMarkers, normalizeFluxReferenceBuilder, ensureAllSegmentRuntimeFields,
  syncInspector, render, toast,
}) {
  let saved = null;
  let savedFolder = "";
  let knownRevision = 0;
  let queue = [];
  let timer = null;
  let gapRetries = 0;
  let processing = false;
  let lastWarningAt = 0;
  let pending = null;  // a reload that has to wait: { folder, reason, freshRevision }
  let pendingTimer = null;
  let baselineAt = 0;  // when the saved state was recorded
  let lastInputAt = 0;  // the last click, key press or field change of the user
  let absorbTimer = null;

  const record = (projectFolder, revision, session) => {
    savedFolder = projectFolderKey(projectFolder);
    knownRevision = Math.max(0, Number(revision || 0));
    saved = savedStateFromSession(session);
    baselineAt = Date.now();
  };
  const INPUT_EVENTS = ["pointerdown", "keydown", "input", "change", "paste", "drop"];
  const noteInput = () => { lastInputAt = Date.now(); };
  for (const type of INPUT_EVENTS) document.addEventListener(type, noteInput, true);

  // The Builder fills in values of its own after a project loads and after a scene changes (a video's history from its backups,
  // the full settings of a locked scene). Those are not edits. While the user has not touched anything since the saved state was
  // recorded, whatever the window holds is taken as the saved state, so only real edits are ever "unsaved".
  const absorbDrift = () => {
    if (!saved || lastInputAt > baselineAt) return false;
    const session = currentSessionData();
    const scenes = (session.segments || []).filter((segment) => segment?.id);
    saved.order = scenes.map((segment) => String(segment.id));
    for (const segment of scenes) saved.segments[String(segment.id)] = clone(durableSegment(segment));
    saved.sections = sectionsKey(session);
    saved.key = unsavedKey(session);
    return true;
  };
  const absorbSoon = (delay) => {
    clearTimeout(absorbTimer);
    absorbTimer = setTimeout(absorbDrift, delay);
  };
  const onLoaded = (event) => {
    record(event.detail?.projectFolder || state.projectFolder, event.detail?.revision, currentSessionData());
    pending = null;  // a project that was just loaded has nothing left to catch up with
    clearTimeout(pendingTimer);
    absorbSoon(1500);  // once the Builder has drawn it and filled in its own values
  };
  const onSaved = (event) => {
    if (projectFolderKey(event.detail?.projectFolder) !== projectFolderKey(state.projectFolder)) return;
    record(state.projectFolder, event.detail?.revision, event.detail?.session || currentSessionData());
  };
  const sectionsChanged = (session) => !saved || sectionsKey(session) !== saved.sections;
  const hasUnsavedEdits = () => {
    if (!saved) return true;
    const session = currentSessionData();
    return sectionsChanged(session) || localSegmentEdits(session.segments, saved).length > 0;
  };
  const isEditing = () => {
    const active = document.activeElement;
    if (!active || !overlay?.contains?.(active)) return false;
    return active.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName);
  };
  const isRendering = () => [...(state.segments || []), ...(state.overlaySegments || [])]
    .some((segment) => segment?.video_status === "running");
  const schedule = (delay = 0) => {
    clearTimeout(timer);
    timer = setTimeout(() => { process().catch((error) => console.warn("[VRGDG Builder] External change sync failed:", error)); }, delay);
  };
  const forwardToStoryboard = (events) => {
    window.dispatchEvent(new CustomEvent(STORYBOARD_EXTERNAL_CHANGE_EVENT, {
      detail: { projectFolder: state.projectFolder, changes: events.map((event) => event.change || {}) },
    }));
  };

  // A reload that cannot happen now is tried again every few seconds until it can, so the user never has to reload by hand.
  function deferReload(folder, reason, freshRevision, why) {
    const tries = (pending?.tries || 0) + 1;
    if (tries > 240) return;  // about ten minutes: leave it to the user
    pending = { folder, reason, freshRevision, tries };
    if (why && Date.now() - lastWarningAt > 20000) {
      lastWarningAt = Date.now();
      toast(`${reason} ${why} The project will reload by itself as soon as it can (revision ${freshRevision}).`, true);
    }
    clearTimeout(pendingTimer);
    pendingTimer = setTimeout(retryPending, 2500);
  }

  async function retryPending() {
    // The job stays pending until the project has loaded (onLoaded clears it), so the tries add up across attempts.
    const job = pending;
    if (!job) return;
    if (projectFolderKey(state.projectFolder) !== projectFolderKey(job.folder)) {
      pending = null;
      return;
    }
    if (isEditing()) {
      deferReload(job.folder, job.reason, job.freshRevision, "");
      return;
    }
    await reloadProject(job.folder, job.reason, job.freshRevision);
  }

  // Reload the project from the server. Scene fields the user changed and has not saved are put back on the reloaded scenes.
  async function reloadProject(folder, reason, freshRevision) {
    if (isRendering()) {
      deferReload(folder, reason, freshRevision, "A render is running in this window.");
      return;
    }
    absorbDrift();
    const session = currentSessionData();
    const edits = saved ? localSegmentEdits(session.segments, saved) : [];
    if (sectionsChanged(session) || edits.some((edit) => !edit.key)) {
      deferReload(folder, reason, freshRevision, "This window has unsaved changes to the project's settings, references or scene list.");
      return;
    }
    const kept = edits.map(({ sceneId, key }) => ({ sceneId, key, value: clone((session.segments || []).find((segment) => String(segment?.id) === sceneId)?.[key]) }));
    if (!(await loadSessionFromProject(folder, { preserveSelection: true, quiet: true }))) {
      deferReload(folder, reason, freshRevision, "");
      return;
    }
    let restored = 0;
    for (const { sceneId, key, value } of kept) {
      const segment = (state.segments || []).find((item) => String(item?.id) === sceneId);
      if (!segment) continue;
      setOrDelete(segment, key, value);
      restored += 1;
    }
    if (restored) {
      // The restored edits are the user's and still unsaved: they must not be taken for values the Builder filled in.
      lastInputAt = Math.max(Date.now(), baselineAt + 1);
      ensureAllSegmentRuntimeFields();
      syncInspector();
      render();
    }
    toast(`${reason} The project was reloaded${restored ? ` and your ${restored} unsaved scene edit${restored === 1 ? " was" : "s were"} kept` : ""}.`);
  }

  async function process() {
    if (processing) return;
    if (!queue.length) return;
    if (isEditing()) {
      schedule(800);
      return;
    }
    processing = true;
    try {
      await whenBuilderSessionSavesIdle();
      absorbDrift();
      const folder = state.projectFolder;
      if (projectFolderKey(folder) !== savedFolder || !saved) {
        queue = [];
        return;
      }
      const data = await postJson("/vrgdg/music_builder/load_session", { project_folder: folder });
      if (projectFolderKey(state.projectFolder) !== projectFolderKey(folder)) {
        queue = [];  // the user opened another project meanwhile
        return;
      }
      const fresh = data.session || {};
      const freshRevision = Number(fresh.revision || 0);
      if (freshRevision <= knownRevision) {
        queue = queue.filter((event) => Number(event.revision) > freshRevision);
        return;
      }
      const plan = planExternalChanges(queue, knownRevision, freshRevision);
      if (!plan.contiguous && gapRetries < 3) {
        gapRetries += 1;  // a later notice may still be on its way
        schedule(400);
        return;
      }
      gapRetries = 0;
      queue = queue.filter((event) => Number(event.revision) > freshRevision);
      if (!plan.mergeable) {
        await reloadProject(folder, "The project was changed through the Agent API / MCP.", freshRevision);
        forwardToStoryboard(plan.covered);
        return;
      }
      const sceneChanges = {};
      const markerIds = new Set();
      for (const event of plan.covered) {
        for (const [sceneId, change] of Object.entries(event.change?.scenes || {})) {
          const target = sceneChanges[sceneId] || (sceneChanges[sceneId] = { segment: [], references: [] });
          target.segment.push(...(change.segment || []));
          target.references.push(...(change.references || []));
        }
        for (const markerId of event.change?.marker_ids || []) markerIds.add(markerId);
      }
      const sectionsBefore = sectionsChanged(currentSessionData());
      const scenes = mergeSceneFields({ liveSegments: [...state.segments, ...state.overlaySegments], freshSegments: [...(fresh.segments || []), ...(fresh.overlay_segments || [])], saved, scenes: sceneChanges });
      if (scenes.missing.length) {
        await reloadProject(folder, "Scenes changed through the Agent API / MCP no longer match this window.", freshRevision);
        return;
      }
      const refs = { ...(state.fluxReferenceBuilder || {}) };
      const references = mergeReferenceMaps({ liveRefs: refs, freshRefs: fresh.flux_reference_builder || {}, saved, scenes: sceneChanges });
      if (references.applied.length) state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
      const markers = mergeTimelineMarkers({ liveMarkers: state.timelineMarkers || [], freshMarkers: fresh.timeline_markers || [], saved, markerIds: [...markerIds] });
      if (markers.applied.length) {
        state.timelineMarkers = normalizeTimelineMarkers(markers.markers);
        if (state.activeTimelineMarkerId && !state.timelineMarkers.some((marker) => marker.id === state.activeTimelineMarkerId)) state.activeTimelineMarkerId = "";
      }
      // The window fills in values of its own when a scene changes (a video's history from its backups, for one). Every scene is
      // then brought up to the server's copy, so a field an API save did not announce (the render's video history) arrives too,
      // and nothing the server and the window agree on counts as unsaved.
      ensureAllSegmentRuntimeFields();
      const rebased = rebaseSegments({ liveSegments: state.segments || [], freshSegments: fresh.segments || [], saved });
      // The server copy now holds everything this window merged; later saves must not be refused as stale.
      knownRevision = freshRevision;
      syncBuilderSessionSaveRevision(fresh.builder_save_revision);
      ensureAllSegmentRuntimeFields();
      syncInspector();
      render();
      if (!sectionsBefore) saved.sections = sectionsKey(currentSessionData());
      absorbSoon(600);  // the redraw above fills in values of its own again
      forwardToStoryboard(plan.covered);
      const conflicts = [
        ...scenes.conflicts.map((item) => `${item.sceneId}: ${item.key}`),
        ...references.conflicts.map((item) => `${item.sceneId}: ${item.key}`),
        ...markers.conflicts.map((id) => `Timeline Note ${id}`),
      ];
      // Values the window derives itself (a video's history) are put back by the redraw, so they are not changes the user sees.
      const liveById = new Map((state.segments || []).map((segment) => [String(segment?.id), segment]));
      const shown = rebased.applied.filter((item) => !sameish(liveById.get(item.sceneId)?.[item.key], item.was));
      const appliedCount = new Set([...scenes.applied, ...shown].map((item) => `${item.sceneId}:${item.key}`)).size
        + references.applied.length + markers.applied.length;
      if (conflicts.length) {
        toast(`Agent API / MCP edits arrived for fields you also changed here. Your unsaved values were kept (saving will replace the API values):\n${conflicts.join("\n")}`, true);
      } else if (appliedCount) {
        toast(`Updated from the Agent API / MCP: ${appliedCount} change${appliedCount === 1 ? "" : "s"}.`);
      }
    } finally {
      processing = false;
      if (queue.length) schedule(50);
    }
  }

  const onProjectChanged = (event) => {
    const detail = event?.detail || {};
    if (detail.source !== "agent_api") return;
    if (projectFolderKey(detail.project_folder) !== projectFolderKey(state.projectFolder)) return;
    if (String(detail.change?.kind || "") === "storyboard") {
      forwardToStoryboard([detail]);  // only the Storyboard's saved copy changed
      return;
    }
    queue.push({ revision: Number(detail.revision || 0), change: detail.change || {} });
    schedule(120);
  };

  api.addEventListener(PROJECT_CHANGED_EVENT, onProjectChanged);
  window.addEventListener("vrgdg:builder-session-loaded", onLoaded);
  window.addEventListener("vrgdg:builder-session-saved", onSaved);
  return {
    dispose() {
      clearTimeout(timer);
      clearTimeout(pendingTimer);
      clearTimeout(absorbTimer);
      for (const type of INPUT_EVENTS) document.removeEventListener(type, noteInput, true);
      api.removeEventListener(PROJECT_CHANGED_EVENT, onProjectChanged);
      window.removeEventListener("vrgdg:builder-session-loaded", onLoaded);
      window.removeEventListener("vrgdg:builder-session-saved", onSaved);
    },
    // What this window holds that the server does not, and whether a reload is waiting. For checks and bug reports.
    status() {
      const session = currentSessionData();
      return {
        edits: saved ? localSegmentEdits(session.segments, saved).slice(0, 20).map((edit) => `${edit.sceneId}:${edit.key}`) : ["(nothing recorded)"],
        sections: sectionsChanged(session),
        editing: isEditing(),
        rendering: isRendering(),
        reloadWaiting: Boolean(pending),
      };
    },
  };
}
