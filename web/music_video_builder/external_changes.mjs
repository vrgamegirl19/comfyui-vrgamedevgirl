// Shows Agent API / MCP edits in the open Video Builder.
//
// agent_api/project_events.py broadcasts "vrgdg.project_changed" after every API save. Scene-card and
// Timeline Note edits name the scenes, fields and notes they changed; those are merged into the open
// project field by field against the last saved state, so unsaved local edits are kept (a field edited on
// both sides keeps the local value and is reported). Any other change reloads the project when nothing
// is unsaved, focused or rendering; otherwise the user is told, and the server keeps rejecting this
// window's older snapshot until it reloads.

export const PROJECT_CHANGED_EVENT = "vrgdg.project_changed";
export const STORYBOARD_EXTERNAL_CHANGE_EVENT = "vrgdg:storyboard-external-change";
const MERGEABLE_KINDS = new Set(["scene_fields", "timeline_markers"]);
const REFERENCE_MAP_KEYS = { subjects: "subject_scene_map", locations: "scene_map" };
const REFERENCE_SWITCHES = { subjects: "use_subject_reference", locations: "use_location_references" };

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

// The saved state the merge compares against, from a session payload (currentSessionData() or a save).
export function savedStateFromSession(session = {}) {
  const refs = session.flux_reference_builder && typeof session.flux_reference_builder === "object" ? session.flux_reference_builder : {};
  return {
    key: unsavedKey(session),
    segments: Object.fromEntries((Array.isArray(session.segments) ? session.segments : [])
      .filter((segment) => segment?.id)
      .map((segment) => [String(segment.id), clone(durableSegment(segment))])),
    markers: Object.fromEntries((Array.isArray(session.timeline_markers) ? session.timeline_markers : [])
      .filter((marker) => marker?.id)
      .map((marker) => [String(marker.id), clone(marker)])),
    referenceMaps: Object.fromEntries(Object.entries(REFERENCE_MAP_KEYS)
      .map(([name, key]) => [name, clone(refs[key] && typeof refs[key] === "object" ? refs[key] : {})])),
  };
}

// Everything an API edit could conflict with. Equal keys mean nothing is unsaved.
export function unsavedKey(session = {}) {
  return JSON.stringify([
    (session.segments || []).map(durableSegment),
    (session.overlay_segments || []).map(durableSegment),
    session.timeline_markers || [],
    session.flux_reference_builder || null,
    session.minimax_h3_settings || null,
    session.builder_story_layer || null,
    session.builder_storyboard_defaults || null,
    session.lyric_mapper || null,
  ]);
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

  const record = (projectFolder, revision, session) => {
    savedFolder = projectFolderKey(projectFolder);
    knownRevision = Math.max(0, Number(revision || 0));
    saved = savedStateFromSession(session);
  };
  const onLoaded = (event) => {
    record(event.detail?.projectFolder || state.projectFolder, event.detail?.revision, currentSessionData());
  };
  const onSaved = (event) => {
    if (projectFolderKey(event.detail?.projectFolder) !== projectFolderKey(state.projectFolder)) return;
    record(state.projectFolder, event.detail?.revision, event.detail?.session || currentSessionData());
  };
  const hasUnsavedEdits = () => !saved || unsavedKey(currentSessionData()) !== saved.key;
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

  async function reloadProject(folder, reason, freshRevision) {
    if (hasUnsavedEdits() || isRendering()) {
      if (Date.now() - lastWarningAt > 20000) {
        lastWarningAt = Date.now();
        toast(`${reason} This window has unsaved edits or a render running, so it was not reloaded. Your edits are kept, but the project must be reloaded before this window can save (revision ${freshRevision}).`, true);
      }
      return;
    }
    if (await loadSessionFromProject(folder, { preserveSelection: true, quiet: true })) toast(`${reason} The project was reloaded.`);
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
      const unsavedBefore = hasUnsavedEdits();
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
      // The server copy now holds everything this window merged; later saves must not be refused as stale.
      knownRevision = freshRevision;
      syncBuilderSessionSaveRevision(fresh.builder_save_revision);
      if (!unsavedBefore) saved.key = unsavedKey(currentSessionData());
      ensureAllSegmentRuntimeFields();
      syncInspector();
      render();
      forwardToStoryboard(plan.covered);
      const conflicts = [
        ...scenes.conflicts.map((item) => `${item.sceneId}: ${item.key}`),
        ...references.conflicts.map((item) => `${item.sceneId}: ${item.key}`),
        ...markers.conflicts.map((id) => `Timeline Note ${id}`),
      ];
      const appliedCount = scenes.applied.length + references.applied.length + markers.applied.length;
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
      api.removeEventListener(PROJECT_CHANGED_EVENT, onProjectChanged);
      window.removeEventListener("vrgdg:builder-session-loaded", onLoaded);
      window.removeEventListener("vrgdg:builder-session-saved", onSaved);
    },
  };
}
