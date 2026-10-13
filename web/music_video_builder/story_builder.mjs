// Story Builder: the Story Mode workspace. One window with tabs in the order a story music video is made
// (Song, Cast, Story, Acts, Scenes, Places, Beats & Prompts). It reads and writes the same session state as the
// Video Builder and calls /vrgdg/storyboard/story_mode/{action} for the LLM steps.
import { audioUrl, postJson } from "./comfy_api.mjs";
import { escapeHtml, makeButton, makeField, makeInput, makeSelect, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { mergeActs, moveActBorder, splitAct, validateStoryActs } from "./story_acts_data.mjs";
import {
  STORY_BUILDER_TABS, castWithoutImages, referenceImageGaps, relinkStoryLocations, sceneRangeText, storyBuilderBlocker,
  storyBuilderProgress, storySubjectCard,
} from "./story_builder_data.mjs";
import { normalizeMiniMaxH3Mode } from "./minimax_h3.mjs";
import { applyStoryContinuationsToSegments } from "./story_continuation.mjs";
import { applyStoryLocations } from "./story_locations_data.mjs";
import { fillStoryStamps, stampStory } from "./story_staleness.mjs";
import {
  applySceneCast, applyStoryBookText, isStoryPersonCard, normalizeSpotMinScenes, normalizeStoryActs, normalizeStoryBook,
  normalizeStoryContinuationMode, normalizeStoryScenePlan, normalizeStorySettings, normalizeStoryVideoStyle,
  planSegmentFields, resolveStoryContinuations, STORY_CAST_ROLES, STORY_CHAPTER_COLORS, STORY_FLOW_VERSION,
  STORY_RELATIONSHIP_TYPES, STORY_VIDEO_STYLES, storyBookToText,
} from "./story_mode_data.mjs";
import { normalizeTimelineMarkers } from "./timeline_state.mjs";

// Below the Reference Builder (100006) and Line Mapping windows, so the tools it opens appear on top.
const STORY_BUILDER_Z = 100004;
const STATUS_COLORS = { empty: "#475569", progress: "#f59e0b", done: "#22c55e" };
const PERFORMANCE_OPTIONS = [
  ["singing", "Singing"], ["speaking", "Speaking"], ["no_lip_sync", "Story only"], ["instrumental", "Instrumental"],
];
const CAST_ROLE_LABELS = { main: "Main character", love_interest: "Love interest", supporting: "Supporting", antagonist: "Antagonist" };
const RELATIONSHIP_LABELS = {
  love_interest: "Love interests", partners: "Partners", exes: "Exes", family: "Family", friends: "Friends",
  rivals: "Rivals", strangers: "Strangers", other: "Other",
};
const CONTINUATION_MODE_LABELS = {
  off: "Off (only the scenes I tick)",
  location: "Same spot: touching scenes at the same spot in a chapter",
  llm: "Let the LLM decide from the story",
};
const CONTINUATION_SOURCE_LABELS = { auto: "same spot", llm: "LLM", manual: "you" };
const MASKED_COLOR = "#a3e635";
const STALE_COLOR = "#f59e0b";

function el(tag, css = "", text = "") {
  const node = document.createElement(tag);
  if (css) node.style.cssText = css;
  if (text) node.textContent = text;
  return node;
}

function textarea(value, placeholder = "", minHeight = 80) {
  const node = el("textarea", `width:100%;box-sizing:border-box;min-height:${minHeight}px;resize:vertical;border:1px solid #3f3f46;border-radius:6px;background:#09090b;color:#f8fafc;padding:8px;font-size:12px;line-height:1.45;`);
  node.value = value || "";
  node.placeholder = placeholder;
  return node;
}

function card(title, note = "") {
  const node = el("div", "border:1px solid #334155;border-radius:8px;background:#0f172a;padding:12px;display:flex;flex-direction:column;gap:10px;");
  node.append(el("div", "font-size:14px;font-weight:900;color:#cffafe;", title));
  if (note) node.append(el("div", "font-size:12px;color:#94a3b8;line-height:1.45;", note));
  return node;
}

function row(...children) {
  const node = el("div", "display:flex;gap:8px;flex-wrap:wrap;align-items:center;");
  node.append(...children.filter(Boolean));
  return node;
}

function clock(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  return `${Math.floor(value / 60)}:${(value % 60).toFixed(1).padStart(4, "0")}`;
}

function reasonText(reasons) {
  return [...new Set(reasons)].join(", ");
}

// An amber box for work that is out of date because something before it changed.
function staleBox(message, ...buttons) {
  const box = el("div", `font-size:12px;color:#fde68a;border:1px solid ${STALE_COLOR};border-left:5px solid ${STALE_COLOR};border-radius:7px;background:#1c1207;padding:9px;display:flex;flex-direction:column;gap:8px;line-height:1.45;`, message);
  if (buttons.filter(Boolean).length) box.append(row(...buttons));
  return box;
}

function staleChip(label, reasons) {
  const chip = el("div", `display:inline-flex;align-self:flex-start;border:1px solid ${STALE_COLOR};border-radius:999px;background:#451a03;color:#fde68a;font-size:11px;font-weight:900;padding:3px 10px;`,
    `⚠ ${label} OUT OF DATE: ${reasonText(reasons)}`);
  return chip;
}

export function createStoryBuilder({
  autoSaveSessionQuiet, createProgressWindow, createSceneBeats, createVideoPrompts, getLlmPayload, normalizeBuilderStoryLayer,
  openFluxReferenceBuilderModal, openLyricMappingWorkflowModal, openLyricReviewModal, openSceneDefaults, openStoryboardClassic,
  pushHistory, referenceBuilderPayload, render, state, storyboardScenePayload, syncInspector = () => {},
  projectFolder = () => "", openInstructions = null,
}) {
  let open = null;

  // The Story Mode instructions each tab's steps use. Beats & Prompts adds the MiniMax prompt instructions of the
  // project's video mode (Story Mode's own copy).
  function tabInstructions(tab) {
    const minimax = isMiniMax()
      ? [[`minimax_h3_story_mode_${normalizeMiniMaxH3Mode(state.miniMaxH3Settings?.video_mode)}`, "MiniMax Prompt"]] : [];
    return {
      story: [["story_mode_book", "Story"], ["story_mode_chapter", "Chapter Rewrite"]],
      acts: [["story_mode_acts", "Acts"]],
      scenes: [["story_mode_scene_plan", "Scene Plan"], ["story_mode_revise_scene", "Update From a Scene"]],
      places: [["story_mode_setting_looks", "Setting Look"], ["story_mode_spots", "Location Card"]],
      prompts: [["story_mode_scene_beat", "Scene Beat"], ...minimax],
    }[tab] || [];
  }

  // A row of "Edit ... Instructions" buttons for Story Mode's own LLM instructions (story_mode_* Builder
  // instructions). Edits apply to Story Mode only and are saved with the project.
  function instructionButtons(items) {
    if (typeof openInstructions !== "function" || !items.length) return null;
    return row(...items.map(([key, label]) => {
      const button = makeButton(`Edit ${label} Instructions`);
      button.title = "Story Mode's own LLM instructions for this step. Changing them never changes another mode.";
      button.onclick = () => {
        const first = segments()[0];
        if (!first) {
          toast("Create the timeline scenes first.", true);
          return;
        }
        openInstructions(key, first);
      };
      return button;
    }));
  }

  // Base timeline scenes only; overlay clips are not part of the story.
  function segments() {
    return (Array.isArray(state.segments) ? state.segments : []).filter((segment) => segment && typeof segment === "object");
  }

  function layer() {
    return normalizeBuilderStoryLayer(state.builderStoryLayer);
  }

  // Reference to Video (MiniMax) and RTV / Ingredients (LTX) send the card images, so every scene needs one.
  function referenceImagesRequired() {
    if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
      return ["reference_to_video", "image_reference_to_video"].includes(normalizeMiniMaxH3Mode(state.miniMaxH3Settings?.video_mode));
    }
    return ["rtv", "ingredients"].includes(String(state.videoModelMode || ""));
  }

  // Toast text for a book or chapter step that wrote people outside the story cast out of the story.
  function removedNote(data, done) {
    const removed = Array.isArray(data?.removed_people) ? data.removed_people : [];
    return removed.length ? `${done} Kept to your cast: wrote out ${removed.join(", ")}.` : done;
  }

  function imageGaps() {
    return referenceImageGaps(normalizeFluxReferenceBuilder(state.fluxReferenceBuilder), segments().map((segment) => segment.id));
  }

  // marks: the steps this save brings up to date, as {book: true, acts|plan|places|beats|prompts: true | [ids] |
  // (layer) => [ids]}. True means every existing output of that step.
  async function saveLayer(fields, label, marks = null) {
    const next = normalizeBuilderStoryLayer({
      ...state.builderStoryLayer, story_flow_version: STORY_FLOW_VERSION, ...fields,
    });
    if (marks) {
      const resolved = Object.fromEntries(Object.entries(marks).map(([key, value]) => [key, typeof value === "function" ? value(next) : value]));
      next.story_stamps = stampStory(staleInput(next), resolved);
    }
    state.builderStoryLayer = next;
    render();
    await autoSaveSessionQuiet(label);
  }

  function isMiniMax() {
    return normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
  }

  function promptKey() {
    return isMiniMax() ? "minimax_h3_prompt" : "i2v_prompt";
  }

  function staleInput(storyLayer = layer()) {
    return { layer: storyLayer, refs: normalizeFluxReferenceBuilder(state.fluxReferenceBuilder), segments: segments(), promptKey: promptKey() };
  }

  // Out-of-date warnings are switched off: nothing is shown as out of date, and every Redo runs only the scenes you
  // choose. Stamps are still kept (story_staleness.mjs) so the warnings can come back without losing history.
  function staleness() {
    return { book: [], acts: {}, plan: {}, places: {}, beats: {}, prompts: {} };
  }

  // Stamps outputs that have none yet, so later changes to what they were made from show as out of date.
  // Also relinks story location cards that lost their link to a chapter setting.
  function fillStamps() {
    const relinked = relinkStoryLocations(normalizeFluxReferenceBuilder(state.fluxReferenceBuilder), normalizeStorySettings(layer().story_settings));
    if (relinked) state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(relinked);
    const filled = fillStoryStamps(staleInput());
    if (filled) state.builderStoryLayer = normalizeBuilderStoryLayer({ ...state.builderStoryLayer, story_stamps: filled });
    if (!relinked && !filled) return;
    Promise.resolve(autoSaveSessionQuiet("story builder change tracking")).catch(() => {});
  }

  // Who two characters are to each other. Every change saves; the story, scene plan and prompts get the list.
  function relationshipEditor(cards, saved) {
    const box = el("div", "display:flex;flex-direction:column;gap:6px;margin-top:8px;border-top:1px dashed #334155;padding-top:8px;");
    box.append(el("div", "font-size:12px;font-weight:800;color:#cffafe;", "Relationships"),
      el("div", "font-size:11px;color:#94a3b8;", "How two characters are connected, e.g. love interests. The story, the scenes and the shots show it in what they do together."));
    const list = el("div", "display:flex;flex-direction:column;gap:6px;");
    const ids = cards.map((subject) => subject.id);
    const names = Object.fromEntries(cards.map((subject) => [subject.id, subject.name || subject.id]));
    const rows = [];
    const save = () => saveLayer({
      story_relationships: rows.map((item) => ({ a: item.a.value, b: item.b.value, type: item.type.value, note: item.note.value })),
    }, "story builder relationships");
    const personSelect = (value) => {
      const select = makeSelect(ids, value);
      for (const option of select.options) option.textContent = names[option.value] || option.value;
      select.style.width = "max-content";
      return select;
    };
    const addRow = (item) => {
      const a = personSelect(item.a);
      const b = personSelect(item.b);
      const type = makeSelect(STORY_RELATIONSHIP_TYPES, item.type || "love_interest");
      for (const option of type.options) option.textContent = RELATIONSHIP_LABELS[option.value] || option.value;
      type.style.width = "max-content";
      const note = makeInput(item.note || "");
      note.placeholder = "Note, e.g. estranged, secretly in love";
      const remove = makeButton("Remove");
      const line = row(a, el("span", "font-size:12px;color:#94a3b8;", "and"), b, type, note, remove);
      const entry = { a, b, type, note };
      remove.onclick = () => {
        rows.splice(rows.indexOf(entry), 1);
        line.remove();
        save();
      };
      for (const control of [a, b, type, note]) control.onchange = save;
      rows.push(entry);
      list.append(line);
    };
    for (const item of saved) if (names[item.a] && names[item.b]) addRow(item);
    const add = makeButton("Add Relationship");
    add.disabled = ids.length < 2;
    if (add.disabled) add.title = "Add a second character first.";
    add.onclick = () => {
      addRow({ a: ids[0], b: ids[1], type: "love_interest" });
      save();
    };
    box.append(list, row(add));
    return box;
  }

  // The characters the story is about: the ones picked in story_cast_ids, or every character card when none are.
  function storyCastCards() {
    const cards = (normalizeFluxReferenceBuilder(state.fluxReferenceBuilder).subjects || []).filter(isStoryPersonCard);
    const chosen = new Set(layer().story_cast_ids || []);
    const picked = cards.filter((subject) => chosen.has(subject.id));
    return chosen.size && picked.length ? picked : cards;
  }

  function sceneNumbers(ids) {
    const index = Object.fromEntries(segments().map((segment, position) => [segment.id, position + 1]));
    return ids.map((id) => index[id]).filter(Boolean).sort((a, b) => a - b);
  }

  // Which scenes continue the one before, from the mode and the manual choices, written onto the timeline scenes as
  // MiniMax masked continuations. Returns the resolved plan; tells the user about scenes their mode cannot continue.
  function applyContinuations(plan, acts, mode = layer().story_continuation_mode) {
    const timeline = segments().map((segment) => ({ id: segment.id, start: segment.start, end: segment.end }));
    const resolved = resolveStoryContinuations(plan, timeline, acts, mode);
    if (!isMiniMax()) return resolved;
    const result = applyStoryContinuationsToSegments(segments(), resolved, state.miniMaxH3Settings);
    if (result.blocked.length) {
      const all = segments();
      const numbers = result.blocked.map((id) => all.findIndex((segment) => segment.id === id) + 1);
      toast(`${sceneRangeText(numbers)} cannot use masked continuation in their MiniMax mode (Image to Video, Image + Reference and 2 Pass Advanced cannot). Change the mode to continue them.`, true);
    }
    return resolved;
  }

  // The scene number each continued scene carries on from, by scene id.
  function continuedFrom(plan) {
    const all = segments();
    const byId = Object.fromEntries(normalizeStoryScenePlan(plan).map((entry) => [entry.scene_id, entry]));
    const result = {};
    all.forEach((segment, index) => {
      if (index > 0 && byId[segment.id]?.continues_previous) result[segment.id] = index;
    });
    return result;
  }

  // A bright chip that says which scene this one is masked to.
  function maskedChip(fromNumber, toNumber, source = "") {
    const chip = el("div", `display:inline-flex;align-items:center;gap:6px;align-self:flex-start;border:1px solid ${MASKED_COLOR};border-radius:999px;background:rgba(54,83,20,.9);color:#ecfccb;font-size:11px;font-weight:900;padding:3px 10px;letter-spacing:.02em;`,
      `\u26d3 MASKED: Scene ${fromNumber} \u2192 Scene ${toNumber}${source ? `  (${CONTINUATION_SOURCE_LABELS[source] || source})` : ""}`);
    chip.title = `Scene ${toNumber} carries straight on from the end of Scene ${fromNumber} in one take (MiniMax Latent Continuation Masked).`;
    return chip;
  }

  function applyPlanToProject(plan, acts) {
    plan = applyContinuations(plan, acts);
    const fields = Object.fromEntries(normalizeStoryScenePlan(plan).map((entry) => [entry.scene_id, planSegmentFields(entry.performance_mode)]));
    for (const segment of segments()) {
      if (fields[segment.id]) Object.assign(segment, fields[segment.id]);
    }
    const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
    const subjectIds = (refs.subjects || []).filter(isStoryPersonCard).map((subject) => subject.id);
    // Each planned scene's characters become its mapping (a cast set in the Scenes tab included).
    const { map, changed } = applySceneCast(refs.subject_scene_map, plan, acts, subjectIds);
    if (changed.length) {
      refs.subject_scene_map = map;
      refs.use_subject_reference = true;
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
    }
    return plan;
  }

  // One LLM step. While it waits, the window counts the seconds and the bar creeps toward `to`, so a long step never
  // looks frozen. `progress` shares one window across a run of steps; the caller then closes it.
  async function runStep(action, extra, title, { progress: shared = null, detail = "", from = 5, to = 95 } = {}) {
    const progress = shared || createProgressWindow(title, { zIndex: STORY_BUILDER_Z + 10 });
    const started = Date.now();
    const tick = () => {
      const seconds = Math.round((Date.now() - started) / 1000);
      const percent = from + (to - from) * (1 - Math.exp(-seconds / 45));
      progress.set(`${title}...\n${detail ? `${detail}\n` : ""}Waiting for the LLM: ${seconds}s\nThis runs on your selected LLM runner.`, percent);
    };
    tick();
    const timer = setInterval(tick, 1000);
    try {
      const baseIds = new Set(segments().map((segment) => segment.id));
      const scenes = storyboardScenePayload().filter((scene) => baseIds.has(scene.id));
      const data = await postJson(`/vrgdg/storyboard/story_mode/${action}`, {
        ...getLlmPayload(),
        // The project folder lets the step use the project's own Story Mode instructions.
        project_folder: projectFolder() || "",
        story_layer: layer(),
        scenes,
        reference_builder: referenceBuilderPayload(),
        timeline_markers: normalizeTimelineMarkers(state.timelineMarkers),
        lyrics: scenes.map((scene) => scene.lyrics || "").filter(Boolean).join("\n"),
        unload_after: true,
        ...extra,
      }, 20 * 60 * 1000);
      if (!shared) {
        progress.set(`${title}: done.`, 100);
        progress.close(900);
      }
      return data;
    } catch (error) {
      progress.set(`${title}: stopped.\n${detail ? `${detail}\n` : ""}Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
      return null;
    } finally {
      clearInterval(timer);
    }
  }

  // Plans the given acts one LLM call at a time, saving each act as it finishes. The window names the act and its
  // scenes. A failed act stops the run; the acts before it stay planned. Returns how many acts were planned.
  // sceneIds: replan only these scenes of the acts (the rest keep their plan). Empty replans every scene of each act.
  async function planActs(acts, list, title, { sceneIds = [] } = {}) {
    const progress = createProgressWindow(title, { zIndex: STORY_BUILDER_Z + 10 });
    const numberById = Object.fromEntries(segments().map((segment, index) => [segment.id, index + 1]));
    const spotMin = layer().story_spot_min_scenes;
    pushHistory();
    let done = 0;
    for (const [index, act] of list.entries()) {
      const numbers = act.scene_ids.filter((id) => !sceneIds.length || sceneIds.includes(id)).map((id) => numberById[id]).filter(Boolean);
      const detail = `Act ${index + 1} of ${list.length}: ${act.title || act.id}\n${numbers.length ? `${sceneRangeText(numbers)} (${numbers.length} scene${numbers.length === 1 ? "" : "s"})` : "No scenes"}\n${done} of ${list.length} act${list.length === 1 ? "" : "s"} planned so far.`;
      const data = await runStep("plan", { act_id: act.id, scene_ids: sceneIds, spot_min_scenes: spotMin, unload_after: index === list.length - 1 }, title, {
        progress, detail, from: (100 * index) / list.length, to: (100 * (index + 1)) / list.length,
      });
      if (!data?.story_layer) return done;
      const planned = applyPlanToProject(data.story_layer.story_scene_plan, acts);
      await saveLayer({ ...data.story_layer, story_scene_plan: planned }, "story builder act replanned", { plan: act.scene_ids });
      done += 1;
    }
    progress.set(`${title}: done.\nPlanned ${done} act${done === 1 ? "" : "s"}.`, 100);
    progress.close(1200);
    return done;
  }

  // The Replan window of an act (a Chapter in the tabs): where it happens and what must happen, then the whole act's
  // scenes or only the scenes picked. Both texts are kept on the act and steer the plan and every beat in it.
  function openReplanDialog(act, onDone = () => {}) {
    const all = segments();
    const numberById = Object.fromEntries(all.map((segment, index) => [segment.id, index + 1]));
    const overlay = el("div", "position:fixed;inset:0;background:rgba(2,6,23,.72);display:flex;align-items:center;justify-content:center;z-index:" + (STORY_BUILDER_Z + 20) + ";");
    const box = el("div", "background:#0b1424;border:1px solid #22d3ee;border-radius:10px;padding:16px;width:min(640px,92vw);max-height:90vh;overflow:auto;display:flex;flex-direction:column;gap:10px;color:#e2e8f0;");
    box.append(el("div", "font-size:16px;font-weight:900;", "Replan " + (act.title || act.id)));
    box.append(el("div", "font-size:12px;color:#94a3b8;", "Your words steer the plan and the beat of every scene in this act. Leave a box empty to keep what is there."));
    const where = textarea(act.location_note || "", "Where it happens: describe the place, e.g. A narrow rooftop at dusk, wet from rain, neon signs below.", 70);
    const what = textarea(act.direction || "", "What must happen: a bit of story for this act, e.g. She finds the letter, hesitates, then burns it.", 100);
    box.append(el("div", "font-size:12px;font-weight:800;", "Location"), where, el("div", "font-size:12px;font-weight:800;", "Story for this act"), what);
    const picks = new Map();
    const scope = el("div", "display:flex;flex-direction:column;gap:6px;border-top:1px dashed #334155;padding-top:8px;");
    const modeSelect = makeSelect(["all", "some"], "all");
    modeSelect.options[0].textContent = "Replan every scene of this act";
    modeSelect.options[1].textContent = "Replan only the scenes I tick (scene by scene)";
    const list = el("div", "display:none;flex-wrap:wrap;gap:8px;");
    for (const id of act.scene_ids) {
      const label = el("label", "display:flex;gap:4px;font-size:12px;align-items:center;cursor:pointer;");
      const tick = document.createElement("input");
      tick.type = "checkbox";
      picks.set(id, tick);
      label.append(tick, document.createTextNode("Scene " + (numberById[id] || "?")));
      list.append(label);
    }
    modeSelect.onchange = () => { list.style.display = modeSelect.value === "some" ? "flex" : "none"; };
    scope.append(modeSelect, list);
    box.append(scope);
    // The plan alone does not change a beat or prompt that already exists, so these are on by default.
    const tickBox = (text) => {
      const label = el("label", "display:flex;gap:6px;font-size:12px;align-items:center;cursor:pointer;");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = true;
      label.append(input, document.createTextNode(text));
      box.append(label);
      return input;
    };
    const redoBeats = tickBox("Then write new beats for these scenes (replaces the beats there now)");
    const redoPrompts = tickBox("Then write new video prompts for these scenes (replaces the prompts there now)");
    const close = () => overlay.remove();
    const cancel = makeButton("Cancel");
    cancel.onclick = close;
    const go = makeButton("Replan", "primary");
    go.onclick = async () => {
      const ids = modeSelect.value === "some" ? act.scene_ids.filter((id) => picks.get(id).checked) : [];
      if (modeSelect.value === "some" && !ids.length) {
        toast("Tick at least one scene, or choose every scene.", true);
        return;
      }
      go.disabled = true;
      const wantBeats = redoBeats.checked;
      const wantPrompts = redoPrompts.checked;
      close();
      const edited = normalizeStoryActs(layer().story_acts).map((item) => {
        if (item.id !== act.id) return item;
        const next = { ...item, location_note: where.value.trim(), direction: what.value.trim() };
        if (!next.location_note) delete next.location_note;
        if (!next.direction) delete next.direction;
        return next;
      });
      await saveLayer({ story_acts: edited }, "story builder act direction");
      const fresh = edited.find((item) => item.id === act.id);
      const done = await planActs(edited, [fresh], "Replanning " + (act.title || act.id), { sceneIds: ids });
      // The scenes just replanned get new beats and prompts from their new plan, so Beats & Prompts shows the rework.
      const targets = ids.length ? ids : act.scene_ids;
      if (done && wantBeats) {
        await createSceneBeats({ overwrite: true, sceneIds: targets });
        await saveLayer({}, "story builder beats redone", { beats: targets });
      }
      if (done && wantPrompts) {
        const noImage = referenceImagesRequired() ? sceneNumbers(targets).filter((number) => imageGaps().scenes.includes(number)) : [];
        if (noImage.length) toast("No video prompt was written for " + sceneRangeText(noImage) + ": it has no character or location image, and Reference to Video needs one.", true);
        else if (targets.some((id) => String(segments().find((segment) => segment.id === id)?.story_beat || "").trim())) {
          await createVideoPrompts({ sceneIds: targets.filter((id) => String(segments().find((segment) => segment.id === id)?.story_beat || "").trim()), storyMode: true });
          await saveLayer({}, "story builder prompts redone", { prompts: targets });
        }
      }
      onDone(done);
    };
    box.append(row(go, cancel));
    overlay.append(box);
    document.body.append(overlay);
  }

  function openStoryBuilder(options = {}) {
    if (open) {
      open.show(options.tab || open.tab);
      return;
    }
    let activeTab = options.tab || "song";
    let selectedSceneId = "";
    let hosted = null;
    const backdrop = el("div", `position:fixed;inset:0;z-index:${STORY_BUILDER_Z};background:rgba(0,0,0,.62);display:flex;align-items:stretch;justify-content:center;padding:18px;box-sizing:border-box;`);
    const shell = el("div", "width:min(1820px,calc(100vw - 36px));height:calc(100vh - 36px);box-sizing:border-box;border:1px solid #155e75;border-radius:10px;background:#111827;color:#e5e7eb;display:grid;grid-template-rows:auto auto auto minmax(0,1fr);overflow:hidden;font-family:system-ui,-apple-system,Segoe UI,sans-serif;");
    const header = el("div", "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 18px;border-bottom:1px solid #1f3b46;background:linear-gradient(180deg,#083344,#111827);");
    const title = el("div");
    title.innerHTML = '<div style="font-size:22px;font-weight:900;color:#cffafe;">Story Builder</div><div style="font-size:13px;color:#cbd5e1;margin-top:2px;">Write the story, break it into chapters, acts and scenes, then build the places it happens.</div>';
    const headerActions = el("div", "display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;");
    const close = makeButton("Close");
    header.append(title, row(headerActions, close));
    const progressStrip = el("div", "display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px;padding:8px 18px 0;");
    const tabBar = el("div", "display:grid;grid-template-columns:repeat(7,minmax(0,1fr));border-bottom:1px solid #1e3a5f;background:#0f172a;");
    const body = el("div", "display:grid;grid-template-columns:minmax(0,1fr) 320px;min-height:0;");
    const content = el("div", "padding:14px;overflow:auto;min-height:0;display:flex;flex-direction:column;gap:12px;");
    const side = el("div", "border-left:1px solid #1e3a5f;background:#0b1220;padding:12px;overflow:auto;min-height:0;");
    body.append(content, side);
    shell.append(header, progressStrip, tabBar, body);
    backdrop.append(shell);
    document.body.append(backdrop);

    const tabButtons = new Map();
    const progressBars = new Map();
    const staleBadges = new Map();
    // Scenes picked in the Beats & Prompts tab to redo; kept while the window is open.
    const picked = new Set();
    for (const tab of STORY_BUILDER_TABS) {
      const bar = el("div", "height:5px;border-radius:999px;background:#475569;");
      progressBars.set(tab.id, bar);
      progressStrip.append(bar);
      const button = el("button", "border:0;border-right:1px solid #1e3a5f;background:transparent;color:#94a3b8;padding:10px 8px;cursor:pointer;text-align:left;");
      button.type = "button";
      button.innerHTML = `<div style="font-size:13px;font-weight:900;">${escapeHtml(tab.label)}</div><div style="font-size:11px;opacity:.8;">${escapeHtml(tab.caption)}</div>`;
      const badge = el("div", `display:none;margin-top:4px;font-size:10px;font-weight:900;color:#1c1207;background:${STALE_COLOR};border-radius:999px;padding:1px 8px;width:max-content;`);
      button.append(badge);
      staleBadges.set(tab.id, badge);
      button.onclick = () => show(tab.id);
      tabButtons.set(tab.id, button);
      tabBar.append(button);
    }

    function progressNow() {
      return storyBuilderProgress({
        sceneIds: segments().map((segment) => segment.id),
        refs: normalizeFluxReferenceBuilder(state.fluxReferenceBuilder),
        layer: layer(),
        segments: segments(),
      });
    }

    function refreshProgress() {
      const progress = progressNow();
      for (const tab of STORY_BUILDER_TABS) {
        progressBars.get(tab.id).style.background = STATUS_COLORS[progress[tab.id]];
        const button = tabButtons.get(tab.id);
        const active = tab.id === activeTab;
        button.style.background = active ? "#10243a" : "transparent";
        button.style.color = active ? "#cffafe" : "#94a3b8";
        button.style.borderBottom = active ? "3px solid #06b6d4" : "3px solid transparent";
      }
      const stale = staleness();
      const counts = {
        story: stale.book.length ? 1 : 0,
        acts: Object.keys(stale.acts).length,
        scenes: Object.keys(stale.plan).length,
        places: Object.keys(stale.places).length,
        prompts: new Set([...Object.keys(stale.beats), ...Object.keys(stale.prompts)]).size,
      };
      for (const [tabId, badge] of staleBadges) {
        const count = counts[tabId] || 0;
        badge.style.display = count ? "block" : "none";
        badge.textContent = count ? (tabId === "story" ? "⚠ out of date" : `⚠ ${count} out of date`) : "";
      }
      return progress;
    }

    function blockerNote(tabId) {
      const message = storyBuilderBlocker(tabId, progressNow());
      if (!message) return null;
      return el("div", "font-size:12px;color:#fbbf24;border:1px solid #78350f;border-radius:7px;background:#1c1207;padding:9px;", message);
    }

    function renderSide() {
      side.textContent = "";
      const all = segments();
      const index = all.findIndex((segment) => segment.id === selectedSceneId);
      if (index < 0) {
        side.append(el("div", "font-size:12px;color:#94a3b8;", "Select a scene in the Song, Acts or Scenes tab to see its story here."));
        return;
      }
      const segment = all[index];
      const current = layer();
      const entry = normalizeStoryScenePlan(current.story_scene_plan).find((item) => item.scene_id === segment.id) || {};
      const act = normalizeStoryActs(current.story_acts).find((item) => item.id === entry.act_id) || {};
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const location = (refs.locations || []).find((item) => item.id === refs.scene_map?.[segment.id]);
      const lines = [
        ["Scene", `${index + 1} (${clock(segment.start)} - ${clock(segment.end)})`],
        ["Lyric (mood)", segment.lyric_text || "[instrumental]"],
        ["Act", act.title || "-"],
        ["Story", entry.story_slice || "-"],
        ["Spot", entry.spot || "-"],
        ["Location card", location?.name || "-"],
        ["Performance", entry.performance_mode || "-"],
        ["Mood", entry.mood || "-"],
        ["Director note", segment.timeline_note || "-"],
        ["Video note", segment.i2v_notes || "-"],
        ["Story beat", segment.story_beat || "-"],
      ];
      side.append(el("div", "font-size:14px;font-weight:900;color:#cffafe;margin-bottom:8px;", "Selected scene"));
      for (const [label, value] of lines) {
        const item = el("div", "margin-bottom:8px;");
        item.append(el("div", "font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.04em;", label));
        item.append(el("div", "font-size:12px;color:#e2e8f0;white-space:pre-wrap;line-height:1.4;", String(value)));
        side.append(item);
      }
    }

    function selectScene(id) {
      selectedSceneId = id;
      renderSide();
    }

    // Plays one scene's stretch of the project song, so notes can be written while listening. One player for the
    // window: starting a scene stops the one playing. Loop repeats the scene until stopped.
    const sceneAudio = { audio: null, playingId: "", loop: false, button: null, label: "" };
    function stopSceneAudio() {
      sceneAudio.audio?.pause();
      if (sceneAudio.button) sceneAudio.button.textContent = sceneAudio.label;
      sceneAudio.playingId = "";
      sceneAudio.button = null;
    }
    function sceneAudioButton(segment) {
      const label = "▶ Play Scene Audio";
      const button = makeButton(label);
      const songPath = String(state.audioPath || "").trim();
      button.disabled = !songPath;
      if (!songPath) button.title = "The project has no song loaded.";
      button.onclick = () => {
        const wasPlaying = sceneAudio.playingId === segment.id;
        stopSceneAudio();
        if (wasPlaying) return;
        if (!sceneAudio.audio) {
          sceneAudio.audio = document.createElement("audio");
          sceneAudio.audio.preload = "auto";
          sceneAudio.audio.src = audioUrl(songPath);
        }
        const audio = sceneAudio.audio;
        const start = Math.max(0, Number(segment.start) || 0);
        const end = Math.max(start + 0.1, Number(segment.end) || start + 0.1);
        Object.assign(sceneAudio, { playingId: segment.id, button, label });
        audio.ontimeupdate = () => {
          if (sceneAudio.playingId !== segment.id) return;
          button.textContent = `■ Stop ${clock(Math.max(0, audio.currentTime - start))} / ${clock(end - start)}`;
          if (audio.currentTime < end - 0.02) return;
          if (sceneAudio.loop) audio.currentTime = start;
          else stopSceneAudio();
        };
        audio.currentTime = start;
        Promise.resolve(audio.play()).catch((error) => {
          stopSceneAudio();
          toast(`Could not play the song: ${String(error?.message || error)}`, true);
        });
      };
      return button;
    }

    // The hosted tool (Reference Builder, or the Storyboard's Scene Defaults, which opens asynchronously) is saved
    // and closed before another tab is drawn.
    async function leaveHostedTab() {
      if (!hosted) return;
      const pending = hosted;
      hosted = null;
      const handle = await Promise.resolve(pending).catch(() => null);
      if (!handle) return;
      try {
        await handle.save();
      } catch (error) {
        console.warn("[VRGDG Story Builder] Saving the hosted tool failed", error);
      }
      handle.close();
    }

    function hostSceneDefaults(container) {
      hosted = Promise.resolve(openSceneDefaults({ host: container })).catch((error) => {
        container.append(el("div", "color:#fca5a5;font-size:12px;", `Scene Defaults failed to open: ${String(error?.message || error)}`));
        return null;
      });
    }

    function hostReferenceBuilder(container, focusedSection) {
      hosted = openFluxReferenceBuilderModal({ focusedSection, host: container, onSaved: refreshProgress }) || null;
    }

    // Opens the Cast or Places tab and its Reference Builder's Generate Missing Images window.
    async function generateMissingImages(kind) {
      const tab = kind === "locations" ? "places" : "cast";
      if (activeTab !== tab || !hosted) await show(tab);
      if (kind === "locations") hosted?.generateMissingLocationImages?.();
      else hosted?.generateMissingSubjectImages?.();
    }

    function imageButton(kind) {
      const button = makeButton(kind === "locations" ? "Generate Missing Location Images" : "Generate Missing Character Images", "primary");
      button.onclick = () => generateMissingImages(kind);
      return button;
    }

    // A note naming the cards without an image, with the button that makes them; null when none are missing. Location
    // images are optional, so that note is information only (blue); characters are amber when Reference to Video
    // sends their images.
    function imageNotice(kind) {
      const names = kind === "locations" ? imageGaps().locations : castWithoutImages(normalizeFluxReferenceBuilder(state.fluxReferenceBuilder));
      if (!names.length) return null;
      const what = kind === "locations" ? "location" : "character";
      const needed = kind !== "locations" && referenceImagesRequired();
      const why = needed
        ? "Reference to Video sends these images, so each one needs an image before prompts and renders."
        : "Images are optional. Make them if you want the renders to match a look.";
      const style = needed
        ? "color:#fbbf24;border:1px solid #78350f;background:#1c1207;"
        : "color:#bae6fd;border:1px solid #1e3a5f;background:#0b1a2b;";
      const note = el("div", `font-size:12px;${style}border-radius:7px;padding:9px;display:flex;flex-direction:column;gap:8px;`,
        `${names.length} ${what}${names.length === 1 ? "" : "s"} ${kind === "locations" ? "used by your scenes " : ""}${names.length === 1 ? "has" : "have"} no image yet: ${names.join(", ")}. ${why}`);
      note.append(row(imageButton(kind)));
      return note;
    }

    // ---------------------------------------------------------------- Song
    function renderSong() {
      const all = segments();
      const duration = all.reduce((end, segment) => Math.max(end, Number(segment.end) || 0), 0);
      const summary = card("Song and scenes", `${all.length} scenes, ${clock(duration)} long. The lyrics set timing, mood and emotion only; they never decide the story or a place.`);
      const lineMapping = makeButton("Line Mapping", "primary");
      lineMapping.onclick = () => openLyricMappingWorkflowModal();
      const review = makeButton("Review Lyrics");
      review.onclick = () => openLyricReviewModal();
      const spotMin = makeInput(String(normalizeSpotMinScenes(layer().story_spot_min_scenes ?? 3)), "number");
      spotMin.min = "1";
      spotMin.max = "20";
      spotMin.onchange = () => saveLayer({ story_spot_min_scenes: normalizeSpotMinScenes(spotMin.value) }, "story builder scenes per spot");
      summary.append(row(lineMapping, review), makeField("Scenes per spot (fewest scenes in a row at one place)", spotMin));

      const notes = card("Timed story events", "A timed note pins an event to the song (for example 42-58 s: she finds the letter). Acts never cut a timed note in two.");
      const list = el("div", "display:flex;flex-direction:column;gap:6px;");
      const markers = normalizeTimelineMarkers(state.timelineMarkers);
      for (const marker of markers) {
        const line = row(el("div", "font-size:12px;color:#e2e8f0;flex:1 1 auto;",
          `${clock(marker.start)}${marker.end != null ? ` - ${clock(marker.end)}` : ""}  ${marker.label}: ${marker.note}`));
        const remove = makeButton("Remove");
        remove.onclick = async () => {
          pushHistory();
          state.timelineMarkers = markers.filter((item) => item.id !== marker.id);
          await autoSaveSessionQuiet("story builder timed note removed");
          show("song");
        };
        line.append(remove);
        list.append(line);
      }
      if (!markers.length) list.append(el("div", "font-size:12px;color:#94a3b8;", "No timed notes yet."));
      const start = makeInput("", "number");
      const end = makeInput("", "number");
      const noteText = makeInput("");
      noteText.placeholder = "What happens, e.g. She finds the letter";
      const add = makeButton("Add Timed Note", "primary");
      add.onclick = async () => {
        if (!noteText.value.trim() || start.value === "") {
          toast("A timed note needs a start time and the event.", true);
          return;
        }
        pushHistory();
        state.timelineMarkers = normalizeTimelineMarkers([
          ...markers, { start: Number(start.value), end: end.value === "" ? null : Number(end.value), label: "Story event", note: noteText.value },
        ]);
        await autoSaveSessionQuiet("story builder timed note added");
        show("song");
      };
      notes.append(list, row(makeField("Start (s)", start), makeField("End (s, optional)", end), makeField("Event", noteText), add));

      const scenes = card("Scenes");
      const table = el("div", "display:flex;flex-direction:column;gap:2px;max-height:420px;overflow:auto;");
      all.forEach((segment, index) => {
        const line = el("button", "display:grid;grid-template-columns:48px 120px 110px minmax(0,1fr);gap:8px;text-align:left;border:0;border-radius:5px;background:#111827;color:#e2e8f0;padding:6px 8px;font-size:12px;cursor:pointer;");
        line.type = "button";
        line.append(el("span", "", String(index + 1)), el("span", "", `${clock(segment.start)}-${clock(segment.end)}`),
          el("span", "color:#94a3b8;", segment.lyric_section || ""), el("span", "", segment.lyric_text || "[instrumental]"));
        line.onclick = () => selectScene(segment.id);
        table.append(line);
      });
      scenes.append(table);
      const defaults = card("Scene Defaults", "The look and feel every scene starts from: speed and music style presets, camera flow, motion, cuts, performance and facial direction. Use Fill Missing or Replace All to update existing scenes. Saved when you leave this tab.");
      const defaultsHost = el("div", "flex:1 1 auto;min-height:680px;display:flex;");
      defaults.append(defaultsHost);
      content.append(summary, defaults, notes, scenes);
      hostSceneDefaults(defaultsHost);
    }

    // ---------------------------------------------------------------- Cast
    function renderCast() {
      const intro = card("Cast", "Add the main characters before writing the story so it can use their names. You can add more cast from the Story tab.");
      const missing = imageNotice("characters");
      if (missing) intro.append(missing);
      const host = el("div", "flex:1 1 auto;min-height:560px;display:flex;");
      content.append(intro, host);
      hostReferenceBuilder(host, "subjects");
    }

    // ---------------------------------------------------------------- Story
    function renderStory() {
      const current = layer();
      const blocked = blockerNote("story");
      if (blocked) content.append(blocked);
      // Who the story is about, their parts and how they are connected. No ids saved means every card. The story
      // uses exactly these people and no one else.
      const people = card("Characters in this story", "Tick who the story is about and give each one a part. The main character leads the story and is in most scenes. The story uses only these people.");
      const cards = (normalizeFluxReferenceBuilder(state.fluxReferenceBuilder).subjects || []).filter(isStoryPersonCard);
      const savedIds = new Set(current.story_cast_ids || []);
      const useAll = !cards.some((subject) => savedIds.has(subject.id));
      const savedRoles = current.story_cast_roles || {};
      const mainId = cards.some((subject) => subject.id === current.story_main_cast_id) ? current.story_main_cast_id : cards[0]?.id || "";
      const boxes = cards.map((subject) => {
        const line = el("div", "display:flex;gap:8px;font-size:12px;color:#e2e8f0;align-items:center;");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = useAll || savedIds.has(subject.id);
        const role = makeSelect(STORY_CAST_ROLES, subject.id === mainId ? "main" : (savedRoles[subject.id] && savedRoles[subject.id] !== "main" ? savedRoles[subject.id] : "supporting"));
        for (const option of role.options) option.textContent = CAST_ROLE_LABELS[option.value] || option.value;
        role.style.width = "max-content";
        line.append(box, el("span", "min-width:160px;", `${subject.name || subject.id} (${subject.reference_type || "character"})`), role);
        people.append(line);
        return { box, role, subject };
      });
      if (!cards.length) people.append(el("div", "font-size:12px;color:#fbbf24;", "No cast yet. Add your characters in the Cast tab first."));
      const saveCast = async (changedRole = null) => {
        // One main character: picking a new one makes the old one supporting.
        if (changedRole?.value === "main") {
          for (const item of boxes) if (item.role !== changedRole && item.role.value === "main") item.role.value = "supporting";
        }
        const picked = boxes.filter((item) => item.box.checked).map((item) => item.subject.id);
        if (cards.length && !picked.length) {
          toast("Pick at least one character for the story.", true);
          return;
        }
        const main = boxes.find((item) => item.role.value === "main" && item.box.checked) || boxes.find((item) => item.box.checked);
        await saveLayer({
          story_cast_ids: picked.length === cards.length ? [] : picked,
          story_allow_new_cast: false,
          story_main_cast_id: main?.subject.id || "",
          story_cast_roles: Object.fromEntries(boxes.map((item) => [item.subject.id, item === main ? "main" : (item.role.value === "main" ? "supporting" : item.role.value)])),
        }, "story builder story cast");
        if (main && main.role.value !== "main") main.role.value = "main";
        drawBookNote();
        refreshProgress();
      };
      for (const { box, role } of boxes) {
        box.onchange = () => saveCast();
        role.onchange = () => saveCast(role);
      }
      people.append(relationshipEditor(cards, current.story_relationships || []));

      // The story's own out-of-date note, redrawn in place so typing elsewhere on the tab is not lost.
      const bookNote = el("div");
      const drawBookNote = () => {
        bookNote.textContent = "";
        const reasons = staleness().book;
        if (!reasons.length) return;
        const keep = makeButton("Keep The Story As Is");
        keep.title = "The story stays as written. Acts, scenes, places, beats and prompts stop showing it as out of date.";
        keep.onclick = async () => {
          await saveLayer({}, "story builder story kept", { book: true });
          drawBookNote();
          refreshProgress();
        };
        bookNote.append(staleBox(`The story is out of date: ${reasonText(reasons)} since it was written. Draft Story again to rewrite it, or keep it as is. Until then the acts, scenes, places, beats and prompts made from it show as out of date too.`, keep));
      };
      drawBookNote();
      content.append(bookNote);

      const idea = card("Premise", "A sentence or a full outline. The LLM writes the book from it, using only the characters ticked above.");
      const premise = textarea(current.overall_story_idea, "A son rides the night train home to find out why his father left.", 70);
      const savePremise = async (announce) => {
        if (premise.value === String(layer().overall_story_idea || "")) {
          if (announce) toast("Premise saved.");
          return;
        }
        await saveLayer({ overall_story_idea: premise.value }, "story builder premise");
        drawBookNote();
        refreshProgress();
        if (announce) toast("Premise saved.");
      };
      premise.onchange = () => savePremise(false);
      const savePremiseButton = makeButton("Save Premise");
      savePremiseButton.onclick = () => savePremise(true);
      const tone = makeInput("");
      tone.placeholder = "Tone, genre, world (optional), e.g. 1970s rural Ohio, bittersweet";
      const draft = makeButton("Draft Story", "primary");
      draft.disabled = Boolean(blocked);
      draft.onclick = async () => {
        await saveLayer({ overall_story_idea: premise.value }, "story builder premise");
        const data = await runStep("book", { story_idea: premise.value, tone_notes: tone.value,
          previous_book: current.story_book_text || "", seed: String(Date.now()) }, "Writing the story");
        if (data?.story_layer) {
          await saveLayer(data.story_layer, "story builder story drafted", { book: true });
          toast(removedNote(data, "Story written. Review it, then create the acts."));
          show("story");
        }
      };
      const telling = makeSelect(STORY_VIDEO_STYLES, normalizeStoryVideoStyle(current.story_video_style));
      for (const option of telling.options) {
        option.textContent = {
          music_video: "Music video: the cast performs the song inside the story",
          spoken: "Spoken: the characters talk, no one sings",
          story_only: "Story only: no singing or speaking on camera",
        }[option.value] || option.value;
      }
      telling.onchange = async () => {
        await saveLayer({ story_video_style: telling.value }, "story builder video style");
        drawBookNote();
        refreshProgress();
      };
      idea.append(premise, tone, makeField("How the story is told (the story follows the song's flow either way)", telling), row(savePremiseButton, draft));

      const own = card("Use your own story", "Paste a finished story. Every word is kept; it is only divided into chapters with a setting each.");
      const pasted = textarea("", "Paste your story here. Chapter headings like 'Chapter 1: The Station' are kept.", 120);
      const split = makeButton("Split Into Chapters", "primary");
      split.disabled = Boolean(blocked);
      split.onclick = async () => {
        if (!pasted.value.trim()) {
          toast("Paste the story first.", true);
          return;
        }
        const data = await runStep("split", { text: pasted.value }, "Splitting the story into chapters");
        if (data?.story_layer) {
          await saveLayer(data.story_layer, "story builder story split", { book: true });
          show("story");
        }
      };
      own.append(pasted, row(split));
      if (current.user_story_arc && !normalizeStoryBook(current.story_book).chapters.length) {
        const convert = makeButton("Convert My Old Story Arc", "primary");
        convert.title = "Turn this project's song-section story arc into chapters. The words are kept; the old arc stays saved.";
        convert.disabled = Boolean(blocked);
        convert.onclick = async () => {
          const data = await runStep("convert", {}, "Converting the story arc");
          if (data?.story_layer) {
            await saveLayer(data.story_layer, "story builder story arc converted", { book: true });
            show("story");
          }
        };
        own.append(el("div", "font-size:12px;color:#a5f3fc;", "This project already has a story arc from before Story Mode."), row(convert));
      }

      const book = card("The story", "Edit freely. Keep 'Chapter N: title' and 'Setting: name' lines; every chapter happens in one general setting.");
      const editor = textarea(current.story_book_text, "Your story appears here.", 320);
      const saveText = makeButton("Save Story Text", "primary");
      saveText.onclick = async () => {
        await saveLayer(applyStoryBookText(current, editor.value), "story builder story edited", { book: true });
        toast("Story saved.");
        show("story");
      };
      book.append(editor, row(saveText));

      const chapters = card("Chapters");
      const settings = Object.fromEntries(normalizeStorySettings(current.story_settings).map((item) => [item.id, item.name]));
      for (const [index, chapter] of normalizeStoryBook(current.story_book).chapters.entries()) {
        const direction = makeInput("");
        direction.placeholder = "What to change, e.g. make it raining, add the conductor";
        const rewrite = makeButton("Rewrite Chapter");
        rewrite.onclick = async () => {
          const data = await runStep("chapter", { chapter_id: chapter.id, direction: direction.value }, `Rewriting chapter ${index + 1}`);
          if (data?.story_layer) {
            await saveLayer(data.story_layer, "story builder chapter rewritten");
            toast(removedNote(data, "Chapter rewritten."));
            show("story");
          }
        };
        chapters.append(el("div", "font-size:12px;color:#e2e8f0;", `Chapter ${index + 1}: ${chapter.title} - ${settings[chapter.setting_id] || "no setting"} - ${chapter.paragraphs.length} paragraphs`),
          row(direction, rewrite));
      }

      content.append(people, idea, own, book, chapters);
    }

    // ---------------------------------------------------------------- Acts
    function renderActs() {
      const current = layer();
      const blocked = blockerNote("acts");
      if (blocked) content.append(blocked);
      const all = segments();
      const sceneIds = all.map((segment) => segment.id);
      const numberById = Object.fromEntries(sceneIds.map((id, index) => [id, index + 1]));
      const chapters = normalizeStoryBook(current.story_book).chapters;
      let acts = normalizeStoryActs(current.story_acts);
      const top = card("Acts", "Each chapter gets a run of scenes and is split into acts: one event each. Timed notes stay inside one act.");
      const create = makeButton(acts.length ? "Rebuild All Acts" : "Create Acts", "primary");
      create.disabled = Boolean(blocked);
      create.onclick = async () => {
        const data = await runStep("acts", {}, "Creating the acts");
        if (data?.story_layer) {
          await saveLayer(data.story_layer, "story builder acts", { acts: true });
          show("acts");
        }
      };
      top.append(row(create));

      const band = el("div", "display:flex;height:26px;border-radius:6px;overflow:hidden;border:1px solid #1e3a5f;");
      const totalLength = all.reduce((sum, segment) => sum + Math.max(0.1, (segment.end || 0) - (segment.start || 0)), 0) || 1;
      for (const act of acts) {
        const length = act.scene_ids.reduce((sum, id) => {
          const segment = all[numberById[id] - 1];
          return sum + (segment ? Math.max(0.1, (segment.end || 0) - (segment.start || 0)) : 0);
        }, 0);
        const chapterIndex = Math.max(0, chapters.findIndex((chapter) => chapter.id === act.chapter_id));
        const piece = el("div", `flex:${length / totalLength} 1 0;background:${STORY_CHAPTER_COLORS[chapterIndex % STORY_CHAPTER_COLORS.length]};border-right:1px solid #0b1220;font-size:10px;color:#fff;padding:5px;overflow:hidden;white-space:nowrap;`, act.title);
        piece.title = `${act.title}: scenes ${act.scene_ids.map((id) => numberById[id]).join(", ")}`;
        band.append(piece);
      }
      top.append(band);
      const problems = validateStoryActs(acts, chapters, sceneIds);
      if (acts.length && problems.length) top.append(el("div", "font-size:12px;color:#fbbf24;white-space:pre-wrap;", problems.join("\n")));
      content.append(top);

      // Acts made before their chapter, the story or the cast changed.
      const stale = staleness().acts;
      const chapterActIds = (chapterId) => (storyLayer) => normalizeStoryActs(storyLayer.story_acts)
        .filter((act) => act.chapter_id === chapterId).map((act) => act.id);
      const staleChapters = chapters.filter((chapter) => acts.some((act) => act.chapter_id === chapter.id && stale[act.id]));
      if (staleChapters.length) {
        const rebuildStale = makeButton(`Rebuild Out-Of-Date Chapters (${staleChapters.length})`, "primary");
        rebuildStale.title = "Rebuild the acts of only the chapters that changed. The other chapters keep their acts.";
        rebuildStale.disabled = Boolean(blocked);
        rebuildStale.onclick = async () => {
          for (const chapter of staleChapters) {
            const data = await runStep("acts", { chapter_id: chapter.id }, `Rebuilding acts of ${chapter.title}`);
            if (!data?.story_layer) break;
            await saveLayer(data.story_layer, "story builder chapter acts", { acts: chapterActIds(chapter.id) });
          }
          show("acts");
        };
        const keepAll = makeButton("Keep All As Is");
        keepAll.title = "Keep these acts. Scenes made from them stop showing the change.";
        keepAll.onclick = async () => {
          await saveLayer({}, "story builder acts kept", { acts: Object.keys(stale) });
          show("acts");
        };
        const count = Object.keys(stale).length;
        top.append(staleBox(`${count} act${count === 1 ? " is" : "s are"} out of date in ${staleChapters.map((chapter) => chapter.title || chapter.id).join(", ")}. Rebuild only those chapters, or keep the acts as they are.`, rebuildStale, keepAll));
      }

      const saveActs = async (next, label) => {
        acts = next;
        const plan = normalizeStoryScenePlan(current.story_scene_plan);
        const actOf = Object.fromEntries(next.flatMap((act) => act.scene_ids.map((id) => [id, act.id])));
        await saveLayer({ story_acts: next, story_scene_plan: plan.map((entry) => ({ ...entry, act_id: actOf[entry.scene_id] || "" })) }, label);
        show("acts");
      };
      for (const chapter of chapters) {
        const chapterCard = card(chapter.title);
        const rebuild = makeButton("Rebuild This Chapter");
        rebuild.onclick = async () => {
          const data = await runStep("acts", { chapter_id: chapter.id }, `Rebuilding acts of ${chapter.title}`);
          if (data?.story_layer) {
            await saveLayer(data.story_layer, "story builder chapter acts", { acts: chapterActIds(chapter.id) });
            show("acts");
          }
        };
        chapterCard.append(row(rebuild));
        const staleHere = acts.filter((act) => act.chapter_id === chapter.id && stale[act.id]);
        if (staleHere.length) {
          chapterCard.style.border = `2px solid ${STALE_COLOR}`;
          const keep = makeButton("Keep As Is");
          keep.onclick = async () => {
            await saveLayer({}, "story builder acts kept", { acts: staleHere.map((act) => act.id) });
            show("acts");
          };
          chapterCard.append(staleBox(`This chapter's acts are out of date: ${reasonText(staleHere.flatMap((act) => stale[act.id]))}. Rebuild This Chapter to redo only its acts.`, keep));
        }
        acts.forEach((act, index) => {
          if (act.chapter_id !== chapter.id) return;
          const numbers = act.scene_ids.map((id) => numberById[id]).filter(Boolean);
          const box = el("div", `border:1px solid ${stale[act.id] ? STALE_COLOR : "#1e3a5f"};border-radius:7px;padding:8px;display:flex;flex-direction:column;gap:6px;`);
          if (stale[act.id]) box.append(staleChip("ACT", stale[act.id]));
          const actTitle = makeInput(act.title);
          const summary = textarea(act.summary, "What happens in this act.", 50);
          const range = el("div", "font-size:12px;color:#94a3b8;", numbers.length ? `Scenes ${numbers[0]}-${numbers[numbers.length - 1]}` : "No scenes");
          const sceneChoice = makeSelect(act.scene_ids.slice(1).map((id) => String(id)), "");
          for (const option of sceneChoice.options) option.textContent = `Scene ${numberById[option.value]}`;
          const startChoices = index > 0 && acts[index - 1].chapter_id === act.chapter_id
            ? [...acts[index - 1].scene_ids.slice(1), ...act.scene_ids.slice(1)] : [];
          const startChoice = makeSelect(startChoices, act.scene_ids[0] || "");
          for (const option of startChoice.options) option.textContent = `Start at scene ${numberById[option.value]}`;
          const saveText = makeButton("Save");
          saveText.onclick = () => saveActs(acts.map((item, i) => (i === index ? { ...item, title: actTitle.value, summary: summary.value } : item)), "story builder act edited");
          const merge = makeButton("Merge With Previous");
          merge.disabled = !(index > 0 && acts[index - 1].chapter_id === act.chapter_id);
          merge.onclick = () => saveActs(mergeActs(acts, index), "story builder acts merged");
          const split = makeButton("Split At");
          split.disabled = act.scene_ids.length < 2;
          split.onclick = () => saveActs(splitAct(acts, index, sceneChoice.value), "story builder act split");
          const move = makeButton("Move Start");
          move.disabled = !startChoices.length;
          move.onclick = () => saveActs(moveActBorder(acts, index, startChoice.value, sceneIds), "story builder act border moved");
          const pick = makeButton("Show First Scene");
          pick.onclick = () => selectScene(act.scene_ids[0]);
          box.append(row(range, pick), actTitle, summary, row(saveText, merge, sceneChoice, split, startChoice, move));
          chapterCard.append(box);
        });
        content.append(chapterCard);
      }
    }

    // ---------------------------------------------------------------- Scenes
    function renderScenes() {
      const current = layer();
      const blocked = blockerNote("scenes");
      if (blocked) content.append(blocked);
      const all = segments();
      const acts = normalizeStoryActs(current.story_acts);
      const actOf = Object.fromEntries(acts.flatMap((act) => act.scene_ids.map((id) => [id, act])));
      const plan = Object.fromEntries(normalizeStoryScenePlan(current.story_scene_plan).map((entry) => [entry.scene_id, entry]));
      const castNames = Object.fromEntries(storyCastCards().map((subject) => [subject.id, subject.name || subject.id]));
      // Clothing, prop and other thing cards, which scenes can use beside their characters.
      const sceneRefs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const itemCards = (sceneRefs.subjects || []).filter((subject) => subject && !subject.extra_reference_for && !isStoryPersonCard(subject));
      const sceneMapIds = (value) => (Array.isArray(value) ? value.map(String) : String(value ?? "").split(",").map((id) => id.trim()).filter(Boolean));
      let reviseFromScene = null;
      const top = card("Scene plan", "Every scene gets its part of the act, a spot inside the chapter's setting, a mood and a performance. Your notes win over the story.");
      const planAll = makeButton(Object.keys(plan).length ? "Replan All Scenes" : "Plan All Scenes", "primary");
      planAll.disabled = Boolean(blocked);
      planAll.onclick = async () => {
        await planActs(acts, acts, "Planning the scenes");
        show("scenes");
      };
      const resetPlan = makeButton("Reset Scene Plan");
      resetPlan.title = "Clear every scene's plan and the masked continuations it set, so you can plan from scratch.";
      resetPlan.disabled = !Object.keys(plan).length;
      resetPlan.onclick = async () => {
        if (window.confirm?.("Clear the scene plan of every scene? Masked continuations it set are turned off. Your scene notes are kept.") === false) return;
        pushHistory();
        if (isMiniMax()) applyStoryContinuationsToSegments(segments(), [], state.miniMaxH3Settings);
        await saveLayer({ story_scene_plan: [] }, "story builder scene plan reset");
        toast("Scene plan cleared. Press Plan All Scenes to plan again.");
        show("scenes");
      };
      // Save sits at the top with the plan buttons, so it is reachable without scrolling past every scene.
      const topActions = row(planAll, resetPlan);
      const loopLabel = el("label", "display:flex;gap:6px;font-size:12px;color:#e2e8f0;align-items:center;cursor:pointer;");
      const loop = document.createElement("input");
      loop.type = "checkbox";
      loop.checked = sceneAudio.loop;
      loop.onchange = () => { sceneAudio.loop = loop.checked; };
      loopLabel.append(loop, document.createTextNode("Loop scene audio while writing notes"));
      top.append(topActions, loopLabel);
      content.append(top);

      // Acts whose scenes were planned before the act, its chapter, the story or the cast changed.
      const stale = staleness();
      const replanAct = async (act) => (await planActs(acts, [act], `Planning ${act.title || act.id}`)) === 1;
      const staleActs = acts.filter((act) => act.scene_ids.some((id) => stale.plan[id]));
      if (staleActs.length) {
        const replanStale = makeButton(`Replan Out-Of-Date Acts (${staleActs.length})`, "primary");
        replanStale.title = "Replan the scenes of only the acts that changed. Every other scene keeps its plan.";
        replanStale.disabled = Boolean(blocked);
        replanStale.onclick = async () => {
          await planActs(acts, staleActs, "Replanning out-of-date acts");
          show("scenes");
        };
        const keepAll = makeButton("Keep All As Is");
        keepAll.title = "Keep these scene plans. Places, beats and prompts made from them stop showing the change.";
        keepAll.onclick = async () => {
          await saveLayer({}, "story builder scene plan kept", { plan: Object.keys(stale.plan) });
          show("scenes");
        };
        const actsFirst = staleActs.some((act) => stale.acts[act.id]);
        top.append(staleBox(`${staleActs.length} act${staleActs.length === 1 ? " has" : "s have"} out-of-date scenes: ${staleActs.map((act) => act.title || act.id).join(", ")}. They are outlined in amber below. Replan only those acts, or keep them as they are.${actsFirst ? " Some of their acts are out of date themselves: rebuild those in the Acts tab first." : ""}`, replanStale, keepAll));
      }

      const mode = normalizeStoryContinuationMode(current.story_continuation_mode);
      const fromById = continuedFrom(current.story_scene_plan);
      const continuedCount = Object.keys(fromById).length;
      const masked = card("Continue scenes in one take", "A continued scene is a MiniMax masked continuation: it starts from the last moments of the scene before, so the two play as one unbroken take. At render its prompt is written from the previous scene's last frame, then follows the direction you give it. Render continued scenes in order, since each needs the scene before it.");
      const modeSelect = makeSelect(Object.keys(CONTINUATION_MODE_LABELS), mode);
      for (const option of modeSelect.options) option.textContent = CONTINUATION_MODE_LABELS[option.value];
      modeSelect.onchange = async () => {
        pushHistory();
        const nextMode = normalizeStoryContinuationMode(modeSelect.value);
        const planned = applyContinuations(current.story_scene_plan, acts, nextMode);
        await saveLayer({ story_continuation_mode: nextMode, story_scene_plan: planned }, "story builder continuation mode");
        show("scenes");
      };
      masked.append(makeField("Which scenes continue", modeSelect),
        el("div", `font-size:12px;font-weight:800;color:${continuedCount ? MASKED_COLOR : "#94a3b8"};`,
          `${continuedCount} scene${continuedCount === 1 ? "" : "s"} continue${continuedCount === 1 ? "s" : ""} from the scene before. Tick or untick any scene below; your choice wins over the mode.`));
      if (!isMiniMax()) {
        masked.append(el("div", "font-size:12px;color:#fbbf24;", "Masked continuation renders with MiniMax H3. Your choices are kept, and they take effect once the project uses MiniMax H3."));
      }
      if (normalizeStoryScenePlan(current.story_scene_plan).some((entry) => entry.continuation_source === "manual")) {
        const clearManual = makeButton("Clear My Ticks");
        clearManual.title = "Forget the scenes you ticked or unticked by hand and use the mode again.";
        clearManual.onclick = async () => {
          pushHistory();
          const reset = normalizeStoryScenePlan(current.story_scene_plan).map((entry) => (
            entry.continuation_source === "manual" ? { ...entry, continuation_source: "", continues_previous: false } : entry));
          const planned = applyContinuations(reset, acts);
          await saveLayer({ story_scene_plan: planned }, "story builder continuation ticks cleared");
          show("scenes");
        };
        masked.append(row(clearManual));
      }
      content.append(masked);
      const rows = [];
      const chapters = normalizeStoryBook(current.story_book).chapters;
      const numberById = Object.fromEntries(all.map((segment, index) => [segment.id, index + 1]));
      // Each act is one section: a framed block in its chapter's colour holding the scenes that tell it.
      const makeSection = (act) => {
        const chapterIndex = act ? chapters.findIndex((chapter) => chapter.id === act.chapter_id) : -1;
        const color = chapterIndex >= 0 ? STORY_CHAPTER_COLORS[chapterIndex % STORY_CHAPTER_COLORS.length] : "#475569";
        const staleIds = act ? act.scene_ids.filter((id) => stale.plan[id]) : [];
        const section = el("div", `border:2px solid ${staleIds.length ? STALE_COLOR : color};border-left:10px solid ${color};border-radius:10px;background:#0b1424;padding:10px;display:flex;flex-direction:column;gap:8px;margin-top:10px;`);
        const numbers = act ? act.scene_ids.map((id) => numberById[id]).filter(Boolean) : [];
        const heading = el("div", "display:flex;flex-direction:column;gap:2px;flex:1 1 auto;");
        heading.append(
          el("div", `font-size:11px;font-weight:900;letter-spacing:.06em;text-transform:uppercase;color:${color};`,
            act ? `Chapter ${chapterIndex + 1}${chapters[chapterIndex]?.title ? `: ${chapters[chapterIndex].title}` : ""}` : "Not in an act"),
          el("div", "font-size:16px;font-weight:900;color:#f8fafc;", act ? act.title || act.id : "Scenes outside every act"),
          el("div", "font-size:12px;color:#94a3b8;", numbers.length ? `${sceneRangeText(numbers)} · ${numbers.length} scene${numbers.length === 1 ? "" : "s"}` : ""),
        );
        const header = el("div", "display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding-bottom:8px;border-bottom:1px solid #1e3a5f;");
        header.append(heading);
        if (act) {
          const replan = makeButton("Replan This Act");
          replan.title = "Tell the LLM where this happens and what must happen, then replan all of this act's scenes or only the ones you pick.";
          replan.onclick = () => openReplanDialog(act, () => show("scenes"));
          header.append(replan);
        }
        section.append(header);
        if (act?.summary) section.append(el("div", "font-size:12px;color:#cbd5e1;line-height:1.45;", act.summary));
        if (staleIds.length) {
          const keep = makeButton("Keep As Is");
          keep.onclick = async () => {
            await saveLayer({}, "story builder scene plan kept", { plan: staleIds });
            show("scenes");
          };
          const actStale = stale.acts[act.id];
          section.append(staleBox(`This act's scenes are out of date: ${reasonText(staleIds.flatMap((id) => stale.plan[id]))}. ${actStale ? "Rebuild this act in the Acts tab first, then replan it here." : "Replan This Act to redo only these scenes."}`, keep));
        }
        content.append(section);
        return section;
      };
      let section = null;
      let lastAct = null;
      all.forEach((segment, index) => {
        const act = actOf[segment.id];
        if (!section || (act?.id || "") !== lastAct) {
          lastAct = act?.id || "";
          section = makeSection(act);
        }
        const entry = plan[segment.id] || { scene_id: segment.id, act_id: act?.id || "", story_slice: "", spot: "", mood: "", performance_mode: "singing", note_sources: [] };
        const box = el("div", "display:grid;grid-template-columns:150px minmax(0,1.4fr) minmax(0,1fr);gap:8px;border:1px solid #1e3a5f;border-radius:7px;padding:8px;");
        const info = el("div", "font-size:12px;color:#94a3b8;display:flex;flex-direction:column;gap:4px;");
        const select = makeButton(`Scene ${index + 1}`);
        select.onclick = () => selectScene(segment.id);
        info.append(select, sceneAudioButton(segment), el("div", "", `${clock(segment.start)}-${clock(segment.end)}`),
          el("div", "color:#cbd5e1;", segment.lyric_text || "[instrumental]"));
        const slice = textarea(entry.story_slice, "What happens in this scene.", 50);
        const spot = makeInput(entry.spot);
        spot.placeholder = "Spot, e.g. Kitchen";
        const mood = makeInput(entry.mood);
        mood.placeholder = "Mood";
        const performance = makeSelect(PERFORMANCE_OPTIONS.map(([value]) => value), entry.performance_mode);
        for (const option of performance.options) option.textContent = PERFORMANCE_OPTIONS.find(([value]) => value === option.value)?.[1] || option.value;
        const story = el("div", "display:flex;flex-direction:column;gap:6px;");
        // Who is in the scene: click a name to add or remove them. A cast you change here wins over the story.
        const picked = new Set(entry.cast_ids || []);
        const castRow = el("div", "display:flex;flex-wrap:wrap;gap:4px;align-items:center;");
        castRow.append(el("span", "font-size:11px;color:#94a3b8;", entry.cast_source === "manual" ? "Characters (yours):" : "Characters:"));
        const drawChip = (chip, id) => {
          const on = picked.has(id);
          chip.style.cssText = `border:1px solid ${on ? "#22d3ee" : "#334155"};border-radius:999px;padding:2px 9px;font-size:11px;font-weight:800;cursor:pointer;background:${on ? "rgba(8,145,178,.35)" : "transparent"};color:${on ? "#cffafe" : "#94a3b8"};`;
          chip.textContent = `${on ? "✓ " : ""}${castNames[id]}${id === entry.performer_id ? " ♪" : ""}`;
        };
        // Clothing and props in the scene: click one to add or remove it, and pick who wears or uses it here.
        const mapped = new Set(sceneMapIds(sceneRefs.subject_scene_map?.[segment.id]));
        const itemsOn = new Set(itemCards.filter((item) => mapped.has(item.id)).map((item) => item.id));
        const itemUsers = { ...(segment.scene_item_users || {}) };
        // Who can wear or use them: the scene's picked characters, else the people already mapped to it.
        const scenePeople = () => {
          const chosen = Object.keys(castNames).filter((id) => picked.has(id));
          return chosen.length ? chosen : [...mapped].filter((id) => castNames[id]);
        };
        const itemsRow = el("div", "display:flex;flex-wrap:wrap;gap:6px;align-items:center;");
        const drawItems = () => {
          itemsRow.textContent = "";
          if (!itemCards.length) return;
          itemsRow.append(el("span", "font-size:11px;color:#94a3b8;", "Clothing & props:"));
          const people = scenePeople();
          for (const item of itemCards) {
            const on = itemsOn.has(item.id);
            const group = el("span", `display:inline-flex;gap:4px;align-items:center;border:1px solid ${on ? "#f0abfc" : "#334155"};border-radius:999px;padding:1px 4px 1px 2px;background:${on ? "rgba(162,28,175,.25)" : "transparent"};`);
            const chip = el("button", `border:0;background:transparent;padding:1px 6px;font-size:11px;font-weight:800;cursor:pointer;color:${on ? "#fae8ff" : "#94a3b8"};`,
              `${on ? "✓ " : ""}${item.name || "Item"}`);
            chip.type = "button";
            chip.title = on ? "In this scene. Click to remove it." : "Click to add it to this scene.";
            chip.onclick = () => {
              if (on) itemsOn.delete(item.id);
              else itemsOn.add(item.id);
              drawItems();
            };
            group.append(chip);
            if (on && people.length) {
              const chosen = people.includes(itemUsers[item.id]) ? itemUsers[item.id] : people.includes(item.wears) ? item.wears : people[0];
              const user = makeSelect(people, chosen);
              for (const option of user.options) option.textContent = `${item.reference_type === "outfit" ? "worn by" : "used by"} ${castNames[option.value]}`;
              user.style.fontSize = "11px";
              user.style.maxWidth = "170px";
              itemUsers[item.id] = user.value;
              user.onchange = () => { itemUsers[item.id] = user.value; };
              group.append(user);
            }
            itemsRow.append(group);
          }
        };
        for (const id of Object.keys(castNames)) {
          const chip = el("button");
          chip.type = "button";
          chip.title = id === entry.performer_id ? "Sings or speaks in this scene." : "Click to add or remove this character.";
          chip.onclick = () => {
            if (picked.has(id)) picked.delete(id);
            else picked.add(id);
            drawChip(chip, id);
            drawItems();
          };
          drawChip(chip, id);
          castRow.append(chip);
        }
        drawItems();
        const interaction = textarea(entry.interaction, "What the characters do with each other, e.g. Mara reaches for Joe's hand; he pulls away.", 34);
        const userNote = textarea(entry.user_note, "Your direction for this scene: what should happen, what a character does. The story follows it.", 34);
        story.append(slice, row(spot, mood, performance), castRow, itemsRow, interaction, userNote);
        if (entry.act_id) {
          const update = makeButton("Update Story From This Scene");
          update.title = "Keeps this scene's characters and your direction, rewrites the scene to match, and adapts its act and chapter. Scenes that may no longer fit are named so you can replan them.";
          update.disabled = Boolean(blocked);
          update.onclick = () => reviseFromScene?.(segment, index + 1);
          story.append(row(update));
        }
        const director = textarea(segment.timeline_note, "Director note: how you want this scene made", 40);
        const video = textarea(segment.i2v_notes, "Video note: motion and action", 40);
        const planning = textarea(segment.notes, "Planning note", 34);
        const notes = el("div", "display:flex;flex-direction:column;gap:6px;");
        notes.append(director, video, planning);
        box.append(info, story, notes);
        let continues = null;
        let direction = null;
        let holdInput = null;
        if (index > 0) {
          const cont = el("div", "grid-column:1/-1;display:flex;flex-direction:column;gap:6px;border-top:1px dashed #334155;padding-top:6px;");
          const toggle = el("label", "display:flex;gap:6px;font-size:12px;color:#e2e8f0;align-items:center;cursor:pointer;");
          continues = document.createElement("input");
          continues.type = "checkbox";
          continues.checked = Boolean(entry.continues_previous);
          toggle.append(continues, document.createTextNode(`Continue from Scene ${index} in one take (masked continuation)`));
          direction = textarea(entry.continuation_direction, "During the continuation: what happens once the carried-on moment settles, e.g. She sets the letter down and walks to the window.", 36);
          holdInput = makeInput(String(segment.minimax_h3_continuation_start_seconds ?? ""), "number");
          holdInput.step = "0.1";
          holdInput.min = "0.5";
          holdInput.placeholder = "0.5";
          holdInput.style.maxWidth = "110px";
          const details = el("div", "display:flex;flex-direction:column;gap:6px;");
          details.append(direction, makeField("Direction starts at (seconds into the scene)", holdInput));
          const chipSlot = el("div");
          const mark = () => {
            const on = continues.checked;
            box.style.borderColor = on ? MASKED_COLOR : "#1e3a5f";
            box.style.borderLeft = on ? `5px solid ${MASKED_COLOR}` : "1px solid #1e3a5f";
            box.style.boxShadow = on ? "0 0 0 1px rgba(163,230,53,.25)" : "none";
            details.style.display = on ? "flex" : "none";
            const manualNow = on !== Boolean(entry.continues_previous);
            chipSlot.textContent = "";
            if (on) chipSlot.append(maskedChip(index, index + 1, manualNow ? "manual" : entry.continuation_source));
          };
          continues.onchange = mark;
          cont.append(chipSlot, toggle, details);
          box.append(cont);
          mark();
        }
        section.append(box);
        rows.push({ segment, entry, slice, spot, mood, performance, director, video, planning, continues, direction, holdInput,
          picked, interaction, userNote, itemsOn, itemUsers, scenePeople });
      });
      // Saves every scene's edits (plan, cast, direction, notes) and returns the saved plan.
      const savePlanEdits = async (label) => {
        const nextPlan = rows.filter((item) => item.entry.act_id).map((item) => {
          const on = item.continues ? item.continues.checked : false;
          // A tick that differs from what the mode decided is your choice, and it wins over the mode from now on.
          const manual = item.continues && on !== Boolean(item.entry.continues_previous);
          const castIds = Object.keys(castNames).filter((id) => item.picked.has(id));
          const castChanged = JSON.stringify(castIds) !== JSON.stringify((item.entry.cast_ids || []).filter((id) => castNames[id]));
          return {
            ...item.entry,
            cast_ids: castIds,
            cast_source: castChanged ? "manual" : item.entry.cast_source,
            performer_id: castIds.includes(item.entry.performer_id) ? item.entry.performer_id : "",
            interaction: item.interaction.value.trim(),
            user_note: item.userNote.value.trim(),
            story_slice: item.slice.value,
            spot: item.spot.value,
            mood: item.mood.value,
            performance_mode: item.performance.value,
            continues_previous: on,
            continuation_source: manual ? "manual" : item.entry.continuation_source,
            continuation_direction: item.direction ? item.direction.value.trim() : item.entry.continuation_direction,
          };
        });
        // Clothing and props: the scene's mapping keeps its people and gets the chosen items after them, and the
        // scene remembers who wears or uses each one. Only scenes whose items changed are touched.
        const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
        const itemIds = new Set(itemCards.map((item) => item.id));
        let itemsChanged = false;
        for (const item of rows) {
          const id = item.segment.id;
          const before = sceneMapIds(refs.subject_scene_map?.[id]);
          const after = [...before.filter((cardId) => !itemIds.has(cardId)), ...itemCards.map((card) => card.id).filter((cardId) => item.itemsOn.has(cardId))];
          if (JSON.stringify(before) !== JSON.stringify(after)) {
            refs.subject_scene_map[id] = after;
            itemsChanged = true;
          }
          const users = Object.fromEntries(Object.entries(item.itemUsers).filter(([cardId, user]) => item.itemsOn.has(cardId) && item.scenePeople().includes(user)));
          if (Object.keys(users).length) item.segment.scene_item_users = users;
          else delete item.segment.scene_item_users;
        }
        if (itemsChanged) state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
        for (const item of rows) {
          item.segment.timeline_note = item.director.value;
          item.segment.i2v_notes = item.video.value;
          item.segment.notes = item.planning.value;
          if (item.holdInput) {
            const hold = String(item.holdInput.value).trim();
            if (hold) item.segment.minimax_h3_continuation_start_seconds = Number(hold);
            else delete item.segment.minimax_h3_continuation_start_seconds;
          }
        }
        const planned = applyPlanToProject(nextPlan, acts);
        await saveLayer({ story_scene_plan: planned }, label);
        return planned;
      };
      const save = makeButton("Save Scene Plan And Notes", "primary");
      save.onclick = async () => {
        pushHistory();
        await savePlanEdits("story builder scene plan edited");
        toast("Scene plan and notes saved.");
        show("scenes");
      };
      topActions.append(save);
      // Update Story From This Scene: the scene's characters and direction are kept, and its act and chapter adapt
      // to them. Only the other scenes the LLM names are marked out of date. One Undo restores everything.
      reviseFromScene = async (segment, number) => {
        pushHistory();
        await savePlanEdits("story builder scene change saved");
        const data = await runStep("revise_scene", { scene_id: segment.id }, `Updating the story from Scene ${number}`);
        if (!data?.story_layer) {
          show("scenes");
          return;
        }
        const others = new Set((data.other_scenes || []).map((item) => item.scene_id));
        const chapterActs = normalizeStoryActs(data.story_layer.story_acts).filter((act) => act.chapter_id === data.chapter_id);
        const keptScenes = chapterActs.flatMap((act) => act.scene_ids).filter((id) => !others.has(id));
        const planned = applyPlanToProject(data.story_layer.story_scene_plan, normalizeStoryActs(data.story_layer.story_acts));
        await saveLayer({ ...data.story_layer, story_scene_plan: planned }, "story builder story updated from a scene",
          { acts: chapterActs.map((act) => act.id), plan: keptScenes });
        const named = (data.other_scenes || []).map((item) => `Scene ${sceneNumbers([item.scene_id])[0]}${item.reason ? ` (${item.reason})` : ""}`);
        toast(named.length
          ? `Story updated from Scene ${number}. These scenes may no longer fit: ${named.join("; ")}. Replan them when you are ready.`
          : `Story updated from Scene ${number}. The rest of the act still fits.`);
        show("scenes");
      };
    }

    // ---------------------------------------------------------------- Places
    function renderPlaces() {
      const current = layer();
      const blocked = blockerNote("places");
      if (blocked) content.append(blocked);
      const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
      const top = card("Places", "Each chapter's setting gets one look, and every spot inside it a location card built on that look. Only the story is used, never the lyrics.");
      const theme = makeInput(refs.location_style_theme || "");
      theme.placeholder = "Look and banned places, e.g. warm 1970s film, no warehouses";
      const regenerate = el("label", "display:flex;gap:6px;font-size:12px;color:#e2e8f0;align-items:center;");
      const regenerateBox = document.createElement("input");
      regenerateBox.type = "checkbox";
      regenerate.append(regenerateBox, document.createTextNode("Rewrite setting looks and card text"));
      const build = makeButton("Build Locations From Story", "primary");
      build.disabled = Boolean(blocked);
      build.onclick = async () => {
        // Save and close the hosted Reference Builder first. It holds its own copy of the cards, and saving that
        // copy after the build would overwrite the new locations.
        await leaveHostedTab();
        const data = await runStep("locations", { style_theme: theme.value, regenerate_looks: regenerateBox.checked }, "Building the locations");
        if (!data?.locations) {
          show("places");
          return;
        }
        pushHistory();
        const merged = applyStoryLocations(normalizeFluxReferenceBuilder(state.fluxReferenceBuilder), data.locations, (taken) => {
          let id = "";
          do id = `loc_${Date.now()}_${Math.floor(Math.random() * 10000)}`; while (taken.has(id));
          taken.add(id);
          return id;
        }, regenerateBox.checked);
        merged.refs.location_style_theme = theme.value;
        state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(merged.refs);
        await saveLayer(data.story_layer || {}, "story builder locations", { places: true });
        toast(`Locations ready: ${merged.added} added, ${merged.updated} updated. Every scene is mapped to its spot.`);
        // Location images are made from the Places tab's button when you want them, never opened for you.
        show("places");
      };
      top.append(makeField("Style and theme", theme), row(regenerate, build));
      const placeReasons = staleness().places;
      const stalePlaces = Object.keys(placeReasons);
      if (stalePlaces.length) {
        const keep = makeButton("Keep Locations As Is");
        keep.onclick = async () => {
          await saveLayer({}, "story builder locations kept", { places: stalePlaces });
          show("places");
        };
        const reasons = stalePlaces.flatMap((id) => placeReasons[id]);
        top.append(staleBox(`${sceneRangeText(sceneNumbers(stalePlaces))} changed since the locations were built: ${reasonText(reasons)}. Build Locations From Story again (existing cards are updated, not duplicated), or keep the locations as they are.`, keep));
      }
      const missingPlaces = imageNotice("locations");
      if (missingPlaces) top.append(missingPlaces);
      const looks = card("Setting looks", "Edit a look to change how every spot of that setting is described next time you build.");
      const lookInputs = normalizeStorySettings(current.story_settings).map((setting) => {
        const input = textarea(setting.look, "The shared look of this setting.", 50);
        looks.append(el("div", "font-size:12px;font-weight:800;color:#e2e8f0;", setting.name), input);
        return { setting, input };
      });
      const saveLooks = makeButton("Save Looks");
      saveLooks.onclick = async () => {
        await saveLayer({ story_settings: lookInputs.map(({ setting, input }) => ({ ...setting, look: input.value })) }, "story builder setting looks");
        toast("Setting looks saved.");
      };
      looks.append(row(saveLooks));
      const host = el("div", "flex:1 1 auto;min-height:560px;display:flex;");
      content.append(top, looks, host);
      hostReferenceBuilder(host, "locations");
    }

    // ---------------------------------------------------------------- Beats & Prompts
    function renderPrompts() {
      const blocked = blockerNote("prompts");
      if (blocked) content.append(blocked);
      const all = segments();
      const promptKey = normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3" ? "minimax_h3_prompt" : "i2v_prompt";
      const withBeat = all.filter((segment) => String(segment.story_beat || "").trim()).length;
      const withPrompt = all.filter((segment) => String(segment[promptKey] || "").trim()).length;
      // Runs a beat or prompt task, then marks the scenes whose beat or prompt it rewrote as up to date.
      const runAndRefresh = async (task, kind = "beats") => {
        const key = kind === "beats" ? "story_beat" : promptKey;
        const before = Object.fromEntries(segments().map((segment) => [segment.id, String(segment[key] || "")]));
        await task();
        const redone = segments().filter((segment) => String(segment[key] || "").trim() && String(segment[key] || "") !== before[segment.id])
          .map((segment) => segment.id);
        if (redone.length) await saveLayer({}, `story builder ${kind} redone`, { [kind]: redone });
        show("prompts");
      };

      const beats = card("Scene beats", `${withBeat} of ${all.length} scenes have a beat. Each beat follows its scene's part of the story and its spot; the lyric only sets the mood.`);
      const missingBeats = makeButton("Create Missing Beats", "primary");
      missingBeats.disabled = Boolean(blocked);
      missingBeats.onclick = () => runAndRefresh(() => createSceneBeats({ overwrite: false }));
      const allBeats = makeButton("Recreate All Beats");
      allBeats.disabled = Boolean(blocked);
      allBeats.onclick = () => {
        if (window.confirm?.("Replace the beat of every scene?") === false) return;
        runAndRefresh(() => createSceneBeats({ overwrite: true }));
      };
      beats.append(row(missingBeats, allBeats));

      const prompts = card("Video prompts", `${withPrompt} of ${all.length} scenes have a video prompt. Creating prompts writes one for every scene from its beat, replacing what is there. Create the beats first.`);
      // Reference to Video cannot write or render a scene that has no image to send.
      const gaps = imageGaps();
      const required = referenceImagesRequired();
      const imagesBlock = required && gaps.scenes.length > 0;
      if (gaps.characters.length || gaps.locations.length || (required && gaps.scenes.length)) {
        const message = imagesBlock
          ? `${sceneRangeText(gaps.scenes)} ${gaps.scenes.length === 1 ? "has" : "have"} no character or location image, and Reference to Video needs at least one. ${gaps.characters.length || gaps.locations.length ? "Make the missing images first." : "Map a character or location to them in the Cast or Places tab."}`
          : `For your information: some cards used by your scenes have no image yet${gaps.characters.length ? `. Characters: ${gaps.characters.join(", ")}` : ""}${gaps.locations.length ? `. Locations: ${gaps.locations.join(", ")}` : ""}. Prompts can still be created.`;
        const note = el("div", `font-size:12px;color:${imagesBlock ? "#fca5a5" : "#bae6fd"};border:1px solid ${imagesBlock ? "#7f1d1d" : "#1e3a5f"};border-radius:7px;background:${imagesBlock ? "#1f0808" : "#0b1a2b"};padding:9px;display:flex;flex-direction:column;gap:8px;`, message);
        note.append(row(gaps.characters.length ? imageButton("characters") : null, gaps.locations.length ? imageButton("locations") : null));
        prompts.append(note);
      }
      const createPrompts = makeButton("Create Video Prompts", "primary");
      createPrompts.disabled = Boolean(blocked) || !withBeat || imagesBlock;
      if (imagesBlock) createPrompts.title = "Make the missing images first.";
      createPrompts.onclick = () => {
        if (withPrompt && window.confirm?.(`Replace the video prompts of all ${all.length} scenes?`) === false) return;
        runAndRefresh(() => createVideoPrompts({ storyMode: true }), "prompts");
      };
      const openBoard = makeButton("Open Storyboard");
      openBoard.title = "The full Storyboard, for per-scene settings and single-scene prompts.";
      openBoard.disabled = Boolean(blocked);
      openBoard.onclick = () => openStoryboardClassic();
      prompts.append(row(createPrompts, openBoard));

      const current = layer();
      const fromById = isMiniMax() ? continuedFrom(current.story_scene_plan) : {};
      const continuedCount = Object.keys(fromById).length;
      if (continuedCount) {
        prompts.append(el("div", `font-size:12px;color:#ecfccb;border:1px solid ${MASKED_COLOR};border-radius:7px;background:rgba(54,83,20,.35);padding:9px;`,
          `${continuedCount} scene${continuedCount === 1 ? " is a masked continuation" : "s are masked continuations"} (marked below). At render, each one's prompt is rewritten from the previous scene's last frame and follows its direction, so the prompt written here is the starting idea. Render them in order.`));
      }
      // Redo only the out-of-date scenes plus any the user ticks, so the rest keep their beats and prompts.
      const stale = staleness();
      for (const id of [...picked]) if (!all.some((segment) => segment.id === id)) picked.delete(id);
      const pickCard = card("Redo ticked scenes", "Tick scenes in the list below, then redo their beats or prompts. Each scene also has its own Redo Beat and Redo Prompt. Every other scene keeps its beat and prompt.");
      const hasBeat = (id) => String(all.find((segment) => segment.id === id)?.story_beat || "").trim();
      const beatIds = () => all.map((segment) => segment.id).filter((id) => stale.beats[id] || picked.has(id));
      const promptIds = () => all.map((segment) => segment.id).filter((id) => (stale.prompts[id] || picked.has(id)) && hasBeat(id));
      const pickBoxes = [];
      const pickedText = el("div", "font-size:12px;color:#94a3b8;");
      const runPicked = (task, kind) => {
        picked.clear();
        return runAndRefresh(task, kind);
      };
      const redoBeats = makeButton("Redo Beats", "primary");
      redoBeats.onclick = () => {
        const ids = beatIds();
        runPicked(() => createSceneBeats({ overwrite: true, sceneIds: ids }), "beats");
      };
      const redoPrompts = makeButton("Redo Prompts", "primary");
      redoPrompts.onclick = () => {
        const ids = promptIds();
        const noImage = required ? sceneNumbers(ids).filter((number) => gaps.scenes.includes(number)) : [];
        if (noImage.length) {
          toast(`${sceneRangeText(noImage)} ${noImage.length === 1 ? "has" : "have"} no character or location image, and Reference to Video needs one. Make the missing images first.`, true);
          return;
        }
        runPicked(() => createVideoPrompts({ sceneIds: ids, storyMode: true }), "prompts");
      };
      const clearPicks = makeButton("Clear Ticks");
      clearPicks.onclick = () => {
        picked.clear();
        for (const { box } of pickBoxes) box.checked = false;
        drawPicked();
      };
      const scenesText = (ids) => (ids.length ? sceneRangeText(sceneNumbers(ids)) : "none");
      const drawPicked = () => {
        const beatsToRun = beatIds();
        const promptsToRun = promptIds();
        redoBeats.textContent = `Redo Beats (${beatsToRun.length})`;
        redoPrompts.textContent = `Redo Prompts (${promptsToRun.length})`;
        redoBeats.disabled = Boolean(blocked) || !beatsToRun.length;
        redoPrompts.disabled = Boolean(blocked) || !promptsToRun.length;
        clearPicks.style.display = picked.size ? "" : "none";
        pickedText.textContent = `Beats: ${scenesText(beatsToRun)}. Prompts: ${scenesText(promptsToRun)}.`;
      };
      pickCard.append(row(redoBeats, redoPrompts, clearPicks), pickedText);

      const list = card("Scenes", "Edit any beat, prompt or continuation direction, then save.");
      // Save sits at the top of the list, above every scene.
      const listActions = row();
      list.append(listActions);
      const plan = Object.fromEntries(normalizeStoryScenePlan(current.story_scene_plan).map((entry) => [entry.scene_id, entry]));
      const actTitles = Object.fromEntries(normalizeStoryActs(current.story_acts).map((act) => [act.id, act.title]));
      const editors = all.map((segment, index) => {
        const entry = plan[segment.id] || {};
        const item = el("div", "border:1px solid #1e293b;border-radius:7px;padding:8px;display:flex;flex-direction:column;gap:6px;");
        const head = el("button", "border:0;background:transparent;color:#cffafe;text-align:left;padding:0;font-size:12px;font-weight:800;cursor:pointer;",
          `Scene ${index + 1} (${clock(segment.start)} - ${clock(segment.end)})  ${[actTitles[entry.act_id], entry.spot, entry.performance_mode].filter(Boolean).join(" · ")}`);
        head.type = "button";
        head.onclick = () => selectScene(segment.id);
        const beat = textarea(segment.story_beat, "No beat yet.", 50);
        const prompt = textarea(segment[promptKey], "No video prompt yet.", 70);
        const from = fromById[segment.id];
        const beatStale = stale.beats[segment.id];
        const promptStale = stale.prompts[segment.id];
        if (beatStale || promptStale) {
          item.style.border = `1px solid ${STALE_COLOR}`;
          item.style.borderLeft = `5px solid ${STALE_COLOR}`;
        }
        if (from) {
          item.style.border = `1px solid ${MASKED_COLOR}`;
          item.style.borderLeft = `5px solid ${MASKED_COLOR}`;
          item.append(maskedChip(from, index + 1, entry.continuation_source));
        }
        if (beatStale) item.append(staleChip("BEAT", beatStale));
        if (promptStale) item.append(staleChip("PROMPT", promptStale));
        const pickLabel = el("label", "display:flex;gap:6px;font-size:12px;font-weight:800;color:#e2e8f0;align-items:center;cursor:pointer;");
        const pickBox = document.createElement("input");
        pickBox.type = "checkbox";
        pickBox.checked = picked.has(segment.id);
        pickBox.onchange = () => {
          if (pickBox.checked) picked.add(segment.id);
          else picked.delete(segment.id);
          drawPicked();
        };
        pickLabel.append(pickBox, document.createTextNode("Redo"));
        pickLabel.title = "Add this scene to Redo Beats and Redo Prompts.";
        pickBoxes.push({ id: segment.id, box: pickBox });
        // This scene alone: every other scene keeps its beat and prompt.
        const oneBeat = makeButton("Redo Beat");
        oneBeat.title = "Write a new beat for this scene only, from its plan, characters, clothing and props.";
        oneBeat.disabled = Boolean(blocked);
        oneBeat.onclick = () => runAndRefresh(() => createSceneBeats({ overwrite: true, sceneIds: [segment.id] }), "beats");
        const onePrompt = makeButton("Redo Prompt");
        onePrompt.title = "Write a new video prompt for this scene only, from the beat shown here.";
        onePrompt.disabled = Boolean(blocked) || (required && gaps.scenes.includes(index + 1));
        onePrompt.onclick = () => {
          const edited = beat.value.trim();
          if (!edited) {
            toast(`Scene ${index + 1} has no beat yet. Redo its beat first.`, true);
            return;
          }
          // A beat edited here and not saved yet is what the prompt is written from.
          segment.story_beat = edited;
          runAndRefresh(() => createVideoPrompts({ sceneIds: [segment.id], storyMode: true }), "prompts");
        };
        item.append(row(pickLabel, head, oneBeat, onePrompt), el("div", "font-size:11px;color:#94a3b8;", `Lyric (mood): ${segment.lyric_text || "[instrumental]"}`),
          el("div", "font-size:11px;color:#94a3b8;", "Beat"), beat, el("div", "font-size:11px;color:#94a3b8;", "Video prompt"), prompt);
        let direction = null;
        if (from) {
          direction = textarea(entry.continuation_direction, "What happens once the carried-on moment settles.", 36);
          item.append(el("div", `font-size:11px;color:${MASKED_COLOR};`, `Continuation direction (rewritten into the prompt from Scene ${from}'s last frame at render)`), direction);
        }
        list.append(item);
        return { segment, beat, prompt, direction };
      });
      const save = makeButton("Save Beats And Prompts", "primary");
      save.onclick = async () => {
        pushHistory();
        const directions = Object.fromEntries(editors.filter((item) => item.direction).map((item) => [item.segment.id, item.direction.value.trim()]));
        if (Object.keys(directions).length) {
          const nextPlan = normalizeStoryScenePlan(layer().story_scene_plan).map((entry) => (
            entry.scene_id in directions ? { ...entry, continuation_direction: directions[entry.scene_id] } : entry));
          await saveLayer({ story_scene_plan: applyContinuations(nextPlan, normalizeStoryActs(layer().story_acts)) }, "story builder continuation directions");
        }
        for (const { segment, beat, prompt } of editors) {
          segment.story_beat = beat.value.trim();
          const text = prompt.value.trim();
          if (text !== String(segment[promptKey] || "").trim()) {
            segment[promptKey] = text;
            segment[`${promptKey}_origin`] = "manual";
          }
        }
        render();
        // Autosave copies the inspector's prompt box into the selected scene; show the edited prompt there first.
        syncInspector();
        await autoSaveSessionQuiet("story builder beats and prompts edited");
        toast("Beats and prompts saved.");
        show("prompts");
      };
      listActions.append(save);
      drawPicked();
      content.append(beats, prompts, pickCard, list);
    }

    const renderers = {
      song: renderSong, cast: renderCast, story: renderStory, acts: renderActs,
      scenes: renderScenes, places: renderPlaces, prompts: renderPrompts,
    };

    async function show(tabId) {
      stopSceneAudio();
      await leaveHostedTab();
      activeTab = renderers[tabId] ? tabId : "song";
      content.textContent = "";
      fillStamps();
      refreshProgress();
      try {
        renderers[activeTab]();
        const instructions = instructionButtons(tabInstructions(activeTab));
        if (instructions) {
          const box = card("Story Mode instructions", "The LLM instructions this tab uses. They belong to Story Mode only, so changing them never changes how the other modes work.");
          box.append(instructions);
          content.append(box);
        }
      } catch (error) {
        console.error("[VRGDG Story Builder] Tab failed to render", error);
        content.append(el("div", "color:#fca5a5;font-size:12px;", `This tab failed to open: ${String(error?.message || error)}`));
      }
      renderSide();
    }

    async function closeStoryBuilder() {
      stopSceneAudio();
      await leaveHostedTab();
      backdrop.remove();
      open = null;
      render();
      options.onClose?.();
    }

    close.onclick = closeStoryBuilder;
    open = { show, get tab() { return activeTab; } };
    show(activeTab);
  }

  return { openStoryBuilder };
}
