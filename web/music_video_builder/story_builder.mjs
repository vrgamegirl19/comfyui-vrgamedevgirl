// Story Builder: the Story Mode workspace. One window with tabs in the order a story music video is made
// (Song, Cast, Story, Acts, Scenes, Places, Beats & Prompts). It reads and writes the same session state as the
// Video Builder and calls /vrgdg/storyboard/story_mode/{action} for the LLM steps.
import { postJson } from "./comfy_api.mjs";
import { escapeHtml, makeButton, makeField, makeInput, makeSelect, normalizeProjectVideoEngine, toast } from "./controls.mjs";
import { normalizeFluxReferenceBuilder } from "./reference_data.mjs";
import { mergeActs, moveActBorder, splitAct, validateStoryActs } from "./story_acts_data.mjs";
import {
  STORY_BUILDER_TABS, castWithoutImages, referenceImageGaps, sceneRangeText, storyBuilderBlocker, storyBuilderProgress,
  storySubjectCard,
} from "./story_builder_data.mjs";
import { normalizeMiniMaxH3Mode } from "./minimax_h3.mjs";
import { applyStoryContinuationsToSegments } from "./story_continuation.mjs";
import { applyStoryLocations } from "./story_locations_data.mjs";
import {
  applyStoryBookText, fillSceneCast, normalizeSpotMinScenes, normalizeStoryActs, normalizeStoryBook,
  normalizeStoryContinuationMode, normalizeStoryScenePlan, normalizeStorySettings, normalizeStoryVideoStyle,
  planSegmentFields, resolveStoryContinuations, STORY_CHAPTER_COLORS, STORY_FLOW_VERSION, STORY_VIDEO_STYLES,
} from "./story_mode_data.mjs";
import { normalizeTimelineMarkers } from "./timeline_state.mjs";

// Below the Reference Builder (100006) and Line Mapping windows, so the tools it opens appear on top.
const STORY_BUILDER_Z = 100004;
const STATUS_COLORS = { empty: "#475569", progress: "#f59e0b", done: "#22c55e" };
const PERFORMANCE_OPTIONS = [
  ["singing", "Singing"], ["speaking", "Speaking"], ["no_lip_sync", "Story only"], ["instrumental", "Instrumental"],
];
const CONTINUATION_MODE_LABELS = {
  off: "Off (only the scenes I tick)",
  location: "Same spot: touching scenes at the same spot in a chapter",
  llm: "Let the LLM decide from the story",
};
const CONTINUATION_SOURCE_LABELS = { auto: "same spot", llm: "LLM", manual: "you" };
const MASKED_COLOR = "#a3e635";

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

export function createStoryBuilder({
  autoSaveSessionQuiet, createProgressWindow, createSceneBeats, createVideoPrompts, getLlmPayload, normalizeBuilderStoryLayer,
  openFluxReferenceBuilderModal, openLyricMappingWorkflowModal, openLyricReviewModal, openSceneDefaults, openStoryboardClassic,
  pushHistory, referenceBuilderPayload, render, state, storyboardScenePayload,
}) {
  let open = null;

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

  async function saveLayer(fields, label) {
    state.builderStoryLayer = normalizeBuilderStoryLayer({
      ...state.builderStoryLayer, story_flow_version: STORY_FLOW_VERSION, ...fields,
    });
    render();
    await autoSaveSessionQuiet(label);
  }

  function isMiniMax() {
    return normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3";
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
    const subjectIds = (refs.subjects || []).filter((subject) => !subject.extra_reference_for).map((subject) => subject.id);
    const { map, changed } = fillSceneCast(refs.subject_scene_map, acts, subjectIds);
    if (changed.length) {
      refs.subject_scene_map = map;
      refs.use_subject_reference = true;
      state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
    }
    return plan;
  }

  async function runStep(action, extra, title) {
    const progress = createProgressWindow(title, { zIndex: STORY_BUILDER_Z + 10 });
    progress.set(`${title}...\nThis runs on your selected LLM runner.`, 15);
    try {
      const baseIds = new Set(segments().map((segment) => segment.id));
      const scenes = storyboardScenePayload().filter((scene) => baseIds.has(scene.id));
      const data = await postJson(`/vrgdg/storyboard/story_mode/${action}`, {
        ...getLlmPayload(),
        story_layer: layer(),
        scenes,
        reference_builder: referenceBuilderPayload(),
        timeline_markers: normalizeTimelineMarkers(state.timelineMarkers),
        lyrics: scenes.map((scene) => scene.lyrics || "").filter(Boolean).join("\n"),
        unload_after: true,
        ...extra,
      }, 20 * 60 * 1000);
      progress.set(`${title}: done.`, 100);
      progress.close(900);
      return data;
    } catch (error) {
      progress.set(`Error:\n${String(error?.message || error)}`, 100);
      toast(String(error?.message || error), true);
      return null;
    }
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
    for (const tab of STORY_BUILDER_TABS) {
      const bar = el("div", "height:5px;border-radius:999px;background:#475569;");
      progressBars.set(tab.id, bar);
      progressStrip.append(bar);
      const button = el("button", "border:0;border-right:1px solid #1e3a5f;background:transparent;color:#94a3b8;padding:10px 8px;cursor:pointer;text-align:left;");
      button.type = "button";
      button.innerHTML = `<div style="font-size:13px;font-weight:900;">${escapeHtml(tab.label)}</div><div style="font-size:11px;opacity:.8;">${escapeHtml(tab.caption)}</div>`;
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

    // An amber note naming the cards without an image, with the button that makes them; null when none are missing.
    function imageNotice(kind) {
      const names = kind === "locations" ? imageGaps().locations : castWithoutImages(normalizeFluxReferenceBuilder(state.fluxReferenceBuilder));
      if (!names.length) return null;
      const what = kind === "locations" ? "location" : "character";
      const why = referenceImagesRequired()
        ? "Reference to Video sends these images, so each one needs an image before prompts and renders."
        : "Make them now so every card is ready.";
      const note = el("div", "font-size:12px;color:#fbbf24;border:1px solid #78350f;border-radius:7px;background:#1c1207;padding:9px;display:flex;flex-direction:column;gap:8px;",
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
      // Who the story is about. No ids saved means every card; the LLM gets only the picked cards.
      const people = card("Characters in this story", "Tick who the story is about. The story uses exactly these people and no one else unless you allow new people below.");
      const cards = (normalizeFluxReferenceBuilder(state.fluxReferenceBuilder).subjects || []).filter((subject) => subject && !subject.extra_reference_for);
      const savedIds = new Set(current.story_cast_ids || []);
      const useAll = !cards.some((subject) => savedIds.has(subject.id));
      const boxes = cards.map((subject) => {
        const label = el("label", "display:flex;gap:8px;font-size:12px;color:#e2e8f0;align-items:center;");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = useAll || savedIds.has(subject.id);
        label.append(box, el("span", "", `${subject.name || subject.id} (${subject.reference_type || "character"})`));
        people.append(label);
        return { box, subject };
      });
      if (!cards.length) people.append(el("div", "font-size:12px;color:#fbbf24;", "No cast yet. Add your characters in the Cast tab first."));
      const allowLabel = el("label", "display:flex;gap:8px;font-size:12px;color:#e2e8f0;align-items:center;");
      const allowBox = document.createElement("input");
      allowBox.type = "checkbox";
      allowBox.checked = Boolean(current.story_allow_new_cast);
      allowLabel.append(allowBox, el("span", "", "Let the story add new people (you add their cards with Find Cast In Story)"));
      people.append(allowLabel);
      const saveCast = async () => {
        const picked = boxes.filter((item) => item.box.checked).map((item) => item.subject.id);
        if (cards.length && !picked.length) {
          toast("Pick at least one character for the story.", true);
          return;
        }
        await saveLayer({
          story_cast_ids: picked.length === cards.length ? [] : picked,
          story_allow_new_cast: allowBox.checked,
        }, "story builder story cast");
      };
      for (const { box } of boxes) box.onchange = saveCast;
      allowBox.onchange = saveCast;

      const idea = card("Premise", "A sentence or a full outline. The LLM writes the book from it, using only the characters ticked above.");
      const premise = textarea(current.overall_story_idea, "A son rides the night train home to find out why his father left.", 70);
      const tone = makeInput("");
      tone.placeholder = "Tone, genre, world (optional), e.g. 1970s rural Ohio, bittersweet";
      const draft = makeButton("Draft Story", "primary");
      draft.disabled = Boolean(blocked);
      draft.onclick = async () => {
        await saveLayer({ overall_story_idea: premise.value }, "story builder premise");
        const data = await runStep("book", { story_idea: premise.value, tone_notes: tone.value,
          previous_book: current.story_book_text || "", seed: String(Date.now()) }, "Writing the story");
        if (data?.story_layer) {
          await saveLayer(data.story_layer, "story builder story drafted");
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
      telling.onchange = () => saveLayer({ story_video_style: telling.value }, "story builder video style");
      idea.append(premise, tone, makeField("How the story is told (the story follows the song's flow either way)", telling), row(draft));

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
          await saveLayer(data.story_layer, "story builder story split");
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
            await saveLayer(data.story_layer, "story builder story arc converted");
            show("story");
          }
        };
        own.append(el("div", "font-size:12px;color:#a5f3fc;", "This project already has a story arc from before Story Mode."), row(convert));
      }

      const book = card("The story", "Edit freely. Keep 'Chapter N: title' and 'Setting: name' lines; every chapter happens in one general setting.");
      const editor = textarea(current.story_book_text, "Your story appears here.", 320);
      const saveText = makeButton("Save Story Text", "primary");
      saveText.onclick = async () => {
        await saveLayer(applyStoryBookText(current, editor.value), "story builder story edited");
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

      const keepToCast = !current.story_allow_new_cast;
      const cast = card("Cast in the story", keepToCast
        ? "Check the story for anyone outside the characters ticked above, and write them out."
        : "Find people in the story who have no character card, and add the ones you want.");
      const find = makeButton(keepToCast ? "Check Story For Other People" : "Find Cast In Story", "primary");
      const found = el("div", "display:flex;flex-direction:column;gap:6px;");
      find.onclick = async () => {
        const data = await runStep("cast", {}, "Finding story characters");
        found.textContent = "";
        const characters = data?.characters || [];
        if (!characters.length) {
          found.append(el("div", "font-size:12px;color:#94a3b8;", keepToCast
            ? "Only your story cast is in the story."
            : "Everyone in the story already has a card."));
          return;
        }
        if (keepToCast) {
          found.append(el("div", "font-size:12px;color:#fbbf24;", "These people are in the story but not in your story cast. Rewrite the story without them, or add them to the cast."));
        }
        const boxes = characters.map((character) => {
          const label = el("label", "display:flex;gap:8px;font-size:12px;color:#e2e8f0;align-items:flex-start;");
          const box = document.createElement("input");
          box.type = "checkbox";
          box.checked = keepToCast || character.role !== "background";
          label.append(box, el("span", "", `${character.name} (${character.role}): ${character.description}`));
          found.append(label);
          return { box, character };
        });
        const addCast = makeButton(keepToCast ? "Add Selected To Cast Instead" : "Add Selected To Cast", keepToCast ? "" : "primary");
        addCast.onclick = async () => {
          const refs = normalizeFluxReferenceBuilder(state.fluxReferenceBuilder);
          const chosen = boxes.filter((item) => item.box.checked).map((item) => item.character);
          if (refs.subjects.length + chosen.length > 12) {
            toast("The Reference Builder holds up to 12 characters. Pick fewer.", true);
            return;
          }
          pushHistory();
          const addedIds = [];
          for (const character of chosen) {
            const subject = storySubjectCard(character, `subj_${Date.now()}_${Math.floor(Math.random() * 10000)}`);
            refs.subjects.push(subject);
            addedIds.push(subject.id);
          }
          refs.subject_count = refs.subjects.length;
          state.fluxReferenceBuilder = normalizeFluxReferenceBuilder(refs);
          // A picked story cast gets the new people too.
          if (current.story_cast_ids?.length) {
            state.builderStoryLayer = normalizeBuilderStoryLayer({
              ...state.builderStoryLayer, story_cast_ids: [...current.story_cast_ids, ...addedIds],
            });
          }
          await autoSaveSessionQuiet("story builder cast added");
          toast(`Added ${chosen.length} character${chosen.length === 1 ? "" : "s"}. Make their images now, or give them a RefMod.`);
          // New cast has no images yet: go straight to making them.
          await generateMissingImages("characters");
        };
        const removeThem = makeButton("Rewrite Without Them", "primary");
        removeThem.title = "The LLM rewrites the story so only your story cast is in it. Acts and the scene plan are cleared.";
        removeThem.onclick = async () => {
          const names = boxes.filter((item) => item.box.checked).map((item) => item.character.name);
          if (!names.length) {
            toast("Tick the people to write out.", true);
            return;
          }
          const data = await runStep("remove_people", { names }, "Rewriting the story without them");
          if (data?.story_layer) {
            await saveLayer(data.story_layer, "story builder people removed");
            toast(`Removed ${names.join(", ")}. Create the acts again.`);
            show("story");
          }
        };
        found.append(keepToCast ? row(removeThem, addCast) : row(addCast));
      };
      cast.append(row(find), found);
      content.append(people, idea, own, book, chapters, cast);
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
          await saveLayer(data.story_layer, "story builder acts");
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
            await saveLayer(data.story_layer, "story builder chapter acts");
            show("acts");
          }
        };
        chapterCard.append(row(rebuild));
        acts.forEach((act, index) => {
          if (act.chapter_id !== chapter.id) return;
          const numbers = act.scene_ids.map((id) => numberById[id]).filter(Boolean);
          const box = el("div", "border:1px solid #1e3a5f;border-radius:7px;padding:8px;display:flex;flex-direction:column;gap:6px;");
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
      const top = card("Scene plan", "Every scene gets its part of the act, a spot inside the chapter's setting, a mood and a performance. Your notes win over the story.");
      const planAll = makeButton(Object.keys(plan).length ? "Replan All Scenes" : "Plan All Scenes", "primary");
      planAll.disabled = Boolean(blocked);
      planAll.onclick = async () => {
        const data = await runStep("plan", { spot_min_scenes: current.story_spot_min_scenes }, "Planning the scenes");
        if (data?.story_layer) {
          pushHistory();
          const planned = applyPlanToProject(data.story_layer.story_scene_plan, acts);
          await saveLayer({ ...data.story_layer, story_scene_plan: planned }, "story builder scene plan");
          show("scenes");
        }
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
      top.append(row(planAll, resetPlan));
      content.append(top);

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
      let lastAct = "";
      all.forEach((segment, index) => {
        const act = actOf[segment.id];
        if (act && act.id !== lastAct) {
          lastAct = act.id;
          const replan = makeButton("Replan This Act");
          replan.onclick = async () => {
            const data = await runStep("plan", { act_id: act.id }, `Planning ${act.title}`);
            if (data?.story_layer) {
              pushHistory();
              const planned = applyPlanToProject(data.story_layer.story_scene_plan, acts);
              await saveLayer({ ...data.story_layer, story_scene_plan: planned }, "story builder act replanned");
              show("scenes");
            }
          };
          content.append(row(el("div", "font-size:13px;font-weight:900;color:#a5f3fc;", `${act.title}`), replan));
        }
        const entry = plan[segment.id] || { scene_id: segment.id, act_id: act?.id || "", story_slice: "", spot: "", mood: "", performance_mode: "singing", note_sources: [] };
        const box = el("div", "display:grid;grid-template-columns:150px minmax(0,1.4fr) minmax(0,1fr);gap:8px;border:1px solid #1e3a5f;border-radius:7px;padding:8px;");
        const info = el("div", "font-size:12px;color:#94a3b8;display:flex;flex-direction:column;gap:4px;");
        const select = makeButton(`Scene ${index + 1}`);
        select.onclick = () => selectScene(segment.id);
        info.append(select, el("div", "", `${clock(segment.start)}-${clock(segment.end)}`), el("div", "color:#cbd5e1;", segment.lyric_text || "[instrumental]"));
        const slice = textarea(entry.story_slice, "What happens in this scene.", 50);
        const spot = makeInput(entry.spot);
        spot.placeholder = "Spot, e.g. Kitchen";
        const mood = makeInput(entry.mood);
        mood.placeholder = "Mood";
        const performance = makeSelect(PERFORMANCE_OPTIONS.map(([value]) => value), entry.performance_mode);
        for (const option of performance.options) option.textContent = PERFORMANCE_OPTIONS.find(([value]) => value === option.value)?.[1] || option.value;
        const story = el("div", "display:flex;flex-direction:column;gap:6px;");
        story.append(slice, row(spot, mood, performance));
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
        content.append(box);
        rows.push({ segment, entry, slice, spot, mood, performance, director, video, planning, continues, direction, holdInput });
      });
      const save = makeButton("Save Scene Plan And Notes", "primary");
      save.onclick = async () => {
        pushHistory();
        const nextPlan = rows.filter((item) => item.entry.act_id).map((item) => {
          const on = item.continues ? item.continues.checked : false;
          // A tick that differs from what the mode decided is your choice, and it wins over the mode from now on.
          const manual = item.continues && on !== Boolean(item.entry.continues_previous);
          return {
            ...item.entry,
            story_slice: item.slice.value,
            spot: item.spot.value,
            mood: item.mood.value,
            performance_mode: item.performance.value,
            continues_previous: on,
            continuation_source: manual ? "manual" : item.entry.continuation_source,
            continuation_direction: item.direction ? item.direction.value.trim() : item.entry.continuation_direction,
          };
        });
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
        await saveLayer({ story_scene_plan: planned }, "story builder scene plan edited");
        toast("Scene plan and notes saved.");
        show("scenes");
      };
      content.append(row(save));
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
        await saveLayer(data.story_layer || {}, "story builder locations");
        toast(`Locations ready: ${merged.added} added, ${merged.updated} updated. Every scene is mapped to its spot.`);
        // New cards have no images yet: go straight to making them.
        if (imageGaps().locations.length) await generateMissingImages("locations");
        else show("places");
      };
      top.append(makeField("Style and theme", theme), row(regenerate, build));
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
      const runAndRefresh = async (task) => {
        await task();
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
          : `Some cards used by your scenes have no image yet${gaps.characters.length ? `. Characters: ${gaps.characters.join(", ")}` : ""}${gaps.locations.length ? `. Locations: ${gaps.locations.join(", ")}` : ""}.`;
        const note = el("div", `font-size:12px;color:${imagesBlock ? "#fca5a5" : "#fbbf24"};border:1px solid ${imagesBlock ? "#7f1d1d" : "#78350f"};border-radius:7px;background:${imagesBlock ? "#1f0808" : "#1c1207"};padding:9px;display:flex;flex-direction:column;gap:8px;`, message);
        note.append(row(gaps.characters.length ? imageButton("characters") : null, gaps.locations.length ? imageButton("locations") : null));
        prompts.append(note);
      }
      const createPrompts = makeButton("Create Video Prompts", "primary");
      createPrompts.disabled = Boolean(blocked) || !withBeat || imagesBlock;
      if (imagesBlock) createPrompts.title = "Make the missing images first.";
      createPrompts.onclick = () => {
        if (withPrompt && window.confirm?.(`Replace the video prompts of all ${all.length} scenes?`) === false) return;
        runAndRefresh(() => createVideoPrompts());
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
      const list = card("Scenes", "Edit any beat, prompt or continuation direction, then save.");
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
        if (from) {
          item.style.border = `1px solid ${MASKED_COLOR}`;
          item.style.borderLeft = `5px solid ${MASKED_COLOR}`;
          item.append(maskedChip(from, index + 1, entry.continuation_source));
        }
        item.append(head, el("div", "font-size:11px;color:#94a3b8;", `Lyric (mood): ${segment.lyric_text || "[instrumental]"}`),
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
        await autoSaveSessionQuiet("story builder beats and prompts edited");
        toast("Beats and prompts saved.");
        show("prompts");
      };
      list.append(row(save));
      content.append(beats, prompts, list);
    }

    const renderers = {
      song: renderSong, cast: renderCast, story: renderStory, acts: renderActs,
      scenes: renderScenes, places: renderPlaces, prompts: renderPrompts,
    };

    async function show(tabId) {
      await leaveHostedTab();
      activeTab = renderers[tabId] ? tabId : "song";
      content.textContent = "";
      refreshProgress();
      try {
        renderers[activeTab]();
      } catch (error) {
        console.error("[VRGDG Story Builder] Tab failed to render", error);
        content.append(el("div", "color:#fca5a5;font-size:12px;", `This tab failed to open: ${String(error?.message || error)}`));
      }
      renderSide();
    }

    async function closeStoryBuilder() {
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
