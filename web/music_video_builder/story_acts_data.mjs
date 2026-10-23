// Story Mode act rules for the Acts tab: validation and border, merge and split edits.
// Pure functions only. Python twin: storyboard/story_acts.py. Both are checked against tests/story_acts_cases.json.

function storyActNumbers(act, numberById) {
  return (act.scene_ids || []).filter((id) => id in numberById).map((id) => numberById[id]).sort((a, b) => a - b);
}

// Problems with an edited act list, as plain sentences. Empty means the acts are valid.
export function validateStoryActs(acts, chapters, sceneIds) {
  const problems = [];
  const numberById = Object.fromEntries(sceneIds.map((id, index) => [id, index + 1]));
  const chapterOrder = Object.fromEntries(chapters.map((chapter, index) => [chapter.id, index]));
  const seen = {};
  let previousLast = 0;
  let previousChapter = -1;
  for (const act of acts) {
    const label = act.title || act.id || "An act";
    if (!(act.chapter_id in chapterOrder)) {
      problems.push(`${label} belongs to no chapter of the story.`);
      continue;
    }
    const unknown = (act.scene_ids || []).filter((id) => !(id in numberById));
    if (unknown.length) problems.push(`${label} lists scenes that are not on the timeline: ${unknown.join(", ")}.`);
    const numbers = storyActNumbers(act, numberById);
    if (!numbers.length) {
      problems.push(`${label} has no scenes.`);
      continue;
    }
    for (const id of act.scene_ids || []) {
      if (id in seen) problems.push(`Scene ${numberById[id] ?? id} is in both ${seen[id]} and ${label}.`);
      seen[id] = label;
    }
    const contiguous = numbers.every((number, index) => number === numbers[0] + index);
    if (!contiguous) problems.push(`${label} skips scenes; an act's scenes must be next to each other.`);
    if (numbers[0] <= previousLast) problems.push(`${label} is out of timeline order.`);
    const order = chapterOrder[act.chapter_id];
    if (order < previousChapter) problems.push(`${label} puts its chapter before an earlier chapter on the timeline.`);
    previousLast = numbers[numbers.length - 1];
    previousChapter = order;
  }
  const missing = sceneIds.filter((id) => !(id in seen)).map((id) => String(numberById[id]));
  if (missing.length) problems.push(`Scenes not in any act: ${missing.join(", ")}.`);
  return problems;
}

function copyActs(acts) {
  return acts.map((act) => ({ ...act }));
}

function renumberActs(acts) {
  return acts.map((act, index) => ({ ...act, id: `act_${index + 1}` }));
}

// Move the border so act actIndex starts at the scene. Both acts keep a scene and share a chapter.
export function moveActBorder(acts, actIndex, newFirstSceneId, sceneIds) {
  if (actIndex <= 0 || actIndex >= acts.length || !sceneIds.includes(newFirstSceneId)) return copyActs(acts);
  const before = acts[actIndex - 1];
  const after = acts[actIndex];
  if (before.chapter_id !== after.chapter_id) return copyActs(acts);
  const joined = [...(before.scene_ids || []), ...(after.scene_ids || [])];
  const cut = joined.indexOf(newFirstSceneId);
  if (cut <= 0) return copyActs(acts);
  const result = copyActs(acts);
  result[actIndex - 1] = { ...before, scene_ids: joined.slice(0, cut) };
  result[actIndex] = { ...after, scene_ids: joined.slice(cut) };
  return result;
}

// Merge act actIndex into the act before it when both are in the same chapter.
export function mergeActs(acts, actIndex) {
  if (actIndex <= 0 || actIndex >= acts.length) return copyActs(acts);
  const before = acts[actIndex - 1];
  const after = acts[actIndex];
  if (before.chapter_id !== after.chapter_id) return copyActs(acts);
  const merged = {
    ...before,
    summary: [before.summary, after.summary].filter(Boolean).join(" "),
    cast_ids: [...new Set([...(before.cast_ids || []), ...(after.cast_ids || [])])],
    scene_ids: [...(before.scene_ids || []), ...(after.scene_ids || [])],
  };
  return renumberActs([...acts.slice(0, actIndex - 1), merged, ...acts.slice(actIndex + 1)]);
}

// Split an act in two; the new act starts at the given scene. Both parts keep at least one scene.
export function splitAct(acts, actIndex, firstSceneIdOfNewAct) {
  if (actIndex < 0 || actIndex >= acts.length) return copyActs(acts);
  const act = acts[actIndex];
  const sceneIds = [...(act.scene_ids || [])];
  const cut = sceneIds.indexOf(firstSceneIdOfNewAct);
  if (cut <= 0) return copyActs(acts);
  const first = { ...act, scene_ids: sceneIds.slice(0, cut) };
  const second = { ...act, title: `${act.title || "Act"} (continued)`, scene_ids: sceneIds.slice(cut) };
  return renumberActs([...acts.slice(0, actIndex), first, second, ...acts.slice(actIndex + 1)]);
}
