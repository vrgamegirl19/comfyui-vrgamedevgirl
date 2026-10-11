// Preserve timeline indexes when applying a flow to a subset of scene cards.
export function sceneDefaultTargets(state, selectedOnly, toast) {
  const targets = state.scenes.map((scene, index) => ({ scene, index }))
    .filter(({ scene }) => !selectedOnly || state.selected?.has(scene.id));
  if (selectedOnly && !targets.length) {
    toast("Select one or more scenes first.");
    return null;
  }
  return targets;
}

export function wireSelectedDefaultActions({ actions, handlers, state }) {
  for (const [key, button] of Object.entries(actions)) {
    button.onclick = () => handlers[key]({ overwrite: true, selectedOnly: true });
  }
  return () => {
    const count = state.scenes.filter((scene) => state.selected?.has(scene.id)).length;
    for (const button of Object.values(actions)) {
      button.hidden = count === 0;
      button.title = `Replace this setting on ${count} selected scene${count === 1 ? "" : "s"} only.`;
    }
  };
}
