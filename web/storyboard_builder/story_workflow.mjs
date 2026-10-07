export const NO_MAPPED_LOCATIONS_MESSAGE = "No locations have been mapped yet. Map locations to scenes before creating the story arc, story brief, or scene beats.";

export function hasMappedStoryboardLocation(state) {
  const locations = Array.isArray(state?.referenceBuilder?.locations) ? state.referenceBuilder.locations : [];
  const ids = new Set(locations.map((location) => String(location?.id || "").trim()).filter(Boolean));
  return (state?.scenes || []).some((scene) => {
    const id = String(scene?.location_ref?.id || scene?.id_lora_location_id || "").trim();
    return id && ids.has(id);
  });
}

export async function runStoryGenerationSequence({ createArc, createBrief, createBeats, save }) {
  const arc = await createArc();
  if (!arc) return { completed: false, stoppedAt: "arc" };
  await save();
  const brief = await createBrief();
  if (!brief) return { completed: false, stoppedAt: "brief" };
  await save();
  const beats = await createBeats();
  if (!beats || beats.failures?.length) return { completed: false, stoppedAt: "beats" };
  return { completed: true, created: beats.created || 0 };
}
