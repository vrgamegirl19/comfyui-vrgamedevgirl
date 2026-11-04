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
