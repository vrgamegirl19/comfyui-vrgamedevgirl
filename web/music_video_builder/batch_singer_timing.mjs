import { postJson } from "./comfy_api.mjs";

// Finish all Whisper work before the first prompt request. Only this phase may load Whisper.
export async function prepareBatchSingerTiming(segments, ensureTiming, { progress, assertNotStopped = () => {} } = {}) {
  const errors = new Map();
  let whisperStarted = false;
  try {
    for (const [index, segment] of segments.entries()) {
      assertNotStopped();
      progress?.set(`Preparing multi-shot lyric timing ${index + 1}/${segments.length}...`, 5);
      try {
        await ensureTiming(segment, {
          multiShotOnly: true,
          unloadAfter: false,
          onTimingStart: () => { whisperStarted = true; },
        });
      } catch (error) {
        errors.set(segment, error);
      }
    }
    return errors;
  } finally {
    if (whisperStarted) await postJson("/vrgdg/music_builder/clear_memory_direct", {}, 120000);
  }
}
