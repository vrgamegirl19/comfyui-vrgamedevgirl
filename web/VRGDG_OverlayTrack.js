const OVERLAY_TRACK_HELP_HTML = `
  <div style="display:flex;flex-direction:column;gap:10px;line-height:1.5;">
    <div><strong style="color:#cffafe;">What it does</strong><br>Adds an optional video track above the normal timeline for alternate takes, B-roll, and covering unwanted parts of a base clip.</div>
    <div><strong style="color:#cffafe;">When it is off</strong><br>The Video Builder behaves exactly like the standard single-timeline workflow. Saved overlay clips are preserved but ignored during playback and stitching.</div>
    <div><strong style="color:#cffafe;">When it is on</strong><br>An enabled overlay clip becomes the visible clip wherever it covers the base timeline. Empty areas and hidden overlay clips automatically show the base video underneath.</div>
    <div><strong style="color:#cffafe;">Eye icon</strong><br>Turns an individual overlay clip on or off without deleting it. A hidden clip stays saved in the project but is ignored during playback and final stitching.</div>
    <div><strong style="color:#cffafe;">Lock icon</strong><br>New overlay clips are locked by default. Unlock a clip before moving or trimming it, then lock it again to freeze its position.</div>
    <div><strong style="color:#cffafe;">Moving clips</strong><br>Unlocked clips can move left or right. They cannot overlap another enabled overlay clip. When Snap Beats is on, movement and trimming use the existing beat-marker snapping assistance.</div>
    <div><strong style="color:#cffafe;">Regenerating a video</strong><br>If the selected base scene already has a video, Add to overlay track keeps the original base clip and renders the new take above it with the same scene prompts, mappings, references, and settings.</div>
    <div><strong style="color:#cffafe;">Audio</strong><br>The base timeline remains the primary audio source. Overlay clips replace only the visible video unless a future audio option is explicitly added.</div>
    <div><strong style="color:#cffafe;">Saving</strong><br>The track setting, clip visibility, lock state, timing, prompts, references, and generated media are saved with the project.</div>
  </div>`;

function normalizeOverlayTrackState(source = {}) {
  return {
    enabled: Boolean(source.enabled),
  };
}

function normalizeOverlayClip(clip = {}) {
  clip.overlay_enabled = clip.overlay_enabled !== false;
  clip.overlay_locked = clip.overlay_locked !== false;
  clip.track = "overlay";
  clip.source = clip.source || "overlay";
  return clip;
}

function overlayClipIsEnabled(clip, trackState) {
  return Boolean(trackState?.enabled) && clip?.overlay_enabled !== false;
}

function cloneSceneAsOverlay(scene, createId, slotNumber) {
  const clone = typeof structuredClone === "function"
    ? structuredClone(scene)
    : JSON.parse(JSON.stringify(scene));
  clone.id = createId();
  clone.track = "overlay";
  clone.source = "overlay_regeneration";
  clone.overlay_slot_number = slotNumber;
  delete clone.scene_slot_number;
  delete clone.slot_number;
  clone.overlay_enabled = true;
  clone.overlay_locked = true;
  clone.label = `${scene.label || "Scene"} - Alternate`;
  clone.video_path = "";
  clone.video_output = null;
  clone.video_status = "none";
  clone.video_history = [];
  clone.video_thumbnail_history = [];
  clone.video_backup_paths = [];
  clone.video_backup_thumbnail_paths = [];
  clone.video_history_index = -1;
  clone.video_thumbnail_path = "";
  clone.video_original_path = "";
  clone.video_original_thumbnail_path = "";
  return clone;
}

function clampOverlayTiming(clip, clips, proposedStart, proposedEnd) {
  const duration = Math.max(0.1, Number(proposedEnd) - Number(proposedStart));
  let start = Math.max(0, Number(proposedStart) || 0);
  let end = start + duration;
  const others = (clips || [])
    .filter((item) => item && item.id !== clip.id && item.overlay_enabled !== false)
    .sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
  for (const other of others) {
    const otherStart = Number(other.start || 0);
    const otherEnd = Number(other.end || otherStart);
    if (end <= otherStart + 0.001 || start >= otherEnd - 0.001) continue;
    const moveLeft = Math.max(0, otherStart - duration);
    const moveRight = otherEnd;
    if (Math.abs(start - moveLeft) <= Math.abs(start - moveRight)) start = moveLeft;
    else start = moveRight;
    end = start + duration;
  }
  return { start, end };
}

export {
  OVERLAY_TRACK_HELP_HTML,
  clampOverlayTiming,
  cloneSceneAsOverlay,
  normalizeOverlayClip,
  normalizeOverlayTrackState,
  overlayClipIsEnabled,
};
