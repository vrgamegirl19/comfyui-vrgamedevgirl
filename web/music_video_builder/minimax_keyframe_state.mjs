// Durable, per-scene I2V inputs. No previous-scene frame resolution belongs here.
export function miniMaxI2VFrameMode(segment = {}) {
  if (["normal", "flf"].includes(segment?.minimax_h3_i2v_frame_mode)) return segment.minimax_h3_i2v_frame_mode;
  return segment?.first_last_frame_end_image_path || segment?.first_last_frame_end_image_data ? "flf" : "normal";
}

export function miniMaxI2VLastFrame(segment = {}) {
  return miniMaxI2VFrameMode(segment) === "flf" ? {
    path: String(segment?.first_last_frame_end_image_path || "").trim(),
    data: String(segment?.first_last_frame_end_image_data || "").trim(),
  } : { path: "", data: "" };
}

export function miniMaxI2VFLFEnabled(segment, engine, mode) {
  return engine === "minimax_h3" && mode === "image_to_video" && miniMaxI2VFrameMode(segment) === "flf";
}

export function miniMaxI2VFramePaths(segment, firstPath) {
  const first = String(firstPath || "").trim();
  if (!first) throw new Error("Image to Video needs a first-frame image.");
  const last = miniMaxI2VLastFrame(segment);
  if (miniMaxI2VFrameMode(segment) === "flf" && !last.path) {
    throw new Error("First–Last Frame needs a saved last-frame image. Choose or upload it before rendering.");
  }
  return { first, last: last.path };
}
