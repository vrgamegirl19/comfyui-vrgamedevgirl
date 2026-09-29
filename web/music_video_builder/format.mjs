export function renameSubjectInDescription(description, previousName, nextName) {
  const before = String(previousName || "").trim();
  const after = String(nextName || "").trim();
  if (!before || !after || before === after) return String(description || "");
  const escaped = before.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`(^|[^\\p{L}\\p{N}_])${escaped}(?=$|[^\\p{L}\\p{N}_])`, "gu");
  return String(description || "").replace(pattern, (_, prefix) => prefix + after);
}

export function formatTime(value) {
  const totalHundredths = Math.max(0, Math.round(Number(value || 0) * 100));
  const minutes = Math.floor(totalHundredths / 6000);
  const seconds = Math.floor((totalHundredths % 6000) / 100);
  const hundredths = totalHundredths % 100;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(hundredths).padStart(2, "0")}`;
}

export function formatDurationSeconds(start, end) {
  const duration = Math.max(0, Number(end || 0) - Number(start || 0));
  return duration.toFixed(duration >= 10 ? 1 : 2);
}
