const BUILDER_REFRESH_KEY = "vrgdg:music-builder:refresh-resume";
const BUILDER_REFRESH_MAX_AGE_MS = 5 * 60 * 1000;

export function queueBuilderRefresh(storage, view, now = Date.now()) {
  storage.setItem(BUILDER_REFRESH_KEY, JSON.stringify({ ...view, expiresAt: now + BUILDER_REFRESH_MAX_AGE_MS }));
}

export function takeBuilderRefresh(storage, now = Date.now()) {
  const raw = storage.getItem(BUILDER_REFRESH_KEY);
  if (!raw) return null;
  storage.removeItem(BUILDER_REFRESH_KEY);
  try {
    const view = JSON.parse(raw);
    if (!view || !String(view.projectFolder || "").trim() || !Number.isFinite(view.expiresAt)
      || view.expiresAt < now) return null;
    return view;
  } catch {
    return null;
  }
}
