export function defaultNotificationSettings() {
  return {
    mode: "off",
    success_sound: "chime",
    error_sound: "warning",
    volume: 0.45,
    success_custom_audio: "",
    success_custom_name: "",
    error_custom_audio: "",
    error_custom_name: "",
  };
}

export function normalizeNotificationSettings(settings = {}) {
  const defaults = defaultNotificationSettings();
  const mode = String(settings.mode || settings.notification_mode || defaults.mode);
  const validModes = new Set(["off", "errors", "batch", "complete", "all"]);
  const volume = Number(settings.volume ?? settings.notification_volume ?? defaults.volume);
  return {
    ...defaults,
    ...settings,
    mode: validModes.has(mode) ? mode : defaults.mode,
    success_sound: String(settings.success_sound || settings.successSound || defaults.success_sound),
    error_sound: String(settings.error_sound || settings.errorSound || defaults.error_sound),
    volume: Number.isFinite(volume) ? Math.max(0, Math.min(1, volume)) : defaults.volume,
    success_custom_audio: String(settings.success_custom_audio || settings.successCustomAudio || ""),
    success_custom_name: String(settings.success_custom_name || settings.successCustomName || ""),
    error_custom_audio: String(settings.error_custom_audio || settings.errorCustomAudio || ""),
    error_custom_name: String(settings.error_custom_name || settings.errorCustomName || ""),
  };
}
