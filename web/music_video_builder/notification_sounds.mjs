import { normalizeVideoType } from "./controls.mjs";
import { normalizeNotificationSettings } from "./notifications.mjs";

function notificationFrequencies(kind, soundName) {
  if (kind === "error") {
    if (soundName === "double_beep") return [220, 180];
    if (soundName === "soft") return [196, 146];
    return [196, 146, 110];
  }
  if (soundName === "bell") return [784, 988, 1175];
  if (soundName === "double_beep") return [660, 880];
  if (soundName === "soft") return [440, 554];
  return [523, 659, 784];
}

export function createNotificationSounds({ state, videoTypeSelect }) {
  function syncVideoTypeControl() {
    state.videoType = normalizeVideoType(state.videoType);
    videoTypeSelect.value = state.videoType;
  }

  let notificationAudioContext = null;
  let lastNotificationAt = 0;

  function shouldNotifyForToast(message, isError) {
    const settings = normalizeNotificationSettings(state.notificationSettings);
    if (settings.mode === "off") return false;
    if (isError) return settings.mode !== "off";
    const text = String(message || "");
    if (settings.mode === "errors") return false;
    if (settings.mode === "all") return true;
    const batchDone = /\b(?:Build Full Video|Image All|Render All|Stitch Preview|(?:Gemma Local|Qwen Local|LM Studio|LLM API|Custom Server) (?:T2I|I2V|T2V|Video) All|Flux\/Klein All|NanoBanana All|Ernie All|ZImage All)\b[\s\S]{0,80}\b(?:complete|finished|ready|saved)\b/i.test(text) ||
      /\b(?:complete after \d+ attempts|final video|stitched video)\b/i.test(text);
    if (settings.mode === "batch") return batchDone;
    const completedThing = /\b(?:complete|completed|finished|created|ready|rendered|stitched|saved)\b/i.test(text);
    return batchDone || completedThing;
  }

  function playGeneratedNotification(kind = "success") {
    const settings = normalizeNotificationSettings(state.notificationSettings);
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    notificationAudioContext ||= new AudioCtx();
    if (notificationAudioContext.state === "suspended") notificationAudioContext.resume().catch(() => {});
    const ctx = notificationAudioContext;
    const volume = Math.max(0, Math.min(1, Number(settings.volume ?? 0.45)));
    const soundName = kind === "error" ? settings.error_sound : settings.success_sound;
    const frequencies = notificationFrequencies(kind, soundName);
    const start = ctx.currentTime + 0.02;
    frequencies.forEach((frequency, index) => {
      const oscillator = ctx.createOscillator();
      const gain = ctx.createGain();
      oscillator.type = kind === "error" ? "square" : "sine";
      oscillator.frequency.value = frequency;
      oscillator.connect(gain);
      gain.connect(ctx.destination);
      const t = start + index * 0.13;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, volume * 0.16), t + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
      oscillator.start(t);
      oscillator.stop(t + 0.14);
    });
  }

  function playCustomNotificationAudio(dataUrl, kind) {
    const settings = normalizeNotificationSettings(state.notificationSettings);
    if (!dataUrl) return false;
    const sound = new Audio(dataUrl);
    sound.volume = Math.max(0, Math.min(1, Number(settings.volume ?? 0.45)));
    sound.play().catch(() => playGeneratedNotification(kind));
    return true;
  }

  function playBuilderNotification(kind = "success", force = false) {
    if (!force && Date.now() - lastNotificationAt < 450) return;
    lastNotificationAt = Date.now();
    const settings = normalizeNotificationSettings(state.notificationSettings);
    const customAudio = kind === "error" ? settings.error_custom_audio : settings.success_custom_audio;
    if (customAudio && playCustomNotificationAudio(customAudio, kind)) return;
    playGeneratedNotification(kind);
  }

  return { playBuilderNotification, shouldNotifyForToast, syncVideoTypeControl };
}
