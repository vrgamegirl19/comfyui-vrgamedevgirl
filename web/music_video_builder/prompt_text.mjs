import {
  storyboardFacialPerformancePreset,
  storyboardPerformancePreset,
} from "../storyboard_builder/performance_presets.mjs";
import { storyboardCutPlanForDuration } from "../storyboard_builder/scenes.mjs";
import { cancelComfyExecutionAndWaitIdle, postJson } from "./comfy_api.mjs";
import { normalizeProjectVideoEngine, normalizeVideoType } from "./controls.mjs";
import { sceneConceptPromptText, sceneLyricTextForPromptValidation } from "./image_prompts.mjs";
import { normalizeMiniMaxH3Voice } from "./minimax_h3.mjs";
import {
  builderMotionSpeedGuidance,
  normalizeBuilderStoryboardDefaults,
  normalizeBuilderStoryLayer,
} from "./model_settings.mjs";
import { timelineSegmentDuration } from "./timeline_state.mjs";

export function cleanGeneratedPromptText(prompt) {
  let text = String(prompt || "").trim();
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  text = text.replace(/<\/?thought>/gi, "").trim();
  text = text.replace(/^[\s\-–—_=~.*#|:;,+/\\]{16,}(?=[\p{L}\p{N}])/u, "").trim();
  const controlPatterns = [
    /^\s*<?\/?end[_\-][a-z0-9_\-]*>?\s*/i,
    /^\s*_?name\s*[:=]\s*/i,
    /^\s*\d+\s*(?:thought|analysis|reasoning)\s*[:\-]?\s*/i,
    /^\s*(?:0f|of)[_\-\s]*(?:thought|analysis|reasoning)\s*[:\-]?\s*/i,
    /^\s*_?\s*(?:thought|analysis|reasoning)\s*<channel\|>\s*/i,
    /^\s*_?\s*<\|?channel\|?>\s*(?:thought|analysis|reasoning)?\s*/i,
    /^\s*_?\s*<channel\|>\s*(?:thought|analysis|reasoning)?\s*/i,
    /^\s*_?\s*(?:thought|analysis|reasoning)\s*[:\-]?\s*/i,
    /^(?:Assistant|Answer|Final prompt)\s*:\s*/i,
  ];
  let previous = "";
  while (text && previous !== text) {
    previous = text;
    for (const pattern of controlPatterns) text = text.replace(pattern, "").trim();
  }
  return text;
}

function looksLikeGeneratedPromptJunk(prompt) {
  const text = String(prompt || "").toLowerCase().replace(/\s+/g, " ").trim();
  if (!text) return false;
  const bracketed = text.match(/\[[^\]]{2,80}\]/g) || [];
  if (bracketed.length >= 2) return true;
  if (/\[[^\]]*(?:subject|setting|environment|camera|motion|weather|lighting|dynamic|framing)[^\]]*\]/i.test(text)) return true;
  const compact = text.replace(/[^a-z0-9_<>\-|]+/g, "");
  const markers = [
    "completion-completion-completion",
    "thought-thought-thought",
    "de-facto-de-facto-de-facto",
    "de-fleshed",
    "thoughtthoughtthought",
    "ownnessownnessownness",
    "nessnessnessness",
    "end_anow",
    "<|channel>",
    "<channel|>",
  ];
  if (markers.some((marker) => compact.includes(marker) || text.includes(marker))) return true;
  if (/([a-z]{2,16})\1{5,}/i.test(compact)) return true;
  const tokens = text.match(/[\p{L}\p{N}_']+/gu) || [];
  if (tokens.length >= 16) {
    const counts = new Map();
    for (const token of tokens) counts.set(token, (counts.get(token) || 0) + 1);
    const maxCount = Math.max(...counts.values());
    if (maxCount >= 10 && maxCount / tokens.length >= 0.20) return true;
    for (const size of [2, 3, 4]) {
      if (tokens.length < size * 4) continue;
      const phraseCounts = new Map();
      for (let index = 0; index <= tokens.length - size; index += 1) {
        const phrase = tokens.slice(index, index + size).join(" ");
        phraseCounts.set(phrase, (phraseCounts.get(phrase) || 0) + 1);
      }
      if (Math.max(...phraseCounts.values()) >= 8) return true;
    }
  }
  return false;
}

export function isRecoverableBuildGemmaError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  if (!message) return false;
  const recoverable = [
    "gemma returned repeated/thought junk",
    "gemma returned repeated/thought text",
    "repeated/thought junk",
    "repeated/thought text",
    "thought junk",
    "unfilled template",
    "square-bracket placeholder",
    "request timed out",
    "backend may still be processing",
    "failed to create a usable prompt",
    "returned an empty i2v prompt",
    "returned an empty t2v prompt",
    "returned an empty flux/klein prompt",
    "returned an empty minimax",
    "returned an empty scene story beat",
    "empty minimax",
    "could not produce a complete minimax h3 prompt within 7,000 characters",
    "returned the scene lyrics instead of a usable",
    "did not return valid json shot descriptions",
    "returned 0 shot descriptions",
    "returned an invalid number of shot descriptions",
    "omitted mapped extra label",
    "incomplete final shot description",
    "orphaned reference-purpose fragment",
    "incomplete <d> dialogue",
    "incomplete reference label",
  ];
  if (recoverable.some((item) => message.includes(item))) return true;
  if (/gemma[\s\S]{0,120}(thought|junk|empty|timed out|timeout|repeated|template|placeholder|scene lyrics)/i.test(message)) return true;
  return false;
}

export function applyTriggerPhrase(prompt, trigger, options = {}) {
  const promptText = cleanGeneratedPromptText(prompt);
  if (options.validateJunk !== false && looksLikeGeneratedPromptJunk(promptText)) {
    const error = new Error("Gemma returned repeated/thought junk instead of a usable prompt. Try again or shorten the notes.");
    error.rawGemmaPrompt = String(prompt || "");
    error.cleanedGemmaPrompt = promptText;
    throw error;
  }
  const triggerText = String(trigger || "").trim().replace(/\s+/g, " ");
  if (!triggerText) return promptText;
  if (!promptText) return triggerText;
  if (promptText.toLowerCase().startsWith(triggerText.toLowerCase())) return promptText;
  return `${triggerText}, ${promptText}`;
}

function unwrapStructuredImagePrompt(value) {
  const original = String(value || "").trim();
  if (!original) return original;
  const fenced = original.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/i, "").trim();
  if (!/^[\[{]/.test(fenced)) return original;
  let parsed;
  try {
    parsed = JSON.parse(fenced);
  } catch (_) {
    return original;
  }
  const promptKeys = ["image_prompt", "t2i_prompt", "text_to_image_prompt", "prompt", "flux_prompt", "nb_prompt", "nano_banana_prompt", "ernie_prompt"];
  const walk = (item) => {
    if (Array.isArray(item)) {
      for (const child of item) {
        const found = walk(child);
        if (found) return found;
      }
      return "";
    }
    if (!item || typeof item !== "object") return "";
    for (const key of promptKeys) {
      const text = typeof item[key] === "string" ? item[key].trim() : "";
      if (text) return text;
    }
    for (const child of Object.values(item)) {
      const found = walk(child);
      if (found) return found;
    }
    return "";
  };
  return walk(parsed) || original;
}

export function syncConceptPromptToStoryBeat(segment, prompt) {
  if (!segment) return "";
  const text = String(prompt || "").trim();
  segment.story_beat = text;
  return text;
}

export async function loadI2VMotionNotesFromPath(path) {
  const notePath = String(path || "").trim();
  if (!notePath) return [];
  const data = await postJson("/vrgdg/music_builder/load_prompt_json", {
    prompt_json_path: notePath,
  });
  return Array.isArray(data.prompts) ? data.prompts : [];
}

export async function loadPromptJsonFromPath(path) {
  const promptPath = String(path || "").trim();
  if (!promptPath) return [];
  const data = await postJson("/vrgdg/music_builder/load_prompt_json", {
    prompt_json_path: promptPath,
  });
  return Array.isArray(data.prompts) ? data.prompts : [];
}

export async function loadLyricSegmentsFromPath(path) {
  const lyricPath = String(path || "").trim();
  if (!lyricPath) return [];
  const data = await postJson("/vrgdg/music_builder/load_prompt_json", {
    prompt_json_path: lyricPath,
  });
  return Array.isArray(data.prompts) ? data.prompts : [];
}

export function isInstrumentalLyricText(text) {
  const value = String(text || "").trim().toLowerCase().replace(/\s+/g, " ");
  if (!value) return false;
  if (value === "instrumental" || value === "[instrumental]" || value === "instrumental section" || value === "instrumental section.") return true;
  const stripped = value
    .replace(/\[(?:intro|outro|bridge|verse|chorus|pre-chorus|prechorus|hook|refrain|interlude|break|section|instrumental|music|no vocals?|no singing|no lip\s*-?\s*sync|no lipsync|b-?roll|visual only|silence)\]/gi, " ")
    .replace(/\b(?:intro|outro|bridge|verse|chorus|pre-chorus|prechorus|hook|refrain|interlude|break|section|instrumental|music|no vocals?|no singing|no lip\s*-?\s*sync|no lipsync|b-?roll|visual only|silence)\b/gi, " ")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
  return /\binstrumental|no vocals?|no singing|no lip\s*-?\s*sync|no lipsync|b-?roll|visual only|silence\b/i.test(value) && !stripped;
}

export function isNoLipSyncSingerChoice(value) {
  const clean = String(value || "").trim().toLowerCase();
  return clean === "b-roll / no lip-sync" || clean === "broll / no lip-sync" || clean === "b-roll" || clean === "broll";
}

export function segmentUsesNoLipSyncPerformance(segment) {
  const style = String(segment?.performance_style || segment?.performanceStyle || segment?.song_style || segment?.songStyle || "").trim();
  if (!style) return Boolean(segment?.lyric_no_lip_sync);
  const preset = storyboardPerformancePreset(style) || {};
  const text = [style, preset.label, preset.value].map((value) => String(value || "").trim()).filter(Boolean).join(" ").toLowerCase();
  return Boolean(segment?.lyric_no_lip_sync)
    || /\bno[_\s-]*vocals?\b/.test(text)
    || /\bno[_\s-]*lip[_\s-]*sync\b/.test(text)
    || /\bb[_\s-]*roll\b/.test(text)
    || /\bvisual[_\s-]*only\b/.test(text);
}

export function quoteOrderedLyricCues(text) {
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
  let changed = false;
  const cuePattern = /^\s*([A-Za-z][A-Za-z0-9 _/-]{0,42}?(?:sings?|answers?|responds?|chants?|whispers?|harmonizes?|says?|first|next|then|female|male|woman|man|girl|boy|group|duet)[A-Za-z0-9 _/-]{0,42}?)\s*:\s*(.+?)\s*$/i;
  const updated = lines.map((line) => {
    const match = line.match(cuePattern);
    if (!match) return line;
    const lyric = String(match[2] || "").trim();
    if (!lyric || /^["'“”‘’]/.test(lyric)) return line;
    changed = true;
    return `${match[1].trim()}: "${lyric.replace(/^["'“”‘’]+|["'“”‘’]+$/g, "")}"`;
  });
  return changed ? updated.join("\n") : String(text || "");
}

function lyricTextHasCueLabels(text) {
  return /:\s*["'“”]/.test(String(text || ""));
}

export function flattenLyricForPrompt(text) {
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .join(" ")
    .replace(/^["'“”‘’]+|["'“”‘’]+$/g, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function videoPromptWritingRulesNote() {
  return [
    "Prompt writing rules:",
    "Use the image reference and text-to-image prompt only for visible first-frame details: subject identity, wardrobe, hair, makeup, props, setting, lighting, color palette, framing, and composition.",
    "Do not use the image prompt for body action, camera motion, performance energy, facial performance, lyric action, story action, or animation pacing.",
    "Use the motion/camera notes, performance direction, vocal direction, facial direction, and scene story beat to decide animation, body action, camera movement, and performance energy.",
    "If the visible subject is identified as the woman, the man, the girl, the boy, the singer, the performer, or a named character, use that natural subject label in the prompt. Do not write generic labels like the visible subject, the subject, the character, or the person unless no subject identity is available.",
    "Each sentence has one job and must add new information. Do not repeat the same mood, trait, motion, authority/defiance language, setting adjective, or descriptive phrase across the face, body, camera, environment, and atmosphere sentences.",
    "If an idea appears in the face sentence, do not repeat it in the body, camera, environment, or atmosphere sentence; use a different concrete visual detail instead.",
    "Do not duplicate adjacent words such as tall, tall or vast, vast.",
  ].join("\n");
}

function cleanNaturalSubjectLabel(value) {
  const original = String(value || "").trim();
  if (!original) return "";
  const firstLine = original.split(/\n/)[0].trim();
  const beforeDetail = firstLine.split(/\s+[:|]\s+/)[0].trim();
  let text = /^(?:character\s*\d*|subject\s*\d*|person|the subject|visible subject|the visible subject)$/i.test(beforeDetail)
    ? original
    : beforeDetail;
  text = text.replace(/\s+-\s+.*$/, "").trim();
  text = text.replace(/\s{2,}/g, " ");
  const lower = text.toLowerCase();
  if (!text || /^(?:character\s*\d*|subject\s*\d*|person|the subject|visible subject|the visible subject)$/i.test(text)) return "";
  const feminine = text.match(/\bthe\s+(?:woman|girl|female singer|female performer|bride|queen|princess)\b/i);
  if (feminine) return feminine[0];
  const masculine = text.match(/\bthe\s+(?:man|boy|male singer|male performer|king|prince)\b/i);
  if (masculine) return masculine[0];
  if (/\b(?:woman|girl|female|feminine|she|her)\b/i.test(lower)) return "the woman";
  if (/\b(?:man|boy|male|masculine|he|him|his)\b/i.test(lower)) return "the man";
  if (/^(?:the\s+)?(?:singer|performer|rapper|dancer|speaker)\b/i.test(text)) return text.startsWith("the ") ? text : `the ${text}`;
  return text;
}

export function normalizeContinuityMode(value, legacyAutoChain = false) {
  const mode = String(value || "").trim().toLowerCase();
  if (mode === "i2v_chain" || mode === "img2img" || mode === "off") return mode;
  return legacyAutoChain ? "i2v_chain" : "off";
}

export function normalizeAutoImg2ImgStartStep(value) {
  return Math.max(1, Math.min(8, Number(value || 5)));
}

export function normalizeAutoImg2ImgCreativity(value) {
  return Math.max(0, Math.min(10, Number(value ?? 5)));
}

export function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function normalizeGemmaGpuLayers(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 99;
  return Math.max(0, Math.min(999, Math.round(parsed)));
}

export function normalizeGemmaContextLimit(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 8000;
  return Math.max(512, Math.min(262144, Math.round(parsed)));
}

export function normalizeLmStudioContextLimit(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 32768;
  return Math.max(512, Math.min(262144, Math.round(parsed)));
}

export function normalizeOutputTokenLimit(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 8192;
  return Math.max(64, Math.min(262144, Math.round(parsed)));
}

function removeQuietFromSingingPrompt(text) {
  return String(text || "")
    .replace(/\bwith\s+a\s+quiet,\s*internal\s+intensity\b/gi, "with controlled internal intensity")
    .replace(/\bwith\s+quiet\s+internal\s+intensity\b/gi, "with controlled internal intensity")
    .replace(/\bquiet,\s*internal\s+intensity\b/gi, "controlled internal intensity")
    .replace(/\bquiet\s+internal\s+intensity\b/gi, "controlled internal intensity")
    .replace(/\bquiet\s+intensity\b/gi, "controlled intensity")
    .replace(/\bquiet\s+performance\b/gi, "controlled performance")
    .replace(/\bquiet\s+emotion\b/gi, "restrained emotion")
    .replace(/\bquiet\s+singing\b/gi, "focused singing")
    .replace(/\s{2,}/g, " ")
    .trim();
}

export function defaultFluxReferenceBuilder() {
  return {
    use_subject_reference: false,
    use_location_references: false,
    include_manual_ingredients: true,
    subject_count: 0,
    subject: { description: "", reference_type: "character", minimax_voice: normalizeMiniMaxH3Voice(), image: { path: "", data: "", name: "" } },
    subjects: [],
    subject_scene_map: {},
    extras_enabled: false,
    extra_subjects: [],
    extra_scene_map: {},
    performer_scene_map: {},
    locations: [],
    scene_map: {},
    scene_trigger_map: {},
    location_style_theme: "",
    max_generated_locations: 8,
    locations_cleared: false,
    cleared: false,
    trigger_position: "start",
    subject_trigger_position: "start",
    location_trigger_position: "start",
    ingredients_sheets: [],
    ingredients_scene_map: {},
    ingredients_auto_map_sources: {
      director_notes: true,
      concept_prompt: true,
      scene_notes: true,
      lyric_text: true,
    },
  };
}

export function createPromptText({
  activeSegment, assertBatchNotStopped, autoSaveSessionQuiet, createProgressWindow, ernieT2IPrompt,
  flowGptPrompt, fluxKleinSettingsForSegment, fluxPrompt, fluxReferenceContextForSegment,
  generateTextOnlyImagePromptFallbackForSegment, idLoraSceneContext, krea2TwoPassT2IPrompt,
  nbImageSettingsForSegment, nbPrompt, render, runClearMemoryWorkflowQuiet, runImageMemoryCleanupQuiet,
  sceneDisplayName, segmentIndexInfo, segmentMappedLocationText, segmentMappedSubjectText, state,
  storyboardScenePayload, t2iPrompt, zEnhancePromptPreview,
}) {
  function removeNegativeAndVocalWordingFromVisualPrompt(text) {
    const forbidden = /\b(?:lip[ -]?sync(?:ing|s)?|sing(?:s|ing)?|sang|sung|rap(?:s|ping)?|vocal(?:s|ization)?|lyric(?:s)?|speak(?:s|ing)?|say(?:s|ing)?|said|dialogue|mouth(?:s|ed|ing)?|lips?)\b/i;
    const negative = /\b(?:no|not|never|without|avoid|omit|exclude|prevent|don['’]t|doesn['’]t|isn['’]t|aren['’]t|cannot|can['’]t|do\s+not|does\s+not)\b/i;
    return String(text || "")
      .split(/(?<=[.!?])\s+|\s*;\s*/)
      .map((part) => part.trim())
      .filter((part) => part && !forbidden.test(part) && !negative.test(part))
      .join(" ")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  async function recoverFromBuildGemmaError(error, attempt, maxRetries, progress) {
    const message = String(error?.message || error);
    const windowTitle = `Build Full Video recovery ${attempt}/${maxRetries}`;
    const recoveryProgress = progress || createProgressWindow(windowTitle);
    recoveryProgress.set(`Recoverable Gemma error detected:\n${message}\n\nInterrupting any stuck backend job and clearing pending queue...`, 100);
    await cancelComfyExecutionAndWaitIdle((status) => {
      recoveryProgress.set(`Recoverable Gemma error detected:\n${message}\n\n${status}`, 100);
    }, { shouldCancel: () => state.batchCancelled });
    recoveryProgress.set(`Recoverable Gemma error detected:\n${message}\n\nClearing memory before retry ${attempt + 1}/${maxRetries + 1}...`, 100);
    try {
      const cleanupOutput = await runClearMemoryWorkflowQuiet(recoveryProgress, `Build Full Video retry ${attempt}/${maxRetries}`, 100);
      recoveryProgress.set(`${cleanupOutput}\n\nRetrying Build Full Video in resume mode so finished work is kept...`, 100);
    } catch (cleanupError) {
      recoveryProgress.set(`Cleanup failed after recoverable Gemma error:\n${String(cleanupError?.message || cleanupError)}\n\nRetrying anyway in resume mode...`, 100);
    }
    await autoSaveSessionQuiet("Build Full Video auto retry recovery").catch(() => null);
    recoveryProgress.close(1600);
  }

  async function runGemmaImagePromptPassWithRetry(segment, progress, basePercent, label, generator, options = {}) {
    const maxRetries = Math.max(1, Number(options.maxRetries || 3));
    let lastError = null;
    let lastErrorWasRecoverable = false;
    for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
      assertBatchNotStopped();
      try {
        const retryLabel = attempt === 1 ? label : `${label} retry ${attempt}/${maxRetries}`;
        const retryOptions = {
          ...(options.generatorOptions || {}),
          clearBeforeLoad: attempt === 1 ? options.clearBeforeLoad !== false : true,
          unloadAfter: attempt === maxRetries ? options.unloadAfter !== false : false,
          seed: Math.floor((Date.now() + Math.random() * 1000000 + attempt * 9973) % 2147483647),
          temperature: Math.min(0.95, Number(options.temperature ?? 0.25) + (attempt - 1) * 0.18),
          topP: Math.max(0.72, Math.min(0.98, Number(options.topP ?? 0.95) - (attempt - 1) * 0.04)),
        };
        progress?.set(`${retryLabel}\n${attempt === 1 ? "Keeping Gemma loaded until this prompt pass finishes..." : "Fresh retry after cleanup..."}`, basePercent);
        return await generator(segment, progress, Math.min(96, basePercent + 4), retryLabel, retryOptions);
      } catch (error) {
        lastError = error;
        lastErrorWasRecoverable = isRecoverableBuildGemmaError(error);
        if (!lastErrorWasRecoverable || attempt >= maxRetries) break;
        progress?.set(`${label}: Gemma returned a recoverable bad response on attempt ${attempt}/${maxRetries}.\nInterrupting and clearing memory before retry...`, Math.min(96, basePercent + 2));
        await cancelComfyExecutionAndWaitIdle((status) => {
          progress?.set(`${label}: cancelling failed Gemma job before retry...\n${status}`, Math.min(96, basePercent + 2));
        }, { shouldCancel: () => state.batchCancelled });
        await runClearMemoryWorkflowQuiet(progress, `${label} retry ${attempt}/${maxRetries}`, Math.min(96, basePercent + 3));
      }
    }
    if (lastErrorWasRecoverable && options.textFallback !== false) {
      progress?.set(`${label}: vision Gemma kept returning junk.\nSwitching to text-only Gemma for this scene only...`, Math.min(96, basePercent + 5));
      await cancelComfyExecutionAndWaitIdle((status) => {
        progress?.set(`${label}: cancelling vision Gemma before text-only fallback...\n${status}`, Math.min(96, basePercent + 5));
      }, { shouldCancel: () => state.batchCancelled });
      await runImageMemoryCleanupQuiet(progress, `${label} text-only fallback`, Math.min(96, basePercent + 6));
      try {
        return await generateTextOnlyImagePromptFallbackForSegment(
          segment,
          progress,
          Math.min(96, basePercent + 8),
          `${label}: text-only fallback`,
          { imageMode: options.imageMode || state.imageModelMode || "zimage" },
        );
      } catch (fallbackError) {
        progress?.set(`${label}: text-only Gemma fallback also failed.\nSaving a local notes-based prompt so the batch can continue...`, Math.min(96, basePercent + 9));
        return buildEmergencyImagePromptForSegment(segment, options.imageMode || state.imageModelMode || "zimage");
      }
    }
    throw lastError || new Error(`${label}: failed to create a usable prompt.`);
  }

  function imageTriggerPhraseForSegment(segment = activeSegment(), imageMode = state.imageModelMode) {
    if (!segment) return state.imageTriggerPhrase || "";
    if (imageMode === "flux_klein") return fluxKleinSettingsForSegment(segment).image_trigger_phrase || "";
    if (imageMode === "nano_banana") return "";
    if (imageMode === "ernie_image") return segment.use_scene_ernie_image_settings ? (segment.ernie_image_settings?.image_trigger_phrase || "") : (state.ernieImageSettings?.image_trigger_phrase || "");
    if (imageMode === "krea2_2pass") return segment.use_scene_krea2_2pass_settings ? (segment.krea2_2pass_settings?.image_trigger_phrase || "") : (state.krea2TwoPassSettings?.image_trigger_phrase || "");
    return segment.use_scene_zimage_settings ? (segment.zimage_settings?.image_trigger_phrase || "") : (state.zimageSettings?.image_trigger_phrase || "");
  }

  function applyImageTriggerToPrompt(prompt, segment = activeSegment(), imageMode = state.imageModelMode, options = {}) {
    return applyTriggerPhrase(prompt, imageTriggerPhraseForSegment(segment, imageMode), options);
  }

  function syncSegmentT2IPrompt(segment, prompt) {
    if (!segment) return "";
    const cleanPrompt = unwrapStructuredImagePrompt(prompt);
    segment.t2i_prompt = cleanPrompt;
    segment.flux_prompt = cleanPrompt;
    segment.nb_prompt = cleanPrompt;
    segment.flow_gpt_prompt = cleanPrompt;
    segment.enhance_prompt = cleanPrompt;
    if (segment.id === activeSegment()?.id) {
      t2iPrompt.value = cleanPrompt;
      ernieT2IPrompt.value = cleanPrompt;
      krea2TwoPassT2IPrompt.value = cleanPrompt;
      fluxPrompt.value = cleanPrompt;
      nbPrompt.value = cleanPrompt;
      flowGptPrompt.value = cleanPrompt;
      zEnhancePromptPreview.value = cleanPrompt;
    }
    return cleanPrompt;
  }

  function syncSegmentFlowGptPrompt(segment, prompt, options = {}) {
    if (!segment) return "";
    const cleanPrompt = options.preserveTypingWhitespace ? String(prompt || "") : unwrapStructuredImagePrompt(prompt);
    segment.flow_gpt_prompt = cleanPrompt;
    segment.t2i_prompt = cleanPrompt;
    segment.flux_prompt = cleanPrompt;
    segment.nb_prompt = cleanPrompt;
    segment.enhance_prompt = cleanPrompt;
    if (segment.id === activeSegment()?.id && !options.skipInputSync) {
      flowGptPrompt.value = cleanPrompt;
      t2iPrompt.value = cleanPrompt;
      fluxPrompt.value = cleanPrompt;
      nbPrompt.value = cleanPrompt;
      zEnhancePromptPreview.value = cleanPrompt;
    }
    return cleanPrompt;
  }

  function ensureSegmentT2IPromptHasTrigger(segment, imageMode = state.imageModelMode, fallback = "") {
    const rawPrompt = String(
      imageMode === "flux_klein"
        ? (segment?.flux_prompt || segment?.t2i_prompt || fallback)
        : imageMode === "nano_banana"
          ? (segment?.nb_prompt || segment?.t2i_prompt || fallback)
        : (segment?.t2i_prompt || segment?.flux_prompt || fallback)
    ).trim();
    const prompt = applyImageTriggerToPrompt(rawPrompt, segment, imageMode, { validateJunk: false });
    return syncSegmentT2IPrompt(segment, prompt);
  }

  function videoTriggerPhraseForSegment(segment = activeSegment()) {
    if (!segment) return state.videoTriggerPhrase || "";
    return segment.use_scene_i2v_video_settings ? (segment.i2v_video_settings?.video_trigger_phrase || "") : (state.i2vVideoSettings?.video_trigger_phrase || "");
  }

  function conceptPromptsTextFromSegments() {
    const prompts = {};
    state.segments.forEach((segment, index) => {
      prompts[`Prompt${index + 1}`] = String(segment?.notes || "");
    });
    return JSON.stringify(prompts, null, 2);
  }

  function i2vMotionNotesTextFromSegments() {
    const notes = {};
    state.segments.forEach((segment, index) => {
      notes[`Motion${index + 1}`] = String(segment?.i2v_notes || "");
    });
    return JSON.stringify(notes, null, 2);
  }

  function hasAnyI2VMotionNotes(segments = state.segments) {
    return Array.isArray(segments) && segments.some((segment) => String(segment?.i2v_notes || "").trim());
  }

  function effectiveVideoPerformanceModeForSegment(segment) {
    return segmentUsesNoLipSyncPerformance(segment) ? "no_lip_sync" : normalizeVideoType(state.videoType);
  }

  function videoGemmaNotesForSegment(segment) {
    const notes = String(segment?.i2v_notes || "").trim();
    const facialNote = facialPerformanceNoteForSegment(segment);
    const writingRules = videoPromptWritingRulesNote();
    const performanceMode = normalizeVideoType(state.videoType);
    const noCharacterNote = segment?.no_character_present
      ? "Subject visibility: no main character is present in this scene. Do not include, mention, show, imply, or describe the mapped character/subject/performer. Build the shot from the location, props, environment, objects, atmosphere, and camera motion instead."
      : "";
    const withNoCharacterNote = (text) => [noCharacterNote, text, facialNote, writingRules, notes].filter(Boolean).join("\n\n");
    const rawLyricText = String(segment?.lyric_text || "").trim();
    const lyricText = quoteOrderedLyricCues(rawLyricText);
    if (performanceMode === "no_lip_sync" || segmentUsesNoLipSyncPerformance(segment)) {
      const visualOnlyNote = "Video Type: no lip sync / visual-only. Do not make any visible subject sing, speak, say dialogue, lip-sync, or move their mouth to the lyric. Use the lyric only as hidden mood/story context, and focus on visual acting, camera motion, environmental motion, dancing, posing, walking, or atmosphere.";
      return withNoCharacterNote(visualOnlyNote);
    }
    if (lyricText && !isInstrumentalLyricText(lyricText)) {
      if (segmentUsesNoLipSyncPerformance(segment)) {
        const brollNote = "Vocal/performance direction: b-roll / no lip-sync. Do not make any visible subject sing or lip-sync in this shot. Use visual acting, camera motion, environmental motion, dancing, posing, walking, or atmosphere instead.";
        return withNoCharacterNote(brollNote);
      }
      const singers = Array.isArray(segment?.lyric_singers) ? segment.lyric_singers.map((value) => String(value || "").trim()).filter(Boolean) : [];
      if (performanceMode === "speaking") {
        const speakingNote = singers.length
          ? `Video Type: speaking / short film. ${singers.length > 1 ? `all listed speakers (${singers.join(", ")}) should say the dialogue line in this shot. Other visible subjects should react, watch, move, or share the moment silently unless also listed.` : `only ${singers[0]} should say the dialogue line in this shot; other visible subjects should react, watch, move, or share the moment silently unless also listed.`} Do not use singing, rapping, vocals, lyrics, or music-performance wording.`
          : `Video Type: speaking / short film. ${performerLabelForSegment(segment)} should say the dialogue line naturally. Do not use singing, rapping, vocals, lyrics, or music-performance wording.`;
        return withNoCharacterNote(segment?.no_character_present ? "" : speakingNote);
      }
      const performanceNote = singers.length
        ? `Vocal/performance direction: ${singers.length > 1 ? `all listed singers (${singers.join(", ")}) must visibly sing together in this shot. Do not describe one listed singer as only listening, watching, reacting, or dancing while another listed singer sings.` : `only ${singers[0]} should visibly sing in this shot; other visible subjects should react, perform, dance, listen, or move without singing unless also listed.`} The exact lyric text will be inserted into the final prompt automatically. Do not describe visible singing as quiet; use controlled, focused, intimate, restrained, inward, tender, or simmering intensity instead.`
        : `Vocal/performance direction: ${performerLabelForSegment(segment)} should perform as if singing in sync with the audio. The exact lyric text will be inserted into the final prompt automatically. Do not describe visible singing as quiet; use controlled, focused, intimate, restrained, inward, tender, or simmering intensity instead.`;
      return withNoCharacterNote(segment?.no_character_present ? "" : performanceNote);
    }
    if (!isInstrumentalLyricText(lyricText)) return withNoCharacterNote("");
    const instrumentalNote = "Lyric/performance status: instrumental / no sung lyrics. In the final prompt, do not mention singing, lip-syncing, mouth movement, instrumental status, or no-vocal status. Use visual acting, camera motion, environmental motion, dancing, posing, walking, or atmosphere instead.";
    return withNoCharacterNote(instrumentalNote);
  }

  function idLoraSpeechTextForSegment(segment) {
    const idContext = idLoraSceneContext(segment);
    const lyricText = flattenLyricForPrompt(quoteOrderedLyricCues(String(idContext.dialogue || segment?.lyric_text || "").trim()));
    if (lyricText && !isInstrumentalLyricText(lyricText)) return lyricText;
    return "";
  }

  function idLoraGemmaNotesForSegment(segment, baseNotes = videoGemmaNotesForSegment(segment)) {
    const idContext = idLoraSceneContext(segment);
    const speech = idLoraSpeechTextForSegment(segment);
    return [
      idContext.contextText ? `ID-LoRA Ref Builder scene casting:\n${idContext.contextText}` : "",
      speech ? `Exact required [SPEECH] line:\n${speech}` : "No exact speech line was provided. Write one short natural spoken line that fits the scene.",
      "ID-LoRA output must include non-empty [VISUAL], [SPEECH], and [SOUNDS] sections.",
      baseNotes,
    ].filter(Boolean).join("\n\n");
  }

  function storyboardVideoExtraNotesForSegment(segment, sceneOverride = null) {
    if (!segment) return "";
    const scene = sceneOverride || storyboardScenePayload().find((item) => item.id === segment.id) || {};
    const defaults = normalizeBuilderStoryboardDefaults(state.builderStoryboardDefaults);
    const storyLayer = normalizeBuilderStoryLayer(state.builderStoryLayer);
    const add = (parts, title, value) => {
      const text = String(value || "").trim();
      if (text) parts.push(`${title}:\n${text}`);
    };
    const parts = [];
    add(parts, "Storyboard scene story beat", scene.story_beat || segment.story_beat);
    add(parts, "FLF storyboard start state", scene.flf_start_state || segment.flf_start_state);
    add(parts, "FLF storyboard continuous transformation", scene.flf_transformation || segment.flf_transformation);
    add(parts, "FLF storyboard end state", scene.flf_end_state || segment.flf_end_state);
    add(parts, "FLF storyboard carry-forward continuity", scene.flf_carry_forward || segment.flf_carry_forward);
    const customMotionSummary = String(scene.motion_summary || segment.i2v_notes || segment.video_notes || "").trim();
    add(parts, "Storyboard motion/video summary", customMotionSummary);
    add(parts, "Storyboard still shot direction", scene.shot_type || segment.shot_type);
    if (!customMotionSummary) add(parts, "Storyboard camera motion", scene.camera_motion || segment.camera_motion || segment.motion_preset);
    add(parts, "Storyboard camera motion speed guidance", defaults.camera_guidance || builderMotionSpeedGuidance(defaults.camera_motion_speed, "camera"));
    if (normalizeProjectVideoEngine(state.projectVideoEngine) === "minimax_h3") {
      const cutPlan = storyboardCutPlanForDuration(timelineSegmentDuration(segment), defaults.minimax_h3_cut_frequency);
      add(parts, "Mandatory MiniMax editing / cut plan", cutPlan.instruction);
    }
    add(parts, "Storyboard character motion guidance", segment.character_motion || defaults.character_guidance || builderMotionSpeedGuidance(defaults.character_motion_speed, "character"));
    const performanceStyle = String(segment.performance_style || defaults.performance_style || "").trim();
    const performancePreset = storyboardPerformancePreset(performanceStyle) || {};
    if (performanceStyle !== "off") add(parts, "Storyboard performance direction", performancePreset.direction || performancePreset.label || performanceStyle);
    add(parts, "Storyboard facial performance direction", resolvedFacialPerformanceText(segment));
    add(parts, "Storyboard lyric section", scene.lyric_section || segment.lyric_section);
    add(parts, "Storyboard global consistency phrase", defaults.global_consistency_phrase);
    if (storyLayer.enabled !== false) {
      add(parts, "Storyboard user story arc", storyLayer.user_story_arc);
      add(parts, "Storyboard song story brief", storyLayer.song_story_brief);
      add(parts, "Storyboard lyric story strength", `${storyLayer.lyric_story_strength}/10`);
    }
    add(parts, "Storyboard first-frame visual inventory", scene.image_prompt || sceneConceptPromptText(segment));
    return parts.join("\n\n");
  }

  function performerLabelForSegment(segment) {
    if (segment?.no_character_present) return "the visible subject";
    const singers = Array.isArray(segment?.lyric_singers)
      ? segment.lyric_singers.map((value) => cleanNaturalSubjectLabel(value)).filter(Boolean)
      : String(segment?.lyric_singers || "").split(/[,;\n]+/).map((value) => cleanNaturalSubjectLabel(value)).filter(Boolean);
    if (singers.length) return singers.join(" and ");
    const directRefs = Array.isArray(segment?.subject_refs) ? segment.subject_refs : [];
    for (const subject of directRefs) {
      const label = cleanNaturalSubjectLabel([
        subject?.trigger_phrase,
        subject?.name,
        subject?.description,
      ].filter(Boolean).join(": "));
      if (label) return label;
    }
    const mappedSubjects = String(segment?.mapped_subjects || segment?.subject || "")
      .split(/[,;\n]+/)
      .map((value) => cleanNaturalSubjectLabel(value))
      .filter(Boolean);
    if (mappedSubjects.length) return mappedSubjects.join(" and ");
    const mappedText = cleanNaturalSubjectLabel(segmentMappedSubjectText(segment));
    if (mappedText) return mappedText;
    const visualText = [segment?.t2i_prompt, segment?.flux_prompt, segment?.notes, segment?.flux_notes].map((value) => String(value || "")).join(" ");
    const inferred = cleanNaturalSubjectLabel(visualText);
    return inferred || "the visible subject";
  }

  function replaceGenericSubjectLabels(prompt, segment) {
    const label = performerLabelForSegment(segment);
    if (!label || /^the visible subject$/i.test(label)) return String(prompt || "").trim();
    return String(prompt || "")
      .replace(/\bthe visible subject\b/gi, label)
      .replace(/\bvisible subject\b/gi, label)
      .replace(/\bthe subject\b/gi, label)
      .replace(/\bthe character\b/gi, label)
      .replace(/\bthe person\b/gi, label)
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  function vocalDirectiveForSegment(segment) {
    if (segment?.no_character_present) return "";
    const performanceMode = normalizeVideoType(state.videoType);
    if (performanceMode === "no_lip_sync" || segmentUsesNoLipSyncPerformance(segment)) return "";
    const rawLyricText = String(segment?.lyric_text || "").trim();
    const lyricText = quoteOrderedLyricCues(rawLyricText).trim();
    if (!lyricText) return "";
    if (isInstrumentalLyricText(lyricText) || segmentUsesNoLipSyncPerformance(segment)) {
      return "";
    }
    const singers = Array.isArray(segment?.lyric_singers) ? segment.lyric_singers.map((value) => cleanNaturalSubjectLabel(value)).filter(Boolean) : [];
    const performer = singers.length ? singers.join(" and ") : performerLabelForSegment(segment);
    const pluralPerformers = singers.length > 1 || /\b(group|duet|all visible)\b/i.test(performer);
    if (lyricTextHasCueLabels(lyricText)) {
      if (performanceMode === "speaking") {
        return `${performer} ${pluralPerformers ? "say" : "says"} the exact dialogue cues naturally: ${lyricText.replace(/\s*\n\s*/g, " ")}`;
      }
      return `${performer} ${pluralPerformers ? "perform" : "performs"} the exact vocal cues in sync with the audio: ${lyricText.replace(/\s*\n\s*/g, " ")}`;
    }
    const cleanLyric = flattenLyricForPrompt(lyricText);
    if (performanceMode === "speaking") {
      return `${performer} ${pluralPerformers ? "say" : "says"} "${cleanLyric}" naturally.`;
    }
    return `${performer} visibly ${pluralPerformers ? "sing" : "sings"} "${cleanLyric}" in sync with the audio.`;
  }

  function i2vAutoChainEnabled() {
    state.continuityMode = normalizeContinuityMode(state.continuityMode, state.autoChainLastFrame);
    return state.continuityMode === "i2v_chain";
  }

  function img2imgContinuityEnabled() {
    state.continuityMode = normalizeContinuityMode(state.continuityMode, state.autoChainLastFrame);
    return state.continuityMode === "img2img";
  }

  function imageModeSupportsImg2ImgContinuity(imageMode = state.imageModelMode || "zimage") {
    return ["zimage", "ernie_image", "krea2_2pass"].includes(String(imageMode || "").trim());
  }

  function imageModeImg2ImgContinuityLabel(imageMode = state.imageModelMode || "zimage") {
    return {
      zimage: "ZImage",
      ernie_image: "Ernie",
      krea2_2pass: "Krea 2",
      flux_klein: "Flux/Klein",
      nano_banana: "NanoBanana",
      flow_gpt: "Flow/GPT",
    }[String(imageMode || "").trim()] || "current image model";
  }

  function vocalClauseForSegment(segment) {
    if (segment?.no_character_present) return null;
    const performanceMode = normalizeVideoType(state.videoType);
    if (performanceMode === "no_lip_sync" || segmentUsesNoLipSyncPerformance(segment)) return null;
    const rawLyricText = String(segment?.lyric_text || "").trim();
    const lyricText = quoteOrderedLyricCues(rawLyricText).trim();
    if (!lyricText || isInstrumentalLyricText(lyricText) || segmentUsesNoLipSyncPerformance(segment)) return null;
    const singers = Array.isArray(segment?.lyric_singers) ? segment.lyric_singers.map((value) => cleanNaturalSubjectLabel(value)).filter(Boolean) : [];
    const performer = singers.length ? singers.join(" and ") : performerLabelForSegment(segment);
    const pluralPerformers = singers.length > 1 || /\b(group|duet|all visible)\b/i.test(performer);
    if (lyricTextHasCueLabels(lyricText)) {
      const cueText = lyricText.replace(/\s*\n\s*/g, " ");
      return {
        performer,
        lyricText,
        cueText,
        clause: performanceMode === "speaking"
          ? `who ${pluralPerformers ? "say" : "says"} the exact dialogue cues naturally: ${cueText}`
          : `who ${pluralPerformers ? "perform" : "performs"} the exact vocal cues in sync with the audio: ${cueText}`,
      };
    }
    const cleanLyric = flattenLyricForPrompt(lyricText);
    return {
      performer,
      lyricText: cleanLyric,
      singleLineLyric: cleanLyric,
      clause: performanceMode === "speaking"
        ? `who ${pluralPerformers ? "say" : "says"} "${cleanLyric}" naturally`
        : `who ${pluralPerformers ? "are" : "is"} singing "${cleanLyric}" in sync with the audio`,
    };
  }

  function applyVocalDirectiveToVideoPrompt(prompt, segment, options = {}) {
    const performanceMode = normalizeVideoType(state.videoType);
    const isVisualOnly = performanceMode === "no_lip_sync" || segmentUsesNoLipSyncPerformance(segment);
    const base = String(prompt || "")
      .replace(/^\s*No visible subject sings or lip-syncs in this shot;\s*this is an instrumental or no-vocal visual moment\.\s*/i, "")
      .trim();
    const facialPresetKey = String(segment?.facial_performance || state.defaultFacialPerformance || "").trim();
    const facialCustomText = String(segment?.facial_performance_custom || state.defaultFacialPerformanceCustom || "").trim();
    // Default natural is guidance for the one-pass LLM request, not literal
    // boilerplate to append to the finished generation prompt. Explicit presets
    // and custom facial direction retain the existing final-output behavior.
    const facialText = options.includeFacialPerformance === false || (!facialPresetKey && !facialCustomText)
      ? ""
      : facialPerformanceNoteForSegment(segment);
    const appendFacial = (text) => {
      const raw = String(text || "").trim();
      const modeClean = isVisualOnly ? removeNegativeAndVocalWordingFromVisualPrompt(raw) : raw;
      const clean = performanceMode === "singing" && vocalDirectiveForSegment(segment)
        ? replaceGenericSubjectLabels(removeQuietFromSingingPrompt(modeClean), segment)
        : replaceGenericSubjectLabels(modeClean, segment);
      if (!facialText || (/facial performance direction/i.test(clean) && /blink/i.test(clean) && /\beye\s+movement\b|\beyes?\s+(?:shift|move|track|glance|flick|dart)\b/i.test(clean))) return clean;
      return clean ? `${clean} ${facialText}` : facialText;
    };
    const directive = vocalDirectiveForSegment(segment);
    if (!directive) {
      const rawLyricText = String(segment?.lyric_text || "").trim();
      const lyricText = quoteOrderedLyricCues(rawLyricText).trim();
      if (isVisualOnly || isInstrumentalLyricText(lyricText)) {
        return appendFacial(base
          .replace(/^\s*(?:the visible subject|the subject|[^.]{1,90}?)\s+(?:visibly\s+)?(?:sings?|singing|lip-syncs?|lip syncs?|lip-syncing)\s+(?:"[^"]*"|'[^']*'|“[^”]*”)?\s*(?:in sync with the audio)?\.\s*/i, "")
          .replace(/\b(?:visibly\s+)?(?:sings?|singing|lip-syncs?|lip syncs?|lip-syncing)\s+(?:"[^"]*"|'[^']*'|“[^”]*”)?\s*(?:in sync with the audio)?/gi, "moves naturally")
          .trim());
      }
      return appendFacial(base);
    }
    if (base.toLowerCase().startsWith(directive.toLowerCase())) return appendFacial(base);
    const vocalClause = vocalClauseForSegment(segment);
    if (vocalClause) {
      const lowerBase = base.toLowerCase();
      const lowerLyric = vocalClause.lyricText.toLowerCase();
      if (lowerLyric && lowerBase.includes(lowerLyric)) return appendFacial(base);
      if (vocalClause.cueText) {
        const cueText = vocalClause.cueText;
        const visibleSinging = /\bvisibly\s+(?:singing|sings|lip-syncing|lip syncs|lip-syncs)(?:\s+together)?\b(?!\s*(?::|["'“”]))/i;
        if (visibleSinging.test(base)) {
          return appendFacial(base.replace(visibleSinging, `visibly singing the exact vocal cues in sync with the audio: ${cueText}`));
        }
        const duetPhrase = /\b(?:synchronized\s+)?vocal duet\b(?!\s*(?::|["'“”]))/i;
        if (duetPhrase.test(base)) {
          return appendFacial(base.replace(duetPhrase, `vocal duet with exact cues in sync with the audio: ${cueText}`));
        }
      }
      if (vocalClause.singleLineLyric) {
        const vocalVerb = /\b(singing|sings|lip-syncing|lip syncs|lip-syncs|speaking|speaks|talking|says|saying)\b(?!\s*["'“”])/i;
        if (vocalVerb.test(base)) {
          return appendFacial(base.replace(vocalVerb, (match) => {
            const cleanMatch = String(match || "").toLowerCase();
            const verb = performanceMode === "speaking"
              ? (/\bspeaks|says\b/i.test(cleanMatch) ? "says" : "saying")
              : cleanMatch.includes("lip") ? "singing" : match;
            return `${verb} "${vocalClause.singleLineLyric}"`;
          }));
        }
      }
      const performerPattern = escapeRegExp(vocalClause.performer);
      const opener = new RegExp(`^(${performerPattern})(\\s+(?:in|inside|at|on|within|during)\\b)`, "i");
      if (opener.test(base)) {
        return appendFacial(base.replace(opener, `$1, ${vocalClause.clause},$2`));
      }
    }
    if (options.suppressPrefix) return appendFacial(base);
    return appendFacial(base ? `${directive} ${base}` : directive);
  }

  function buildEmergencyImagePromptForSegment(segment, imageMode = state.imageModelMode || "zimage") {
    const context = imageMode === "nano_banana" ? nbImageSettingsForSegment(segment).reference_context : fluxReferenceContextForSegment(segment);
    const source = [
      segment?.notes,
      segment?.flux_notes,
      segment?.nb_notes,
      segmentMappedSubjectText(segment),
      segmentMappedLocationText(segment),
      segment?.story_beat,
      segment?.shot_type,
      context?.subject_description,
      context?.location_name,
      context?.location_description,
    ].map((value) => String(value || "").trim()).filter(Boolean).join(", ").replace(/\s+/g, " ").trim();
    const lyricHint = sceneLyricTextForPromptValidation(segment) ? "symbolic mood inspired by the scene lyric, " : "";
    const base = source || `${lyricHint}cinematic image for ${sceneDisplayName(segment, segmentIndexInfo(segment).index)}`;
    const clipped = base.length > 850 ? `${base.slice(0, 847).trim()}...` : base;
    const prompt = `cinematic image, ${clipped}, visually specific composition, clear subject, atmospheric lighting, high detail`;
    syncSegmentT2IPrompt(segment, applyImageTriggerToPrompt(prompt, segment, imageMode, { validateJunk: false }));
    render();
    return { prompt: segment.t2i_prompt, used_local_notes_fallback: true };
  }

  function resolvedFacialPerformanceText(segment = null) {
    if (segment?.no_character_present) return "";
    const presetKey = String(segment?.facial_performance || state.defaultFacialPerformance || "").trim();
    const custom = String(segment?.facial_performance_custom || state.defaultFacialPerformanceCustom || "").trim();
    if (presetKey === "off") return "";
    const preset = storyboardFacialPerformancePreset(presetKey);
    const base = presetKey === "custom" && custom
      ? custom
      : [preset.direction, custom].filter(Boolean).join(" ");
    let facialText = base || "Use natural expressive facial performance with visible emotion, engaged eyes, active brows, subtle cheek and jaw movement, subtle eye movement, and occasional natural blinking.";
    if (!/blink/i.test(facialText)) facialText = `${facialText} Include occasional natural blinking.`;
    if (!/\beye\s+movement\b|\beyes?\s+(?:shift|move|track|glance|flick|dart)\b/i.test(facialText)) facialText = `${facialText} Include subtle natural eye movement.`;
    return facialText;
  }

  function facialPerformanceNoteForSegment(segment = null) {
    const facialText = resolvedFacialPerformanceText(segment);
    if (!facialText) return "";
    return `Facial performance direction: ${facialText}`;
  }

  return {
    applyImageTriggerToPrompt, applyVocalDirectiveToVideoPrompt, conceptPromptsTextFromSegments,
    effectiveVideoPerformanceModeForSegment, ensureSegmentT2IPromptHasTrigger,
    facialPerformanceNoteForSegment, hasAnyI2VMotionNotes, i2vAutoChainEnabled,
    i2vMotionNotesTextFromSegments, idLoraGemmaNotesForSegment, idLoraSpeechTextForSegment,
    imageModeImg2ImgContinuityLabel, imageModeSupportsImg2ImgContinuity, img2imgContinuityEnabled,
    recoverFromBuildGemmaError, runGemmaImagePromptPassWithRetry, storyboardVideoExtraNotesForSegment,
    syncSegmentFlowGptPrompt, syncSegmentT2IPrompt, videoGemmaNotesForSegment, videoTriggerPhraseForSegment,
  };
}
