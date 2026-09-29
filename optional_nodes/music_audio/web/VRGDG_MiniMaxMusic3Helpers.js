import { app } from "../../scripts/app.js";

const NODE_NAME = "VRGDG_Music3PromptBuilder";
const PRESET_URL = new URL("./VRGDG_Music3CaptionPresets.json", import.meta.url);
const TUNING_NODE_NAME = "VRGDG_Music3TuningPreset";
const TUNING_DETAILS_URL = new URL("./VRGDG_Music3TuningDetails.json", import.meta.url);

let presetPromise;
let tuningDetailsPromise;

function loadPresets() {
  if (!presetPromise) {
    presetPromise = fetch(PRESET_URL)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .catch((error) => {
        presetPromise = undefined;
        console.error("[VRGDG Music3] Could not load caption presets:", error);
        return {};
      });
  }
  return presetPromise;
}

function loadTuningDetails() {
  if (!tuningDetailsPromise) {
    tuningDetailsPromise = fetch(TUNING_DETAILS_URL)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .catch((error) => {
        tuningDetailsPromise = undefined;
        console.error("[VRGDG Music3] Could not load tuning details:", error);
        return {};
      });
  }
  return tuningDetailsPromise;
}

const FIELD_MAP = {
  singer: "singer_name",
  style: "style_name",
  genre: "genre_subgenre",
  mood: "mood_themes",
  instruments: "instruments",
  vocal: "vocal_qualities",
  production: "production",
  tempo: "tempo_range",
  arc: "song_arc",
  avoid: "avoid_characteristics",
};

const VOCAL_PROFILES = {
  "Female Lead": {
    singer: "solo female lead vocalist",
    vocal: "expressive female lead vocal with a natural register and clear melodic delivery",
  },
  "Female Alto / Mezzo": {
    singer: "solo female alto-to-mezzo lead vocalist",
    vocal: "expressive female alto-to-mezzo lead, warm lower register, controlled chest voice and natural emotional detail",
  },
  "Female Soprano / Bright": {
    singer: "solo female soprano lead vocalist",
    vocal: "bright clear female soprano lead, agile upper register, controlled vibrato and clean sustained notes",
  },
  "Female Rock / Rasp": {
    singer: "solo female rock lead vocalist",
    vocal: "powerful female rock lead, conversational lower verses, open-throated chorus, controlled rasp and natural emotional cracks",
  },
  "Female Soft / Airy": {
    singer: "solo female intimate lead vocalist",
    vocal: "soft airy female lead, close breath detail, gentle dynamics, natural vibrato and restrained harmonies",
  },
  "Male Lead": {
    singer: "solo male lead vocalist",
    vocal: "expressive male lead vocal with a natural register and clear melodic delivery",
  },
  "Male Baritone": {
    singer: "solo male baritone lead vocalist",
    vocal: "warm resonant male baritone lead, grounded lower register, controlled projection and natural phrasing",
  },
  "Male Tenor": {
    singer: "solo male tenor lead vocalist",
    vocal: "clear expressive male tenor lead, strong upper register, controlled vibrato and open melodic chorus",
  },
  "Male Rock / Grit": {
    singer: "solo male rock lead vocalist",
    vocal: "powerful male rock lead, firm chest voice, gritty chorus, controlled rasp and emotionally direct phrasing",
  },
  "Male Soft / Intimate": {
    singer: "solo male intimate lead vocalist",
    vocal: "soft close male lead, warm breath detail, restrained projection, gentle dynamics and minimal doubling",
  },
  "Androgynous / Neutral": {
    singer: "solo androgynous lead vocalist",
    vocal: "androgynous gender-neutral lead vocal, balanced register, natural phrasing and an intimate unforced delivery",
  },
  "Female + Male Duet": {
    singer: "female and male duet vocalists",
    vocal: "alternating female and male lead vocals, clear call-and-response, shared chorus harmonies and distinct vocal registers",
  },
};

function widget(node, name) {
  return node?.widgets?.find((item) => item.name === name);
}

async function applyPreset(node, presetName, onlyBlank = false) {
  if (!node || presetName === "Custom / fields only") return;
  const presets = await loadPresets();
  const values = presets[presetName];
  if (!values) return;

  for (const [jsonName, widgetName] of Object.entries(FIELD_MAP)) {
    const target = widget(node, widgetName);
    if (!target) continue;
    if (onlyBlank && String(target.value ?? "").trim()) continue;
    target.value = values[jsonName] ?? "";
  }

  const extra = widget(node, "extra_direction");
  if (extra && (!onlyBlank || !String(extra.value ?? "").trim())) {
    extra.value = values.extra ?? "";
  }

  if (!onlyBlank) {
    const vocalProfile = widget(node, "vocal_profile");
    if (vocalProfile) await applyVocalProfile(node, String(vocalProfile.value));
  }

  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

async function applyVocalProfile(node, profileName) {
  if (!node || profileName === "Custom / keep fields") return;

  const presets = await loadPresets();
  const styleName = String(widget(node, "preset")?.value ?? "");
  const style = presets[styleName] ?? {};
  let singer = style.singer ?? "solo lead vocalist";
  let vocal = style.vocal ?? "natural expressive lead vocal";

  if (profileName !== "Use Style Default") {
    const profile = VOCAL_PROFILES[profileName];
    if (!profile) return;
    singer = profile.singer;
    vocal = style.vocal ? `${profile.vocal}, adapted to ${style.vocal}` : profile.vocal;
  }

  const singerWidget = widget(node, "singer_name");
  const vocalWidget = widget(node, "vocal_qualities");
  if (singerWidget) singerWidget.value = singer;
  if (vocalWidget) vocalWidget.value = vocal;
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "vrgdg.minimax.music3.caption.presets",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalCreated?.apply(this, args);
      const owner = this;
      const presetWidget = widget(owner, "preset");
      if (presetWidget) {
        const originalCallback = presetWidget.callback;
        presetWidget.callback = function (value) {
          const callbackResult = originalCallback?.apply(this, arguments);
          void applyPreset(owner, String(value), false);
          return callbackResult;
        };
        setTimeout(() => void applyPreset(owner, String(presetWidget.value), true), 0);
      }
      const vocalProfileWidget = widget(owner, "vocal_profile");
      if (vocalProfileWidget) {
        const originalVocalCallback = vocalProfileWidget.callback;
        vocalProfileWidget.callback = function (value) {
          const callbackResult = originalVocalCallback?.apply(this, arguments);
          void applyVocalProfile(owner, String(value));
          return callbackResult;
        };
      }
      return result;
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = originalConfigure?.apply(this, args);
      setTimeout(() => {
        const presetWidget = widget(this, "preset");
        if (presetWidget) void applyPreset(this, String(presetWidget.value), true);
      }, 0);
      return result;
    };
  },
});

async function updateTuningTooltip(node, presetWidget) {
  if (!node || !presetWidget) return;
  const details = await loadTuningDetails();
  const detail = details[String(presetWidget.value)] ?? "Select a behavior preset for coordinated Music3 planning and rendering values.";
  const text = `${presetWidget.value}\n\n${detail}\n\nCaption preset = WHAT style. Tuning preset = HOW Music3 interprets it.`;
  presetWidget.options ??= {};
  presetWidget.options.tooltip = text;
  presetWidget.tooltip = text;
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "vrgdg.minimax.music3.tuning.tooltips",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TUNING_NODE_NAME) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalCreated?.apply(this, args);
      const owner = this;
      const presetWidget = widget(owner, "preset");
      if (presetWidget) {
        const originalCallback = presetWidget.callback;
        presetWidget.callback = function () {
          const callbackResult = originalCallback?.apply(this, arguments);
          void updateTuningTooltip(owner, presetWidget);
          return callbackResult;
        };
        setTimeout(() => void updateTuningTooltip(owner, presetWidget), 0);
      }
      return result;
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = originalConfigure?.apply(this, args);
      setTimeout(() => void updateTuningTooltip(this, widget(this, "preset")), 0);
      return result;
    };
  },
});
