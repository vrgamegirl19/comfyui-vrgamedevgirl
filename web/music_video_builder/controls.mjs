import { app } from "../../../scripts/app.js";
import { BUILDER_FONT_STACK, BUY_ME_A_COFFEE_URL } from "./constants.mjs";

export function getWidget(node, name) {
  return (node?.widgets || []).find((widget) => widget?.name === name);
}

export function setWidgetValue(node, name, value) {
  const widget = getWidget(node, name);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse);
  const index = (node.widgets || []).indexOf(widget);
  if (Array.isArray(node.widgets_values) && index >= 0) node.widgets_values[index] = value;
  app.graph?.setDirtyCanvas?.(true, true);
}

export function makeButton(label, kind = "neutral") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.style.cssText = `
    border: 1px solid ${kind === "primary" ? "#0891b2" : "#3f3f46"};
    border-radius: 6px;
    background: ${kind === "primary" ? "#06b6d4" : "#27272a"};
    color: ${kind === "primary" ? "#082f49" : "#f4f4f5"};
    font-family: ${BUILDER_FONT_STACK};
    font-size: 12px;
    font-weight: 600;
    padding: 8px 11px;
    cursor: pointer;
    white-space: nowrap;
    line-height: 1.2;
  `;
  return button;
}

export const COMPACT_TOOLBAR_ICONS = {
  save: '<path d="M13 2H5a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V8z"/><path d="M13 2v6h6"/><path d="M8 13h8v6H8z"/>',
  wizard: '<path d="m15 4 5 5L7 22l-5-5Z"/><path d="m14 5 5 5"/><path d="M6 3v4"/><path d="M4 5h4"/><path d="M19 14v4"/><path d="M17 16h4"/>',
  auto: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="m7 5 2 4 2-4 2 4 2-4 2 4 2-4"/><path d="M8 13h4"/><path d="M10 11v4"/><path d="M17 12v4"/><path d="M15 14h4"/>',
  story: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 4v16"/><path d="M16 4v16"/><path d="M3 9h5"/><path d="M16 15h5"/>',
  reference: '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m5 18 4-4 3 3 2-2 5 3"/>',
  mapping: '<circle cx="5" cy="12" r="2"/><circle cx="12" cy="5" r="2"/><circle cx="19" cy="12" r="2"/><circle cx="12" cy="19" r="2"/><path d="m6.5 10.5 4-4"/><path d="m13.5 6.5 4 4"/><path d="m17.5 13.5-4 4"/><path d="m10.5 17.5-4-4"/>',
  brain: '<path d="M9.5 4A3.5 3.5 0 0 0 6 7.5v.2A3.5 3.5 0 0 0 4 11v2a3.5 3.5 0 0 0 2 3.2v.3A3.5 3.5 0 0 0 9.5 20H11V4Z"/><path d="M14.5 4A3.5 3.5 0 0 1 18 7.5v.2a3.5 3.5 0 0 1 2 3.3v2a3.5 3.5 0 0 1-2 3.2v.3a3.5 3.5 0 0 1-3.5 3.5H13V4Z"/><path d="M8 9h3"/><path d="M13 14h3"/>',
  prompt: '<path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4Z"/><path d="M8 9h8"/><path d="M8 13h5"/>',
  download: '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
  memory: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',
  fullscreen: '<path d="M8 3H3v5"/><path d="m3 3 6 6"/><path d="M16 3h5v5"/><path d="m21 3-6 6"/><path d="M8 21H3v-5"/><path d="m3 21 6-6"/><path d="M16 21h5v-5"/><path d="m21 21-6-6"/>',
  restore: '<path d="M8 3H3v5"/><path d="m3 3 6 6"/><path d="M16 3h5v5"/><path d="m21 3-6 6"/><path d="M8 21H3v-5"/><path d="m3 21 6-6"/><path d="M16 21h5v-5"/><path d="m21 21-6-6"/>',
  close: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
};

export function styleCompactToolbarButton(button, options = {}) {
  const lines = Array.isArray(options.lines) ? options.lines.filter(Boolean) : [String(options.label || button.textContent || "")];
  const accessibleLabel = String(options.ariaLabel || lines.join(" ") || button.textContent || "Toolbar button");
  const width = Math.max(36, Number(options.width || (options.iconOnly ? 40 : 58)));
  button.style.cssText += `
    width:${width}px;
    min-width:${width}px;
    height:${options.iconOnly ? 42 : 54}px;
    box-sizing:border-box;
    padding:${options.iconOnly ? "6px" : "5px 4px"};
    display:inline-flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    gap:2px;
    white-space:normal;
    line-height:1.08;
    text-align:center;
    font-family:${BUILDER_FONT_STACK};
    font-size:11px;
    font-weight:500;
    letter-spacing:0;
  `;
  button.setAttribute("aria-label", accessibleLabel);
  if (options.title) button.title = options.title;
  button.replaceChildren();
  if (options.icon && COMPACT_TOOLBAR_ICONS[options.icon]) {
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("width", options.iconOnly ? "20" : "15");
    icon.setAttribute("height", options.iconOnly ? "20" : "15");
    icon.setAttribute("fill", "none");
    icon.setAttribute("stroke", "currentColor");
    icon.setAttribute("stroke-width", "2");
    icon.setAttribute("stroke-linecap", "round");
    icon.setAttribute("stroke-linejoin", "round");
    icon.setAttribute("aria-hidden", "true");
    icon.innerHTML = COMPACT_TOOLBAR_ICONS[options.icon];
    button.append(icon);
  }
  if (!options.iconOnly) {
    const label = document.createElement("span");
    label.style.cssText = "display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:11px;font-weight:500;line-height:1.08;letter-spacing:0;";
    for (const line of lines) {
      const row = document.createElement("span");
      row.textContent = line;
      label.append(row);
    }
    button.append(label);
  }
  return button;
}

function compactButtonLabelText(label) {
  const text = String(label || "");
  const map = {
    "Text to Video": "T2V",
    "Image to Video": "I2V",
    "Reference to Video": "Ref to\nVideo",
    "Video to Video": "V2V",
    "Image Settings": "Image\nSettings",
    "Video Settings": "Video\nSettings",
    "Singer Assignment": "Singer\nAssignment",
    "Speaker Assignment": "Speaker\nAssignment",
    "LLM Prompting": "LLM\nPrompting",
  };
  return map[text] || text;
}

export function applyCompactButtonLabel(button, label, options = {}) {
  const compact = options.noMap ? String(label || "") : compactButtonLabelText(label);
  button.textContent = compact;
  button.title = options.title || String(label || compact).replace(/\s+/g, " ").trim();
  button.style.minWidth = `${Number(options.minWidth || 0)}px`;
  button.style.whiteSpace = "pre-line";
  button.style.lineHeight = "1.05";
  button.style.textAlign = "center";
  button.style.padding = options.padding || "7px 8px";
  return button;
}

export function makeGptLinkButton(label, url) {
  const button = makeButton(label, "primary");
  button.title = url;
  button.onclick = () => window.open(url, "_blank", "noopener,noreferrer");
  return button;
}

export function makeBuyMeACoffeeButton() {
  const wrapper = document.createElement("span");
  wrapper.style.cssText = `
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:100%;
    min-height:52px;
    line-height:1;
    margin-bottom:4px;
  `;

  const fallback = document.createElement("a");
  fallback.href = BUY_ME_A_COFFEE_URL;
  fallback.target = "_blank";
  fallback.rel = "noopener noreferrer";
  fallback.title = "Support VRGameDevGirl on Buy Me a Coffee";
  fallback.setAttribute("aria-label", "Support VRGameDevGirl on Buy Me a Coffee");
  fallback.textContent = "\u2615 Buy me a coffee";
  fallback.style.cssText = `
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:100%;
    min-height:52px;
    box-sizing:border-box;
    border:1px solid #000000;
    border-radius:8px;
    background:#ffdd00;
    color:#000000;
    font-family:Lato, Arial, sans-serif;
    font-size:22px;
    font-weight:900;
    padding:10px 18px;
    text-decoration:none;
    white-space:nowrap;
  `;
  wrapper.append(fallback);

  const script = document.createElement("script");
  script.type = "text/javascript";
  script.src = "https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js";
  script.dataset.name = "bmc-button";
  script.dataset.slug = "vrgamedevgirl";
  script.dataset.color = "#FFDD00";
  script.dataset.emoji = "\u2615";
  script.dataset.font = "Lato";
  script.dataset.text = "Buy me a coffee";
  script.dataset.outlineColor = "#000000";
  script.dataset.fontColor = "#000000";
  script.dataset.coffeeColor = "#ffffff";
  script.onload = () => {
    const officialButton = Array.from(wrapper.querySelectorAll(".bmc-button, a[href*='buymeacoffee.com']"))
      .find((element) => element !== fallback);
    if (officialButton) {
      officialButton.style.width = "100%";
      officialButton.style.minHeight = "52px";
      officialButton.style.boxSizing = "border-box";
      officialButton.style.borderRadius = "8px";
      officialButton.style.justifyContent = "center";
      fallback.remove();
    }
  };
  wrapper.append(script);
  return wrapper;
}

export async function copyTextToClipboard(text) {
  const value = String(text || "");
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return true;
  }
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "readonly");
  textarea.style.cssText = "position:fixed;left:-9999px;top:-9999px;opacity:0;";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  return copied;
}

export function makeInput(value = "", type = "text") {
  const input = document.createElement("input");
  input.type = type;
  input.value = value;
  input.style.cssText = `width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-family:${BUILDER_FONT_STACK};font-size:12px;font-weight:400;`;
  return input;
}

export function makeCheckbox(label, checked = false) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = `display:flex;align-items:center;gap:8px;font-family:${BUILDER_FONT_STACK};font-size:12px;color:#f4f4f5;font-weight:500;`;
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(checked);
  wrapper.append(input, document.createTextNode(label));
  return { wrapper, input };
}

export function makeSelect(options = [], value = "") {
  const select = document.createElement("select");
  select.style.cssText = `width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:8px;font-family:${BUILDER_FONT_STACK};font-size:12px;font-weight:400;`;
  for (const optionValue of options) {
    const option = document.createElement("option");
    if (optionValue && typeof optionValue === "object") {
      option.value = optionValue.value ?? optionValue.label ?? "";
      option.textContent = optionValue.label ?? option.value;
      if (optionValue.description || optionValue.direction) option.title = optionValue.description || optionValue.direction;
    } else {
      option.value = optionValue;
      option.textContent = optionValue;
    }
    select.append(option);
  }
  select.value = value;
  return select;
}

export const VIDEO_TYPE_OPTIONS = [
  { value: "singing", label: "Singing (music video)" },
  { value: "speaking", label: "Speaking (short film)" },
  { value: "no_lip_sync", label: "No lip sync" },
];

export function normalizeVideoType(value) {
  const text = String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (["speaking", "short_film", "dialogue", "dialog"].includes(text)) return "speaking";
  if (["no_lip_sync", "nolipsync", "no_lipsync", "no_sync", "silent", "visual_only"].includes(text)) return "no_lip_sync";
  return "singing";
}

export function normalizeProjectVideoEngine(value) {
  return String(value || "").trim().toLowerCase() === "minimax_h3" ? "minimax_h3" : "ltx";
}

export function makeVideoTypeSelect(value = "singing") {
  const select = makeSelect([], normalizeVideoType(value));
  for (const optionInfo of VIDEO_TYPE_OPTIONS) {
    const option = document.createElement("option");
    option.value = optionInfo.value;
    option.textContent = optionInfo.label;
    select.append(option);
  }
  select.value = normalizeVideoType(value);
  return select;
}

export function makeSearchableLoraPicker(value = "[none]") {
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:4px;position:relative;z-index:1;";
  const input = makeInput(value || "[none]");
  const list = document.createElement("div");
  list.style.cssText = "display:none;width:100%;max-height:180px;overflow:auto;border:1px solid #3f3f46;border-radius:6px;background:#18181b;box-shadow:0 8px 18px rgba(0,0,0,.32);";
  wrapper.append(input, list);
  return { wrapper, input, list, options: [], matches: [], activeIndex: -1 };
}

export function makeField(label, control) {
  const wrapper = document.createElement("label");
  wrapper.style.cssText = `display:flex;flex-direction:column;gap:5px;font-family:${BUILDER_FONT_STACK};font-size:12px;color:#d4d4d8;font-weight:500;`;
  const text = document.createElement("span");
  text.textContent = label;
  wrapper.append(text, control);
  return wrapper;
}

export function makePickerField(label, input, button) {
  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  row.append(input, button);
  return makeField(label, row);
}

export function makeEditField(label, input, button) {
  const row = document.createElement("div");
  row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;";
  button.style.padding = "8px 10px";
  row.append(input, button);
  return makeField(label, row);
}

export function makeMiniButton(label) {
  const button = makeButton(label);
  button.style.padding = "5px 8px";
  button.style.fontSize = "10px";
  button.style.borderRadius = "5px";
  return button;
}

export function makeSettingsSection(title, children = [], open = true) {
  const details = document.createElement("details");
  details.open = Boolean(open);
  details.style.cssText = "border:1px solid #3f3f46;border-radius:7px;background:#18181b;overflow:visible;";
  const summary = document.createElement("summary");
  summary.textContent = title;
  summary.style.cssText = "cursor:pointer;list-style:none;padding:10px 10px;font-size:12px;font-weight:900;color:#f4f4f5;background:linear-gradient(180deg,#34343a,#29292f);border-bottom:1px solid #4b5563;border-radius:7px 7px 0 0;";
  const body = document.createElement("div");
  body.style.cssText = "display:flex;flex-direction:column;gap:8px;padding:9px;";
  body.append(...children);
  details.append(summary, body);
  return details;
}

export function makeSettingsPanel(children = []) {
  const panel = document.createElement("div");
  panel.style.cssText = "display:flex;flex-direction:column;gap:8px;border:1px solid #303038;border-radius:7px;background:#18181b;padding:9px;";
  panel.append(...children);
  return panel;
}

export function makeI2VNodeOverridePassPanel(passLabel, controls) {
  const panel = document.createElement("div");
  panel.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #272b35;border-radius:7px;background:#151821;padding:12px;box-shadow:inset 0 1px 0 rgba(255,255,255,.025);";
  const title = document.createElement("div");
  title.style.cssText = "display:flex;align-items:center;gap:7px;color:#f4f4f5;font-size:16px;font-weight:900;";
  const dot = document.createElement("span");
  dot.style.cssText = "width:8px;height:8px;border-radius:999px;background:#a855f7;box-shadow:0 0 10px rgba(168,85,247,.5);flex:0 0 auto;";
  title.append(dot, document.createTextNode(passLabel));

  const topGrid = document.createElement("div");
  topGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;";
  const samplerCard = document.createElement("div");
  samplerCard.style.cssText = "display:flex;flex-direction:column;gap:10px;border:1px solid #2f3440;border-radius:6px;background:#171b25;padding:12px;";
  const samplerTitle = document.createElement("div");
  samplerTitle.textContent = "Sampler";
  samplerTitle.style.cssText = "font-size:13px;font-weight:900;color:#f4f4f5;";
  samplerCard.append(samplerTitle, makeField("Sampler Name", controls.sampler));

  const sigmasCard = document.createElement("div");
  sigmasCard.style.cssText = samplerCard.style.cssText;
  const sigmasTitle = document.createElement("div");
  sigmasTitle.textContent = "Sigmas";
  sigmasTitle.style.cssText = samplerTitle.style.cssText;
  sigmasCard.append(sigmasTitle, controls.sigmas);
  topGrid.append(samplerCard, sigmasCard);

  const inplaceCard = document.createElement("div");
  inplaceCard.style.cssText = "display:flex;flex-direction:column;gap:12px;border:1px solid #2f3440;border-radius:6px;background:#171b25;padding:12px;";
  const inplaceTitle = document.createElement("div");
  inplaceTitle.textContent = "LTXVImgToVideoInplace";
  inplaceTitle.style.cssText = samplerTitle.style.cssText;
  const strengthRow = document.createElement("div");
  strengthRow.style.cssText = "display:grid;grid-template-columns:54px minmax(0,1fr) 84px;gap:10px;align-items:center;";
  const strengthLabel = document.createElement("div");
  strengthLabel.textContent = "Strength";
  strengthLabel.style.cssText = "font-size:12px;color:#d4d4d8;";
  strengthRow.append(strengthLabel, controls.strength, controls.strengthNumber);
  const bypassRow = document.createElement("div");
  bypassRow.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;";
  const bypassLabel = document.createElement("div");
  bypassLabel.textContent = "Bypass";
  bypassLabel.style.cssText = strengthLabel.style.cssText;
  bypassRow.append(bypassLabel, controls.bypass.wrapper);
  inplaceCard.append(inplaceTitle, strengthRow, bypassRow);

  panel.append(title, topGrid, inplaceCard);
  panel.inplaceCard = inplaceCard;
  return panel;
}

export function makeSubTabs(tabs = []) {
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  const tabBar = document.createElement("div");
  tabBar.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(74px,1fr));gap:6px;position:sticky;top:44px;z-index:2;background:#202024;padding-bottom:2px;";
  const panels = document.createElement("div");
  panels.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  const buttons = [];
  const setActive = (value) => {
    for (const item of tabs) {
      const active = item.value === value;
      item.content.style.display = active ? "flex" : "none";
    }
    for (const button of buttons) {
      const active = button.dataset.value === value;
      button.style.background = active ? "#06b6d4" : "#27272a";
      button.style.borderColor = active ? "#0891b2" : "#3f3f46";
      button.style.color = active ? "#082f49" : "#f4f4f5";
    }
  };
  for (const tab of tabs) {
    const button = makeButton(tab.label);
    applyCompactButtonLabel(button, tab.label, { minWidth: 0, padding: "7px 6px" });
    button.dataset.value = tab.value;
    button.onclick = () => setActive(tab.value);
    buttons.push(button);
    tabBar.append(button);
    panels.append(tab.content);
  }
  wrapper.append(tabBar, panels);
  setActive(tabs[0]?.value || "");
  return { wrapper, setActive };
}

export function toast(message, isError = false) {
  window.dispatchEvent(new CustomEvent("vrgdg:builder-toast", {
    detail: { message: String(message || ""), isError: Boolean(isError) },
  }));
  const element = document.createElement("div");
  element.textContent = message;
  element.style.cssText = `
    position: fixed;
    right: 18px;
    bottom: 18px;
    z-index: 100003;
    max-width: min(560px, calc(100vw - 36px));
    border: 1px solid ${isError ? "#991b1b" : "#155e75"};
    border-radius: 8px;
    background: ${isError ? "#450a0a" : "#083344"};
    color: ${isError ? "#fecaca" : "#cffafe"};
    padding: 12px 14px;
    white-space: pre-wrap;
    font-size: 12px;
    line-height: 1.4;
    box-shadow: 0 18px 60px rgba(0,0,0,.45);
  `;
  document.body.appendChild(element);
  setTimeout(() => element.remove(), 6500);
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
