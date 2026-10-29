// Small helpers of the transition screens: elements, buttons, names and colours.

export const transitionName = (id) => String(id).replace(/([a-z0-9])([A-Z])/g, "$1 $2");

export const transitionSceneName = (segment, index) => String(segment?.label || "").trim() || `Scene ${index + 1}`;

export const transitionSeconds = (value) => `${Number(value).toFixed(2)} s`;

export function transitionElement(tag, css, props = {}) {
  const node = document.createElement(tag);
  if (css) node.style.cssText = css;
  Object.assign(node, props);
  return node;
}

export function transitionButton(text, primary) {
  return transitionElement("button", `border-radius:5px;padding:6px 12px;cursor:pointer;font-weight:800;color:#fff;border:1px solid ${primary ? "#a78bfa" : "#52525b"};background:${primary ? "#7c3aed" : "#27272a"};`, { type: "button", textContent: text });
}

export function transitionKeyIsTyping(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  const tag = String(target.tagName || "").toUpperCase();
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag !== "INPUT") return false;
  return !["range", "checkbox", "radio", "button", "submit", "reset", "file", "image"].includes(String(target.type || "text").toLowerCase());
}

export const transitionToHex = (rgb) => `#${rgb.slice(0, 3).map((c) => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, "0")).join("")}`;

export const transitionFromHex = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
