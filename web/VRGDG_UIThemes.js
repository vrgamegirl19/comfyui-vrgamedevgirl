(() => {
  "use strict";

  const STORAGE_KEY = "vrgdg.uiTheme";
  const STYLE_ID = "vrgdg-ui-theme-style";
  const CONTROL_ID = "vrgdg-ui-theme-control";
  const CONTROL_CLASS = "vrgdg-ui-theme-control";
  const ROOT_ATTR = "data-vrgdg-theme-root";

  const THEMES = {
    current: {
      label: "Current",
      colors: null,
    },
    light: {
      label: "Light",
      colors: {
        bg0: "#f8fafc",
        bg1: "#eef2f7",
        bg2: "#e2e8f0",
        bg3: "#cbd5e1",
        text0: "#0f172a",
        text1: "#334155",
        text2: "#64748b",
        border0: "#cbd5e1",
        border1: "#94a3b8",
        primary: "#0284c7",
        primary2: "#0ea5e9",
        primaryText: "#f8fafc",
        success: "#15803d",
        danger: "#b91c1c",
        warning: "#a16207",
      },
    },
    dark: {
      label: "Dark",
      colors: {
        bg0: "#0b1020",
        bg1: "#111827",
        bg2: "#1f2937",
        bg3: "#374151",
        text0: "#f9fafb",
        text1: "#d1d5db",
        text2: "#9ca3af",
        border0: "#374151",
        border1: "#4b5563",
        primary: "#38bdf8",
        primary2: "#0ea5e9",
        primaryText: "#082f49",
        success: "#22c55e",
        danger: "#ef4444",
        warning: "#facc15",
      },
    },
    studio: {
      label: "Studio",
      colors: {
        bg0: "#101114",
        bg1: "#191b20",
        bg2: "#262a31",
        bg3: "#3a404a",
        text0: "#f5f5f0",
        text1: "#d6d3c8",
        text2: "#9ca3af",
        border0: "#3a404a",
        border1: "#525a66",
        primary: "#d6a84f",
        primary2: "#f0c96b",
        primaryText: "#1c1608",
        success: "#5fbf8f",
        danger: "#df6b6b",
        warning: "#f0c96b",
      },
    },
    neon: {
      label: "Neon",
      colors: {
        bg0: "#070814",
        bg1: "#0d1024",
        bg2: "#191b3d",
        bg3: "#2a2d61",
        text0: "#f8f7ff",
        text1: "#d9d6ff",
        text2: "#9ca3ff",
        border0: "#3340a3",
        border1: "#5b6cff",
        primary: "#00e5ff",
        primary2: "#ff4fd8",
        primaryText: "#061018",
        success: "#39ff88",
        danger: "#ff3b6b",
        warning: "#fff36a",
      },
    },
    forest: {
      label: "Forest",
      colors: {
        bg0: "#07140f",
        bg1: "#0f1f17",
        bg2: "#1b3325",
        bg3: "#31513e",
        text0: "#f4fff8",
        text1: "#cbe8d4",
        text2: "#8fb69b",
        border0: "#31513e",
        border1: "#4b7359",
        primary: "#34d399",
        primary2: "#86efac",
        primaryText: "#052e1a",
        success: "#22c55e",
        danger: "#f87171",
        warning: "#fde68a",
      },
    },
    candy: {
      label: "Candy",
      colors: {
        bg0: "#fff7fb",
        bg1: "#ffe4f1",
        bg2: "#fbcfe8",
        bg3: "#f9a8d4",
        text0: "#3b1230",
        text1: "#6b244f",
        text2: "#9d4f7b",
        border0: "#f9a8d4",
        border1: "#f472b6",
        primary: "#ec4899",
        primary2: "#8b5cf6",
        primaryText: "#ffffff",
        success: "#10b981",
        danger: "#dc2626",
        warning: "#d97706",
      },
    },
    minimal: {
      label: "Minimal",
      colors: {
        bg0: "#ffffff",
        bg1: "#f4f4f5",
        bg2: "#e4e4e7",
        bg3: "#d4d4d8",
        text0: "#18181b",
        text1: "#3f3f46",
        text2: "#71717a",
        border0: "#d4d4d8",
        border1: "#a1a1aa",
        primary: "#2563eb",
        primary2: "#0f766e",
        primaryText: "#ffffff",
        success: "#16a34a",
        danger: "#dc2626",
        warning: "#ca8a04",
      },
    },
  };

  const COLOR_ROLES = {
    "#020617": "bg0",
    "#0b0f16": "bg0",
    "#0b1118": "bg0",
    "#0d1017": "bg0",
    "#101114": "bg0",
    "#111113": "bg0",
    "#18181b": "bg1",
    "#1b1b1f": "bg1",
    "#202024": "bg2",
    "#27272a": "bg2",
    "#29292f": "bg2",
    "#303038": "border0",
    "#34343a": "bg3",
    "#3f3f46": "border0",
    "#4b5563": "border1",
    "#52525b": "border1",
    "#fafafa": "text0",
    "#f8fafc": "text0",
    "#f4f4f5": "text0",
    "#e5e7eb": "text1",
    "#d4d4d8": "text1",
    "#cbd5e1": "text1",
    "#a1a1aa": "text2",
    "#94a3b8": "text2",
    "#71717a": "text2",
    "#06b6d4": "primary",
    "#0891b2": "primary2",
    "#0e7490": "primary2",
    "#155e75": "primary2",
    "#164e63": "primary2",
    "#082f49": "primaryText",
    "#cffafe": "primary",
    "#67e8f9": "primary",
    "#a5f3fc": "primary",
    "#bae6fd": "primary",
    "#a3e635": "success",
    "#22c55e": "success",
    "#064e3b": "success",
    "#b91c1c": "danger",
    "#991b1b": "danger",
    "#7f1d1d": "danger",
    "#fee2e2": "text0",
    "#fecaca": "danger",
    "#fca5a5": "danger",
    "#fde68a": "warning",
  };

  let currentThemeId = safeLoadThemeId();
  let applyingTheme = false;
  const originalStyles = new WeakMap();
  const themeRoots = new Set();
  let observer = null;

  function safeLoadThemeId() {
    try {
      const value = localStorage.getItem(STORAGE_KEY) || "current";
      return THEMES[value] ? value : "current";
    } catch (_error) {
      return "current";
    }
  }

  function saveThemeId(themeId) {
    try {
      localStorage.setItem(STORAGE_KEY, themeId);
    } catch (_error) {
      // localStorage can be disabled; the theme still applies for this session.
    }
  }

  function replacementFor(hex, colors) {
    const role = COLOR_ROLES[String(hex || "").toLowerCase()];
    return role && colors[role] ? colors[role] : hex;
  }

  function normalizeRgbColor(value) {
    const match = String(value || "").match(/^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([0-9.]+)\s*)?\)$/i);
    if (!match) return null;
    const parts = match.slice(1, 4).map((part) => Math.max(0, Math.min(255, Number(part) || 0)));
    return `#${parts.map((part) => part.toString(16).padStart(2, "0")).join("")}`;
  }

  function replacementForCssColor(match, colors) {
    const normalized = match.startsWith("#") ? match.toLowerCase() : normalizeRgbColor(match);
    if (!normalized) return match;
    const replacement = replacementFor(normalized, colors);
    return replacement === normalized ? match : replacement;
  }

  function translateStyleText(styleText, colors) {
    if (!styleText || !colors) return styleText || "";
    return String(styleText)
      .replace(/#[0-9a-fA-F]{3,8}\b/g, (match) => {
        const hex = match.toLowerCase();
        if (hex.length !== 7) return match;
        return replacementForCssColor(hex, colors);
      })
      .replace(/rgba?\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}(?:\s*,\s*[0-9.]+\s*)?\)/gi, (match) => replacementForCssColor(match, colors));
  }

  function rememberOriginal(element) {
    if (!element || isThemeControl(element) || originalStyles.has(element)) return;
    const styleText = element.getAttribute("style") || "";
    if (styleText) originalStyles.set(element, styleText);
  }

  function themeElement(element, colors) {
    if (!(element instanceof HTMLElement)) return;
    if (isThemeControl(element)) return;
    const styleText = element.getAttribute("style") || "";
    if (!styleText) return;
    rememberOriginal(element);
    const source = originalStyles.get(element) || styleText;
    const themed = translateStyleText(source, colors);
    if (themed && themed !== styleText) element.setAttribute("style", themed);
  }

  function restoreElement(element) {
    if (!(element instanceof HTMLElement)) return;
    if (!originalStyles.has(element)) return;
    element.setAttribute("style", originalStyles.get(element));
  }

  function walkElements(root, callback) {
    if (!root) return;
    if (root instanceof HTMLElement) callback(root);
    const elements = root.querySelectorAll?.("[style]");
    if (!elements) return;
    for (const element of elements) callback(element);
  }

  function applyTheme(themeId = currentThemeId) {
    currentThemeId = THEMES[themeId] ? themeId : "current";
    saveThemeId(currentThemeId);
    updateRootThemeAttributes();
    updateControlValues();
    applyingTheme = true;
    try {
      const colors = THEMES[currentThemeId].colors;
      for (const root of Array.from(themeRoots)) {
        if (!root?.isConnected) {
          themeRoots.delete(root);
          continue;
        }
        if (!colors) {
          walkElements(root, restoreElement);
        } else {
          walkElements(root, (element) => themeElement(element, colors));
        }
      }
    } finally {
      applyingTheme = false;
    }
  }

  function isThemeControl(element) {
    return Boolean(element?.id === CONTROL_ID || element?.classList?.contains(CONTROL_CLASS) || element?.closest?.(`.${CONTROL_CLASS}`));
  }

  function updateRootThemeAttributes() {
    for (const root of Array.from(themeRoots)) {
      if (!root?.isConnected) {
        themeRoots.delete(root);
        continue;
      }
      root.dataset.vrgdgUiTheme = currentThemeId;
    }
  }

  function updateControlValues() {
    for (const select of document.querySelectorAll(`.${CONTROL_CLASS} select`)) {
      if (select.value !== currentThemeId) select.value = currentThemeId;
    }
  }

  function ensureThemeStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      [${ROOT_ATTR}][data-vrgdg-ui-theme="light"],
      [${ROOT_ATTR}][data-vrgdg-ui-theme="candy"],
      [${ROOT_ATTR}][data-vrgdg-ui-theme="minimal"] {
        color-scheme: light;
      }
      [${ROOT_ATTR}][data-vrgdg-ui-theme="dark"],
      [${ROOT_ATTR}][data-vrgdg-ui-theme="studio"],
      [${ROOT_ATTR}][data-vrgdg-ui-theme="neon"],
      [${ROOT_ATTR}][data-vrgdg-ui-theme="forest"] {
        color-scheme: dark;
      }
      .${CONTROL_CLASS} {
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 7px 8px;
        border: 1px solid rgba(148, 163, 184, 0.55);
        border-radius: 8px;
        background: rgba(17, 24, 39, 0.92);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.35);
        color: #f8fafc;
        font-family: Arial, sans-serif;
        font-size: 12px;
      }
      .${CONTROL_CLASS} label {
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 800;
        white-space: nowrap;
      }
      .${CONTROL_CLASS} select {
        border: 1px solid rgba(148, 163, 184, 0.75);
        border-radius: 6px;
        background: #020617;
        color: #f8fafc;
        padding: 5px 7px;
        font-size: 12px;
      }
    `;
    document.head.append(style);
  }

  function createControl() {
    const control = document.createElement("div");
    control.className = CONTROL_CLASS;
    const label = document.createElement("label");
    label.textContent = "Theme";
    const select = document.createElement("select");
    for (const [value, theme] of Object.entries(THEMES)) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = theme.label;
      select.append(option);
    }
    select.value = currentThemeId;
    select.addEventListener("change", () => applyTheme(select.value));
    label.append(select);
    control.append(label);
    return control;
  }

  function mountControl(container) {
    if (!(container instanceof HTMLElement)) return null;
    ensureThemeStyle();
    const existing = container.querySelector(`.${CONTROL_CLASS}`);
    if (existing) return existing;
    const control = createControl();
    container.append(control);
    return control;
  }

  function registerRoot(root, options = {}) {
    if (!(root instanceof HTMLElement)) return null;
    ensureThemeStyle();
    root.setAttribute(ROOT_ATTR, "true");
    themeRoots.add(root);
    if (options.controlContainer instanceof HTMLElement) {
      mountControl(options.controlContainer);
    }
    observeThemeTargets();
    applyTheme(currentThemeId);
    return root;
  }

  function observeThemeTargets() {
    if (observer) return;
    observer = new MutationObserver((mutations) => {
      if (applyingTheme || currentThemeId === "current") return;
      const colors = THEMES[currentThemeId]?.colors;
      if (!colors) return;
      applyingTheme = true;
      try {
        for (const mutation of mutations) {
          if (mutation.type === "attributes" && mutation.attributeName === "style") {
            const element = mutation.target;
            if (element instanceof HTMLElement && element.closest?.(`[${ROOT_ATTR}]`) && !isThemeControl(element)) {
              originalStyles.set(element, element.getAttribute("style") || "");
              themeElement(element, colors);
            }
          }
          for (const node of mutation.addedNodes || []) {
            if (!(node instanceof HTMLElement)) continue;
            const roots = node.matches?.(`[${ROOT_ATTR}]`)
              ? [node]
              : Array.from(node.querySelectorAll?.(`[${ROOT_ATTR}]`) || []);
            for (const root of roots) themeRoots.add(root);
            const scopedRoot = node.matches?.(`[${ROOT_ATTR}]`) ? node : node.closest?.(`[${ROOT_ATTR}]`);
            if (scopedRoot) walkElements(node, (element) => themeElement(element, colors));
          }
        }
      } finally {
        applyingTheme = false;
      }
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["style"],
    });
  }

  function init() {
    if (!document.body) {
      window.addEventListener("DOMContentLoaded", init, { once: true });
      return;
    }
    document.getElementById(CONTROL_ID)?.remove();
    ensureThemeStyle();
    observeThemeTargets();
    for (const root of document.querySelectorAll(`[${ROOT_ATTR}]`)) {
      themeRoots.add(root);
    }
    applyTheme(currentThemeId);
    window.VRGDG_UIThemes = {
      themes: THEMES,
      apply: applyTheme,
      registerRoot,
      mountControl,
      get current() {
        return currentThemeId;
      },
    };
  }

  init();
})();
