const BUTTON_COLORS = {
  primary: ["#12b5cb", "#0891b2"],
  purple: ["#0e7490", "#06b6d4"],
  danger: ["#b91c1c", "#ef4444"],
  default: ["#2b2b30", "#3f3f46"],
};

export function makeButton(label, variant = "default") {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.style.cssText = "border:1px solid;border-radius:6px;color:#f8fafc;padding:9px 13px;font-weight:800;cursor:pointer;";
  setButtonVariant(button, variant);
  return button;
}

export function setButtonVariant(button, variant) {
  const [bg, border] = BUTTON_COLORS[variant] || BUTTON_COLORS.default;
  button.style.background = bg;
  button.style.borderColor = border;
}

export function setButtonDisabled(button, disabled) {
  button.disabled = disabled;
  button.style.opacity = disabled ? "0.5" : "";
  button.style.cursor = disabled ? "not-allowed" : "pointer";
}

export function makeInput(value = "", placeholder = "") {
  const input = document.createElement("input");
  input.value = value || "";
  input.placeholder = placeholder;
  input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#0b1220;color:#e5e7eb;padding:9px;font:12px monospace;";
  return input;
}

export function makeTextarea(value = "", placeholder = "", rows = 4) {
  const textarea = document.createElement("textarea");
  textarea.value = value || "";
  textarea.placeholder = placeholder;
  textarea.rows = rows;
  textarea.style.cssText = "width:100%;box-sizing:border-box;resize:vertical;border:1px solid #334155;border-radius:6px;background:#050814;color:#e5e7eb;padding:9px;font:12px monospace;line-height:1.45;";
  return textarea;
}

const sortLabelOf = (option) => String(option?.label ?? option?.value ?? option ?? "");
const isPinnedOption = (option) => {
  if (option === null || typeof option !== "object") return false;
  const value = String(option.value ?? "");
  return value === "" || /^_*custom_*$/i.test(value);
};
const compareLabels = (a, b) => sortLabelOf(a).localeCompare(sortLabelOf(b), undefined, { sensitivity: "base", numeric: true });

// Alphabetizes a flat option list. The first entry, the blank/default entry, and Custom entries stay
// at the top in their original order.
export function sortOptionsAlphabetically(options) {
  const list = Array.isArray(options) ? options : [];
  if (list.length < 2) return list;
  const [first, ...rest] = list;
  const pinned = rest.filter(isPinnedOption);
  const sorted = rest.filter((option) => !isPinnedOption(option)).sort(compareLabels);
  return [first, ...pinned, ...sorted];
}

// Alphabetizes grouped options: groups by label, and the options inside each group. Entries without
// options (the leading placeholder) stay at the top.
export function sortGroupsAlphabetically(groups) {
  const list = Array.isArray(groups) ? groups : [];
  const loose = list.filter((group) => !group.options);
  const grouped = list
    .filter((group) => group.options)
    .map((group) => ({ ...group, options: [...group.options].sort(compareLabels) }))
    .sort(compareLabels);
  return [...loose, ...grouped];
}

export function makeSelect(options, value = "") {
  const select = document.createElement("select");
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:9px;";
  for (const option of options) {
    const item = document.createElement("option");
    item.value = option.value;
    item.textContent = option.label;
    select.append(item);
  }
  select.value = value || options[0]?.value || "";
  return select;
}

export function makeGroupedSelect(groups, value = "") {
  const select = document.createElement("select");
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:9px;";
  for (const group of groups) {
    if (group.options) {
      const optgroup = document.createElement("optgroup");
      optgroup.label = group.label;
      for (const option of group.options) {
        const item = document.createElement("option");
        item.value = option.value ?? option;
        item.textContent = option.label ?? option;
        optgroup.append(item);
      }
      select.append(optgroup);
    } else {
      const item = document.createElement("option");
      item.value = group.value ?? "";
      item.textContent = group.label ?? "";
      select.append(item);
    }
  }
  select.value = value || "";
  return select;
}

export function makeMultiSelect(options, values = []) {
  const select = document.createElement("select");
  select.multiple = true;
  select.size = Math.min(6, Math.max(3, options.length || 3));
  select.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #334155;border-radius:6px;background:#18181b;color:#f8fafc;padding:7px;min-height:104px;";
  const selected = new Set(Array.isArray(values) ? values.map(String) : []);
  for (const option of options) {
    const item = document.createElement("option");
    item.value = option.value;
    item.textContent = option.label;
    item.selected = selected.has(String(option.value));
    select.append(item);
  }
  return select;
}

export function makeCollapsiblePanel(title, summary = "", content = null, { open = false } = {}) {
  const panel = document.createElement("div");
  panel.style.cssText = "margin:8px 24px 0;border:1px solid #334155;border-radius:8px;background:#0f172a;overflow:hidden;min-width:0;max-width:100%;box-sizing:border-box;";
  const header = document.createElement("button");
  header.type = "button";
  header.style.cssText = "width:100%;min-width:0;box-sizing:border-box;border:0;background:#0f172a;color:#e5e7eb;padding:9px 12px;display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:10px;align-items:center;text-align:left;cursor:pointer;";
  const caret = document.createElement("span");
  caret.style.cssText = "color:#67e8f9;font-size:13px;";
  const label = document.createElement("span");
  label.style.cssText = "font-weight:900;color:#cffafe;font-size:13px;white-space:nowrap;";
  label.textContent = title;
  const summaryNode = document.createElement("span");
  summaryNode.style.cssText = "min-width:0;max-width:100%;color:#94a3b8;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
  summaryNode.textContent = summary;
  const body = document.createElement("div");
  body.style.cssText = "min-width:0;max-width:100%;box-sizing:border-box;border-top:1px solid #1f3347;padding:10px 12px;";
  if (content) body.append(content);
  let expanded = Boolean(open);
  const sync = () => {
    caret.textContent = expanded ? "▾" : "▸";
    body.style.display = expanded ? "" : "none";
  };
  header.onclick = () => {
    expanded = !expanded;
    sync();
  };
  header.append(caret, label, summaryNode);
  panel.append(header, body);
  panel.setSummary = (value) => {
    summaryNode.textContent = String(value || "");
  };
  panel.setOpen = (value) => {
    expanded = Boolean(value);
    sync();
  };
  panel.isOpen = () => expanded;
  sync();
  return panel;
}

export function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function truncate(text, length = 130) {
  const clean = String(text || "").trim();
  if (clean.length <= length) return clean;
  return `${clean.slice(0, Math.max(0, length - 1)).trim()}...`;
}

export function replaceLabeledPlanningLine(value, labelName, selectedValue) {
  const cleanLabel = String(labelName || "").trim();
  const cleanValue = String(selectedValue || "").trim();
  if (!cleanLabel || !cleanValue) return String(value || "").trim();
  const prefix = `${cleanLabel}:`;
  const replacement = `${prefix} ${cleanValue}.`;
  const lines = String(value || "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .filter((line) => !line.trim().toLowerCase().startsWith(prefix.toLowerCase()));
  lines.push(replacement);
  return lines.map((line) => line.trim()).filter(Boolean).join("\n");
}

export function tagsHtml(tags) {
  const list = Array.isArray(tags) ? tags : [];
  if (!list.length) return `<span style="color:#94a3b8;">-</span>`;
  return list.map((tag) => `<span style="display:inline-flex;border-radius:5px;background:#1e1b4b;color:#ddd6fe;padding:4px 7px;margin:2px;font-size:11px;">${escapeHtml(tag)}</span>`).join("");
}

export function createToast(message, error = false) {
  const toast = document.createElement("div");
  toast.textContent = message;
  toast.style.cssText = `position:fixed;right:24px;bottom:24px;z-index:100020;max-width:520px;border:1px solid ${error ? "#991b1b" : "#155e75"};border-radius:8px;background:${error ? "#3f0808" : "#083344"};color:#f8fafc;padding:12px 14px;box-shadow:0 12px 40px rgba(0,0,0,.45);white-space:pre-wrap;font-size:13px;`;
  document.body.append(toast);
  setTimeout(() => toast.remove(), error ? 8500 : 4200);
}

export function createStoryboardProgressWindow(title = "Storyboard LLM") {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100030;background:rgba(0,0,0,.18);pointer-events:none;display:flex;align-items:flex-start;justify-content:center;padding-top:72px;";
  const box = document.createElement("div");
  box.style.cssText = "width:min(760px,calc(100vw - 48px));border:1px solid #0891b2;border-radius:9px;background:#0f172a;color:#e5e7eb;box-shadow:0 22px 70px rgba(0,0,0,.55);overflow:hidden;pointer-events:auto;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;background:#083f4f;border-bottom:1px solid #0891b2;";
  const titleEl = document.createElement("div");
  titleEl.textContent = title;
  titleEl.style.cssText = "font-weight:900;color:#cffafe;";
  const close = makeButton("Close");
  close.style.padding = "8px 12px";
  header.append(titleEl, close);
  const body = document.createElement("div");
  body.style.cssText = "padding:14px;display:flex;flex-direction:column;gap:12px;";
  const message = document.createElement("div");
  message.style.cssText = "white-space:pre-wrap;line-height:1.45;font-size:13px;color:#e2e8f0;min-height:38px;";
  const track = document.createElement("div");
  track.style.cssText = "height:8px;border-radius:999px;background:#155e75;overflow:hidden;";
  const fill = document.createElement("div");
  fill.style.cssText = "height:100%;width:0%;background:#22d3ee;border-radius:999px;transition:width .18s ease;";
  track.append(fill);
  body.append(message, track);
  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  close.onclick = () => backdrop.remove();
  return {
    set(text, percent = 0) {
      message.textContent = String(text || "");
      const pct = Number.isFinite(Number(percent)) ? Math.max(0, Math.min(100, Number(percent))) : 0;
      fill.style.width = `${pct}%`;
    },
    showDiagnostics(diagnostics = {}) {
      if (!diagnostics || typeof diagnostics !== "object") return;
      const details = document.createElement("details");
      details.style.cssText = "border:1px solid #475569;border-radius:7px;background:#111827;padding:9px 10px;color:#cbd5e1;font-size:12px;";
      const summary = document.createElement("summary");
      summary.textContent = "Show raw model output (diagnostics)";
      summary.style.cssText = "cursor:pointer;font-weight:800;color:#bae6fd;";
      const meta = document.createElement("div");
      meta.style.cssText = "margin-top:8px;line-height:1.45;white-space:pre-wrap;";
      const runner = String(diagnostics.runner || "LLM");
      const expected = Array.isArray(diagnostics.expected_sections) ? diagnostics.expected_sections.join(" → ") : "";
      meta.textContent = `Runner: ${runner}${expected ? `\nExpected sections: ${expected}` : ""}`;
      const rawLabel = document.createElement("div");
      rawLabel.textContent = "Raw response:";
      rawLabel.style.cssText = "margin-top:8px;font-weight:800;color:#fda4af;";
      const raw = document.createElement("pre");
      raw.textContent = String(diagnostics.raw_output || "[empty]");
      raw.style.cssText = "max-height:220px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0 8px;padding:8px;background:#020617;border-radius:5px;color:#fecdd3;";
      const cleanedLabel = document.createElement("div");
      cleanedLabel.textContent = "Cleaned response:";
      cleanedLabel.style.cssText = "font-weight:800;color:#bae6fd;";
      const cleaned = document.createElement("pre");
      cleaned.textContent = String(diagnostics.cleaned_output || "[empty]");
      cleaned.style.cssText = "max-height:180px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0 8px;padding:8px;background:#020617;border-radius:5px;color:#bae6fd;";
      const copy = makeButton("Copy output");
      copy.style.padding = "5px 9px";
      copy.onclick = async () => {
        try { await navigator.clipboard.writeText(String(diagnostics.raw_output || "")); copy.textContent = "Copied"; setTimeout(() => { copy.textContent = "Copy output"; }, 1200); }
        catch { copy.textContent = "Copy failed"; }
      };
      details.append(summary, meta, rawLabel, raw, cleanedLabel, cleaned, copy);
      body.insertBefore(details, track);
    },
    close(delay = 0) {
      if (delay > 0) {
        setTimeout(() => backdrop.remove(), delay);
      } else {
        backdrop.remove();
      }
    },
  };
}

export async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.cssText = "position:fixed;left:-9999px;top:-9999px;";
  document.body.append(textarea);
  textarea.focus();
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

export function storyField(label, control) {
  const wrap = document.createElement("label");
  wrap.style.cssText = "display:flex;flex-direction:column;gap:6px;font-size:12px;font-weight:900;color:#cbd5e1;";
  wrap.textContent = label;
  wrap.append(control);
  return wrap;
}
