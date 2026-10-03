import { makeButton, makeCheckbox } from "./controls.mjs";

// Styled OK / Cancel confirmation for destructive actions. Used instead of window.confirm so every delete
// in the Builder warns the same way. Resolves { confirmed, optionChecked }; Cancel, Escape and a click
// outside the box all cancel, and Cancel has focus so Enter never deletes by accident.
//   title        heading, shown in red
//   message      a line or an array of lines explaining what is about to happen
//   details      optional lines shown in a box (what exactly will be affected)
//   option       optional checkbox { label, checked } (e.g. "Also delete scene images")
export function confirmDestructiveAction({
  title,
  message = [],
  details = [],
  option = null,
  confirmLabel = "OK",
  cancelLabel = "Cancel",
} = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.setAttribute("role", "alertdialog");
    backdrop.setAttribute("aria-modal", "true");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:20px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(560px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;border:1px solid #7f1d1d;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = String(title || "Are you sure?");
    heading.style.cssText = "font-size:16px;font-weight:900;color:#fecaca;";
    box.append(heading);
    for (const line of [].concat(message || []).filter(Boolean)) {
      const paragraph = document.createElement("div");
      paragraph.textContent = String(line);
      paragraph.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
      box.append(paragraph);
    }
    const detailLines = [].concat(details || []).filter(Boolean);
    if (detailLines.length) {
      const detailBox = document.createElement("div");
      detailBox.style.cssText = "border:1px solid #3f3f46;border-radius:6px;background:#18181b;padding:9px;color:#bae6fd;font-size:11px;overflow-wrap:anywhere;display:flex;flex-direction:column;gap:3px;";
      for (const line of detailLines) {
        const row = document.createElement("div");
        row.textContent = String(line);
        detailBox.append(row);
      }
      box.append(detailBox);
    }
    const optionCheckbox = option ? makeCheckbox(String(option.label || ""), Boolean(option.checked)) : null;
    if (optionCheckbox) box.append(optionCheckbox.wrapper);

    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton(cancelLabel);
    const confirm = makeButton(confirmLabel, "danger");
    confirm.style.borderColor = "#7f1d1d";
    confirm.style.background = "#991b1b";
    confirm.style.color = "#fee2e2";

    let settled = false;
    const finish = (confirmed) => {
      if (settled) return;
      settled = true;
      document.removeEventListener("keydown", onKeyDown, true);
      backdrop.remove();
      resolve({ confirmed, optionChecked: Boolean(optionCheckbox?.input.checked) });
    };
    // Capture phase, so Escape here never reaches the timeline shortcuts underneath.
    const onKeyDown = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      finish(false);
    };
    cancel.onclick = () => finish(false);
    confirm.onclick = () => finish(true);
    backdrop.onclick = (event) => {
      if (event.target === backdrop) finish(false);
    };
    document.addEventListener("keydown", onKeyDown, true);

    actions.append(cancel, confirm);
    box.append(actions);
    backdrop.append(box);
    document.body.append(backdrop);
    cancel.focus();
  });
}

// Styled single-line text prompt (a window.prompt replacement). Resolves the trimmed text, or null when the
// user cancels. Enter confirms, Escape and a click outside the box cancel, and an empty name cannot be submitted.
export function promptForText({
  title,
  message = [],
  label = "",
  placeholder = "",
  initialValue = "",
  maxLength = 60,
  confirmLabel = "OK",
  cancelLabel = "Cancel",
} = {}) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.setAttribute("role", "dialog");
    backdrop.setAttribute("aria-modal", "true");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:20px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(460px,calc(100vw - 40px));border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;";
    const heading = document.createElement("div");
    heading.textContent = String(title || "Enter a name");
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    box.append(heading);
    for (const line of [].concat(message || []).filter(Boolean)) {
      const paragraph = document.createElement("div");
      paragraph.textContent = String(line);
      paragraph.style.cssText = "font-size:13px;color:#d4d4d8;line-height:1.45;";
      box.append(paragraph);
    }
    if (label) {
      const labelNode = document.createElement("div");
      labelNode.textContent = String(label);
      labelNode.style.cssText = "font-size:12px;font-weight:700;color:#a5f3fc;";
      box.append(labelNode);
    }
    const input = document.createElement("input");
    input.type = "text";
    input.value = String(initialValue || "");
    input.placeholder = String(placeholder || "");
    input.maxLength = Number(maxLength) || 60;
    input.style.cssText = "width:100%;box-sizing:border-box;border:1px solid #3f3f46;border-radius:6px;background:#18181b;color:#fafafa;padding:9px;font-size:13px;";
    box.append(input);
    const actions = document.createElement("div");
    actions.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";
    const cancel = makeButton(cancelLabel);
    const confirm = makeButton(confirmLabel, "primary");

    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      document.removeEventListener("keydown", onKeyDown, true);
      backdrop.remove();
      resolve(value);
    };
    const submit = () => {
      const text = String(input.value || "").trim();
      if (text) finish(text);
      else input.focus();
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        finish(null);
      } else if (event.key === "Enter" && event.target === input) {
        event.preventDefault();
        event.stopPropagation();
        submit();
      }
    };
    cancel.onclick = () => finish(null);
    confirm.onclick = submit;
    backdrop.onclick = (event) => {
      if (event.target === backdrop) finish(null);
    };
    document.addEventListener("keydown", onKeyDown, true);
    actions.append(cancel, confirm);
    box.append(actions);
    backdrop.append(box);
    document.body.append(backdrop);
    input.focus();
    if (input.select) input.select();
  });
}
