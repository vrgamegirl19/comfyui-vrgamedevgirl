import { MINIMAX_H3_DMAD_LORA_NAME } from "./minimax_h3.mjs";

export function miniMaxPickerLoraName(picker) {
  return String(picker.input.value || picker.missingLoraName || "[none]").trim();
}

export function syncInstalledMiniMaxLoraPicker(picker, requested = miniMaxPickerLoraName(picker)) {
  const name = String(requested || "[none]").trim();
  const installed = name === "[none]" || (picker.options || []).includes(name);
  picker.missingLoraName = installed ? "" : name;
  picker.input.value = installed ? name : "";
  const message = installed ? "" : `Missing installed LoRA: ${name}. Select an installed file or install this version in Models.`;
  picker.input.placeholder = installed ? "" : message;
  picker.input.title = message;
  picker.input.style.borderColor = installed ? "" : "#f87171";
  picker.input.setAttribute?.("aria-invalid", String(!installed));
  picker.input.setCustomValidity?.(message);
}

export function assertMiniMaxLorasInstalled(settings, installed) {
  const choices = new Set(installed);
  const active = settings.use_loras
    ? (settings.loras || []).slice(0, Number(settings.lora_count || 0)).filter(item => item.name && item.name !== "[none]")
    : [];
  if (settings.use_dmad) {
    const dmad = active.find(item => item.name.split(/[\\/]/).pop() === MINIMAX_H3_DMAD_LORA_NAME
      && (settings.render_pass === "single" || ["pass1", "both"].includes(item.apply_to || "pass1"))
      && Number(item.strength ?? 1) !== 0);
    if (!dmad || !choices.has(dmad.name)) {
      throw new Error(`DMAD rendering requires ${MINIMAX_H3_DMAD_LORA_NAME} installed and enabled on pass 1. Open Models → DMAD Models to install it, then select it in MiniMax Video Settings.`);
    }
  }
  const required = [...active.map(item => item.name)];
  if (settings.render_pass === "two_pass") required.push(settings.two_pass_lora_name);
  if (settings.render_pass === "three_pass") required.push(settings.three_pass_lightx_lora_name);
  const missing = required.filter(name => !name || name === "[none]" || !choices.has(name));
  if (missing.length) throw new Error(`Cannot render: selected MiniMax LoRA is not installed: ${[...new Set(missing)].join(", ")}. Install it in Models or choose an installed LoRA.`);
}
