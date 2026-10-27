// The GPU list of the LLM Runner (Gemma Local and Qwen Local). A local model runs inside ComfyUI, so it can only go on the cards ComfyUI sees, and the
// number it takes is the one ComfyUI gives each card. That is not always Task Manager's or nvidia-smi's number (CUDA puts the fastest card first,
// and a launcher can hide cards), so every card is listed with its name and memory, and the hidden ones are listed too, with the reason.
// Python twin: llm/gpus.py, which makes the list (the `gpus` answer of GET /vrgdg/music_builder/gemma_choices).

export const AUTO_GPU_LABEL = "Automatic (every GPU ComfyUI can see)";

function gpuCardLabel(card) {
  const memory = card.total_gb != null ? `${card.total_gb} GB${card.free_gb != null ? `, ${card.free_gb} GB free` : ""}` : "";
  const detail = memory ? ` (${memory})` : "";
  if (!card.visible) return `${card.name}${detail} - not visible to ComfyUI`;
  return `GPU ${card.number}: ${card.name}${detail}${card.comfyui_gpu ? " - ComfyUI renders here" : ""}`;
}

// One line under the select: the card's number in ComfyUI, its name and memory, and the number Windows uses for it when that is a different one.
export function gpuCardLine(card) {
  const memory = card.total_gb != null ? ` ${card.total_gb} GB${card.free_gb != null ? `, ${card.free_gb} GB free` : ""}` : "";
  const where = card.visible ? `GPU ${card.number}` : "Hidden from ComfyUI";
  const slot = card.slot != null && card.slot !== card.number ? `  (Task Manager and nvidia-smi: GPU ${card.slot})` : "";
  return `${where}: ${card.name},${memory}${card.comfyui_gpu ? " - ComfyUI renders here" : ""}${slot}`;
}

// The options of the GPU select and the text under it. `info` is the `gpus` answer (null, or without a list, when this ComfyUI is too old to say),
// `saved` the number chosen before (-1 is automatic). A saved number that is not on the list is kept as an option, so opening the window and
// saving never drops the choice by accident, and the text says it is not available.
export function gpuChoices(info, saved = -1) {
  const choice = Number.isInteger(saved) && saved >= 0 ? saved : -1;
  const options = [{ value: "-1", label: AUTO_GPU_LABEL, disabled: false }];
  const lines = [];
  const cards = info && Array.isArray(info.gpus) ? info.gpus : null;
  if (!cards) {
    lines.push("ComfyUI cannot list its GPUs yet. Update the comfyui-vrgamedevgirl pack, restart ComfyUI and reopen this window to choose a GPU.");
  } else if (!cards.length) {
    lines.push("No NVIDIA GPU was found, so a local model runs on the CPU.");
  } else {
    cards.forEach((card, index) => {
      options.push({ value: card.visible ? String(card.number) : `hidden-${card.slot ?? index}`, label: card.label || gpuCardLabel(card), disabled: !card.visible });
      lines.push(gpuCardLine(card));
    });
    if (info.note) lines.push(info.note);
    lines.push("Automatic spreads a model over every GPU ComfyUI can see. A number puts the whole model on that GPU, which leaves the others free for ComfyUI's renders.");
  }
  if (choice >= 0 && !options.some((option) => option.value === String(choice) && !option.disabled)) {
    options.push({ value: String(choice), label: `GPU ${choice} (not visible to ComfyUI now)`, disabled: false });
    if (cards) lines.push(`The saved choice, GPU ${choice}, is not visible to ComfyUI now. Pick another GPU or Automatic before running.`);
  }
  return { options, details: lines.join("\n"), selected: String(choice) };
}
