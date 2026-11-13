// What the Custom Nodes dialog shows for each pack and when installing is possible. The status comes from
// `GET /vrgdg/video_builder/custom_nodes/status`: the pack's own route inside ComfyUI, or H3 Studio passing that call on to ComfyUI.

export const MANAGER_MISSING_TEXT = "ComfyUI-Manager is not installed in ComfyUI’s Python, so nodes cannot be installed from here. Start ComfyUI with --enable-manager, or run pip install -r manager_requirements.txt from the ComfyUI folder, then restart ComfyUI.";

// The text under "cannot install": a reason the server gave (H3 Studio cannot reach the pack in ComfyUI), else the ComfyUI-Manager one.
export function managerNoteText(result) {
  return String(result?.manager_note || "").trim() || MANAGER_MISSING_TEXT;
}

// A pack ComfyUI already has built in (the version it came with is in `built_in_since`) is not installed from here.
export function isBuiltIn(status) {
  return Boolean(status?.built_in);
}

// The badge of a card: `item` is the pack from VIDEO_BUILDER_CUSTOM_NODES, `status` its row of the status answer.
export function nodeBadge(item, status) {
  if (isBuiltIn(status)) return { text: "Built in", kind: "ok" };
  if (item?.current || status?.installed) return { text: "Installed", kind: "ok" };
  return { text: "Missing", kind: "missing" };
}

// The extra line of a built-in card, with the ComfyUI version when the server told it.
export function builtInLine(status, comfyuiVersion = "") {
  if (!isBuiltIn(status)) return "";
  const since = status.built_in_since ? ` ${status.built_in_since} and newer` : "";
  const yours = comfyuiVersion ? ` Yours is ${comfyuiVersion}.` : "";
  return `Built into ComfyUI${since}, so this pack is not needed.${yours} Only older ComfyUI needs it.`;
}

// A Python library installed with pip (`source: "pip"`), not a custom node: it needs no ComfyUI-Manager.
export function isPipLibrary(status) {
  return status?.source === "pip";
}

// The ids "Install Missing Nodes" asks for: not installed and not built in. A pip library is large and needs an NVIDIA GPU,
// so it is installed from its own card only.
export function missingNodeIds(nodes) {
  return (nodes || []).filter((node) => !node.installed && !isBuiltIn(node) && !isPipLibrary(node)).map((node) => node.id);
}

// The button of a card: its label and whether it can be clicked.
export function installButtonState(item, status, managerAvailable) {
  if (item?.current) return { text: "Included", disabled: true };
  if (isBuiltIn(status)) return { text: "Built in", disabled: true };
  if (isPipLibrary(status)) return status.installed ? { text: "Installed", disabled: true } : { text: "Install", disabled: false };
  return { text: status?.installed ? "Reinstall Requirements" : "Install", disabled: !managerAvailable };
}
