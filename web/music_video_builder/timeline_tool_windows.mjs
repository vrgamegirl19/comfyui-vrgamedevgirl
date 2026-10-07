import { makeButton } from "./controls.mjs";

export function timelineDeleteAvailability(segments) {
  const items = segments.filter((segment) => segment && typeof segment === "object" && !Array.isArray(segment));
  const hasValue = (value) => Boolean(String(value || "").trim());
  return {
    images: items.some((segment) => Boolean(
      (Array.isArray(segment.image_history) && segment.image_history.some(hasValue))
      || ["approved_image_path", "custom_image_path", "custom_image_data", "first_last_frame_end_image_path",
        "first_last_frame_end_image_data", "flf_rendered_start_frame_path", "flf_rendered_start_frame_data"]
        .some((key) => hasValue(segment[key]))
    )),
    videos: items.some((segment) => Boolean(
      ["video_path", "video_original_path"].some((key) => hasValue(segment[key]))
      || ["video_history", "video_backup_paths"].some((key) =>
        Array.isArray(segment[key]) && segment[key].some(hasValue))
    )),
    segments: items.length > 0,
  };
}

export function createTimelineToolWindows({ overlay, toolButtons, deleteButtons, getDeleteAvailability }) {
  const toolsButton = makeButton("Timeline Tools");
  toolsButton.title = "Open movable timeline range, gap, track, and location controls.";
  const toolsWindow = document.createElement("div");
  toolsWindow.style.cssText = "position:fixed;z-index:100004;display:none;width:340px;max-width:calc(100vw - 16px);box-sizing:border-box;border:1px solid #155e75;border-radius:8px;background:#202024;color:#fafafa;box-shadow:0 18px 60px rgba(0,0,0,.6);overflow:hidden;";
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;padding:8px 10px;background:#083344;cursor:move;user-select:none;";
  const title = document.createElement("strong");
  title.textContent = "Timeline Tools";
  const close = makeButton("×");
  close.title = "Close Timeline Tools";
  close.style.padding = "2px 8px";
  header.append(title, close);
  const body = document.createElement("div");
  body.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:12px;";
  for (const button of toolButtons) {
    button.style.width = "100%";
    button.style.whiteSpace = "normal";
    body.append(button);
  }
  toolsWindow.append(header, body);
  overlay.append(toolsWindow);
  const clamp = (value, maximum) => Math.max(8, Math.min(value, Math.max(8, maximum)));
  toolsButton.onclick = () => {
    if (toolsWindow.style.display !== "none") {
      toolsWindow.style.display = "none";
      return;
    }
    const source = toolsButton.getBoundingClientRect();
    toolsWindow.style.display = "block";
    toolsWindow.style.left = `${clamp(source.left, window.innerWidth - toolsWindow.offsetWidth - 8)}px`;
    toolsWindow.style.top = `${clamp(source.top - 190, window.innerHeight - toolsWindow.offsetHeight - 8)}px`;
  };
  close.onclick = () => { toolsWindow.style.display = "none"; };
  header.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || close.contains(event.target)) return;
    event.preventDefault();
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = toolsWindow.offsetLeft;
    const startTop = toolsWindow.offsetTop;
    const move = (moveEvent) => {
      toolsWindow.style.left = `${clamp(startLeft + moveEvent.clientX - startX, window.innerWidth - toolsWindow.offsetWidth - 8)}px`;
      toolsWindow.style.top = `${clamp(startTop + moveEvent.clientY - startY, window.innerHeight - toolsWindow.offsetHeight - 8)}px`;
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  });

  const deleteAllButton = makeButton("Delete All…");
  deleteAllButton.title = "Open bulk timeline deletion actions. Each action asks for confirmation.";
  deleteAllButton.style.display = "none";
  const deleteMenu = document.createElement("div");
  deleteMenu.style.cssText = "position:fixed;z-index:100004;display:none;flex-direction:column;gap:5px;min-width:190px;padding:7px;border:1px solid #7f1d1d;border-radius:7px;background:#18181b;box-shadow:0 14px 40px rgba(0,0,0,.55);";
  const closeDeleteMenu = () => {
    deleteMenu.style.display = "none";
    window.removeEventListener("pointerdown", closeOnOutside, true);
    window.removeEventListener("keydown", closeOnEscape, true);
  };
  const closeOnOutside = (event) => {
    if (!deleteMenu.contains(event.target) && event.target !== deleteAllButton) closeDeleteMenu();
  };
  const closeOnEscape = (event) => { if (event.key === "Escape") closeDeleteMenu(); };
  for (const button of deleteButtons) {
    button.style.width = "100%";
    button.addEventListener("click", closeDeleteMenu);
    deleteMenu.append(button);
  }
  const refreshDeleteActions = () => {
    const availability = getDeleteAvailability();
    const visible = [availability.images, availability.videos, availability.segments];
    deleteButtons.forEach((button, index) => {
      button.style.display = visible[index] ? "" : "none";
      button.disabled = !visible[index];
    });
    deleteAllButton.style.display = visible.some(Boolean) ? "" : "none";
    if (!visible.some(Boolean)) closeDeleteMenu();
  };
  overlay.append(deleteMenu);
  deleteAllButton.onclick = () => {
    refreshDeleteActions();
    if (deleteAllButton.style.display === "none") return;
    if (deleteMenu.style.display !== "none") {
      closeDeleteMenu();
      return;
    }
    const source = deleteAllButton.getBoundingClientRect();
    deleteMenu.style.display = "flex";
    deleteMenu.style.left = `${clamp(source.left, window.innerWidth - deleteMenu.offsetWidth - 8)}px`;
    deleteMenu.style.top = `${clamp(source.top - deleteMenu.offsetHeight - 8, window.innerHeight - deleteMenu.offsetHeight - 8)}px`;
    window.addEventListener("pointerdown", closeOnOutside, true);
    window.addEventListener("keydown", closeOnEscape, true);
  };
  return { toolsButton, deleteAllButton, refreshDeleteActions };
}
