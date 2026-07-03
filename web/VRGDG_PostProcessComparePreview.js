export function createPostProcessComparePreview({ makeImageUrl }) {
  const root = document.createElement("div");
  root.className = "vrgdg-post-process-compare-preview";
  root.style.cssText = [
    "display:none",
    "position:absolute",
    "inset:0",
    "background:#050505",
    "overflow:hidden",
    "align-items:center",
    "justify-content:center",
    "z-index:2",
    "user-select:none",
  ].join(";");

  const before = document.createElement("img");
  const after = document.createElement("img");
  for (const image of [before, after]) {
    image.alt = "";
    image.draggable = false;
    image.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#050505;";
  }

  const afterClip = document.createElement("div");
  afterClip.style.cssText = "position:absolute;inset:0;overflow:hidden;";
  afterClip.append(after);

  const divider = document.createElement("div");
  divider.style.cssText = [
    "position:absolute",
    "top:0",
    "bottom:0",
    "width:2px",
    "background:#ffffff",
    "box-shadow:0 0 0 1px rgba(0,0,0,.65),0 0 16px rgba(255,255,255,.35)",
    "transform:translateX(-1px)",
    "pointer-events:none",
  ].join(";");

  const handle = document.createElement("div");
  handle.style.cssText = [
    "position:absolute",
    "top:50%",
    "width:34px",
    "height:34px",
    "border-radius:999px",
    "background:rgba(255,255,255,.92)",
    "border:1px solid rgba(0,0,0,.45)",
    "box-shadow:0 3px 14px rgba(0,0,0,.45)",
    "transform:translate(-50%,-50%)",
    "display:flex",
    "align-items:center",
    "justify-content:center",
    "color:#111113",
    "font-size:15px",
    "font-weight:900",
    "pointer-events:none",
  ].join(";");
  handle.textContent = "<>";

  const beforeLabel = document.createElement("div");
  const afterLabel = document.createElement("div");
  beforeLabel.textContent = "Before";
  afterLabel.textContent = "After";
  for (const label of [beforeLabel, afterLabel]) {
    label.style.cssText = [
      "position:absolute",
      "top:10px",
      "padding:5px 8px",
      "border-radius:5px",
      "background:rgba(0,0,0,.62)",
      "color:#f4f4f5",
      "font-size:11px",
      "font-weight:900",
      "line-height:1",
      "pointer-events:none",
    ].join(";");
  }
  beforeLabel.style.left = "10px";
  afterLabel.style.right = "10px";

  const caption = document.createElement("div");
  caption.style.cssText = [
    "position:absolute",
    "left:10px",
    "right:10px",
    "bottom:10px",
    "padding:7px 9px",
    "border-radius:5px",
    "background:rgba(0,0,0,.58)",
    "color:#d4d4d8",
    "font-size:11px",
    "font-weight:800",
    "text-align:center",
    "pointer-events:none",
  ].join(";");

  root.append(before, afterClip, divider, handle, beforeLabel, afterLabel, caption);

  let slider = 0.5;
  let dragging = false;

  function syncSlider() {
    const percent = Math.max(0, Math.min(1, slider)) * 100;
    afterClip.style.clipPath = `inset(0 0 0 ${percent}%)`;
    divider.style.left = `${percent}%`;
    handle.style.left = `${percent}%`;
  }

  function setSliderFromEvent(event) {
    const rect = root.getBoundingClientRect();
    if (!rect.width) return;
    slider = (event.clientX - rect.left) / rect.width;
    syncSlider();
  }

  root.addEventListener("pointerdown", (event) => {
    dragging = true;
    root.setPointerCapture?.(event.pointerId);
    setSliderFromEvent(event);
  });
  root.addEventListener("pointermove", (event) => {
    if (dragging) setSliderFromEvent(event);
  });
  root.addEventListener("pointerup", (event) => {
    dragging = false;
    root.releasePointerCapture?.(event.pointerId);
  });
  root.addEventListener("pointercancel", () => {
    dragging = false;
  });

  syncSlider();

  return {
    element: root,
    show({ beforePath, afterPath, title = "Post-process preview" }) {
      if (!beforePath || !afterPath) return false;
      before.src = makeImageUrl(beforePath);
      after.src = makeImageUrl(afterPath);
      caption.textContent = title;
      root.style.display = "flex";
      syncSlider();
      return true;
    },
    hide() {
      root.style.display = "none";
      before.removeAttribute("src");
      after.removeAttribute("src");
    },
  };
}
