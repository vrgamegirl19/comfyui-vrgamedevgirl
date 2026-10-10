import { BUILDER_FONT_STACK } from "./constants.mjs";
import { makeButton } from "./controls.mjs";
import { refmodPreviewUrl } from "./refmod_card.mjs";
import { createdCardRows, prettyRefmodName } from "./refmods_viewer_data.mjs";

// The confirmation RefMods Studio shows once a RefMod is created: a card with its picture and details.
// `info` is the create result plus what the Studio knew: { name, folder, path, kind, tokens, canvas, quality,
// latent_frames, imageCount, mode, description, typeLabel }.

const div = (css, text) => {
  const element = document.createElement("div");
  if (css) element.style.cssText = css;
  if (text !== undefined) element.textContent = text;
  return element;
};

export function showRefmodCreatedCard(info) {
  return new Promise((resolve) => {
    const backdrop = document.createElement("div");
    backdrop.setAttribute("role", "dialog");
    backdrop.setAttribute("aria-modal", "true");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:20px;";
    const box = div(`width:min(480px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow:auto;box-sizing:border-box;border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`);
    box.append(div("font-size:16px;font-weight:900;color:#a7f3d0;", "Your RefMod was created"));

    const card = div("border:1px solid #27272a;border-radius:8px;background:#0f172a;overflow:hidden;display:flex;flex-direction:column;");
    const name = `${info.folder}/${info.name}`;
    const picture = div("width:100%;aspect-ratio:16/10;background:#09090b;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;");
    const placeholder = div("font-size:40px;font-weight:900;color:#3f3f46;", prettyRefmodName(info.name).charAt(0) || "?");
    picture.append(placeholder);
    const image = document.createElement("img");
    image.alt = prettyRefmodName(info.name);
    image.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;";
    image.onload = () => placeholder.remove();
    image.onerror = () => image.remove();
    image.src = `${refmodPreviewUrl(name)}&t=${Date.now()}`;
    picture.append(image);
    card.append(picture);

    const body = div("padding:12px;display:flex;flex-direction:column;gap:6px;");
    body.append(div("font-size:16px;font-weight:900;overflow-wrap:anywhere;", prettyRefmodName(info.name)), div("font-size:11px;color:#71717a;overflow-wrap:anywhere;", info.path || name));
    for (const [label, value] of createdCardRows(info)) {
      const row = div("display:flex;justify-content:space-between;gap:10px;font-size:12px;padding:4px 0;border-bottom:1px solid #27272a;");
      row.append(div("color:#a1a1aa;", label), div("color:#f4f4f5;font-weight:700;text-align:right;", value));
      body.append(row);
    }
    if (info.description) {
      body.append(div("font-size:11px;font-weight:800;color:#a1a1aa;margin-top:4px;", "DESCRIPTION"), div("font-size:12px;line-height:1.5;color:#e4e4e7;white-space:pre-wrap;", info.description));
    }
    card.append(body);
    box.append(card);

    const finish = () => {
      document.removeEventListener("keydown", onKey, true);
      backdrop.remove();
      resolve();
    };
    const onKey = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        finish();
      }
    };
    document.addEventListener("keydown", onKey, true);
    const done = makeButton("Done", "primary");
    done.onclick = finish;
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) finish();
    });
    box.append(done);
    backdrop.append(box);
    document.body.append(backdrop);
    done.focus();
  });
}
