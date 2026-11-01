import { api } from "../../../scripts/api.js";
import { BUILDER_FONT_STACK } from "./constants.mjs";
import { confirmDestructiveAction } from "./confirm_dialog.mjs";
import { makeButton, makeInput, makeSelect, toast } from "./controls.mjs";
import { loadRefmodLibrary, refmodLibraryChanged, refmodPreviewUrl } from "./refmod_card.mjs";
import {
  addServerRefmod, askReportReason, loadServerLibrary, openServerDialog, removeServerRefmod, reportServerRefmod, serverPreviewUrl, shareRefmod,
} from "./refmods_server.mjs";
import { REFMOD_ACTIVE_TYPES } from "./refmods_studio.mjs";
import {
  addedMessage, filterRefmods, formatBytes, formatCanvas, formatTokens, groupByType, isSignInRequired, prettyRefmodName, refmodKindLabel,
  serverHost, sharedMessage, statusLabel, typeLabel, VIEWER_SORTS,
} from "./refmods_viewer_data.mjs";

// RefMods Viewer: browse the saved RefMods by category with their pictures. RefMods are made in RefMods Studio. The left
// list is the categories with a count, the middle is a grid of cards, and clicking a card shows its details on the
// right. Data comes from GET /vrgdg/refmod/library, pictures from GET /vrgdg/refmod/preview.
// "Server RefMods" shows what other people shared on the RefMod server (GET /vrgdg/refmod/server/library): the same
// categories and cards, with who shared each one. A shared RefMod is added to your library from its details, and one of
// yours is sent to the server from its details with "Share to server". The server needs a sign-in (GitHub or Discord),
// which the dialog in refmods_server.mjs handles. A new account's uploads wait for approval and show only to their owner
// until then, and any shared RefMod can be reported.

const TYPE_LABELS = Object.fromEntries(REFMOD_ACTIVE_TYPES.map((item) => [item.value, item.label]));
const TYPE_ORDER = REFMOD_ACTIVE_TYPES.map((item) => item.value);
const TYPE_HINTS = Object.fromEntries(REFMOD_ACTIVE_TYPES.map((item) => [item.value, item.hint]));

const BACKDROP_STYLE = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:16px;";
const BOX_STYLE = `width:min(1500px,calc(100vw - 32px));height:calc(100vh - 32px);box-sizing:border-box;overflow:hidden;border:1px solid #3f3f46;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.55);padding:16px;display:flex;flex-direction:column;gap:12px;font-family:${BUILDER_FONT_STACK};`;
const PANEL_STYLE = "border:1px solid #27272a;border-radius:8px;background:#18181b;min-height:0;min-width:0;";
const BADGE_STYLE = "display:inline-block;border-radius:999px;padding:2px 8px;font-size:10px;font-weight:800;line-height:1.5;";

const div = (css, text) => {
  const element = document.createElement("div");
  if (css) element.style.cssText = css;
  if (text !== undefined) element.textContent = text;
  return element;
};

const badge = (text, background, color) => div(`${BADGE_STYLE}background:${background};color:${color};`, text);

// A picture that falls back to the first letter of the name when the RefMod has no preview or it fails to load.
export function pictureElement(item, css, previewUrl = refmodPreviewUrl(item.name)) {
  const wrap = div(`${css}position:relative;overflow:hidden;background:#09090b;display:flex;align-items:center;justify-content:center;`);
  const placeholder = div("font-size:34px;font-weight:900;color:#3f3f46;", prettyRefmodName(item.name).charAt(0) || "?");
  wrap.append(placeholder);
  if (item.has_preview) {
    const image = document.createElement("img");
    image.loading = "lazy";
    image.alt = prettyRefmodName(item.name);
    image.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;";
    image.onload = () => placeholder.remove();
    image.onerror = () => image.remove();
    image.src = previewUrl;
    wrap.append(image);
  }
  return wrap;
}

export function openRefModsViewer({ onOpenStudio } = {}) {
  document.getElementById("vrgdg-refmods-viewer")?.remove();

  let source = "mine"; // "mine": the RefMods in this library, "server": what others shared on the RefMod server
  let library = [];
  let loaded = false;
  let loadError = "";
  let needsSignIn = false;
  let serverUrl = "";
  let serverAccount = "";
  let activeType = "";
  let selectedKey = "";
  const view = { query: "", sort: "name" };
  const onServer = () => source === "server";
  // A RefMod in this library is known by its name (type/name). A shared one by its id, because two people can share the same name.
  const keyOf = (item) => (onServer() ? item.id : item.name);
  const pictureOf = (item, css) => pictureElement(item, css, onServer() ? serverPreviewUrl(item.id) : refmodPreviewUrl(item.name));

  const backdrop = document.createElement("div");
  backdrop.id = "vrgdg-refmods-viewer";
  backdrop.setAttribute("role", "dialog");
  backdrop.setAttribute("aria-modal", "true");
  backdrop.style.cssText = BACKDROP_STYLE;
  const box = div(BOX_STYLE);

  const close = () => {
    document.removeEventListener("keydown", onKey, true);
    backdrop.remove();
  };
  const onKey = (event) => {
    // Escape belongs to the delete confirmation while it is open.
    if (event.key === "Escape" && !document.querySelector('[role="alertdialog"]')) {
      event.stopPropagation();
      close();
    }
  };

  // Header: title, search, sort, refresh, close.
  const header = div("display:flex;align-items:center;gap:12px;flex:0 0 auto;flex-wrap:wrap;");
  const titleWrap = div("flex:1 1 220px;min-width:0;");
  const title = div("font-size:18px;font-weight:900;", "RefMods Viewer");
  const subtitle = div("font-size:12px;color:#a1a1aa;margin-top:2px;", "Loading RefMods...");
  titleWrap.append(title, subtitle);
  const search = makeInput("");
  search.placeholder = "Search name, description or tag";
  search.style.maxWidth = "280px";
  const sort = makeSelect(VIEWER_SORTS, "name");
  sort.title = "Sort the cards.";
  sort.style.width = "140px";
  const refresh = makeButton("Refresh");
  const tabs = div("display:flex;gap:4px;flex:0 0 auto;");
  const tabButtons = [["mine", "My RefMods", "The RefMods saved on this computer."], ["server", "Server RefMods", "RefMods other people shared on the RefMod server."]].map(([value, label, hint]) => {
    const button = makeButton(label);
    button.title = hint;
    button.onclick = () => switchSource(value);
    return [value, button];
  });
  tabs.append(...tabButtons.map(([, button]) => button));
  const serverSettings = makeButton("Server settings");
  serverSettings.title = "The address of the RefMod server and the account you share with.";
  serverSettings.onclick = async () => {
    const { confirmed, changed } = await openServerDialog();
    if (confirmed || changed) load(true);
  };
  const studioButton = makeButton("Open RefMods Studio");
  studioButton.onclick = () => {
    close();
    onOpenStudio?.();
  };
  if (!onOpenStudio) studioButton.style.display = "none";
  const closeButton = makeButton("Close");
  closeButton.onclick = close;
  header.append(titleWrap, tabs, search, sort, refresh, serverSettings, studioButton, closeButton);

  // Body: categories | grid | details.
  const body = div("display:grid;grid-template-columns:210px minmax(0,1fr) 340px;gap:12px;flex:1 1 auto;min-height:0;");
  const categories = div(`${PANEL_STYLE}padding:8px;display:flex;flex-direction:column;gap:4px;overflow:auto;`);
  const gridPanel = div(`${PANEL_STYLE}padding:12px;overflow:auto;`);
  const grid = div("display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;align-content:start;");
  gridPanel.append(grid);
  const details = div(`${PANEL_STYLE}padding:12px;overflow:auto;display:flex;flex-direction:column;gap:10px;`);
  body.append(categories, gridPanel, details);

  const emptyMessage = (headline, text, buttons = []) => {
    const wrap = div("grid-column:1/-1;display:flex;flex-direction:column;align-items:center;gap:8px;padding:48px 16px;text-align:center;");
    wrap.append(div("font-size:15px;font-weight:800;color:#e4e4e7;", headline), div("font-size:12px;color:#a1a1aa;max-width:420px;line-height:1.5;", text));
    if (buttons.length) {
      const row = div("display:flex;gap:8px;flex-wrap:wrap;justify-content:center;");
      row.append(...buttons);
      wrap.append(row);
    }
    return wrap;
  };

  const studioPrompt = () => {
    if (!onOpenStudio) return [];
    const open = makeButton("Open RefMods Studio", "primary");
    open.onclick = () => {
      close();
      onOpenStudio();
    };
    return [open];
  };

  const retryButtons = () => {
    const retry = makeButton("Try again", "primary");
    retry.onclick = () => load(true);
    const settings = makeButton("Server settings");
    settings.onclick = serverSettings.onclick;
    return [retry, settings];
  };

  const signInButtons = () => {
    const signIn = makeButton("Sign in", "primary");
    signIn.onclick = async () => {
      const { changed } = await openServerDialog({
        title: "Sign in to the RefMod server",
        message: ["The server library is for signed-in people. Sign in with GitHub or Discord. Your account name is shown with what you share."],
        confirmLabel: "Done",
        requireSignIn: true,
      });
      if (changed) load(true);
    };
    return [signIn];
  };

  const renderCategories = () => {
    categories.replaceChildren();
    const groups = groupByType(library, TYPE_ORDER);
    const entries = [{ type: "", label: "All RefMods", count: library.length }, ...groups.map((group) => ({ ...group, label: typeLabel(group.type, TYPE_LABELS) }))];
    for (const entry of entries) {
      const active = entry.type === activeType;
      const row = document.createElement("button");
      row.type = "button";
      row.style.cssText = `display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;box-sizing:border-box;text-align:left;border:1px solid ${active ? "#0891b2" : "transparent"};border-radius:6px;background:${active ? "#164e63" : "transparent"};color:${active ? "#ecfeff" : "#d4d4d8"};font-family:${BUILDER_FONT_STACK};font-size:13px;font-weight:${active ? 800 : 600};padding:8px 10px;cursor:pointer;`;
      row.title = entry.type ? TYPE_HINTS[entry.type] || "" : "Every category.";
      row.append(div("min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", entry.label), badge(String(entry.count), active ? "#0e7490" : "#27272a", "#f4f4f5"));
      row.onclick = () => {
        activeType = entry.type;
        render();
      };
      categories.append(row);
    }
  };

  const card = (item) => {
    const selected = keyOf(item) === selectedKey;
    const element = document.createElement("button");
    element.type = "button";
    element.style.cssText = `display:flex;flex-direction:column;gap:0;padding:0;overflow:hidden;text-align:left;border:2px solid ${selected ? "#22d3ee" : "#27272a"};border-radius:8px;background:#0f172a;color:#f8fafc;font-family:${BUILDER_FONT_STACK};cursor:pointer;`;
    element.title = item.description || item.name;
    element.append(pictureOf(item, "width:100%;aspect-ratio:1/1;"));
    const info = div("padding:8px 10px;display:flex;flex-direction:column;gap:5px;min-width:0;");
    info.append(div("font-size:13px;font-weight:800;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", prettyRefmodName(item.name)));
    const badges = div("display:flex;gap:5px;flex-wrap:wrap;");
    badges.append(
      badge(typeLabel(item.type, TYPE_LABELS), "#134e4a", "#99f6e4"),
      badge(item.kind === "video" ? "Video" : "Picture", item.kind === "video" ? "#4c1d95" : "#1e3a8a", item.kind === "video" ? "#ddd6fe" : "#bfdbfe"),
    );
    if (onServer() && statusLabel(item.status)) badges.append(badge(statusLabel(item.status), "#713f12", "#fde68a"));
    info.append(badges);
    if (onServer()) info.append(div("font-size:11px;color:#a1a1aa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", item.mine ? "Shared by you" : `Shared by ${item.uploader || "Anonymous"}`));
    element.append(info);
    element.onclick = () => {
      selectedKey = keyOf(item);
      render();
    };
    return element;
  };

  const detailRow = (label, value) => {
    const row = div("display:flex;justify-content:space-between;gap:10px;font-size:12px;padding:5px 0;border-bottom:1px solid #27272a;");
    row.append(div("color:#a1a1aa;flex:0 0 auto;", label), div("color:#f4f4f5;font-weight:700;text-align:right;min-width:0;overflow-wrap:anywhere;", value));
    return row;
  };

  const renderDetails = (visible) => {
    details.replaceChildren();
    const item = visible.find((entry) => keyOf(entry) === selectedKey) || library.find((entry) => keyOf(entry) === selectedKey);
    if (!item) {
      details.append(
        div("font-size:14px;font-weight:800;color:#e4e4e7;", "Select a RefMod"),
        div("font-size:12px;color:#a1a1aa;line-height:1.5;", "Click a card to see its picture, type, size, description and tags."),
      );
      return;
    }
    details.append(pictureOf(item, "width:100%;aspect-ratio:1/1;border-radius:8px;flex:0 0 auto;"));
    details.append(div("font-size:16px;font-weight:900;overflow-wrap:anywhere;", prettyRefmodName(item.name)));
    details.append(div("font-size:11px;color:#71717a;overflow-wrap:anywhere;margin-top:-6px;", item.name));
    const rows = [
      ["Category", typeLabel(item.type, TYPE_LABELS)],
      ["Kind", refmodKindLabel(item)],
      ["Tokens", formatTokens(item.tokens)],
      ["Canvas", formatCanvas(item.canvas)],
      ["File size", formatBytes(item.size)],
      ["Mode", item.mode],
      ["Shared by", onServer() ? (item.mine ? "You" : item.uploader || "Anonymous") : ""],
      ["Status", onServer() ? statusLabel(item.status) : ""],
    ].filter(([, value]) => value);
    for (const [label, value] of rows) details.append(detailRow(label, value));
    if (item.description) {
      details.append(div("font-size:11px;font-weight:800;color:#a1a1aa;margin-top:4px;", "DESCRIPTION"), div("font-size:12px;line-height:1.5;color:#e4e4e7;white-space:pre-wrap;", item.description));
    }
    if (item.tags?.length) {
      const tags = div("display:flex;gap:5px;flex-wrap:wrap;");
      for (const tag of item.tags) tags.append(badge(tag, "#27272a", "#d4d4d8"));
      details.append(div("font-size:11px;font-weight:800;color:#a1a1aa;margin-top:4px;", "TAGS"), tags);
    }
    const copy = makeButton("Copy name");
    copy.title = "Copy this RefMod's name (type/name).";
    copy.onclick = async () => {
      try {
        await navigator.clipboard.writeText(item.name);
        toast(`Copied ${item.name}`);
      } catch (error) {
        toast("Could not copy to the clipboard.", true);
      }
    };
    const dangerous = (label, title, onClick) => {
      const button = makeButton(label);
      button.title = title;
      button.style.cssText += "border-color:#7f1d1d;background:#450a0a;color:#fecaca;";
      button.onclick = onClick;
      return button;
    };
    const actions = div("display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;");
    if (onServer()) {
      const add = makeButton("Add to my library", "primary");
      add.title = "Copy this RefMod and its picture into your models/refmods folder. Nothing of yours is replaced.";
      add.onclick = () => addFromServer(item, add);
      actions.append(add, copy);
      if (item.mine) actions.append(dangerous("Remove from server", "Take this RefMod off the server. Copies other people already added stay with them.", () => removeFromServer(item)));
      else {
        const report = makeButton("Report");
        report.title = "Tell the owner of the server this RefMod should not be here. Enough reports hide it until the owner has looked.";
        report.onclick = () => reportToServer(item, report);
        actions.append(report);
      }
    } else {
      const share = makeButton("Share to server");
      share.title = "Send this RefMod and its picture to the RefMod server so other people can add it.";
      share.onclick = () => shareToServer(item, share);
      actions.append(share, copy, dangerous("Delete RefMod", "Permanently delete this RefMod from models/refmods.", () => deleteRefmod(item)));
    }
    details.append(actions);
  };

  // Runs a server action behind its button and tells the user how it went.
  const withServer = async (button, work) => {
    button.disabled = true;
    try {
      await work();
    } catch (error) {
      toast(String(error?.message || error), true);
    } finally {
      button.disabled = false;
    }
  };

  const shareToServer = async (item, button) => {
    const { confirmed } = await openServerDialog({
      title: "Share with the server",
      message: [
        `This sends "${prettyRefmodName(item.name)}" and its picture to the RefMod server, where anyone using it can see it and add it to their library, under the name of your account.`,
        "Only share RefMods you made or have permission to share. The first RefMods from a new account are checked by the owner of the server before others see them.",
      ],
      confirmLabel: "Share",
      requireSignIn: true,
    });
    if (!confirmed) return;
    await withServer(button, async () => {
      toast(sharedMessage(await shareRefmod(item.name), item.name));
    });
  };

  const reportToServer = async (item, button) => {
    const reason = await askReportReason({
      title: "Report this RefMod",
      message: [`Tell the owner of the server why "${prettyRefmodName(item.name)}" shared by ${item.uploader || "Anonymous"} should not be here.`],
    });
    if (reason === null) return;
    await withServer(button, async () => {
      await reportServerRefmod(item.id, reason);
      toast("Thanks. Your report was sent to the owner of the server.");
      await load(true);
    });
  };

  const addFromServer = (item, button) => withServer(button, async () => {
    const result = await addServerRefmod(item.id);
    refmodLibraryChanged();
    toast(addedMessage(result));
  });

  const removeFromServer = async (item) => {
    const { confirmed } = await confirmDestructiveAction({
      title: "Remove from the server?",
      message: [
        `This takes "${prettyRefmodName(item.name)}" off the RefMod server so no one else can add it.`,
        "Copies other people already added stay with them. Your own copy is not touched.",
      ],
      details: [item.name],
      confirmLabel: "Remove from server",
    });
    if (!confirmed) return;
    try {
      await removeServerRefmod(item.id);
    } catch (error) {
      toast(String(error?.message || error), true);
      return;
    }
    if (selectedKey === item.id) selectedKey = "";
    toast(`Removed ${prettyRefmodName(item.name)} from the server.`);
    await load(true);
  };

  const deleteRefmod = async (item) => {
    const { confirmed } = await confirmDestructiveAction({
      title: "Delete this RefMod?",
      message: [
        `This permanently deletes the RefMod "${prettyRefmodName(item.name)}" and its picture from your models/refmods folder.`,
        "This cannot be undone. Scenes and Reference Builder cards that use it will no longer find it.",
      ],
      details: [item.name],
      confirmLabel: "Delete RefMod",
    });
    if (!confirmed) return;
    try {
      const response = await api.fetchApi("/vrgdg/refmod/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: item.name }),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || !result?.ok) throw new Error(result?.error || `Delete failed (${response.status}).`);
    } catch (error) {
      toast(String(error?.message || error), true);
      return;
    }
    refmodLibraryChanged();
    if (selectedKey === item.name) selectedKey = "";
    toast(`Deleted ${prettyRefmodName(item.name)}`);
    await load(true);
  };

  const render = () => {
    const server = onServer();
    const visible = filterRefmods(library, { type: activeType, query: view.query, sort: view.sort });
    const total = library.length;
    const place = server ? `shared on ${serverHost(serverUrl) || "the server"}${serverAccount ? `, signed in as ${serverAccount}` : ""}` : "";
    subtitle.textContent = !loaded
      ? "Loading RefMods..."
      : needsSignIn
        ? "Sign in to see the server RefMods"
        : loadError
        ? "Could not load the server RefMods"
        : total
          ? `${visible.length} of ${total} RefMod${total === 1 ? "" : "s"}${activeType ? ` in ${typeLabel(activeType, TYPE_LABELS)}` : ""}${place ? `, ${place}` : ""}`
          : server ? `Nothing shared yet on ${serverHost(serverUrl) || "the server"}` : "No RefMods yet";
    refresh.title = server ? "Ask the RefMod server again." : "Read the RefMods folders again.";
    serverSettings.style.display = server ? "" : "none";
    for (const [value, button] of tabButtons) {
      const active = value === source;
      button.style.borderColor = active ? "#0891b2" : "#3f3f46";
      button.style.background = active ? "#164e63" : "#27272a";
      button.style.color = active ? "#ecfeff" : "#f4f4f5";
    }
    renderCategories();
    grid.replaceChildren();
    if (!loaded) {
      grid.append(emptyMessage("Loading...", server ? "Asking the RefMod server." : "Reading the RefMods folders."));
    } else if (needsSignIn) {
      grid.append(emptyMessage("Sign in to see what others shared", "The RefMod server is for signed-in people. Sign in with GitHub or Discord. It takes a moment and your account name is shown with what you share.", signInButtons()));
    } else if (loadError) {
      grid.append(emptyMessage("Could not use the server", loadError, retryButtons()));
    } else if (!total) {
      grid.append(server
        ? emptyMessage("Nothing shared yet", "Open My RefMods, pick a RefMod and choose Share to server. It shows up here for everyone using this server.")
        : emptyMessage("No RefMods yet", "RefMods you make in RefMods Studio are saved under models/refmods and show up here with their pictures.", studioPrompt()));
    } else if (!visible.length) {
      grid.append(emptyMessage("Nothing matches", view.query ? `No RefMod matches "${view.query}"${activeType ? " in this category" : ""}.` : "This category is empty."));
    } else {
      for (const item of visible) grid.append(card(item));
    }
    renderDetails(visible);
  };

  const load = async (force) => {
    const wanted = source;
    refresh.disabled = true;
    loadError = "";
    needsSignIn = false;
    try {
      if (wanted === "server") {
        const result = await loadServerLibrary();
        if (wanted !== source) return;
        library = result.refmods || [];
        serverUrl = result.url || "";
        serverAccount = result.account || "";
      } else {
        library = await loadRefmodLibrary(force);
      }
    } catch (error) {
      if (wanted !== source) return;
      library = [];
      needsSignIn = isSignInRequired(error);
      loadError = needsSignIn ? "" : String(error?.message || error);
    } finally {
      refresh.disabled = false;
    }
    if (wanted !== source) return;
    loaded = true;
    if (activeType && !library.some((item) => item.type === activeType)) activeType = "";
    if (selectedKey && !library.some((item) => keyOf(item) === selectedKey)) selectedKey = "";
    render();
  };

  const switchSource = (value) => {
    if (value === source) return;
    source = value;
    library = [];
    loaded = false;
    loadError = "";
    needsSignIn = false;
    selectedKey = "";
    activeType = "";
    render();
    load(true);
  };

  search.addEventListener("input", () => {
    view.query = search.value;
    render();
  });
  sort.addEventListener("change", () => {
    view.sort = sort.value;
    render();
  });
  refresh.onclick = () => load(true);
  backdrop.addEventListener("click", (event) => {
    if (event.target === backdrop) close();
  });
  document.addEventListener("keydown", onKey, true);

  box.append(header, body);
  backdrop.append(box);
  document.body.append(backdrop);
  render();
  load(true);
  search.focus();
}
