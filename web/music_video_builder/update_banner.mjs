import { api } from "../../../scripts/api.js";
import { BUILDER_FONT_STACK } from "./constants.mjs";
import { makeButton } from "./controls.mjs";

export function buildUpdateBanner({ updateStatus }) {
  const updateStatusBanner = document.createElement("div");
  updateStatusBanner.style.cssText = `display:flex;align-items:center;gap:10px;min-height:34px;box-sizing:border-box;padding:5px 10px 5px 12px;border-bottom:1px solid #52525b;background:#27272a;color:#f4f4f5;font-family:${BUILDER_FONT_STACK};font-size:12px;font-weight:500;line-height:1.3;letter-spacing:0;`;
  const updateStatusDot = document.createElement("span");
  updateStatusDot.style.cssText = "width:9px;height:9px;flex:0 0 9px;border-radius:999px;background:#a1a1aa;box-shadow:0 0 0 3px rgba(161,161,170,.16);";
  const updateStatusText = document.createElement("span");
  updateStatusText.style.cssText = "min-width:0;flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
  updateStatusText.textContent = "AI Video Builder — Checking for updates…";
  const updateWhatsNewAction = makeButton("What's New");
  updateWhatsNewAction.style.cssText += `display:none;padding:4px 10px;min-height:24px;background:#164e63;border-color:#22d3ee;color:#ecfeff;font-family:${BUILDER_FONT_STACK};font-size:11px;font-weight:600;letter-spacing:0;`;
  const updateStatusAction = makeButton("Update");
  updateStatusAction.style.cssText += `display:none;padding:4px 10px;min-height:24px;background:#7f1d1d;border-color:#f87171;color:#fff;font-family:${BUILDER_FONT_STACK};font-size:11px;font-weight:600;letter-spacing:0;`;
  const updateStatusClose = document.createElement("button");
  updateStatusClose.type = "button";
  updateStatusClose.textContent = "×";
  updateStatusClose.title = "Dismiss this version status";
  updateStatusClose.setAttribute("aria-label", "Dismiss AI Video Builder version status");
  updateStatusClose.style.cssText = "width:24px;height:24px;flex:0 0 24px;display:flex;align-items:center;justify-content:center;padding:0;border:1px solid rgba(255,255,255,.32);border-radius:5px;background:rgba(0,0,0,.18);color:inherit;font:700 18px/1 sans-serif;cursor:pointer;";
  updateStatusBanner.append(updateStatusDot, updateStatusText, updateWhatsNewAction, updateStatusAction, updateStatusClose);

  const setUpdateStatusAppearance = (kind, text, detail = "") => {
    const styles = {
      current: { background: "#14532d", border: "#22c55e", dot: "#86efac", shadow: "rgba(134,239,172,.22)" },
      outdated: { background: "#7f1d1d", border: "#ef4444", dot: "#fecaca", shadow: "rgba(254,202,202,.24)" },
      unavailable: { background: "#713f12", border: "#eab308", dot: "#fde047", shadow: "rgba(253,224,71,.22)" },
      checking: { background: "#27272a", border: "#52525b", dot: "#a1a1aa", shadow: "rgba(161,161,170,.16)" },
    };
    const style = styles[kind] || styles.checking;
    updateStatusBanner.style.background = style.background;
    updateStatusBanner.style.borderBottomColor = style.border;
    updateStatusDot.style.background = style.dot;
    updateStatusDot.style.boxShadow = `0 0 0 3px ${style.shadow}`;
    updateStatusText.textContent = text;
    updateStatusBanner.title = detail || text;
    updateStatusAction.style.display = kind === "outdated" ? "inline-flex" : "none";
    const hasReleaseNotes = Array.isArray(updateStatus.payload?.release_notes?.releases)
      && updateStatus.payload.release_notes.releases.length > 0;
    updateWhatsNewAction.style.display = hasReleaseNotes ? "inline-flex" : "none";
    updateWhatsNewAction.title = kind === "outdated"
      ? "Review what will change before updating."
      : "Review recently installed features and fixes.";
  };

  const updateReleaseNotes = (payload = updateStatus.payload) => {
    const releases = payload?.release_notes?.releases;
    return Array.isArray(releases) ? releases.filter((release) => release && typeof release === "object") : [];
  };

  updateStatusClose.onclick = () => {
    updateStatusBanner.style.display = "none";
  };

  const refreshV10UpdateStatus = async () => {
    updateStatus.payload = null;
    setUpdateStatusAppearance("checking", "AI Video Builder — Checking for updates…");
    try {
      const response = await api.fetchApi("/vrgdg/update/v10/status");
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || `Status check failed (HTTP ${response.status}).`);
      }
      updateStatus.payload = payload;

      const installed = String(payload.installed_commit || "").slice(0, 7) || "unknown";
      const latest = String(payload.latest_commit || "").slice(0, 7) || "unknown";
      const behind = Math.max(0, Number(payload.behind || 0));
      const wrongBranch = Boolean(payload.expected_branch) && payload.branch !== payload.expected_branch;
      const isOutdated = Boolean(payload.outdated);
      // A feature/PR branch is a local development checkout, not an outdated
      // production checkout. Only offer the main-branch updater while we are
      // actually on the branch it is designed to fast-forward.
      if (isOutdated && !wrongBranch) {
        const countText = `${behind} commit${behind === 1 ? "" : "s"} behind`;
        const reason = countText;
        setUpdateStatusAppearance(
          "outdated",
          `AI Video Builder build ${installed} — Update available (${reason}; latest ${latest})`,
          `This installation is not at the latest production version on main (${reason}). Click Update to run the safe updater.`
        );
      } else {
        const localNotes = [];
        if (wrongBranch || Number(payload.ahead || 0) > 0) localNotes.push("local development build");
        if (payload.tracked_changes) localNotes.push("modified locally");
        const localNote = localNotes.length ? ` · ${localNotes.join(" · ")}` : "";
        setUpdateStatusAppearance(
          "current",
          `AI Video Builder build ${installed} — Up to date${localNote}`,
          `Installed AI Video Builder commit: ${payload.installed_commit}`
        );
      }
    } catch (error) {
      updateStatus.payload = null;
      setUpdateStatusAppearance(
        "unavailable",
        "AI Video Builder — Could not check for updates",
        String(error?.message || error)
      );
    }
  };

  return { refreshV10UpdateStatus, updateReleaseNotes, updateStatusAction, updateStatusBanner, updateWhatsNewAction };
}

export function createWhatsNewModal({ updateReleaseNotes, updateStatus, updateV10Button }) {
  function openWhatsNewModal({ mode = "current" } = {}) {
    const payload = updateStatus.payload || {};
    const allReleases = updateReleaseNotes(payload);
    const availableIds = new Set(
      Array.isArray(payload.available_release_ids)
        ? payload.available_release_ids.map((value) => String(value || ""))
        : []
    );
    const currentReleaseId = String(payload.current_release_id || "");
    let releases = allReleases;
    if (mode === "available" && availableIds.size) {
      releases = allReleases.filter((release) => availableIds.has(String(release.id || "")));
    } else if (mode === "current" && currentReleaseId) {
      releases = allReleases.filter((release) => (
        !String(release.commit || "").trim()
        || String(release.id || "") === currentReleaseId
      ));
    }
    if (!releases.length) releases = allReleases.slice(0, mode === "history" ? 8 : 3);

    document.querySelector(".vrgdg-builder-whats-new")?.remove();
    const backdrop = document.createElement("div");
    backdrop.className = "vrgdg-builder-whats-new";
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100030;background:rgba(0,0,0,.74);display:flex;align-items:center;justify-content:center;padding:20px;";
    const box = document.createElement("div");
    box.style.cssText = "width:min(760px,calc(100vw - 40px));max-height:min(820px,calc(100vh - 40px));border:1px solid #155e75;border-radius:11px;background:#0f172a;color:#f8fafc;box-shadow:0 26px 90px rgba(0,0,0,.65);display:flex;flex-direction:column;overflow:hidden;";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:flex-start;gap:12px;padding:16px 18px;border-bottom:1px solid #164e63;background:#082f49;";
    const headingWrap = document.createElement("div");
    headingWrap.style.cssText = "min-width:0;flex:1 1 auto;";
    const heading = document.createElement("div");
    heading.textContent = "What's New in AI Video Builder";
    heading.style.cssText = "font-size:19px;font-weight:950;color:#cffafe;";
    const summary = document.createElement("div");
    const installed = String(payload.installed_commit || "").slice(0, 7);
    const latest = String(payload.latest_commit || "").slice(0, 7);
    summary.textContent = payload.outdated
      ? `Review the changes available before updating${latest ? ` to build ${latest}` : ""}.`
      : `Recent changes in your installed build${installed ? ` ${installed}` : ""}.`;
    summary.style.cssText = "margin-top:4px;font-size:12px;line-height:1.4;color:#bae6fd;";
    headingWrap.append(heading, summary);
    const closeTop = makeButton("Close");
    header.append(headingWrap, closeTop);

    const content = document.createElement("div");
    content.style.cssText = "display:flex;flex-direction:column;gap:12px;padding:16px 18px;overflow:auto;min-height:0;";
    if (!releases.length) {
      const empty = document.createElement("div");
      empty.textContent = "Release notes are not available for this build yet.";
      empty.style.cssText = "border:1px dashed #475569;border-radius:8px;padding:18px;text-align:center;color:#94a3b8;font-size:13px;";
      content.append(empty);
    }

    for (const release of releases) {
      const card = document.createElement("section");
      card.style.cssText = "border:1px solid #334155;border-radius:9px;background:#111827;padding:14px;display:flex;flex-direction:column;gap:10px;";
      const releaseTitle = document.createElement("div");
      releaseTitle.textContent = String(release.title || release.version || "Builder update");
      releaseTitle.style.cssText = "font-size:16px;font-weight:900;color:#e0f2fe;";
      const meta = document.createElement("div");
      const metaParts = [
        release.version ? `Version ${release.version}` : "",
        release.date ? String(release.date) : "",
        release.restart_required ? "ComfyUI restart required" : "",
      ].filter(Boolean);
      meta.textContent = metaParts.join(" · ");
      meta.style.cssText = "font-size:11px;color:#67e8f9;";
      card.append(releaseTitle, meta);

      const sections = Array.isArray(release.sections) ? release.sections : [];
      for (const section of sections) {
        const sectionTitle = document.createElement("div");
        sectionTitle.textContent = String(section?.title || "Changes");
        sectionTitle.style.cssText = "margin-top:2px;font-size:12px;font-weight:900;color:#fef3c7;text-transform:uppercase;letter-spacing:.04em;";
        const list = document.createElement("ul");
        list.style.cssText = "margin:0;padding-left:20px;display:flex;flex-direction:column;gap:5px;color:#dbeafe;font-size:13px;line-height:1.45;";
        for (const itemText of Array.isArray(section?.items) ? section.items : []) {
          const item = document.createElement("li");
          item.textContent = String(itemText || "");
          list.append(item);
        }
        if (list.childElementCount) card.append(sectionTitle, list);
      }

      if (release.guide_url) {
        const guideButton = makeButton("Open Related Guide");
        guideButton.style.alignSelf = "flex-start";
        guideButton.onclick = () => window.open(String(release.guide_url), "_blank", "noopener,noreferrer");
        card.append(guideButton);
      }
      content.append(card);
    }

    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;justify-content:flex-end;gap:9px;padding:12px 18px;border-top:1px solid #334155;background:#111827;";
    const close = makeButton("Close");
    actions.append(close);
    if (payload.outdated) {
      const updateNow = makeButton("Update Now", "primary");
      updateNow.onclick = () => {
        backdrop.remove();
        updateV10Button.click();
      };
      actions.append(updateNow);
    }

    const finish = () => backdrop.remove();
    closeTop.onclick = finish;
    close.onclick = finish;
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) finish();
    });
    backdrop.addEventListener("keydown", (event) => {
      if (event.key === "Escape") finish();
    });
    box.append(header, content, actions);
    backdrop.append(box);
    document.body.append(backdrop);
    backdrop.tabIndex = -1;
    backdrop.focus();
  }

  return { openWhatsNewModal };
}
