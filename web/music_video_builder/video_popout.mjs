// Pop-out for the preview player. The whole player stage (video, image, compare view, status) moves into a real
// browser window that can be dragged to another monitor. It is the same element, so playback, the timeline sync and
// every handler keep working. Closing the window, or clicking the button again, puts the player back in the Builder.
// Uses a Picture-in-Picture document window where the browser has one (always on top), otherwise a normal pop-up.

const POPOUT_NAME = "vrgdg_video_popout";
const POPOUT_WIDTH = 960;
const POPOUT_HEIGHT = 560;
const BUTTON_STYLE = "position:absolute;right:8px;top:8px;z-index:3;padding:3px 8px;border:1px solid #155e75;border-radius:5px;background:rgba(8,51,68,.82);color:#cffafe;font-size:11px;font-weight:700;cursor:pointer;opacity:.75;";

function copyStyles(fromDocument, toDocument) {
  for (const node of fromDocument.querySelectorAll('style, link[rel="stylesheet"]')) {
    try {
      const copy = node.cloneNode(true);
      // A relative link such as user.css would be looked up from about:blank and fail, so use the full address.
      if (node.href) copy.href = node.href;
      toDocument.head.append(copy);
    } catch {
      // A style that cannot be copied only affects looks.
    }
  }
}

// Returns { target, reason }. ``reason`` says why the borderless floating window was not used, or is empty when it was.
async function openPopoutWindow() {
  const pip = window.documentPictureInPicture;
  let reason = "";
  if (pip?.requestWindow) {
    try {
      return { target: await pip.requestWindow({ width: POPOUT_WIDTH, height: POPOUT_HEIGHT }), reason };
    } catch (error) {
      reason = `the browser refused it (${String(error?.message || error)})`;
    }
  } else if (!window.isSecureContext) {
    reason = `this page is not a secure context. Open ComfyUI at http://127.0.0.1:${location.port || 8188} or http://localhost:${location.port || 8188} instead of ${location.origin}`;
  } else {
    reason = "this browser has no Picture-in-Picture window support";
  }
  console.warn(`[VRGDG Video] Floating player window unavailable: ${reason}`);
  const target = window.open("", POPOUT_NAME, `popup=yes,width=${POPOUT_WIDTH},height=${POPOUT_HEIGHT},resizable=yes,scrollbars=no,location=no,toolbar=no,menubar=no,status=no`);
  return { target, reason };
}

export function createVideoPopout({ previewStage, previewVideo, previewVideoState, toast }) {
  const button = document.createElement("button");
  button.type = "button";
  button.style.cssText = BUTTON_STYLE;
  button.onmouseenter = () => { button.style.opacity = "1"; };
  button.onmouseleave = () => { button.style.opacity = ".75"; };

  let popout = null;
  let placeholder = null;
  let watchTimer = null;
  let opening = false;

  function refreshButton() {
    button.textContent = popout ? "Pop in" : "Pop out";
    button.title = popout
      ? "Put the player back in the Builder. Closing the pop-out window does the same."
      : "Move the player into its own window you can drag to another screen.";
  }

  // Moving a playing video between documents pauses it. That pause is not the user's, so it must not stop the
  // timeline, and playback continues where it was.
  function moveStage(place) {
    const wasPlaying = !previewVideo.paused && !previewVideo.ended;
    previewVideoState.syncPause = true;
    try {
      place();
    } finally {
      setTimeout(() => { previewVideoState.syncPause = false; }, 150);
    }
    if (wasPlaying) previewVideo.play().catch(() => {});
  }

  function popIn() {
    if (!popout) return;
    const closing = popout;
    popout = null;
    clearInterval(watchTimer);
    watchTimer = null;
    if (placeholder?.isConnected) moveStage(() => placeholder.replaceWith(previewStage));
    placeholder = null;
    refreshButton();
    try {
      closing.close();
    } catch {
      // Already closed.
    }
  }

  async function popOut() {
    if (popout || opening) return;
    opening = true;
    let target = null;
    let reason = "";
    try {
      ({ target, reason } = await openPopoutWindow());
    } finally {
      opening = false;
    }
    if (!target) {
      toast("The browser blocked the pop-out window. Allow pop-ups for this site and try again.", true);
      return;
    }
    if (reason) toast(`Using a normal window because ${reason}.`, true);
    popout = target;
    const doc = target.document;
    doc.title = "VRGDG Video Player";
    copyStyles(document, doc);
    const base = doc.createElement("style");
    base.textContent = "html,body{margin:0;width:100%;height:100%;background:#09090b;overflow:hidden;}";
    doc.head.append(base);
    placeholder = document.createElement("div");
    placeholder.textContent = "The player is in its own window. Close that window to bring it back.";
    placeholder.style.cssText = "display:flex;align-items:center;justify-content:center;min-height:0;color:#71717a;font-size:13px;text-align:center;padding:16px;";
    moveStage(() => {
      previewStage.replaceWith(placeholder);
      doc.body.append(previewStage);
    });
    refreshButton();
    target.addEventListener("pagehide", popIn);
    // The Builder closing (or reloading) must not leave the player stranded in a window.
    watchTimer = setInterval(() => {
      if (!placeholder?.isConnected || target.closed) popIn();
    }, 1000);
  }

  button.onclick = () => {
    if (popout) popIn();
    else popOut();
  };
  window.addEventListener("pagehide", () => {
    try {
      popout?.close();
    } catch {
      // Nothing to close.
    }
  });
  refreshButton();
  return { button, popIn, isPoppedOut: () => Boolean(popout) };
}
