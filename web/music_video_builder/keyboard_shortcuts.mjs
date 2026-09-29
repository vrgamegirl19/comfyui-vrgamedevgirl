import { toast } from "./controls.mjs";

export function wireKeyboardShortcuts({
  activeSegment, addSegmentButton, builderLifecycle, moveActiveSceneSelection, overlay, playButton, redo,
  segmentTrack, snapSceneEdgeToNearestBeat, undo,
}) {
  builderLifecycle.keydownHandler = (event) => {
    if (!overlay.isConnected) return;
    const target = event.target;
    const targetIsBuilder = target === document || target === document.body || target === document.documentElement || overlay.contains(target);
    if (!targetIsBuilder) return;
    const tag = String(event.target?.tagName || "").toLowerCase();
    const isTyping = tag === "input" || tag === "textarea" || tag === "select" || event.target?.isContentEditable;
    if (isTyping) return;
    const shortcutKey = event.key.toLowerCase();
    const isPlainShortcut = !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey;
    const snapSide = event.ctrlKey && !event.shiftKey && !event.altKey
      ? shortcutKey === "s"
        ? "start"
        : shortcutKey === "e"
          ? "end"
          : ""
      : "";
    if (isPlainShortcut && (event.code === "Space" || event.key === " ")) {
      event.preventDefault();
      event.stopPropagation();
      if (!event.repeat) playButton.click();
    } else if (isPlainShortcut && shortcutKey === "s") {
      event.preventDefault();
      event.stopPropagation();
      if (!event.repeat) addSegmentButton.click();
    } else if (snapSide) {
      event.preventDefault();
      event.stopPropagation();
      if (event.repeat) return;
      const segment = activeSegment();
      if (!segment || segmentTrack(segment) === "overlay") {
        toast("Select a base scene before using the beat-snap shortcut.", true);
        return;
      }
      snapSceneEdgeToNearestBeat(segment, snapSide).catch((error) => {
        toast(`Could not snap the scene ${snapSide}:\n${String(error?.message || error)}`, true);
      });
    } else if (event.ctrlKey && !event.shiftKey && shortcutKey === "z") {
      event.preventDefault();
      event.stopPropagation();
      undo();
    } else if ((event.ctrlKey && shortcutKey === "y") || (event.ctrlKey && event.shiftKey && shortcutKey === "z")) {
      event.preventDefault();
      event.stopPropagation();
      redo();
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key === "ArrowRight") {
      if (moveActiveSceneSelection(1)) {
        event.preventDefault();
        event.stopPropagation();
      }
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key === "ArrowLeft") {
      if (moveActiveSceneSelection(-1)) {
        event.preventDefault();
        event.stopPropagation();
      }
    }
  };
  document.addEventListener("keydown", builderLifecycle.keydownHandler, true);
  overlay.tabIndex = -1;
  setTimeout(() => overlay.focus(), 0);
}
