export function timelineEmotionNotesVisible(state) {
  return state.videoType === "speaking" && Boolean(state.showTimelineEmotionTags);
}

export function createTimelineEmotionNote({ segment, state, left, width, top, height, isActive, pushHistory, autoSaveSessionQuiet }) {
  const box = document.createElement("textarea");
  box.value = String(segment.emotion_expression_tags || "");
  box.placeholder = "Mood, emotion, expression…";
  box.setAttribute("aria-label", "Emotion Tag");
  box.maxLength = 1200;
  box.title = "Speaking direction for this scene. Describe a mood, emotion, expression, or progression; the LLM uses it to choose acting and speech delivery tags.";
  box.style.cssText = `position:absolute;left:${left}px;top:${top}px;width:${Math.max(24, width)}px;height:${height}px;box-sizing:border-box;resize:none;z-index:3;pointer-events:auto;border:1px solid ${isActive ? "#ef4444" : "#a16207"};border-radius:5px;background:rgba(66,32,6,.9);color:#fef3c7;padding:6px;font-size:11px;line-height:1.25;`;
  box.onpointerdown = box.onclick = box.onkeydown = event => event.stopPropagation();
  const liveSegment = () => state.segments.find(item => item.id === segment.id) || segment;
  let previous = box.value;
  let saved = box.value;
  let historyPushed = false;
  box.onfocus = () => {
    previous = box.value;
    saved = box.value;
    historyPushed = false;
  };
  box.oninput = () => {
    if (!historyPushed && box.value !== previous) {
      pushHistory();
      historyPushed = true;
    }
    liveSegment().emotion_expression_tags = box.value;
  };
  const save = () => {
    liveSegment().emotion_expression_tags = box.value.trim();
    saved = box.value;
    autoSaveSessionQuiet("timeline emotion tag edited");
  };
  box.onchange = save;
  box.onblur = () => { if (saved !== box.value) save(); };
  return box;
}
