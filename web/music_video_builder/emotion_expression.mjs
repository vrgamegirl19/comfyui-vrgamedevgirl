// Emotion direction is interpreted by the LLM, not expanded into fixed acting prose.
export function emotionExpressionInput(segment = {}, state = {}) {
  return {
    emotion_expression_tags: String(segment.emotion_expression_tags || "").trim(),
    facial_performance: String(segment.facial_performance || state.defaultFacialPerformance || "").trim(),
    facial_performance_custom: String(segment.facial_performance_custom || state.defaultFacialPerformanceCustom || "").trim(),
  };
}

export function hasEmotionExpressionInput(segment = {}, state = {}) {
  const input = emotionExpressionInput(segment, state);
  return Boolean(input.emotion_expression_tags || (input.facial_performance !== "off"
    && (input.facial_performance || input.facial_performance_custom)));
}
