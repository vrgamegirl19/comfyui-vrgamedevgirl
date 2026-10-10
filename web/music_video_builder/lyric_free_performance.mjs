// Custom-audio performance policy. Python twin: minimax/lyric_free_performance.py.
export function lyricFreePerformanceEnabled(enabled, performanceMode, audioMode) {
  return Boolean(enabled) && performanceMode === "singing" && audioMode !== "built_in_audio";
}

export function cleanLyricFreeDescription(text, lyrics = [], options = {}) {
  let clean = String(text || "").replace(/<d>[\s\S]*?<\/d>/gi, "");
  for (const lyric of lyrics.filter(Boolean).sort((a, b) => b.length - a.length)) {
    const pattern = lyric.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
    clean = clean.replace(new RegExp(pattern, "gi"), "");
  }
  clean = clean.replace(/["“”']\s*["“”']/g, "");
  // Remove stale performance sentences, including ones returned despite the LLM instructions.
  const vocal = /\b(?:sing\w*|sang|sung|lip[ -]?sync\w*|lyrics?|vocals?|dialogue|mouth\w*|lips?|jaw)\b/i;
  const protectedText = clean.replace(/(\d)\.(?=\d)/g, "$1VRGDGDECIMALTOKEN");
  if (options.preservePerformance) {
    // Keep LLM-chosen emotion tags and acting rather than replacing them with stock prose.
    const safe = options.singleShot ? protectedText : protectedText.split(/(?<=[.!?…])\s+/).map((sentence) =>
      sentence.split(",").filter((clause) => !/\b(?:mouth\w*|lips?|jaw)\b/i.test(clause)).join(",")
    ).join(" ");
    return safe.replace(/VRGDGDECIMALTOKEN/g, ".").replace(/\s+/g, " ").trim();
  }
  return (protectedText.match(/[^.!?…]+[.!?…]+|[^.!?…]+$/g) || [])
    .filter((sentence) => !vocal.test(sentence)).join(" ").replace(/VRGDGDECIMALTOKEN/g, ".").replace(/\s+/g, " ").trim();
}

export function cleanLyricFreeFacialDirection(text) {
  const clauses = String(text || "").split(/[,.;]/).filter((clause) =>
    !/\b(?:mouth\w*|lips?|jaw|sing\w*|sang|sung|vocals?|lyrics?|teeth|notes)\b/i.test(clause)
  ).map((clause) => clause.trim()).filter(Boolean).join(", ").replace(/,\s*and\s+/g, ", ");
  return clauses ? `${clauses}.` : "";
}

export function lyricFreeShotDirection(performer, singleShot, timing = "", includeExpression = true) {
  return `${performer} sings with passion in sync with the vocals in <Audio 1>${timing ? ` during ${timing}` : ""}.`
    + (singleShot ? ` ${performer}'s natural mouth and jaw movement follows only the audible vocal phrasing.` : "")
    + (includeExpression ? ` ${performer}'s engaged eyes and expressive brows convey the song's intensity.` : "");
}

export function remainingLyricFreeContract(description, contract) {
  const [singing, ...rest] = contract.split(/(?<=[.!?])\s+/);
  const actor = singing.match(/^(.*?) sings\b/i)?.[1];
  const timing = singing.match(/ during (.+)\.$/)?.[1];
  const sentences = description.replace(/(\d)\.(?=\d)/g, "$1DECIMAL").split(/(?<=[.!?])\s+/);
  const actorPattern = actor && new RegExp(`${actor.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s+sings\\b.*\\bin sync\\b.*<Audio 1>`, "i");
  const covered = actor && sentences.some(s => actorPattern.test(s)
    && (!timing || s.includes(timing.replace(/(\d)\.(?=\d)/g, "$1DECIMAL"))));
  if (!covered) return contract;
  return rest.filter(s => !(/\bmouth|\bjaw/i.test(s)
    && sentences.some(existing => existing.includes(actor) && /\bmouth|\bjaw/i.test(existing)
      && /audible vocal phrasing/i.test(existing)))).join(" ");
}

export function lyricFreePromptContext(text, lyrics, directions, singleShot) {
  // Keep the shot task, visual planning and reference contracts; replace vocal contracts as a unit.
  const context = String(text || "").split(/\n\n/).filter((part) =>
    !/^(?:MANDATORY VOCAL PERFORMANCE|Vocal performance|Exact lyric|Timed singer|Performer \/ vocal|Multi-performer rule|AUTHORITATIVE PERFORMER)/i.test(part.trim())
  ).join("\n\n");
  let clean = context;
  for (const lyric of lyrics.filter(Boolean).sort((a, b) => b.length - a.length)) clean = clean.split(lyric).join("[audio vocal cue]");
  clean = clean.replace(/[^\n.!?]*\b(?:mouth\w*|lips?|jaw)\b[^\n.!?]*[.!?]?/gi, "");
  return `${clean}\n\nLYRIC-FREE CUSTOM AUDIO — MANDATORY:\nDo not quote, invent, or add lyric lines or <d> tags. ${singleShot ? "Mouth and jaw synchronization is allowed only during audible vocals." : "Do not mention mouth, lip, or jaw movement anywhere in this multi-shot prompt."} Instrumental intervals use only visual action and camera direction; do not mention singing in them. Preserve the supplied audio unchanged. Use selected facial direction through eyes, brows and expression.\n${directions.join("\n")}`;
}
