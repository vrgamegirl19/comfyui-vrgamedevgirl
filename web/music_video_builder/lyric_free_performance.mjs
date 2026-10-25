// Custom-audio performance policy. Python twin: minimax/lyric_free_performance.py.
export function lyricFreePerformanceEnabled(enabled, performanceMode, audioMode) {
  return Boolean(enabled) && performanceMode === "singing" && audioMode !== "built_in_audio";
}

export function stripSingingArticulation(text) {
  const saved = [];
  // A silent person's closed lips ("<Subject 2> keeps his lips pressed together") are kept: they stop that person mouthing the song.
  let clean = String(text || '').replace(/<d>[\s\S]*?<\/d>|“[^”]*”|"[^"]*"|\[[^\]]*\]|\blips\s+(?:(?:stay|stays|staying|remain|remains|remaining|still|kept|firmly|tightly)\s+){0,2}pressed\s+(?:together|shut|closed)\b/gi, value => {
    saved.push(value);
    return `VRGDGPROTECTED${saved.length - 1}TOKEN`;
  }).replace(/(\d)\.(?=\d)/g, '$1VRGDGDECIMALTOKEN');
  // Remove oral-rest directions without losing the actor of the following action.
  clean = clean.replace(/(<Subject\s+\d+>(?:\s+\([^)]*\))?|\b(?:she|he|they))\s+(?:keeps?|holds?|lets?|maintains?)\s+(?:(?:her|his|their|a|an|the)\s+)?(?:(?:relaxed|closed|soft|neutral|gently|pressed|shut|still|natural|loose|calm)[ -]+){0,6}(?:mouth|lips?|jaw)\b[^,;.!?]*,\s*(?:then|and)\s+/gi, '$1 ');
  clean = clean.replace(/\b(?:her|his|their)\s+(?:mouth|lips?|jaw)\s+(?:close|closes|relax|relaxes|settle|settles)\b[^,;.!?]*?\s+and\s+(?=(?:she|he|they)\b)/gi, '');
  // Keep the adjacent eye/brow expression when an articulation clause introduces it.
  clean = clean.replace(/\bwith\s+(?:natural\s+)?(?:mouth|lips?|jaw)\b[^,;.!?]*,\s*/gi, 'with ');
  clean = clean.split(/(?<=[.!?…])\s+/).map(sentence => {
    const parts = sentence.split(/([,;])\s*/);
    let result = '';
    for (let i = 0; i < parts.length; i += 2) {
      const clause = parts[i].replace(/\s+(?:with|and|while)\s+(?:(?:her|his|their|natural)\s+)*(?:mouth|lips?|jaw)\b[^.!?]*/gi, '').trim();
      if (!clause || /\b(?:mouth\w*|lips?|jaw)\b/i.test(clause)) continue;
      if (/^(?:after|before|between|during)\s+(?:the\s+)?(?:vocal\s+)?(?:cues?|lines?|phrases?|vocals?|singing)(?:\s+(?:ends?|finishes?))?$/i.test(clause)
        && (!parts[i + 2] || /\b(?:mouth\w*|lips?|jaw)\b/i.test(parts[i + 2]))) continue;
      result += (result ? `${parts[i - 1] || ','} ` : '') + clause;
    }
    result = result.replace(/[,;]\s*([.!?])/g, '$1').replace(/[ ,;]+$/, '');
    if (result && /[.!?…]$/.test(sentence) && !/[.!?…]$/.test(result)) result += sentence.slice(-1);
    return result;
  }).filter(Boolean).join(' ');
  return clean.replace(/VRGDGDECIMALTOKEN/g, '.').replace(/VRGDGPROTECTED(\d+)TOKEN/g, (_m, i) => saved[Number(i)])
    .replace(/\s+/g, ' ').trim();
}

export function mergeLyricFreeContract(description, contract) {
  const protect = value => value.replace(/(\d)\.(?=\d)/g, '$1DECIMAL');
  const [singing, ...rest] = protect(contract).split(/(?<=[.!?])\s+/);
  const actor = singing.match(/^(.*?) sings\b/i)?.[1];
  if (!actor) return [description, contract];
  const maskQuotes = value => value.replace(/<d>[\s\S]*?<\/d>|“[^”]*”|"[^"]*"/gi, match => ' '.repeat(match.length));
  const escaped = actor.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const tags = '(?:\\s*\\[[^\\]]+\\])*';
  const suffix = `${tags}\\s+sings(?:\\s+[a-z]+ly|\\s+with passion)?\\b`;
  const owned = new RegExp(`${escaped}${suffix}`, 'i');
  const pronoun = new RegExp(`\\b(?:she|he|they)${suffix}`, 'i');
  const sentences = protect(description).split(/(?<=[.!?])\s+/);
  const requestedTime = singing.match(/ during (.+)\.$/)?.[1];
  let preceding = '';
  let index = -1;
  let cue = null;
  for (let i = 0; i < sentences.length; i += 1) {
    const visible = maskQuotes(sentences[i]);
    const named = owned.exec(visible);
    const inferred = pronoun.exec(visible);
    const labels = new Set((preceding + visible.slice(0, inferred?.index || 0)).match(/<Subject\s+\d+>/gi) || []);
    const match = named || (inferred && labels.size === 1 && labels.has(actor) ? inferred : null);
    const existingTime = sentences[i].match(/ during (\d+(?:DECIMAL\d+)?s\s*[–—-]\s*\d+(?:DECIMAL\d+)?s)/)?.[1];
    if (match && (!requestedTime || !existingTime || existingTime === requestedTime)) {
      index = i;
      cue = match;
      break;
    }
    preceding += visible + ' ';
  }
  if (index < 0) return [description, contract];
  const sync = singing.replace(/^.*? sings(?: with passion)?\s*/i, '').replace(/\.$/, '');
  const start = cue.index + cue[0].length;
  let tail = sentences[index].slice(start);
  // Replace only this cue's adjacent sync/timing, preserving other singers and action.
  tail = tail.replace(/^(\s*(?:\[[^\]]+\]\s*)*)in sync with\s+(?:the\s+)?(?:audible\s+)?(?:vocals(?:\s+in)?\s*(?:<Audio 1>)?|<Audio 1>)/i, '$1');
  tail = tail.replace(/^(\s*(?:\[[^\]]+\]\s*)*)during\s+\d+(?:DECIMAL\d+)?s\s*[–—-]\s*\d+(?:DECIMAL\d+)?s/i, '$1');
  tail = tail.replace(/^\s+([.,;!?])/, '$1').replace(/^\s+/, ' ');
  sentences[index] = sentences[index].slice(0, start) + ` ${sync}` + tail;
  const canonical = singing.replace(/ sings with passion/i, ' sings').replace(/\s+/g, ' ').trim();
  const kept = sentences.filter((sentence, i) => i === index || sentence.replace(/ sings with passion/i, ' sings').replace(/\s+/g, ' ').trim() !== canonical);
  return [kept.join(' ').replace(/DECIMAL/g, '.'), rest.join(' ').replace(/DECIMAL/g, '.')];
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
    const safe = stripSingingArticulation(protectedText);
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
  return `${clean}\n\nLYRIC-FREE CUSTOM AUDIO — MANDATORY:\nDo not quote, invent, or add lyric lines or <d> tags. Do not mention mouth, lip, or jaw movement anywhere in this prompt. Write each assigned singing cue once, only during its audible vocals in Audio 1. Instrumental intervals use only visual action and camera direction; do not mention singing in them. Preserve the supplied audio unchanged. Use selected facial direction through eyes, brows and expression.\n${directions.join("\n")}`;
}
