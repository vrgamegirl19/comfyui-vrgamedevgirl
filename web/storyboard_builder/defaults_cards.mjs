// Presentation only: move existing controls without recreating them or changing state.
let defaultsSettingId = 0;

// The value a setting row shows in its header: the chosen option of its first list, else its slider or text.
export function currentValueText(controls) {
  const select = controls.querySelector("select");
  const range = controls.querySelector('input[type="range"]');
  const chosen = select
    ? String(Array.from(select.options || select.children || []).find((item) => item.value === select.value)?.textContent || select.value || "").trim()
    : "";
  if (range) return chosen ? `${range.value}/10 · ${chosen}` : `${range.value}/10`;
  if (select) return chosen || "Not set";
  const text = controls.querySelector("textarea, input:not([type='checkbox'])");
  const words = String(text?.value || "").trim();
  return words ? (words.length > 40 ? `${words.slice(0, 40)}…` : words) : "Not set";
}

export function buildDefaultsCards(groups) {
  if (!document.getElementById("vrgdg-scene-defaults-card-styles")) {
    const style = document.createElement("style");
    style.id = "vrgdg-scene-defaults-card-styles";
    style.textContent = `
      .vrgdg-scene-defaults-cards { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px; }
      @container (max-width:720px) { .vrgdg-scene-defaults-cards { grid-template-columns:minmax(0,1fr); } }
      .vrgdg-scene-defaults-card { min-width:0;padding:10px;border:1px solid #334155;border-radius:8px;background:#192638; }
      .vrgdg-scene-defaults-card h3 { display:flex;justify-content:space-between;align-items:center;margin:0 0 10px;color:#67e8f9;font:inherit;font-weight:800; }
      .vrgdg-scene-defaults-card h3 svg { width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:1.5; }
      .vrgdg-scene-defaults-setting { margin-top:7px;border-radius:6px;background:#1e2c40;overflow:hidden; }
      .vrgdg-scene-defaults-setting:has(.vrgdg-scene-defaults-primary[style*="display: none"]) { display:none; }
      .vrgdg-scene-defaults-toggle { display:flex;align-items:center;justify-content:space-between;gap:10px;width:100%;box-sizing:border-box;padding:9px 10px;border:0;background:transparent;color:#e2e8f0;font:inherit;text-align:left;cursor:pointer; }
      .vrgdg-scene-defaults-toggle:hover,.vrgdg-scene-defaults-toggle[aria-expanded="true"] { background:#26374d; }
      .vrgdg-scene-defaults-toggle:focus-visible { outline:2px solid #22d3ee;outline-offset:-2px; }
      .vrgdg-scene-defaults-chevron { flex:none;width:12px;height:12px;fill:none;stroke:#22d3ee;stroke-width:1.5;transition:transform .15s ease; }
      .vrgdg-scene-defaults-toggle[aria-expanded="true"] .vrgdg-scene-defaults-chevron { transform:rotate(180deg); }
      .vrgdg-scene-defaults-body { padding:9px 10px 11px;display:grid;gap:9px;animation:vrgdg-defaults-reveal .15s ease; }
      .vrgdg-scene-defaults-body[hidden] { display:none; }
      .vrgdg-scene-defaults-selected-action[hidden] { display:none!important; }
      .vrgdg-scene-defaults-primary > :first-child { display:none; }
      .vrgdg-scene-defaults-body > div { min-width:0!important;max-width:100%;gap:8px; }
      .vrgdg-scene-defaults-body select,.vrgdg-scene-defaults-body textarea,.vrgdg-scene-defaults-body input:not([type="checkbox"]) { min-width:0!important;max-width:100%;box-sizing:border-box; }
      .vrgdg-scene-defaults-body select { flex:1 1 170px;width:100%!important; }
      .vrgdg-scene-defaults-body input[type="range"] { flex:1 1 150px!important; }
      .vrgdg-scene-defaults-body textarea { flex-basis:100%!important; }
      .vrgdg-scene-defaults-body label { min-width:0!important;max-width:100%;flex-wrap:wrap; }
      .vrgdg-scene-defaults-body > div:not(.vrgdg-scene-defaults-primary) > div:first-child { min-width:0!important;text-align:left!important; }
      .vrgdg-scene-defaults-help { font-size:12px;color:#94a3b8;line-height:1.45; }
      .vrgdg-scene-defaults-value { flex:1 1 auto;min-width:0;margin-left:auto;color:#a5f3fc;font-size:11px;font-weight:700;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap; }
      @keyframes vrgdg-defaults-reveal { from { opacity:0;transform:translateY(-3px); } to { opacity:1;transform:translateY(0); } }
      @media (prefers-reduced-motion:reduce) { .vrgdg-scene-defaults-body { animation:none; }.vrgdg-scene-defaults-chevron { transition:none; } }
    `;
    document.head.append(style);
  }
  const grid = document.createElement("div");
  grid.className = "vrgdg-scene-defaults-cards";
  let active = null;
  // Each setting's header shows its current value, so a reopened panel shows what was chosen without opening rows.
  const summaries = [];
  grid.refreshSummaries = () => {
    for (const update of summaries) update();
  };
  for (const { title, settings } of groups) {
    const card = document.createElement("section");
    card.className = "vrgdg-scene-defaults-card";
    const heading = document.createElement("h3");
    heading.textContent = title;
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 16 16");
    icon.setAttribute("aria-hidden", "true");
    icon.innerHTML = '<path d="M2 4h5m3 0h4M2 12h8m3 0h1"/><circle cx="8.5" cy="4" r="1.5"/><circle cx="11.5" cy="12" r="1.5"/>';
    heading.append(icon);
    card.append(heading);
    for (const { name, controls, content, info } of settings) {
      const row = document.createElement("div");
      row.className = "vrgdg-scene-defaults-setting";
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "vrgdg-scene-defaults-toggle";
      const label = document.createElement("span");
      label.textContent = name;
      const chevron = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      chevron.setAttribute("viewBox", "0 0 12 12");
      chevron.setAttribute("aria-hidden", "true");
      chevron.classList.add("vrgdg-scene-defaults-chevron");
      chevron.innerHTML = '<path d="m2 4 4 4 4-4"/>';
      const value = document.createElement("span");
      value.className = "vrgdg-scene-defaults-value";
      const update = () => {
        value.textContent = currentValueText(controls);
        value.title = value.textContent;
      };
      summaries.push(update);
      controls.addEventListener("input", update);
      controls.addEventListener("change", update);
      update();
      toggle.append(label, value, chevron);
      const body = document.createElement("div");
      body.className = "vrgdg-scene-defaults-body";
      body.id = `vrgdg-scene-default-setting-${++defaultsSettingId}`;
      toggle.setAttribute("aria-controls", body.id);
      controls.classList.add("vrgdg-scene-defaults-primary");
      for (const control of controls.querySelectorAll("input, select, textarea")) {
        control.setAttribute("aria-label", name);
      }
      body.append(controls, ...(content || []));
      if (info) {
        info.classList.add("vrgdg-scene-defaults-help");
        body.append(info);
      }
      const setExpanded = (expanded) => {
        toggle.setAttribute("aria-expanded", String(expanded));
        body.hidden = !expanded;
      };
      setExpanded(false);
      toggle.addEventListener("click", () => {
        const open = body.hidden;
        if (active) active(false);
        setExpanded(open);
        active = open ? setExpanded : null;
      });
      row.append(toggle, body);
      card.append(row);
    }
    grid.append(card);
  }
  return grid;
}
