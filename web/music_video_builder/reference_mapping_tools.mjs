import { makeField } from "./controls.mjs";

// A focused Wizard route that reuses the Reference Builder's existing actions.
export function createWizardMappingTools({ cardStyle, assignScenes, extractLocations, autoMapLocations, locationStyleTheme }) {
  const card = document.createElement("div");
  card.style.cssText = cardStyle;

  function addToolGroup(title, tools) {
    const available = tools.filter(([button]) => !button.hidden);
    if (!available.length) return;
    const group = document.createElement("div");
    group.style.cssText = "display:flex;flex-direction:column;gap:12px;padding:14px;border:1px solid #334155;border-radius:8px;background:#0b1220;";
    const label = document.createElement("div");
    label.textContent = title;
    label.style.cssText = "font-size:14px;font-weight:900;color:#cffafe;";
    group.append(label);
    for (const [button, explanation] of available) {
      const row = document.createElement("div");
      row.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:12px;";
      button.style.minWidth = "210px";
      button.style.minHeight = "36px";
      const help = document.createElement("div");
      help.textContent = explanation;
      help.style.cssText = "flex:1 1 280px;color:#cbd5e1;font-size:12px;line-height:1.5;";
      row.append(button, help);
      group.append(row);
    }
    card.append(group);
  }

  function render() {
    card.replaceChildren();
    const subjectHint = document.createElement("div");
    subjectHint.textContent = "Subject mappings update automatically when you save Align lyrics / dialogue. Use Adjust & View All Mappings to review who is present and who performs each line.";
    subjectHint.style.cssText = "font-size:12px;line-height:1.5;color:#cbd5e1;";
    card.append(subjectHint);
    addToolGroup("Assign subjects and locations in bulk", [
      [assignScenes, "Choose saved subjects and locations for multiple scenes using random, rotating, or repeating block patterns. Preview before applying and choose whether to replace existing mappings. This does not use an LLM."],
    ]);
    addToolGroup("Locations — build a list, then assign locations to scenes", [
      [extractLocations, "Ask your configured LLM to extract a reusable location list from scene text, notes, and lyrics. Create or update locations, then use Auto Map."],
      [autoMapLocations, "Ask your configured LLM to choose from saved locations for each scene. Reference images are optional."],
    ]);
    if (!extractLocations.hidden) card.append(makeField("Optional style/theme for location extraction", locationStyleTheme));
    const hint = document.createElement("div");
    hint.textContent = "These tools update the same references and mappings as the original editors. Save when finished, then use Adjust & View All Mappings in the Wizard to review each scene.";
    hint.style.cssText = "font-size:12px;line-height:1.5;color:#94a3b8;";
    card.append(hint);
  }

  return { card, render };
}
