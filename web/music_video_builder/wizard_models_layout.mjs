// Reuse the live Builder controls in Wizard Beta, then put every control back on exit.
export function organizeWizardVideoModels({ holder, contents, miniMax, miniMaxPassChooser, movePanel }) {
  const restores = [];
  const models = contents[0];
  if (miniMax) {
    const passControl = document.createElement("div");
    passControl.className = "wb-pass-control";
    holder.append(passControl);
    restores.push(() => passControl.remove(), movePanel(passControl, miniMaxPassChooser));

    const fields = Array.from(models.children);
    if (fields.length >= 7) {
      for (const [title, controls] of [["Video Models", fields.slice(1, 4)], ["Audio Model", fields.slice(4, 5)]]) {
        const card = document.createElement("section");
        card.className = "wb-model-card";
        const heading = document.createElement("h3");
        heading.textContent = title;
        card.append(heading);
        models.append(card);
        restores.push(() => card.remove());
        for (const control of controls) restores.push(movePanel(card, control));
      }
      const help = document.createElement("details");
      help.className = "wb-model-help";
      const summary = document.createElement("summary");
      summary.textContent = "Model compatibility";
      help.append(summary);
      models.append(help);
      restores.push(() => help.remove(), movePanel(help, fields[0]));
      for (const card of Array.from(models.children).filter((child) => child.className === "wb-model-card").reverse()) {
        models.prepend(card);
      }
    }
  }

  // LLM model selection belongs to the Wizard's LLM Runner, not its video model page.
  const hidden = document.createElement("div");
  for (const section of Array.from(models.children)) {
    const title = section.children[0]?.textContent;
    if (title === "Non-Vision LLM Models" || title === "Vision LLM Models") {
      restores.push(movePanel(hidden, section));
    }
  }
  return restores;
}
