import { BUILDER_FONT_STACK } from "./constants.mjs";

// "Rules and uses": a short guide to RefMods that opens on top of RefMods Studio. The text is data (SECTIONS) so it is
// easy to extend. Each section has a title and a list of items. An item is a string, or { term, text } for a labelled line.

export const REFMOD_RULES_SECTIONS = [
  {
    title: "What a RefMod is",
    items: [
      "A RefMod is a reference saved as a file under models/refmods/<type>. The Builder shows it to the video model with a numbered label, so one scene can use several of them at once.",
      "One photo makes an image RefMod (<Picture n>). Several photos stacked in one RefMod make a video RefMod (<Video n>). The label follows what is saved in the file, not what you started from.",
      "A RefMod keeps no trigger word. Its name is not a trigger. It only works through its label in the prompt, such as <Video 1> (Darrel).",
    ],
  },
  {
    title: "Types and what goes in them",
    items: [
      { term: "Identity", text: "One person per RefMod. Slots 1 to 3 are close-ups of the face (front, left, right). Extra images are other angles and expressions. Full-body shots also teach the clothes in them, so keep outfits out of an identity RefMod when you plan to dress the character with a Clothing RefMod." },
      { term: "Clothing (men / women)", text: "The outfit only: front, back and sides on a mannequin or flat lay work well. Avoid a person's face and body. It is worn by a character you link on the Reference Builder card (the \"Worn by\" choice)." },
      { term: "Background", text: "A place or set. Use it for the setting of a scene." },
      { term: "Style", text: "A look, grade or medium. It applies to the whole scene, not to one subject." },
      { term: "Object / Prop", text: "A held or placed item. It is placed in the scene next to the main character, held, used, or stood beside." },
      { term: "Vehicle", text: "The main character drives it, or stands in front of or beside it." },
      { term: "Creature", text: "An animal or monster that is present next to the main character. It does not speak or sing." },
      { term: "Pose / Motion, Generic", text: "A pose, dance, gesture or camera move, or anything that fits no other type." },
    ],
  },
  {
    title: "Making a good RefMod",
    items: [
      "Use clear images of one subject. Keep different characters in separate RefMods. Mixing people in one file blends them.",
      "Leave the quality and mode on their defaults first. Full Reference keeps the most detail and is the baseline to compare against.",
      "Trim to subject keeps the token cost down by cropping around the subject. Check the outline before you create.",
      "The description is optional. It is stored in the file for you, and the prompt writer uses it as extra context when it exists. A RefMod without a description still works and is named by its card.",
    ],
  },
  {
    title: "Using RefMods in a project",
    items: [
      "Switch the project to the RefMod pipeline in the MiniMax panel. Then every reference card is a saved RefMod.",
      "In the Reference Builder, pick a RefMod on a card, map the card to the scenes that use it, and set its strength from 0 to 1. A strength of 0 leaves it out.",
      "A scene can hold up to 24 RefMods. The scene shows its token total and warns above 6,000. A very large RefMod can dominate a smaller one.",
      "Order in a scene: characters, extras, clothing, objects, background, then style. Labels are numbered per kind in that order, so the prompt and the render always agree.",
      "Clothing follows the character it is linked to. To dress a character differently in a scene, map that scene to the clothing card in Review Lines + Map Performers (the props, clothing and vehicles box).",
    ],
  },
  {
    title: "How the prompt names them",
    items: [
      "Every mention of a RefMod in a prompt is its label with the card name: <Video 1> (Darrel) sings the lyric line.",
      "Clothing is written as part of what the character wears: <Video 1> (Darrel), wearing <Video 2> (Warm Clothing). It is never a person in the scene.",
      "A prop or vehicle the writer forgets is added in one short sentence, for example: <Video 1> (Darrel) stands beside <Picture 2> (Black Charger).",
    ],
  },
  {
    title: "Limits and fixes",
    items: [
      "RefMods do not guarantee that only the wanted trait is copied. Identity, clothing, background and composition can still mix.",
      "The character keeps the old clothes: make the identity RefMod from the face and head only, or lower that character's strength for scenes that use a Clothing RefMod.",
      "Two characters blend into one: give the lighter one more images, or lower the stronger one's strength. The scene status warns when one is more than 2x lighter.",
      "Voice transfer from a reference does not work reliably. Use the voice settings on the character card instead.",
      "Compare a result with a render without the RefMod, using the same prompt and seed, before you judge it.",
    ],
  },
];

const BACKDROP_STYLE = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;padding:16px;";
const BOX_STYLE = `width:min(860px,calc(100vw - 32px));max-height:calc(100vh - 32px);box-sizing:border-box;display:flex;flex-direction:column;gap:12px;padding:16px;border:1px solid #3f3f46;border-radius:8px;background:#111827;color:#f8fafc;font-family:${BUILDER_FONT_STACK};box-shadow:0 20px 70px rgba(0,0,0,.6);`;

function renderItem(item) {
  const row = document.createElement("li");
  row.style.cssText = "margin:0 0 6px 0;line-height:1.5;font-size:12px;color:#d4d4d8;";
  if (item && typeof item === "object") {
    const term = document.createElement("strong");
    term.textContent = `${item.term}: `;
    term.style.color = "#e0f2fe";
    row.append(term, document.createTextNode(item.text));
  } else {
    row.textContent = String(item);
  }
  return row;
}

export function openRefModsRules() {
  if (document.getElementById("vrgdg-refmods-rules")) return;
  const backdrop = document.createElement("div");
  backdrop.id = "vrgdg-refmods-rules";
  backdrop.setAttribute("role", "dialog");
  backdrop.setAttribute("aria-modal", "true");
  backdrop.setAttribute("aria-label", "RefMods rules and uses");
  backdrop.style.cssText = BACKDROP_STYLE;
  const box = document.createElement("div");
  box.style.cssText = BOX_STYLE;

  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;flex:0 0 auto;";
  const title = document.createElement("div");
  title.textContent = "RefMods: rules and uses";
  title.style.cssText = "font-size:18px;font-weight:900;";
  const closeButton = document.createElement("button");
  closeButton.type = "button";
  closeButton.textContent = "Close";
  closeButton.style.cssText = `border:1px solid #3f3f46;border-radius:6px;background:#27272a;color:#fafafa;padding:7px 12px;font-family:${BUILDER_FONT_STACK};font-size:12px;font-weight:700;cursor:pointer;`;
  header.append(title, closeButton);

  const body = document.createElement("div");
  body.style.cssText = "overflow:auto;min-height:0;display:flex;flex-direction:column;gap:14px;padding-right:4px;";
  for (const section of REFMOD_RULES_SECTIONS) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "border:1px solid #27272a;border-radius:8px;background:#18181b;padding:12px;";
    const heading = document.createElement("div");
    heading.textContent = section.title;
    heading.style.cssText = "font-size:13px;font-weight:800;color:#e4e4e7;margin-bottom:8px;";
    const list = document.createElement("ul");
    list.style.cssText = "margin:0;padding-left:18px;";
    for (const item of section.items) list.append(renderItem(item));
    wrap.append(heading, list);
    body.append(wrap);
  }
  box.append(header, body);
  backdrop.append(box);

  const close = () => {
    document.removeEventListener("keydown", onKey, true);
    backdrop.remove();
  };
  // Escape closes this guide only, not the Studio behind it.
  const onKey = (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    close();
  };
  closeButton.onclick = close;
  backdrop.addEventListener("mousedown", (event) => {
    if (event.target === backdrop) close();
  });
  document.addEventListener("keydown", onKey, true);
  document.body.append(backdrop);
  closeButton.focus();
}
