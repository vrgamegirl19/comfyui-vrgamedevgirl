import { chromium } from "playwright";
import fs from "node:fs/promises";
import path from "node:path";

const projectDir = path.dirname(new URL(import.meta.url).pathname).replace(/^\/(.:\/)/, "$1");
const outDir = path.join(projectDir, "debug");
await fs.mkdir(outDir, { recursive: true });

const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
const context = browser.contexts()[0];
if (!context) throw new Error("No Chrome context found on port 9222.");
const page = context.pages().find((p) => p.url().startsWith("https://labs.google/")) || context.pages()[0];
if (!page) throw new Error("No page found.");

console.log(`Inspecting: ${page.url()}`);

const result = await page.evaluate(() => {
  const nodes = Array.from(document.querySelectorAll("button, [role='button'], a, mat-icon, svg"));
  const clean = (value) => (value || "").replace(/\s+/g, " ").trim();
  const short = (value, max = 500) => {
    value = clean(value);
    return value.length > max ? value.slice(0, max) + " ..." : value;
  };
  const cssPath = (el) => {
    const parts = [];
    let current = el;
    while (current && current.nodeType === 1 && parts.length < 6) {
      let part = current.tagName.toLowerCase();
      if (current.id) {
        part += `#${current.id}`;
        parts.unshift(part);
        break;
      }
      const cls = Array.from(current.classList || []).slice(0, 3).join(".");
      if (cls) part += `.${cls}`;
      const parent = current.parentElement;
      if (parent) {
        const sameTag = Array.from(parent.children).filter((child) => child.tagName === current.tagName);
        if (sameTag.length > 1) part += `:nth-of-type(${sameTag.indexOf(current) + 1})`;
      }
      parts.unshift(part);
      current = parent;
    }
    return parts.join(" > ");
  };
  const items = [];
  for (const node of nodes) {
    const box = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    const visible = box.width > 0 && box.height > 0 && style.visibility !== "hidden" && style.display !== "none";
    if (!visible) continue;
    const clickable = node.closest("button, [role='button'], a") || node;
    const cbox = clickable.getBoundingClientRect();
    const parent = clickable.parentElement;
    const grand = parent?.parentElement;
    items.push({
      tag: node.tagName.toLowerCase(),
      clickableTag: clickable.tagName.toLowerCase(),
      text: short(clickable.innerText || clickable.textContent, 160),
      ariaLabel: clickable.getAttribute("aria-label"),
      title: clickable.getAttribute("title"),
      type: clickable.getAttribute("type"),
      role: clickable.getAttribute("role"),
      className: clickable.getAttribute("class"),
      dataAttrs: Array.from(clickable.attributes || [])
        .filter((attr) => attr.name.startsWith("data-") || attr.name.startsWith("js"))
        .map((attr) => `${attr.name}=${attr.value}`),
      box: { x: Math.round(cbox.x), y: Math.round(cbox.y), width: Math.round(cbox.width), height: Math.round(cbox.height) },
      path: cssPath(clickable),
      html: short(clickable.outerHTML, 800),
      parentHtml: short(parent?.outerHTML, 1200),
      grandParentHtml: short(grand?.outerHTML, 1200),
    });
  }
  return { url: location.href, title: document.title, count: items.length, items };
});

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const jsonPath = path.join(outDir, `flow-buttons-${stamp}.json`);
const txtPath = path.join(outDir, `flow-buttons-${stamp}.txt`);
await fs.writeFile(jsonPath, JSON.stringify(result, null, 2), "utf8");
await fs.writeFile(txtPath, result.items.map((item, index) => [
  `#${index + 1}`,
  `tag: ${item.tag} clickable: ${item.clickableTag}`,
  `text: ${item.text}`,
  `aria: ${item.ariaLabel}`,
  `title: ${item.title}`,
  `role: ${item.role}`,
  `class: ${item.className}`,
  `box: ${JSON.stringify(item.box)}`,
  `data: ${(item.dataAttrs || []).join(", ")}`,
  `path: ${item.path}`,
  `html: ${item.html}`,
  `parent: ${item.parentHtml}`,
  ""
].join("\n")).join("\n"), "utf8");

console.log(`Wrote ${txtPath}`);
console.log(`Wrote ${jsonPath}`);
await browser.close();
