import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { test } from "node:test";
import vm from "node:vm";
import { closeBrowserImageViewer, waitForBrowserChat } from "../flow_automation/browser-image-viewer.mjs";

const require = createRequire(new URL("../flow_automation/package.json", import.meta.url));
const { chromium } = require("playwright");
const findComposer = async (page) => page.locator("textarea");
const fixture = (label = 'aria-label="Close image"', escape = false) => `
  <textarea></textarea><button aria-label="Close Codes project">Project</button>
  <div id="viewer" style="position:fixed;inset:0;background:#222;z-index:99">
    <button ${label} onclick="setTimeout(()=>document.querySelector('#viewer').remove(),100)">X</button>
  </div>${escape ? '<script>document.addEventListener("keydown",e=>{if(e.key==="Escape")document.querySelector("#viewer")?.remove()})</script>' : ''}`;

test("viewer toolbar labels and Escape restore the composer without closing unrelated controls", async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  try {
    for (const label of ['aria-label="Close image"', 'aria-label="Close"', 'title="Close"', 'aria-label="Dismiss"']) {
      await page.setContent(fixture(label, true));
      await closeBrowserImageViewer(page);
      await waitForBrowserChat(page, findComposer, "Test");
      assert.equal(await page.locator("#viewer").count(), 0);
      assert.equal(await page.getByRole("button", { name: "Close Codes project" }).count(), 1);
      await page.locator("textarea").fill("Next reference");
      assert.equal(await page.locator("textarea").inputValue(), "Next reference");
    }
    await page.setContent("<textarea>Keep prompt</textarea>");
    await closeBrowserImageViewer(page);
    await waitForBrowserChat(page, findComposer, "Test");
    assert.equal(await page.locator("textarea").inputValue(), "Keep prompt");
  } finally { await browser.close(); }
});

test("a visible composer covered by a viewer cannot report success", async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  try {
    await page.setContent(fixture());
    assert.equal(await page.locator("textarea").isVisible(), true);
    await assert.rejects(waitForBrowserChat(page, findComposer, "ChatGPT Images", 600), /did not return to a usable chat prompt/);
  } finally { await browser.close(); }
});

for (const script of ["chatgpt-images-poc.mjs", "meta-ai-poc.mjs"]) {
  test(`${script} saves then closes the viewer for consecutive requests and failed downloads`, async () => {
    const source = await readFile(new URL(`../flow_automation/${script}`, import.meta.url), "utf8");
    const start = source.indexOf("let outputPath;");
    const block = source.slice(start, source.indexOf("if (shouldCloseContext)", start));
    assert.ok(start > 0);
    const browser = await chromium.launch({ channel: "chrome", headless: true });
    const page = await browser.newPage();
    try {
      for (const outcome of ["toolbar", "direct", "failure"]) {
        await page.setContent(fixture());
        const logs = [];
        const download = async () => outcome === "toolbar" ? "saved.png" : null;
        const context = {
          page, image: { src: "image-url" }, outputDir: "outputs", prompt: "reference", findComposer,
          closeBrowserImageViewer, waitForBrowserChat,
          retryOpenViewerAndDownload: download, downloadGeneratedImageFromOverlay: download,
          newestVisibleImageUrl: async () => "image-url",
          saveImageUrl: async () => { if (outcome === "failure") throw new Error("Download failed"); return "saved.png"; },
          console: { log: (line) => logs.push(line) },
        };
        const run = vm.runInNewContext(`(async () => { ${block} })()`, context);
        if (outcome === "failure") {
          await assert.rejects(run, /Download failed/);
          assert.equal(logs.some((line) => line.startsWith("Saved:")), false);
        } else {
          await run;
          assert.ok(logs.includes("Saved: saved.png"));
        }
        assert.equal(await page.locator("#viewer").count(), 0);
        await page.locator("textarea").fill("Next reference");
      }
    } finally { await browser.close(); }
  });
}

test("bundled provider scripts and viewer helper match the primary copies", async () => {
  for (const file of ["browser-image-viewer.mjs", "chatgpt-images-poc.mjs", "meta-ai-poc.mjs"]) {
    assert.equal(await readFile(new URL(`../flow_automation/${file}`, import.meta.url), "utf8"),
      await readFile(new URL(`../optional_nodes/ui_tools/flow_automation/${file}`, import.meta.url), "utf8"));
  }
});
