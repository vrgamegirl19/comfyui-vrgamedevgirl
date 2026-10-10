import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import vm from "node:vm";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { dismissFlowAnnouncement } from "../flow_automation/flow-announcements.mjs";
import { addUploadedFlowImageToPrompt } from "../flow_automation/flow-upload-selection.mjs";

const require = createRequire(new URL("../flow_automation/package.json", import.meta.url));
const { chromium } = require("playwright");

export async function loadFlowFunctions(filename) {
  const source = await fs.readFile(new URL(`../flow_automation/${filename}`, import.meta.url), "utf8");
  const sandbox = { fs, path, Buffer, URL, console, setTimeout, dismissFlowAnnouncement, addUploadedFlowImageToPrompt };
  vm.createContext(sandbox);
  // Load function declarations without executing browser launch or generation.
  const declarations = source.slice(source.search(/\n(?:async )?function /));
  vm.runInContext(declarations, sandbox);
  return sandbox;
}

for (const filename of ["flow-poc.mjs", "manual-bridge.mjs"]) {
  test(`${filename}: uploads through Add ingredients and adds reference to prompt`, async () => {
    const functions = await loadFlowFunctions(filename);
    const browser = await chromium.launch({ channel: "chrome", headless: true });
    try {
      const page = await browser.newPage();
      const uploadedSrc = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="red"/></svg>');
      const avatarSrc = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="blue"/></svg>');
      await page.setContent(`<button aria-label="Add media menu" onclick="throw Error('Wrong menu')">add</button>
        <button aria-label="Add ingredients to the prompt box" onclick="document.querySelector('#picker').hidden=false">add</button>
        <section id="picker" hidden>
          <button onclick="document.querySelector('input').click()">Upload media</button>
          <input type="file" hidden onchange="setTimeout(() => { document.querySelector('#uploaded').hidden=false; }, 100)">
          <button role="option" id="uploaded" hidden onclick="this.classList.add('asset-item-active'); setTimeout(() => { document.querySelector('#preview').src=this.querySelector('img').src; }, 100)"><img src="${uploadedSrc}"><span>flow_controls.mjs</span><span>Image</span></button>
          <button role="option"><img src="${avatarSrc}"><span>Me</span><span>Avatar</span></button>
          <section><img id="preview" src="${avatarSrc}"><button id="attach" onclick="document.body.dataset.attached=document.querySelector('#preview').src; document.querySelector('#picker').hidden=true">Add to prompt</button></section>
        </section>`);
      const originalWait = page.waitForTimeout.bind(page);
      page.waitForTimeout = (ms) => originalWait(ms >= 3000 ? 10 : ms);
      const upload = functions.uploadImageAndAddToPrompt || functions.uploadFlowImageAndAddToPrompt;
      await upload(page, fileURLToPath(import.meta.url));
      assert.equal(await page.locator("body").getAttribute("data-attached"), uploadedSrc);
      assert.equal(await page.locator("input").evaluate((el) => el.files.length), 1);
    } finally { await browser.close(); }
  });
}

test("standalone automation reuses the new Flow domain and recognizes Start generation", async () => {
  const functions = await loadFlowFunctions("flow-poc.mjs");
  const unrelated = { url: () => "https://example.com/" };
  const flow = { url: () => "https://flow.google.com/project/test" };
  assert.equal(await functions.getOrCreatePage({ pages: () => [unrelated, flow] }), flow);
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage();
    await page.setContent('<button aria-label="Start generation" onclick="this.dataset.clicked=\'true\'">arrow_forward</button>');
    assert.equal(await functions.clickFirstVisible(functions.submitLocators(page)), true);
    assert.equal(await page.locator("button").getAttribute("data-clicked"), "true");
  } finally { await browser.close(); }
});

test("Flow automation copies remain synchronized", async () => {
  for (const filename of ["flow-poc.mjs", "manual-bridge.mjs", "flow-announcements.mjs", "flow-upload-selection.mjs"]) {
    const root = fileURLToPath(new URL("../", import.meta.url));
    assert.deepEqual(await fs.readFile(path.join(root, "flow_automation", filename)),
      await fs.readFile(path.join(root, "optional_nodes/ui_tools/flow_automation", filename)));
  }
});

test("refuses Add to prompt when the selected file still previews the avatar", async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage();
    const image = (color) => 'data:image/svg+xml,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2"><rect width="2" height="2" fill="${color}"/></svg>`);
    await page.setContent(`<button role="option" onclick="this.classList.add('asset-item-active')"><span>reference.png</span><img src="${image('red')}"></button>
      <section><img src="${image('blue')}"><button onclick="document.body.dataset.added='true'">Add to prompt</button></section>`);
    await assert.rejects(addUploadedFlowImageToPrompt(page, "reference.png", 400), /refusing to add a different image or avatar/);
    assert.equal(await page.locator("body").getAttribute("data-added"), null);
  } finally { await browser.close(); }
});

test("detects completed generated tiles, excludes uploads, and ignores renewed CDN signatures", async () => {
  const functions = await loadFlowFunctions("flow-poc.mjs");
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    const pixel = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1sAAAAASUVORK5CYII=", "base64");
    await page.context().route("https://flow-content.google/image/**", (route) => route.fulfill({ contentType: "image/png", body: pixel }));
    await page.setContent(`<flow-grid-tile-container><flow-image-tile><img width="100" height="100" alt="Tile displaying a user's image" src="https://flow-content.google/image/reference?Signature=one"></flow-image-tile></flow-grid-tile-container>
      <flow-grid-tile-container><flow-image-tile><img width="100" height="100" alt="Tile displaying a user's image" src="https://flow-content.google/image/output?Signature=one"></flow-image-tile><button aria-label="Reuse prompt"></button></flow-grid-tile-container>
      <img width="100" height="100" alt="generated image" style="display:none" src="https://flow-content.google/image/hidden">`);
    await page.locator("img").evaluateAll((images) => Promise.all(images.map((img) => img.decode())));
    assert.deepEqual(Array.from(await functions.getGeneratedImageUrls(page)), ["https://flow-content.google/image/output?Signature=one"]);
    await page.locator('img[src*="/output?"]').evaluate((img) => { img.src = "https://flow-content.google/image/output?Signature=two"; });
    await page.locator('img[src*="/output?"]').evaluate((img) => img.decode());
    const originalWait = page.waitForTimeout.bind(page);
    page.waitForTimeout = (ms) => originalWait(Math.min(ms, 10));
    await assert.rejects(functions.waitForNewGeneratedImageUrl(page, new Set(["https://flow-content.google/image/output?Signature=one"]), 40), /new completed Flow image/);
    await page.evaluate(() => {
      const tile = document.createElement("flow-grid-tile-container");
      tile.innerHTML = '<flow-image-tile><img width="100" height="100" src="https://flow-content.google/image/new-output"></flow-image-tile><button aria-label="Reuse prompt"></button>';
      document.body.append(tile);
    });
    await page.locator('img[src$="new-output"]').evaluate((img) => img.decode());
    assert.equal(await functions.waitForNewGeneratedImageUrl(page, new Set(["https://flow-content.google/image/output?Signature=one"]), 1000), "https://flow-content.google/image/new-output");

    const directory = await fs.mkdtemp(path.join(os.tmpdir(), "flow-download-test-"));
    let saved;
    try {
      const beforePages = page.context().pages().length;
      saved = await functions.saveGeneratedImageUrl(page, "https://flow-content.google/image/new-output", directory, "test-image");
      assert.deepEqual(await fs.readFile(saved), pixel);
      assert.equal(page.context().pages().length, beforePages);
    } finally {
      if (saved) await fs.unlink(saved);
      await fs.rmdir(directory);
    }
  } finally { await browser.close(); }
});
