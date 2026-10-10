import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { test } from "node:test";
import { dismissFlowAnnouncement } from "../flow_automation/flow-announcements.mjs";

const require = createRequire(new URL("../flow_automation/package.json", import.meta.url));
const { chromium } = require("playwright");

const announcement = (version) => `<section id="announcement">
  <h2>Nano Banana ${version}</h2>
  <footer><a href="#">View all changelogs</a>
  <button onclick="document.querySelector('#announcement').remove()">Get started</button></footer>
</section>`;

test("dismisses announcements without depending on version or dialog roles", async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  try {
    for (const version of ["2.1", "3.0"]) {
      await page.setContent(`<textarea placeholder="What do you want to create?"></textarea>${announcement(version)}`);
      assert.equal(await dismissFlowAnnouncement(page), true);
      assert.equal(await page.locator("#announcement").count(), 0);
      await page.locator("textarea").fill("Test prompt");
      assert.equal(await page.locator("textarea").inputValue(), "Test prompt");
    }
  } finally { await browser.close(); }
});

test("leaves unrelated Get started controls alone", async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  try {
    await page.setContent('<a>View all changelogs</a><section><button onclick="this.remove()">Get started</button></section>');
    assert.equal(await dismissFlowAnnouncement(page), false);
    assert.equal(await page.getByRole("button", { name: "Get started" }).count(), 1);
    await page.setContent("<textarea></textarea>");
    assert.equal(await dismissFlowAnnouncement(page), false);
  } finally { await browser.close(); }
});

test("handles delayed announcements and ignores hidden ones", async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage();
  try {
    await page.setContent(`<div style="display:none">${announcement("2.1")}</div>`);
    assert.equal(await dismissFlowAnnouncement(page), false);
    await page.setContent("<main></main>");
    await page.evaluate((html) => {
      setTimeout(() => { document.querySelector("main").innerHTML = html; }, 100);
    }, announcement("2.1"));
    assert.equal(await dismissFlowAnnouncement(page, 2000), true);
  } finally { await browser.close(); }
});
