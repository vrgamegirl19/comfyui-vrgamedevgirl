import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs/promises";

const args = parseArgs(process.argv.slice(2));
const projectDir = path.dirname(new URL(import.meta.url).pathname).replace(/^\/(.:\/)/, "$1");
const url = args.url || "https://www.meta.ai/";
const prompt = normalizePrompt(args.prompt || "create an image");
const outputDir = path.resolve(projectDir, args.out || "meta_outputs");
const timeout = Number(args.timeout || 360000);
const cdpUrl = args.cdp || args["connect-cdp"] || "";
const noNavigate = Boolean(args["no-navigate"]);
const imagePaths = normalizeList(args.image).map((imagePath) => path.resolve(imagePath));
const profileDir = path.resolve(projectDir, "meta-browser-profile");

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(profileDir, { recursive: true });

let browser = null;
let context = null;
let shouldCloseContext = true;

if (cdpUrl) {
  console.log(`Connecting to real Chrome at ${cdpUrl}`);
  browser = await chromium.connectOverCDP(cdpUrl);
  context = browser.contexts()[0];
  shouldCloseContext = false;
  if (!context) throw new Error("Connected to Chrome, but no browser context was available.");
} else {
  console.log(`Using automation profile: ${profileDir}`);
  context = await chromium.launchPersistentContext(profileDir, {
    channel: "chrome",
    headless: false,
    acceptDownloads: true,
    viewport: { width: 1600, height: 950 },
  });
}

let page = await getOrCreatePage(context);
page.setDefaultTimeout(30000);

if (noNavigate) {
  console.log(`Using current tab: ${page.url()}`);
} else {
  console.log(`Opening ${url}`);
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle", { timeout: 60000 }).catch(() => {});
}
await allowDownloadsForAttachedChrome(context, page, outputDir);
await ensureComposerReady(page);

if (imagePaths.length > 0) {
  console.log(`Attaching ${imagePaths.length} image(s).`);
  for (let index = 0; index < imagePaths.length; index += 1) {
    const imagePath = imagePaths[index];
    console.log(`Attaching image ${index + 1}/${imagePaths.length}: ${imagePath}`);
    await attachImage(page, imagePath);
  }
  console.log("Waiting for Meta AI to finish preparing all attachments...");
  await page.waitForTimeout(2000);
}

const beforeKeys = new Set(await getCandidateImageKeys(page));
console.log(`Images before submit: ${beforeKeys.size}`);

console.log("Entering prompt...");
const composer = await findComposer(page);
if (!composer) throw new Error("Could not find the Meta AI prompt box.");
await enterPrompt(page, composer, prompt, { preserveAttachments: imagePaths.length > 0 });
const historyLinksBeforeSubmit = new Set(await sidebarChatHrefs(page));

console.log("Submitting prompt...");
const submitted = await clickMetaSubmitBesideInstant(page);
if (!submitted) throw new Error("Could not find the Meta AI submit arrow immediately beside the Instant button.");
console.log("Prompt submitted; waiting for Meta AI to create the conversation...");
const openedConversation = await openNewSidebarConversation(page, composer, historyLinksBeforeSubmit, 30000);
if (openedConversation) console.log("Opened the newly created Meta AI conversation.");
else console.log("No new sidebar conversation link appeared; continuing on the current page.");

console.log("Waiting for generated image...");
const generated = await waitForNewImage(context, page, beforeKeys, timeout);
page = generated.page;
const image = generated.image;
console.log("Saving newly generated image directly...");
if (!image?.src) throw new Error("Meta AI generated an image, but its image URL was unavailable.");
const buttonDownload = await downloadGeneratedImageFromOverlay(page, image, outputDir, prompt);
const outputPath = buttonDownload || await saveImageUrl(page, image.src, outputDir, prompt);
console.log(`Saved: ${outputPath}`);

if (shouldCloseContext) {
  await context.close();
} else if (browser) {
  await browser.close();
}

function normalizePrompt(value) {
  return String(value ?? "").replace(/[\r\n\t]+/g, " ").replace(/\s{2,}/g, " ").trim();
}

function normalizeList(value) {
  if (value === undefined || value === null || value === false) return [];
  return Array.isArray(value) ? value.filter(Boolean) : [value];
}

function parseArgs(raw) {
  const parsed = {};
  for (let i = 0; i < raw.length; i += 1) {
    const item = raw[i];
    if (!item.startsWith("--")) continue;
    const key = item.slice(2);
    const next = raw[i + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      if (key === "image" && parsed[key] !== undefined) {
        parsed[key] = Array.isArray(parsed[key]) ? [...parsed[key], next] : [parsed[key], next];
      } else {
        parsed[key] = next;
      }
      i += 1;
    }
  }
  return parsed;
}

async function getOrCreatePage(context) {
  const pages = context.pages();
  const metaPage = pages.find((candidate) => candidate.url().startsWith("https://www.meta.ai/") || candidate.url().startsWith("https://meta.ai/"));
  if (metaPage) return metaPage;
  const nonBlank = pages.find((candidate) => candidate.url() !== "about:blank");
  return nonBlank || pages[0] || await context.newPage();
}

async function allowDownloadsForAttachedChrome(context, page, downloadPath) {
  try {
    const session = await context.newCDPSession(page);
    await session.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath });
  } catch {}
}

async function ensureComposerReady(page) {
  await page.bringToFront().catch(() => {});
  const composer = await waitForComposer(page, 120000);
  if (!composer) {
    if (await loginPromptVisible(page)) {
      throw new Error("Meta AI is asking for login. Click Open Meta Login, sign in, then run again.");
    }
    throw new Error("Could not find the Meta AI prompt box. Sign into Meta AI in the automation Chrome window, then run again.");
  }
}

async function waitForComposer(page, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const composer = await findComposer(page);
    if (composer) return composer;
    if (await loginPromptVisible(page)) return null;
    await page.waitForTimeout(1000);
  }
  return null;
}

async function findComposer(page) {
  const locators = [
    page.getByPlaceholder(/ask meta ai/i),
    page.getByPlaceholder(/where should we start/i),
    page.getByPlaceholder(/ask anything/i),
    page.getByPlaceholder(/message/i),
    page.getByRole("textbox", { name: /ask meta ai|message|prompt/i }),
    page.locator("[contenteditable='true'][aria-label*='Ask Meta' i]"),
    page.locator("textarea[placeholder*='Describe' i]"),
    page.locator("textarea[placeholder*='Ask' i]"),
    page.locator("[contenteditable='true'][data-placeholder*='Describe' i]"),
    page.locator("[contenteditable='true'][aria-label*='message' i]"),
    page.locator("[contenteditable='true']"),
    page.locator("textarea"),
  ];
  return await firstVisibleLocator(locators);
}

async function loginPromptVisible(page) {
  return await page.evaluate(() => {
    const text = document.body?.innerText || "";
    return /\b(log in|sign up|get more from meta ai)\b/i.test(text)
      && Array.from(document.querySelectorAll("button,a,[role='button']")).some((el) => {
        const rect = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden" && /log in/i.test(el.textContent || el.getAttribute("aria-label") || "");
      });
  }).catch(() => false);
}

async function firstVisibleLocator(locators) {
  for (const locator of locators) {
    const count = await locator.count().catch(() => 0);
    for (let i = count - 1; i >= 0; i -= 1) {
      const candidate = locator.nth(i);
      if (await candidate.isVisible().catch(() => false)) return candidate;
    }
  }
  return null;
}

async function attachImage(page, filePath) {
  await fs.access(filePath).catch(() => {
    throw new Error(`Image file does not exist: ${filePath}`);
  });

  const beforeAttachmentKeys = await getAttachmentPreviewKeys(page);
  const opened = await clickFirstVisible([
    page.getByRole("button", { name: /^\+$/i }),
    page.locator("button[aria-label*='Add' i]"),
    page.locator("button[aria-label*='Attach' i]"),
    page.locator("button[aria-label*='Upload' i]"),
    page.getByRole("button", { name: /attach|upload|add photo|add files|photo|image/i }),
    page.locator("[data-testid*='attach' i]"),
    page.locator("[data-testid*='upload' i]"),
    page.locator("button").filter({ hasText: /^\s*\+\s*$/ }),
  ]);
  if (!opened) {
    const directInput = await firstExistingLocator([
      page.locator("input[type='file'][accept*='image' i]"),
      page.locator("input[type='file']"),
    ]);
    if (directInput) {
      await directInput.setInputFiles(filePath);
      console.log("Image set on file input; waiting for Meta AI attachment preview.");
      await waitForAttachmentPreview(page, beforeAttachmentKeys, filePath);
      return;
    }
    throw new Error("Could not find the Meta AI plus/upload button.");
  }

  // The plus button opens Meta's Add media and files modal; it does not open
  // the OS chooser. Give the modal a moment to mount, then use its input
  // directly whenever possible instead of waiting for a chooser that never fires.
  await page.waitForTimeout(300);
  const directInput = await firstExistingLocator([
    page.locator("[role='dialog'] input[type='file'][accept*='image' i]"),
    page.locator("[role='dialog'] input[type='file']"),
    page.locator("input[type='file'][accept*='image' i]"),
    page.locator("input[type='file']"),
  ]);
  if (directInput) {
    await directInput.setInputFiles(filePath);
    console.log("Image set directly on Meta AI media-modal input; waiting for attachment preview.");
    await waitForAttachmentPreview(page, beforeAttachmentKeys, filePath);
    return;
  }

  const chooserPromise = page.waitForEvent("filechooser", { timeout: 5000 }).catch(() => null);
  const browseClicked = await clickFirstVisible([
    page.getByText(/click to browse/i, { exact: false }),
    page.getByRole("button", { name: /click to browse|browse|upload/i }),
    page.locator("[role='dialog']").getByText(/browse/i),
  ]);
  if (!browseClicked) throw new Error("Meta AI media modal opened, but its browse control was not found.");
  const chooser = await chooserPromise;
  if (!chooser) throw new Error("Meta AI browse control did not open a file chooser.");
  await chooser.setFiles(filePath);
  console.log("Image selected; waiting for Meta AI attachment preview.");
  await waitForAttachmentPreview(page, beforeAttachmentKeys, filePath);
}

async function getAttachmentPreviewKeys(page) {
  return await page.evaluate(() => {
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 900;
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1600;
    const visible = (el) => {
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden" && rect.bottom > 0 && rect.right > 0 && rect.top < viewportHeight && rect.left < viewportWidth;
    };
    const keys = [];
    for (const img of Array.from(document.querySelectorAll("img"))) {
      if (!visible(img)) continue;
      const rect = img.getBoundingClientRect();
      const src = img.currentSrc || img.src || img.getAttribute("src") || "";
      const thumbnailLike = rect.width >= 24 && rect.height >= 24 && rect.width <= 180 && rect.height <= 180;
      const inComposerArea = rect.top > viewportHeight * 0.18 && rect.left > viewportWidth * 0.18 && rect.left < viewportWidth * 0.86;
      if (thumbnailLike && inComposerArea) keys.push(`img:${src}|${Math.round(rect.left)},${Math.round(rect.top)}|${Math.round(rect.width)}x${Math.round(rect.height)}`);
    }
    for (const el of Array.from(document.querySelectorAll("[style*='background-image']"))) {
      if (!visible(el)) continue;
      const rect = el.getBoundingClientRect();
      const background = getComputedStyle(el).backgroundImage || "";
      const thumbnailLike = rect.width >= 24 && rect.height >= 24 && rect.width <= 180 && rect.height <= 180;
      const inComposerArea = rect.top > viewportHeight * 0.18 && rect.left > viewportWidth * 0.18 && rect.left < viewportWidth * 0.86;
      if (thumbnailLike && inComposerArea && /url\(/i.test(background)) keys.push(`bg:${background}|${Math.round(rect.left)},${Math.round(rect.top)}|${Math.round(rect.width)}x${Math.round(rect.height)}`);
    }
    return keys;
  }).catch(() => []);
}

async function waitForAttachmentPreview(page, beforeKeys, filePath, maxMs = 15000) {
  const before = new Set(beforeKeys || []);
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const after = await getAttachmentPreviewKeys(page);
    const newKeys = after.filter((key) => !before.has(key));
    if (newKeys.length || after.length > before.size) {
      console.log(`Detected Meta AI attachment preview for ${path.basename(filePath)}.`);
      await page.waitForTimeout(1500);
      return true;
    }
    await page.waitForTimeout(1000);
  }
  console.log(`Attachment preview was not detected for ${path.basename(filePath)}; continuing because the file input accepted the image.`);
  return false;
}

async function firstExistingLocator(locators) {
  for (const locator of locators) {
    const count = await locator.count().catch(() => 0);
    if (count > 0) return locator.first();
  }
  return null;
}

async function enterPrompt(page, composer, text, options = {}) {
  const normalized = normalizePrompt(text);
  if (options.preserveAttachments) {
    await clickComposerTextArea(page, composer);
    await page.keyboard.insertText(normalized);
    await page.waitForTimeout(500);
    if (await promptLooksEntered(composer, normalized) || await promptTextVisibleOnPage(page, normalized)) return;
    throw new Error("Could not enter prompt text in Meta AI without disturbing uploaded attachments.");
  }
  await composer.click();
  await composer.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
  const filled = await composer.fill(normalized, { timeout: 10000 }).then(() => true).catch(() => false);
  if (filled && await promptLooksEntered(composer, normalized)) return;
  await composer.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
  await composer.press("Backspace").catch(() => {});
  await page.keyboard.insertText(normalized);
  await page.waitForTimeout(500);
  if (await promptLooksEntered(composer, normalized)) return;
  throw new Error("Could not enter prompt text in Meta AI.");
}

async function clickComposerTextArea(page, composer) {
  const box = await composer.boundingBox().catch(() => null);
  if (box) {
    await page.mouse.click(box.x + Math.min(Math.max(30, box.width * 0.45), box.width - 30), box.y + Math.max(24, box.height - 42));
    await page.waitForTimeout(250);
    return;
  }
  await composer.click();
  await page.waitForTimeout(250);
}

async function promptLooksEntered(locator, normalizedText) {
  if (!normalizedText) return true;
  const wanted = normalizedText.slice(0, Math.min(40, normalizedText.length)).toLowerCase();
  const text = await locator.evaluate((el) => {
    return String(el.value || el.innerText || el.textContent || "").replace(/[\r\n\t]+/g, " ").replace(/\s{2,}/g, " ").trim();
  }).catch(() => "");
  return text.toLowerCase().includes(wanted);
}

async function promptTextVisibleOnPage(page, normalizedText) {
  if (!normalizedText) return true;
  const wanted = normalizedText.slice(0, Math.min(40, normalizedText.length)).toLowerCase();
  return await page.evaluate((needle) => {
    const text = String(document.body?.innerText || "").replace(/[\r\n\t]+/g, " ").replace(/\s{2,}/g, " ").toLowerCase();
    return text.includes(needle);
  }, wanted).catch(() => false);
}

async function clickFirstVisible(locators) {
  for (const locator of locators) {
    const count = await locator.count().catch(() => 0);
    for (let i = count - 1; i >= 0; i -= 1) {
      const candidate = locator.nth(i);
      if (!(await candidate.isVisible().catch(() => false))) continue;
      const disabled = await candidate.getAttribute("disabled").catch(() => null);
      const ariaDisabled = await candidate.getAttribute("aria-disabled").catch(() => null);
      if (disabled !== null || ariaDisabled === "true") continue;
      const clicked = await candidate.click().then(() => true).catch(() => false);
      if (clicked) return true;
    }
  }
  return false;
}

async function clickMetaSubmitBesideInstant(page) {
  const instantButtons = page.getByRole("button", { name: /^instant$/i });
  const instantCount = await instantButtons.count().catch(() => 0);
  let instantBox = null;
  for (let index = instantCount - 1; index >= 0; index -= 1) {
    const candidate = instantButtons.nth(index);
    if (!(await candidate.isVisible().catch(() => false))) continue;
    instantBox = await candidate.boundingBox().catch(() => null);
    if (instantBox) break;
  }
  if (!instantBox) return false;

  const arrowButtons = page
    .locator("svg path[d^='M16 6.125a.89.89'][d*='V25'][d*='7.5-7.5']")
    .locator("xpath=ancestor::*[self::button or @role='button'][1]");
  const arrowCount = await arrowButtons.count().catch(() => 0);
  const instantCenterY = instantBox.y + instantBox.height / 2;
  const candidates = [];
  for (let index = 0; index < arrowCount; index += 1) {
    const button = arrowButtons.nth(index);
    if (!(await button.isVisible().catch(() => false))) continue;
    if (await button.isDisabled().catch(() => false)) continue;
    if (await button.getAttribute("aria-disabled").catch(() => null) === "true") continue;
    const box = await button.boundingBox().catch(() => null);
    if (!box) continue;
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    if (centerX <= instantBox.x + instantBox.width) continue;
    if (Math.abs(centerY - instantCenterY) > 36) continue;
    candidates.push({ box, distance: centerX - (instantBox.x + instantBox.width) });
  }
  candidates.sort((a, b) => a.distance - b.distance);
  const target = candidates[0]?.box;
  if (!target) return false;
  const x = target.x + target.width / 2;
  const y = target.y + target.height / 2;
  console.log(`Clicking Meta AI submit arrow beside Instant at ${Math.round(x)},${Math.round(y)}.`);
  await page.mouse.click(x, y);
  return true;
}

async function sidebarChatHrefs(page) {
  return await page.evaluate(() => Array.from(document.querySelectorAll("a[href]"))
    .filter((anchor) => {
      const rect = anchor.getBoundingClientRect();
      const style = getComputedStyle(anchor);
      return rect.left >= 0 && rect.left < 450 && rect.top > 180
        && rect.width > 40 && rect.height > 12
        && style.display !== "none" && style.visibility !== "hidden";
    })
    .map((anchor) => anchor.href)
    .filter(Boolean)
  ).catch(() => []);
}

async function openNewSidebarConversation(page, composer, oldHrefs, maxMs) {
  const started = Date.now();
  let composerCleared = false;
  while (Date.now() - started < maxMs) {
    if (!composerCleared) {
      const text = await composer.evaluate((el) => String(el.value || el.innerText || el.textContent || "").trim()).catch(() => "");
      composerCleared = !text;
      if (!composerCleared) {
        await page.waitForTimeout(250);
        continue;
      }
      console.log("Meta AI cleared the submitted prompt; looking for its new sidebar conversation link.");
    }
    const links = page.locator("a[href]");
    const count = await links.count().catch(() => 0);
    for (let index = 0; index < count; index += 1) {
      const link = links.nth(index);
      const href = await link.getAttribute("href").catch(() => "");
      if (!href) continue;
      const absoluteHref = await link.evaluate((el) => el.href).catch(() => "");
      if (!absoluteHref || oldHrefs.has(absoluteHref)) continue;
      const box = await link.boundingBox().catch(() => null);
      if (!box || box.x < 0 || box.x >= 450 || box.y <= 180 || box.width <= 40 || box.height <= 12) continue;
      if (!(await link.isVisible().catch(() => false))) continue;
      console.log(`Opening new Meta AI conversation: ${absoluteHref}`);
      await link.click({ timeout: 5000 });
      await page.waitForTimeout(750);
      return true;
    }
    await page.waitForTimeout(250);
  }
  return false;
}

async function getCandidateImageKeys(page) {
  return page.evaluate(() => Array.from(document.querySelectorAll("img"))
    .filter((img) => {
      const rect = img.getBoundingClientRect();
      const style = getComputedStyle(img);
      return rect.width > 160 && rect.height > 160 && style.display !== "none" && style.visibility !== "hidden";
    })
    .map((img) => `${img.currentSrc || img.src || img.getAttribute("src") || ""}|${Math.round(img.getBoundingClientRect().width)}x${Math.round(img.getBoundingClientRect().height)}`)
  ).catch(() => []);
}

async function newestOpenMetaPage(context, previousPage = null) {
  const pages = context.pages().filter((candidate) => !candidate.isClosed());
  const metaPages = pages.filter((candidate) => /^https:\/\/(?:www\.)?meta\.ai\//i.test(candidate.url()));
  return metaPages.at(-1) || (previousPage && !previousPage.isClosed() ? previousPage : null) || pages.at(-1) || null;
}

async function waitForNewImage(context, initialPage, beforeKeys, maxMs) {
  const started = Date.now();
  let activePage = initialPage;
  while (Date.now() - started < maxMs) {
    if (!activePage || activePage.isClosed()) {
      const replacement = await newestOpenMetaPage(context, activePage);
      if (!replacement) {
        await new Promise((resolve) => setTimeout(resolve, 250));
        continue;
      }
      activePage = replacement;
      activePage.setDefaultTimeout(30000);
      console.log(`Following replacement Meta AI page: ${activePage.url()}`);
    }
    const handles = await activePage.locator("img").elementHandles().catch(async (error) => {
      if (/closed|target page|context/i.test(String(error?.message || error))) {
        activePage = await newestOpenMetaPage(context, activePage);
        return [];
      }
      throw error;
    });
    for (let i = handles.length - 1; i >= 0; i -= 1) {
      const handle = handles[i];
      const info = await handle.evaluate((img) => {
        const rect = img.getBoundingClientRect();
        const style = getComputedStyle(img);
        const src = img.currentSrc || img.src || img.getAttribute("src") || "";
        return {
          src,
          key: `${src}|${Math.round(rect.width)}x${Math.round(rect.height)}`,
          visible: rect.width > 200 && rect.height > 200 && style.display !== "none" && style.visibility !== "hidden",
          complete: img.complete && img.naturalWidth > 0 && img.naturalHeight > 0,
        };
      }).catch(() => null);
      if (!info?.visible || !info.complete) continue;
      if (!beforeKeys.has(info.key)) {
        console.log("Found new visible image.");
        return { image: info, page: activePage };
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("Timed out waiting for a generated Meta AI image.");
}

async function downloadGeneratedImageFromOverlay(page, imageInfo, outputDir, promptText) {
  const images = page.locator("img");
  const count = await images.count().catch(() => 0);
  let target = null;
  for (let index = count - 1; index >= 0; index -= 1) {
    const candidate = images.nth(index);
    const src = await candidate.evaluate((img) => img.currentSrc || img.src || img.getAttribute("src") || "").catch(() => "");
    if (src !== imageInfo.src || !(await candidate.isVisible().catch(() => false))) continue;
    target = candidate;
    break;
  }
  if (!target) {
    console.log("Generated image element was not found for Download-button hover.");
    return null;
  }

  await target.scrollIntoViewIfNeeded().catch(() => {});
  await target.hover({ force: true }).catch(() => {});
  await page.waitForTimeout(350);
  const downloadControls = [
    page.getByRole("button", { name: /^download$/i }),
    page.locator("button[aria-label='Download' i]"),
    page.locator("button[title='Download' i]"),
  ];
  let downloadButton = null;
  for (const locator of downloadControls) {
    const buttonCount = await locator.count().catch(() => 0);
    for (let index = buttonCount - 1; index >= 0; index -= 1) {
      const candidate = locator.nth(index);
      if (await candidate.isVisible().catch(() => false)) {
        downloadButton = candidate;
        break;
      }
    }
    if (downloadButton) break;
  }
  if (!downloadButton) {
    console.log("Generated image Download button did not appear after hover.");
    return null;
  }

  console.log("Clicking generated image Download button...");
  const downloadPromise = page.waitForEvent("download", { timeout: 30000 }).catch(() => null);
  const clicked = await downloadButton.click({ timeout: 5000 }).then(() => true).catch(() => false);
  if (!clicked) return null;
  const download = await downloadPromise;
  if (!download) {
    console.log("Download button was clicked, but Meta AI did not emit a browser download event.");
    return null;
  }
  const suggested = download.suggestedFilename() || `${safeFilename(promptText)}.png`;
  const outputPath = path.join(outputDir, uniqueFilename(suggested));
  await download.saveAs(outputPath);
  return outputPath;
}

async function waitForThoughtComplete(page, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const ready = await page.evaluate(() => {
      const text = document.body?.innerText || "";
      return /Thought\s+for\s+\d+\s*(s|sec|secs|second|seconds)\b/i.test(text);
    }).catch(() => false);
    if (ready) {
      console.log("Detected Thought-for completion marker.");
      await page.waitForTimeout(5000);
      return true;
    }
    await page.waitForTimeout(3000);
  }
  throw new Error("Timed out waiting for Thought-for completion marker.");
}

async function downloadFromViewer(page, outputDir, promptText) {
  const downloadPromise = page.waitForEvent("download", { timeout: 15000 }).catch(() => null);
  const clicked = await clickDownloadButton(page);
  if (!clicked) return null;

  const download = await downloadPromise;
  if (!download) return null;
  const suggested = download.suggestedFilename() || `${safeFilename(promptText)}.png`;
  const outputPath = path.join(outputDir, uniqueFilename(suggested));
  try {
    await download.saveAs(outputPath);
    return outputPath;
  } catch {
    return await findExistingDownloadedFile(outputDir, suggested);
  }
}

async function clickDownloadButton(page) {
  const labeled = await clickFirstVisible([
    page.getByRole("button", { name: /download/i }),
    page.getByRole("link", { name: /download/i }),
    page.locator("button[aria-label*='Download' i]"),
    page.locator("a[aria-label*='Download' i]"),
    page.locator("button[title*='Download' i]"),
    page.locator("a[title*='Download' i]"),
    page.locator("a[download]"),
  ]);
  if (labeled) return true;

  return await page.evaluate(() => {
    const isVisible = (el) => {
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
    };
    const nameOf = (el) => [
      el.getAttribute("aria-label"),
      el.getAttribute("title"),
      el.getAttribute("data-testid"),
      el.textContent,
    ].filter(Boolean).join(" ").trim();
    const controls = Array.from(document.querySelectorAll("button,a,[role='button']"))
      .filter((el) => isVisible(el))
      .map((el) => {
        const rect = el.getBoundingClientRect();
        return {
          el,
          rect,
          name: nameOf(el),
          svg: el.querySelector("svg")?.outerHTML || "",
        };
      });

    const named = controls.find((item) => /download/i.test(item.name) || item.el.hasAttribute("download"));
    if (named) {
      named.el.click();
      return true;
    }

    const iconMatch = controls.find((item) => {
      const svg = item.svg.toLowerCase();
      return svg.includes("download") || svg.includes("m21 15v4") || svg.includes("m7 10l5 5") || svg.includes("arrow-down");
    });
    if (iconMatch) {
      iconMatch.el.click();
      return true;
    }

    // Never guess based on toolbar position. Meta frequently places New Project
    // beside Share, and clicking it strands the generated image in History.
    return false;
  }).catch(() => false);
}

async function retryDownloadFromViewer(page, outputDir, promptText, attempts, delayMs) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    console.log(`Trying Meta AI download button, attempt ${attempt}/${attempts}...`);
    const downloaded = await downloadFromViewer(page, outputDir, promptText);
    if (downloaded) return downloaded;
    if (attempt < attempts) {
      console.log(`Download not ready; waiting ${Math.round(delayMs / 1000)} seconds before retry...`);
      await page.waitForTimeout(delayMs);
    }
  }
  return null;
}

async function retryOpenViewerAndDownload(page, imageInfo, outputDir, promptText, attempts, delayMs) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    console.log(`Opening generated image viewer, attempt ${attempt}/${attempts}...`);
    const opened = await clickGeneratedImage(page, imageInfo);
    if (!opened) {
      console.log("Could not click generated image; retrying with newest visible image.");
      imageInfo = await newestVisibleImageInfo(page);
      if (!(await clickGeneratedImage(page, imageInfo))) {
        console.log("Generated image could not be opened on this attempt.");
        if (attempt < attempts) await page.waitForTimeout(delayMs);
        continue;
      }
    }
    await page.waitForTimeout(3000);

    console.log("Saving visible viewer image directly...");
    const directSaved = await saveNewestVisibleImage(page, outputDir, promptText).catch((error) => {
      console.log(`Direct viewer image save did not work: ${error.message}`);
      return null;
    });
    if (directSaved) return directSaved;

    console.log("Trying toolbar download button...");
    const downloaded = await downloadFromViewer(page, outputDir, promptText);
    if (downloaded) return downloaded;

    console.log("Download button did not work; leaving the active Meta AI generation open.");
    if (attempt < attempts) {
      console.log(`Waiting ${Math.round(delayMs / 1000)} seconds before retrying without closing the generation...`);
      await page.waitForTimeout(delayMs);
    }
  }
  return null;
}

async function clickGeneratedImage(page, imageInfo) {
  for (let retry = 0; retry < 3; retry += 1) {
    const index = await page.evaluate((wanted) => {
      const images = Array.from(document.querySelectorAll("img"))
        .map((img, index) => {
          const rect = img.getBoundingClientRect();
          const style = getComputedStyle(img);
          const src = img.currentSrc || img.src || img.getAttribute("src") || "";
          return {
            index,
            src,
            key: `${src}|${Math.round(rect.width)}x${Math.round(rect.height)}`,
            area: rect.width * rect.height,
            visible: rect.width > 200 && rect.height > 200 && style.display !== "none" && style.visibility !== "hidden",
            complete: img.complete && img.naturalWidth > 0 && img.naturalHeight > 0,
          };
        })
        .filter((item) => item.visible && item.complete && item.src);
      const exact = images.findLast((item) => item.key === wanted.key);
      const sameSrc = images.findLast((item) => item.src === wanted.src);
      const largest = images.sort((a, b) => b.area - a.area)[0];
      return (exact || sameSrc || largest)?.index ?? -1;
    }, imageInfo).catch(() => -1);
    if (index < 0) return false;
    const image = page.locator("img").nth(index);
    await image.scrollIntoViewIfNeeded().catch(() => {});
    const clicked = await image.click({ timeout: 10000 }).then(() => true).catch(() => false);
    if (clicked) return true;
    await page.waitForTimeout(1000);
  }
  return false;
}

async function closeImageViewer(page) {
  const closed = await clickFirstVisible([
    page.getByRole("button", { name: /^close$/i }),
    page.locator("button[aria-label*='Close' i]"),
  ]).catch(() => false);
  if (!closed) {
    await page.keyboard.press("Escape").catch(() => {});
  }
  await page.waitForTimeout(1500);
}

async function newestVisibleImageUrl(page) {
  const info = await newestVisibleImageInfo(page);
  if (!info.src) throw new Error("Could not find a visible image URL to save.");
  return info.src;
}

async function saveNewestVisibleImage(page, outputDir, promptText) {
  const imageUrl = await newestVisibleImageUrl(page);
  return await saveImageUrl(page, imageUrl, outputDir, promptText);
}

async function newestVisibleImageInfo(page) {
  const info = await page.evaluate(() => {
    const images = Array.from(document.querySelectorAll("img"))
      .map((img) => {
        const rect = img.getBoundingClientRect();
        const style = getComputedStyle(img);
        const src = img.currentSrc || img.src || img.getAttribute("src") || "";
        return {
          src,
          key: `${src}|${Math.round(rect.width)}x${Math.round(rect.height)}`,
          area: rect.width * rect.height,
          visible: rect.width > 200 && rect.height > 200 && style.display !== "none" && style.visibility !== "hidden",
          complete: img.complete && img.naturalWidth > 0 && img.naturalHeight > 0,
        };
      })
      .filter((item) => item.visible && item.complete && item.src)
      .sort((a, b) => b.area - a.area);
    return images[0] || null;
  });
  if (!info) throw new Error("Could not find a visible image URL to save.");
  return info;
}

async function saveImageUrl(page, imageUrl, outputDir, promptText) {
  const contextResponse = await page.context().request.get(imageUrl, {
    timeout: 60000,
    failOnStatusCode: false,
  }).catch((error) => ({ error: error.message }));
  if (contextResponse && !contextResponse.error && contextResponse.ok()) {
    const bytes = await contextResponse.body();
    if (bytes?.length) {
      const contentType = contextResponse.headers()["content-type"] || "image/png";
      const ext = extensionForContentType(contentType);
      const outputPath = path.join(outputDir, uniqueFilename(`${safeFilename(promptText)}${ext}`));
      await fs.writeFile(outputPath, bytes);
      return outputPath;
    }
  }

  const contextError = contextResponse?.error
    || (contextResponse && typeof contextResponse.status === "function" ? `${contextResponse.status()} ${contextResponse.statusText()}` : "no data");
  console.log(`Browser-context image download did not work (${contextError}); trying in-page fetch fallback.`);
  const result = await page.evaluate(async (url) => {
    const response = await fetch(url, { credentials: "include" });
    if (!response.ok) return { error: `${response.status} ${response.statusText}` };
    const buffer = await response.arrayBuffer();
    return {
      contentType: response.headers.get("content-type") || "image/png",
      bytes: Array.from(new Uint8Array(buffer)),
    };
  }, imageUrl).catch((error) => ({ error: error.message }));

  if (!result || result.error || !result.bytes?.length) {
    throw new Error(`Direct image save failed. Browser-context request: ${contextError}. In-page request: ${result?.error || "no data"}`);
  }

  const ext = extensionForContentType(result.contentType);
  const outputPath = path.join(outputDir, uniqueFilename(`${safeFilename(promptText)}${ext}`));
  await fs.writeFile(outputPath, Buffer.from(result.bytes));
  return outputPath;
}

async function findExistingDownloadedFile(downloadPath, suggestedName) {
  const files = await fs.readdir(downloadPath, { withFileTypes: true }).catch(() => []);
  const wantedBase = path.basename(suggestedName).toLowerCase();
  const candidates = [];
  for (const entry of files) {
    if (!entry.isFile()) continue;
    const fullPath = path.join(downloadPath, entry.name);
    const stat = await fs.stat(fullPath).catch(() => null);
    if (!stat) continue;
    const lower = entry.name.toLowerCase();
    const isLikely = lower === wantedBase || lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".webp");
    if (isLikely) candidates.push({ fullPath, mtimeMs: stat.mtimeMs });
  }
  candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return candidates[0]?.fullPath || null;
}

function extensionForContentType(contentType) {
  const normalized = contentType.toLowerCase();
  if (normalized.includes("jpeg") || normalized.includes("jpg")) return ".jpg";
  if (normalized.includes("webp")) return ".webp";
  if (normalized.includes("png")) return ".png";
  return ".png";
}

function safeFilename(value) {
  return String(value || "meta-ai-image").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80) || "meta-ai-image";
}

function uniqueFilename(name) {
  const ext = path.extname(name);
  const base = path.basename(name, ext);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${base}-${stamp}${ext || ".png"}`;
}
