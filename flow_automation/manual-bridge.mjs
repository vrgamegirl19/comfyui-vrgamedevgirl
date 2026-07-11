import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs/promises";

const args = parseArgs(process.argv.slice(2));
const projectDir = path.dirname(new URL(import.meta.url).pathname).replace(/^\/(.:\/)/, "$1");
const provider = normalizeProvider(args.provider || "flow_nano_banana");
const action = String(args.action || "wait-download").trim().toLowerCase();
const url = args.url || providerUrl(provider);
const outputDir = path.resolve(projectDir, args.out || "manual_downloads");
const timeout = Number(args.timeout || 600000);
const cdpUrl = args.cdp || args["connect-cdp"] || "";
const imagePaths = normalizeList(args.image).map((imagePath) => path.resolve(imagePath));
const prompt = String(args.prompt || "").trim();

await fs.mkdir(outputDir, { recursive: true });

if (!cdpUrl) {
  throw new Error("Manual bridge requires --connect-cdp so it can use the existing provider browser.");
}

const browser = await chromium.connectOverCDP(cdpUrl);
const context = browser.contexts()[0];
if (!context) throw new Error("Connected to Chrome, but no browser context was available.");

const page = await getOrCreateProviderPage(context, provider);
page.setDefaultTimeout(30000);

await ensureProviderPage(page, provider, url);
await allowDownloadsForAttachedChrome(context, page, outputDir);

if (action === "upload") {
  if (!imagePaths.length) throw new Error("No image paths were provided for upload.");
  for (let index = 0; index < imagePaths.length; index += 1) {
    const imagePath = imagePaths[index];
    console.log(`Uploading ${index + 1}/${imagePaths.length}: ${imagePath}`);
    if (provider === "gpt_image") await attachChatGptImage(page, imagePath);
    else if (provider === "meta_ai") await attachMetaAIImage(page, imagePath);
    else await uploadFlowImageAndAddToPrompt(page, imagePath);
  }
  if (provider === "meta_ai") {
    console.log("Waiting for Meta AI to finish preparing all manual attachments...");
    await page.waitForTimeout(2000);
  }
  if (prompt) {
    console.log("Copying manual chat prompt into the provider composer...");
    await fillManualChatPrompt(page, prompt);
  }
  console.log(`Uploaded ${imagePaths.length} image(s).`);
} else if (action === "wait-download") {
  console.log(`Waiting for next ${providerLabel(provider)} browser download...`);
  const startedAt = Date.now();
  const savedPath = await waitForDownloadOrNewFile(context, outputDir, timeout, startedAt);
  console.log(`Saved: ${savedPath}`);
} else if (action === "open") {
  console.log(`${providerLabel(provider)} browser ready: ${page.url()}`);
} else {
  throw new Error(`Unknown manual bridge action: ${action}`);
}

function normalizeProvider(value) {
  const key = String(value || "").trim().toLowerCase().replace(/[-\s]+/g, "_");
  if (["gpt", "gpt_image", "gpt_images", "chatgpt", "chatgpt_image", "chatgpt_images"].includes(key)) return "gpt_image";
  if (["meta", "meta_ai", "metaai", "meta_image", "meta_images"].includes(key)) return "meta_ai";
  return "flow_nano_banana";
}

function providerLabel(value) {
  if (value === "meta_ai") return "Meta AI";
  return value === "gpt_image" ? "GPT Image" : "Flow";
}

function providerUrl(value) {
  if (value === "gpt_image") return "https://chatgpt.com/images";
  if (value === "meta_ai") return "https://www.meta.ai/";
  return "https://labs.google/fx/tools/flow";
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

async function fillManualChatPrompt(page, promptText) {
  const composer = await findVisibleLocator([
    page.locator("textarea[placeholder*='message' i]"),
    page.locator("textarea[placeholder*='ask' i]"),
    page.locator("textarea[placeholder*='prompt' i]"),
    page.locator("textarea"),
    page.locator("[contenteditable='true'][data-placeholder]"),
    page.locator("[contenteditable='true'][role='textbox']"),
    page.locator("[contenteditable='true']"),
  ], 20000);
  if (!composer) throw new Error("Reference images uploaded, but the provider prompt box was not found.");
  await composer.scrollIntoViewIfNeeded().catch(() => {});
  await composer.click();
  const tagName = await composer.evaluate((element) => element.tagName.toLowerCase());
  if (tagName === "textarea" || tagName === "input") {
    await composer.fill(promptText);
  } else {
    await composer.evaluate((element, value) => {
      element.focus();
      element.textContent = value;
      element.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    }, promptText);
  }
  await page.waitForTimeout(500);
}

async function getOrCreateProviderPage(context, providerName) {
  const pages = context.pages();
  const matcher = providerName === "gpt_image"
    ? (candidate) => candidate.url().startsWith("https://chatgpt.com/")
    : providerName === "meta_ai"
      ? (candidate) => candidate.url().startsWith("https://www.meta.ai/") || candidate.url().startsWith("https://meta.ai/")
      : (candidate) => candidate.url().startsWith("https://labs.google/") && !candidate.url().includes("/signin");
  return pages.find(matcher) || pages.find((candidate) => candidate.url() !== "about:blank") || pages[0] || await context.newPage();
}

async function ensureProviderPage(page, providerName, targetUrl) {
  await page.bringToFront().catch(() => {});
  const currentUrl = page.url() || "";
  const valid = providerName === "gpt_image"
    ? currentUrl.startsWith("https://chatgpt.com/")
    : providerName === "meta_ai"
      ? currentUrl.startsWith("https://www.meta.ai/") || currentUrl.startsWith("https://meta.ai/")
      : currentUrl.startsWith("https://labs.google/");
  if (!valid) {
    await page.goto(targetUrl, { waitUntil: "domcontentloaded" });
    await page.waitForLoadState("networkidle", { timeout: 60000 }).catch(() => {});
  }
}

async function allowDownloadsForAttachedChrome(context, page, downloadPath) {
  try {
    const session = await context.newCDPSession(page);
    await session.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath });
  } catch {}
}

async function uploadFlowImageAndAddToPrompt(page, filePath) {
  await fs.access(filePath).catch(() => {
    throw new Error(`Image file does not exist: ${filePath}`);
  });
  const beforeUrls = new Set(await getPromptAttachmentUrls(page));
  const opened = await clickFirstVisible([
    page.getByRole("button", { name: /^create$/i }),
    page.locator("button[aria-haspopup='dialog']:has(i.google-symbols:text-is('add_2'))"),
    page.locator("button:has(i.google-symbols:text-is('add_2'))"),
    page.locator("button:has(i.google-symbols:text-is('add'))"),
  ], 12000);
  if (!opened) throw new Error("Could not find the bottom + / Create button to open media upload.");

  await page.waitForTimeout(800);
  const uploadTarget = await findVisibleLocator([
    page.getByText(/upload media/i),
    page.getByRole("button", { name: /upload media/i }),
    page.locator("[role='button']:has-text('Upload media')"),
    page.locator("button:has-text('Upload media')"),
    page.locator("text=Upload media"),
  ], 15000);
  if (!uploadTarget) throw new Error("Could not find Upload media in Flow.");

  const chooserPromise = page.waitForEvent("filechooser", { timeout: 15000 });
  await uploadTarget.click();
  const chooser = await chooserPromise;
  await chooser.setFiles(filePath);
  await page.waitForTimeout(10000);

  const addToPrompt = await findAddToPromptAfterUpload(page, beforeUrls, 90000);
  await addToPrompt.click();
  await page.waitForTimeout(3000);
}

async function attachChatGptImage(page, filePath) {
  await fs.access(filePath).catch(() => {
    throw new Error(`Image file does not exist: ${filePath}`);
  });
  const directInput = await firstExistingLocator([
    page.locator("input[type='file'][accept*='image' i]"),
    page.locator("input[type='file']"),
  ]);
  if (directInput) {
    await directInput.setInputFiles(filePath);
    await page.waitForTimeout(5000);
    return;
  }

  const chooserPromise = page.waitForEvent("filechooser", { timeout: 20000 }).catch(() => null);
  const opened = await clickFirstVisible([
    page.getByRole("button", { name: /^add photos$/i }),
    page.locator("button[aria-label*='Add photos' i]"),
    page.getByRole("button", { name: /attach|upload|add photo|add files/i }),
    page.locator("button[aria-label*='Attach' i]"),
    page.locator("button[aria-label*='Upload' i]"),
    page.locator("[data-testid*='attach' i]"),
    page.locator("[data-testid*='upload' i]"),
  ], 15000);
  if (!opened) throw new Error("Could not find the ChatGPT paperclip/attach button.");
  const chooser = await chooserPromise;
  if (!chooser) {
    const directInput = await firstExistingLocator([
      page.locator("input[type='file'][accept*='image' i]"),
      page.locator("input[type='file']"),
    ]);
    if (!directInput) throw new Error("Meta AI upload button opened, but no file chooser or file input appeared.");
    await directInput.setInputFiles(filePath);
    await page.waitForTimeout(5000);
    return;
  }
  await chooser.setFiles(filePath);
  await page.waitForTimeout(5000);
}

async function attachMetaAIImage(page, filePath) {
  await fs.access(filePath).catch(() => {
    throw new Error(`Image file does not exist: ${filePath}`);
  });
  const beforeAttachmentKeys = await getMetaAttachmentPreviewKeys(page);
  const opened = await clickFirstVisible([
    page.getByRole("button", { name: /^\+$/i }),
    page.locator("button[aria-label*='Add' i]"),
    page.locator("button[aria-label*='Attach' i]"),
    page.locator("button[aria-label*='Upload' i]"),
    page.getByRole("button", { name: /attach|upload|add photo|add files|photo|image/i }),
    page.locator("[data-testid*='attach' i]"),
    page.locator("[data-testid*='upload' i]"),
    page.locator("button").filter({ hasText: /^\s*\+\s*$/ }),
  ], 15000);
  if (!opened) {
    const directInput = await firstExistingLocator([
      page.locator("input[type='file'][accept*='image' i]"),
      page.locator("input[type='file']"),
    ]);
    if (!directInput) throw new Error("Could not find the Meta AI plus/upload button.");
    await directInput.setInputFiles(filePath);
    await waitForMetaAttachmentPreview(page, beforeAttachmentKeys, filePath);
    return;
  }

  await page.waitForTimeout(300);
  const directInput = await firstExistingLocator([
    page.locator("[role='dialog'] input[type='file'][accept*='image' i]"),
    page.locator("[role='dialog'] input[type='file']"),
    page.locator("input[type='file'][accept*='image' i]"),
    page.locator("input[type='file']"),
  ]);
  if (directInput) {
    await directInput.setInputFiles(filePath);
    console.log("Image set directly on Meta AI manual media-modal input; waiting for attachment preview.");
    await waitForMetaAttachmentPreview(page, beforeAttachmentKeys, filePath);
    return;
  }

  const chooserPromise = page.waitForEvent("filechooser", { timeout: 5000 }).catch(() => null);
  const browseClicked = await clickFirstVisible([
    page.getByText(/click to browse/i, { exact: false }),
    page.getByRole("button", { name: /click to browse|browse|upload/i }),
    page.locator("[role='dialog']").getByText(/browse/i),
  ], 3000);
  if (!browseClicked) throw new Error("Meta AI manual media modal opened, but its browse control was not found.");
  const chooser = await chooserPromise;
  if (!chooser) throw new Error("Meta AI manual browse control did not open a file chooser.");
  await chooser.setFiles(filePath);
  await waitForMetaAttachmentPreview(page, beforeAttachmentKeys, filePath);
}

async function getMetaAttachmentPreviewKeys(page) {
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

async function waitForMetaAttachmentPreview(page, beforeKeys, filePath, maxMs = 15000) {
  const before = new Set(beforeKeys || []);
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const after = await getMetaAttachmentPreviewKeys(page);
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

async function getPromptAttachmentUrls(page) {
  return await page.evaluate(() => Array.from(document.querySelectorAll("img"))
    .map((img) => img.currentSrc || img.src || "")
    .filter(Boolean)
    .slice(-80)).catch(() => []);
}

async function findAddToPromptAfterUpload(page, beforeUrls, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const candidate = await findVisibleLocator([
      page.getByRole("button", { name: /add to prompt/i }),
      page.locator("button:has-text('Add to Prompt')"),
      page.locator("button:has-text('Add to prompt')"),
      page.locator("[role='button']:has-text('Add to Prompt')"),
      page.locator("[role='button']:has-text('Add to prompt')"),
    ], 1000);
    if (candidate) {
      const disabled = await candidate.getAttribute("aria-disabled").catch(() => null);
      if (disabled !== "true") return candidate;
    }
    const urls = await getPromptAttachmentUrls(page);
    if (urls.some((candidateUrl) => !beforeUrls.has(candidateUrl))) {
      const afterUpload = await findVisibleLocator([
        page.getByRole("button", { name: /add to prompt/i }),
        page.locator("button:has-text('Add to Prompt')"),
        page.locator("button:has-text('Add to prompt')"),
      ], 3000);
      if (afterUpload) return afterUpload;
    }
    await page.waitForTimeout(1000);
  }
  throw new Error("Uploaded image appeared to finish, but Flow did not show Add to prompt.");
}

async function clickFirstVisible(locators, timeoutMs = 30000) {
  const locator = await findVisibleLocator(locators, timeoutMs);
  if (!locator) return false;
  await locator.click();
  return true;
}

async function findVisibleLocator(locators, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    for (const locator of locators) {
      const count = await locator.count().catch(() => 0);
      for (let i = count - 1; i >= 0; i -= 1) {
        const candidate = locator.nth(i);
        if (await candidate.isVisible().catch(() => false)) return candidate;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return null;
}

async function firstExistingLocator(locators) {
  for (const locator of locators) {
    const count = await locator.count().catch(() => 0);
    if (count > 0) return locator.first();
  }
  return null;
}

function uniqueFilename(name) {
  const ext = path.extname(name) || ".png";
  const base = path.basename(name, ext).replace(/[<>:"/\\|?*\x00-\x1F]+/g, "_").slice(0, 120) || "manual";
  return `${base}_${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14)}${ext}`;
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
    const isLikely = lower === wantedBase || /\.(png|jpe?g|webp)$/i.test(lower);
    if (isLikely) candidates.push({ fullPath, mtimeMs: stat.mtimeMs });
  }
  candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return candidates[0]?.fullPath || null;
}

async function waitForDownloadOrNewFile(context, downloadPath, timeoutMs, startedAtMs) {
  try {
    return await Promise.any([
      waitForAnyPageDownload(context, downloadPath, timeoutMs),
      waitForNewDownloadedImage(downloadPath, timeoutMs, startedAtMs),
    ]);
  } catch (error) {
    throw new Error(`No completed image download appeared in ${downloadPath} before timeout. ${error.message || ""}`.trim());
  }
}

async function waitForAnyPageDownload(context, downloadPath, timeoutMs) {
  const pages = context.pages();
  const waits = pages.map((candidate) => candidate.waitForEvent("download", { timeout: timeoutMs }));
  const download = await Promise.any(waits);
  const suggested = download.suggestedFilename() || `manual-${Date.now()}.png`;
  const outputPath = path.join(downloadPath, uniqueFilename(suggested));
  try {
    await download.saveAs(outputPath);
    return outputPath;
  } catch (error) {
    const existingPath = await findExistingDownloadedFile(downloadPath, suggested);
    if (existingPath) return existingPath;
    throw error;
  }
}

async function waitForNewDownloadedImage(downloadPath, timeoutMs, startedAtMs) {
  const deadline = Date.now() + timeoutMs;
  const earliestMtimeMs = startedAtMs - 30000;
  while (Date.now() < deadline) {
    const candidate = await newestDownloadedImageSince(downloadPath, earliestMtimeMs);
    if (candidate && await isStableFile(candidate)) return candidate;
    await new Promise((resolve) => setTimeout(resolve, 600));
  }
  throw new Error("Timed out waiting for a completed image file.");
}

async function newestDownloadedImageSince(downloadPath, startedAtMs) {
  const files = await fs.readdir(downloadPath, { withFileTypes: true }).catch(() => []);
  const candidates = [];
  for (const entry of files) {
    if (!entry.isFile()) continue;
    const lower = entry.name.toLowerCase();
    if (lower.endsWith(".crdownload") || lower.endsWith(".tmp")) continue;
    if (!/\.(png|jpe?g|webp)$/i.test(lower)) continue;
    const fullPath = path.join(downloadPath, entry.name);
    const stat = await fs.stat(fullPath).catch(() => null);
    if (!stat || stat.mtimeMs < startedAtMs - 2000 || stat.size <= 0) continue;
    candidates.push({ fullPath, mtimeMs: stat.mtimeMs });
  }
  candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return candidates[0]?.fullPath || null;
}

async function isStableFile(filePath) {
  const first = await fs.stat(filePath).catch(() => null);
  if (!first || first.size <= 0) return false;
  await new Promise((resolve) => setTimeout(resolve, 900));
  const second = await fs.stat(filePath).catch(() => null);
  return Boolean(second && second.size === first.size && second.mtimeMs === first.mtimeMs);
}
