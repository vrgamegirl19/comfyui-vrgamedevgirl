import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs/promises";

const args = parseArgs(process.argv.slice(2));
const projectDir = path.dirname(new URL(import.meta.url).pathname).replace(/^\/(.:\/)/, "$1");
const url = args.url || "https://labs.google/fx/tools/flow";
const prompt = normalizePromptForFlow(args.prompt || "a woman in a red dress");
const outputDir = path.resolve(projectDir, args.out || "downloads");
const timeout = Number(args.timeout || 240000);
const keepOpen = Boolean(args["keep-open"]);
const cdpUrl = args.cdp || args["connect-cdp"] || "";
const noNavigate = Boolean(args["no-navigate"]);
const imagePaths = normalizeList(args.image).map((imagePath) => path.resolve(imagePath));
const profileDir = path.resolve(projectDir, "browser-profile");

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

const page = await getOrCreatePage(context);
page.setDefaultTimeout(30000);

if (noNavigate) {
  console.log(`Using current tab: ${page.url()}`);
} else {
  console.log(`Opening ${url}`);
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForLoadState("networkidle", { timeout: 60000 }).catch(() => {});
}

await allowDownloadsForAttachedChrome(context, page, outputDir);
await ensureProjectPage(page);

if (imagePaths.length > 0) {
  console.log(`Uploading ${imagePaths.length} image(s) for edit.`);
  for (let index = 0; index < imagePaths.length; index += 1) {
    const imagePath = imagePaths[index];
    console.log(`Uploading image ${index + 1}/${imagePaths.length}: ${imagePath}`);
    await uploadImageAndAddToPrompt(page, imagePath);
  }
}

console.log("Finding prompt box...");
const promptBox = await findPromptBox(page);
if (promptBox) {
  await promptBox.click();
  await clearAndType(page, promptBox, prompt);
} else {
  await coordinatePromptFallback(page, prompt);
}
console.log("Prompt entered; waiting 2 seconds before submit.");
await page.waitForTimeout(2000);

const beforeImageUrls = new Set(await getGeneratedImageUrls(page));
console.log(`Generated images before submit: ${beforeImageUrls.size}`);

console.log("Submitting prompt...");
const submitted = await clickFirstVisible(submitLocators(page));
if (!submitted) {
  await page.keyboard.press("Enter");
}

console.log("Waiting 30 seconds before checking output...");
await page.waitForTimeout(30000);

console.log("Waiting for a new generated image URL...");
const imageUrl = await waitForNewGeneratedImageUrl(page, beforeImageUrls, timeout);

console.log("Saving generated image directly from Flow media URL...");
let outputPath = await saveGeneratedImageUrl(page, imageUrl, outputDir, prompt).catch((error) => {
  console.log(`Direct image save failed: ${error.message}`);
  return null;
});

if (!outputPath) {
  outputPath = await retryDownloadViaContextMenu(page, imageUrl, outputDir, prompt, 5, 10000);
}

console.log(`Saved: ${outputPath}`);

if (keepOpen) {
  console.log("Keeping browser open. Press Ctrl+C when finished.");
  await new Promise(() => {});
}

if (shouldCloseContext) {
  await context.close();
} else if (browser) {
  await browser.close();
}

function normalizePromptForFlow(value) {
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
  const flowPage = pages.find((candidate) => candidate.url().startsWith("https://labs.google/") && !candidate.url().includes("/signin"));
  if (flowPage) return flowPage;
  const nonBlank = pages.find((candidate) => candidate.url() !== "about:blank");
  return nonBlank || pages[0] || await context.newPage();
}

async function allowDownloadsForAttachedChrome(context, page, downloadPath) {
  try {
    const session = await context.newCDPSession(page);
    await session.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath });
  } catch {}
}

async function ensureProjectPage(page) {
  await page.bringToFront().catch(() => {});
  await page.waitForLoadState("domcontentloaded").catch(() => {});
  if (await findPromptBox(page)) return;

  const newProjectClicked = await clickFirstVisible([
    page.getByRole("button", { name: /new project/i }),
    page.getByText(/new project/i),
    page.locator("button:has-text('New project')"),
    page.locator("[role='button']:has-text('New project')"),
  ]);

  if (!newProjectClicked) {
    throw new Error("Could not find the New project button on the Flow home page.");
  }

  console.log("Clicked New project.");
  await page.waitForTimeout(2500);
  await page.waitForLoadState("networkidle", { timeout: 60000 }).catch(() => {});
  const promptBox = await waitForPromptBox(page, 60000);
  if (!promptBox) {
    throw new Error("Clicked New project, but the project prompt box did not appear.");
  }
}

async function waitForPromptBox(page, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const found = await findPromptBox(page);
    if (found) return found;
    await page.waitForTimeout(1000);
  }
  return null;
}

async function findPromptBox(page) {
  const roots = [page, ...page.frames()];
  for (const root of roots) {
    const locators = [
      root.getByPlaceholder(/what do you want to create/i),
      root.getByRole("textbox", { name: /what do you want to create/i }),
      root.locator("textarea[placeholder*='create' i]"),
      root.locator("[contenteditable='true'][aria-label*='create' i]"),
      root.locator("[role='textbox'][aria-label*='create' i]"),
      root.locator("textarea"),
      root.locator("[contenteditable='true']"),
      root.locator("[contenteditable='plaintext-only']"),
      root.locator("[role='textbox']"),
      root.locator(".ProseMirror"),
      root.locator("input[type='text']"),
    ];
    const found = await firstWorkingLocator(locators);
    if (found) return found;
  }
  return null;
}

async function firstWorkingLocator(locators) {
  for (const locator of locators) {
    try {
      const count = await locator.count();
      for (let i = count - 1; i >= 0; i -= 1) {
        const candidate = locator.nth(i);
        if (await candidate.isVisible().catch(() => false)) return candidate;
      }
    } catch {}
  }
  return null;
}

async function coordinatePromptFallback(page, text) {
  await page.bringToFront().catch(() => {});
  const viewport = page.viewportSize() || { width: 1600, height: 950 };
  const x = Math.round(viewport.width * 0.43);
  const y = Math.round(viewport.height * 0.93);
  console.log(`Could not identify Flow's prompt box by selector; clicking prompt area at ${x}, ${y}.`);
  await page.mouse.click(x, y);
  await page.waitForTimeout(300);
  await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
  await page.keyboard.press("Backspace").catch(() => {});
  await enterPromptText(page, normalizePromptForFlow(text));
}

async function clearAndType(page, locator, text) {
  const normalized = normalizePromptForFlow(text);
  await locator.click();
  await locator.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
  const filled = await locator.fill(normalized, { timeout: 10000 }).then(() => true).catch(() => false);
  if (filled && await promptLooksEntered(locator, normalized)) return;

  await locator.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
  await locator.press("Backspace").catch(async () => {
    await locator.press(process.platform === "darwin" ? "Meta+A" : "Control+A");
    await locator.press("Backspace");
  });
  await enterPromptText(page, normalized);
  await page.waitForTimeout(500);
  if (await promptLooksEntered(locator, normalized)) return;

  await locator.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
  await locator.press("Backspace").catch(() => {});
  await pastePromptText(page, normalized);
}

async function enterPromptText(page, normalizedText) {
  console.log(`Entering prompt text (${normalizedText.length} chars) with fast insert.`);
  await page.keyboard.insertText(normalizedText);
}

async function pastePromptText(page, normalizedText) {
  console.log("Fast insert did not stick; trying clipboard paste.");
  const modifier = process.platform === "darwin" ? "Meta" : "Control";
  const copied = await page.evaluate(async (text) => {
    await navigator.clipboard.writeText(text);
    return true;
  }, normalizedText).catch(() => false);
  if (!copied) {
    throw new Error("Could not enter prompt text: fast insert failed and browser clipboard access was blocked.");
  }
  await page.keyboard.press(`${modifier}+V`);
}

async function promptLooksEntered(locator, normalizedText) {
  if (!normalizedText) return true;
  const wanted = normalizedText.slice(0, Math.min(40, normalizedText.length)).toLowerCase();
  const text = await locator.evaluate((el) => {
    return String(el.value || el.innerText || el.textContent || "").replace(/[\r\n\t]+/g, " ").replace(/\s{2,}/g, " ").trim();
  }).catch(() => "");
  return text.toLowerCase().includes(wanted);
}

function submitLocators(page) {
  return [
    page.getByRole("button", { name: /submit|send|create|generate/i }),
    page.locator("button[aria-label*='Submit' i]"),
    page.locator("button[aria-label*='Send' i]"),
    page.locator("button[aria-label*='Create' i]"),
    page.locator("button[aria-label*='Generate' i]"),
    page.locator("button:has(i.google-symbols:text-is('arrow_forward'))"),
    page.locator("[role='button'][aria-label*='Submit' i]"),
    page.locator("[role='button'][aria-label*='Send' i]"),
  ];
}

async function waitAndClickFirstVisible(locators, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    if (await clickFirstVisible(locators)) return true;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}
async function clickFirstVisible(locators) {
  for (const locator of locators) {
    const count = await locator.count().catch(() => 0);
    for (let i = count - 1; i >= 0; i -= 1) {
      const candidate = locator.nth(i);
      if (!(await candidate.isVisible().catch(() => false))) continue;
      const disabled = await candidate.getAttribute("aria-disabled").catch(() => null);
      if (disabled === "true") continue;
      await candidate.click();
      return true;
    }
  }
  return false;
}

async function getGeneratedImageUrls(page) {
  return page.evaluate(() => {
    const urls = [];
    for (const img of document.querySelectorAll("img")) {
      const src = img.currentSrc || img.src || img.getAttribute("src") || "";
      const alt = img.getAttribute("alt") || "";
      const rect = img.getBoundingClientRect();
      const visible = rect.width > 50 && rect.height > 50 && getComputedStyle(img).display !== "none" && getComputedStyle(img).visibility !== "hidden";
      const isGenerated = alt.toLowerCase().includes("generated image") || src.includes("media.getMediaUrlRedirect");
      if (!visible || !isGenerated || !src) continue;
      urls.push(new URL(src, window.location.href).href);
    }
    return urls;
  }).catch(() => []);
}

async function waitForNewGeneratedImageUrl(page, beforeUrls, maxMs) {
  const started = Date.now();
  let lastUrls = [];
  while (Date.now() - started < maxMs) {
    lastUrls = await getGeneratedImageUrls(page);
    const freshUrls = lastUrls.filter((candidate) => !beforeUrls.has(candidate));
    if (freshUrls.length > 0) {
      const newest = freshUrls[freshUrls.length - 1];
      console.log(`Found new generated image URL: ${newest}`);
      await waitForImageUrlToSettle(page, newest, 10000);
      return newest;
    }
    await page.waitForTimeout(2500);
  }

  if (lastUrls.length > 0) {
    console.log("No brand-new image URL was detected; using newest visible generated image as fallback.");
    return lastUrls[lastUrls.length - 1];
  }

  throw new Error("Timed out waiting for a generated image URL.");
}

async function waitForImageUrlToSettle(page, imageUrl, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const ready = await page.evaluate((targetUrl) => {
      for (const img of document.querySelectorAll("img")) {
        const src = new URL(img.currentSrc || img.src || img.getAttribute("src") || "", window.location.href).href;
        if (src === targetUrl && img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) return true;
      }
      return false;
    }, imageUrl).catch(() => false);
    if (ready) return;
    await page.waitForTimeout(500);
  }
}

async function retryDownloadViaContextMenu(page, imageUrl, outputDir, promptText, attempts, delayMs) {
  let lastError = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    console.log(`Trying Flow right-click Download > 2K menu, attempt ${attempt}/${attempts}...`);
    try {
      return await downloadViaContextMenu(page, imageUrl, outputDir, promptText);
    } catch (error) {
      lastError = error;
      console.log(`Download attempt ${attempt} failed: ${error.message}`);
      if (attempt < attempts) {
        console.log(`Waiting ${Math.round(delayMs / 1000)} seconds before retrying download...`);
        await page.keyboard.press("Escape").catch(() => {});
        await page.waitForTimeout(delayMs);
      }
    }
  }
  throw lastError || new Error("Download failed after retries.");
}
async function findVisibleLocator(locators, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    for (const locator of locators) {
      const count = await locator.count().catch(() => 0);
      for (let i = count - 1; i >= 0; i -= 1) {
        const candidate = locator.nth(i);
        if (await candidate.isVisible().catch(() => false)) return candidate;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return null;
}

async function find2kOption(page, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const handle = await page.evaluateHandle(() => {
      const visible = (el) => {
        const rect = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
      };
      const candidates = Array.from(document.querySelectorAll("[role='menuitem'], [role='option'], button, div"));
      for (const el of candidates) {
        if (!visible(el)) continue;
        const text = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
        if (/^2K(\s|$)/i.test(text) || text.toLowerCase().includes("2k upscaled")) {
          return el.closest("[role='menuitem'], button") || el;
        }
      }
      return null;
    });
    const element = handle.asElement();
    if (element) return element;
    await page.waitForTimeout(500);
  }
  return null;
}

async function dumpVisibleMenuText(page) {
  const menuText = await page.evaluate(() => Array.from(document.querySelectorAll("[role='menu'], [role='menuitem'], [role='option'], button, div"))
    .filter((el) => {
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
    })
    .map((el) => (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .slice(-80));
  console.log("Visible menu/debug text:");
  for (const line of menuText) console.log(`  ${line}`);
}
async function downloadViaContextMenu(page, imageUrl, downloadPath, promptText) {
  const image = await findImageElementForUrl(page, imageUrl);
  if (!image) {
    throw new Error("Could not find the generated image element for the Flow download menu.");
  }

  await page.bringToFront().catch(() => {});
  await image.scrollIntoViewIfNeeded().catch(() => {});
  await image.click({ button: "right" });
  await page.waitForTimeout(700);

  const downloadItem = await findVisibleLocator([
    page.getByRole("menuitem", { name: /^download$/i }),
    page.locator("[role='menuitem']:has-text('Download')"),
    page.locator("text=Download").last(),
  ], 10000);
  if (!downloadItem) {
    throw new Error("Could not find Download in the Flow context menu.");
  }

  console.log("Opening Download submenu...");
  await downloadItem.hover();
  await page.waitForTimeout(1500);

  const option2k = await find2kOption(page, 20000);
  if (!option2k) {
    await dumpVisibleMenuText(page);
    throw new Error("Could not find the 2K download option in the Flow context menu.");
  }

  console.log("Clicked Download > 2K. Waiting for upscale/download...");
  const downloadPromise = page.waitForEvent("download", { timeout: 180000 }).catch(() => null);
  await option2k.click();

  const download = await downloadPromise;
  if (!download) {
    throw new Error("Clicked Download > 2K, but Chrome did not report a downloaded file within 180 seconds.");
  }

  const suggested = download.suggestedFilename() || `${safeFilename(promptText)}-2k.png`;
  const outputPath = path.join(downloadPath, uniqueFilename(suggested));
  try {
    await download.saveAs(outputPath);
    return outputPath;
  } catch (error) {
    const existingPath = await findExistingDownloadedFile(downloadPath, suggested);
    if (existingPath) {
      console.log(`Chrome already saved the download: ${existingPath}`);
      return existingPath;
    }
    throw error;
  }
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

async function findImageElementForUrl(page, imageUrl) {
  const handles = await page.locator("img").elementHandles();
  for (let i = handles.length - 1; i >= 0; i -= 1) {
    const handle = handles[i];
    const matches = await handle.evaluate((img, targetUrl) => {
      const src = new URL(img.currentSrc || img.src || img.getAttribute("src") || "", window.location.href).href;
      const rect = img.getBoundingClientRect();
      return src === targetUrl && rect.width > 50 && rect.height > 50;
    }, imageUrl).catch(() => false);
    if (matches) return handle;
  }
  return null;
}
async function saveGeneratedImageUrl(page, imageUrl, downloadPath, promptText) {
  const result = await page.evaluate(async (url) => {
    const response = await fetch(url, { credentials: "include" });
    if (!response.ok) {
      return { error: `${response.status} ${response.statusText}`, url };
    }
    const buffer = await response.arrayBuffer();
    return {
      url,
      contentType: response.headers.get("content-type") || "image/png",
      bytes: Array.from(new Uint8Array(buffer)),
    };
  }, imageUrl).catch((error) => ({ error: error.message, url: imageUrl }));

  if (!result || result.error || !result.bytes?.length) {
    throw new Error(`Direct image save failed: ${result?.error || "no data"}`);
  }

  const ext = extensionForContentType(result.contentType);
  const outputPath = path.join(downloadPath, uniqueFilename(`${safeFilename(promptText)}${ext}`));
  await fs.writeFile(outputPath, Buffer.from(result.bytes));
  return outputPath;
}

function extensionForContentType(contentType) {
  const normalized = contentType.toLowerCase();
  if (normalized.includes("jpeg") || normalized.includes("jpg")) return ".jpg";
  if (normalized.includes("webp")) return ".webp";
  if (normalized.includes("png")) return ".png";
  return ".png";
}

async function uploadImageAndAddToPrompt(page, filePath) {
  await fs.access(filePath).catch(() => {
    throw new Error(`Image file does not exist: ${filePath}`);
  });

  const beforeUrls = new Set(await getPromptAttachmentUrls(page));

  const opened = await clickFirstVisible([
    page.getByRole("button", { name: /^create$/i }),
    page.locator("button[aria-haspopup='dialog']:has(i.google-symbols:text-is('add_2'))"),
    page.locator("button:has(i.google-symbols:text-is('add_2'))"),
    page.locator("button:has(i.google-symbols:text-is('add'))"),
  ]);
  if (!opened) {
    throw new Error("Could not find the bottom + / Create button to open media upload.");
  }

  await page.waitForTimeout(800);
  const uploadTarget = await findVisibleLocator([
    page.getByText(/upload media/i),
    page.getByRole("button", { name: /upload media/i }),
    page.locator("[role='button']:has-text('Upload media')"),
    page.locator("button:has-text('Upload media')"),
    page.locator("text=Upload media"),
  ], 15000);
  if (!uploadTarget) {
    await dumpVisibleMenuText(page);
    throw new Error("Could not find Upload media in the media picker.");
  }

  const chooserPromise = page.waitForEvent("filechooser", { timeout: 15000 });
  await uploadTarget.click();
  const chooser = await chooserPromise;
  await chooser.setFiles(filePath);
  console.log("Upload selected; waiting 10 seconds for Flow to process it...");
  await page.waitForTimeout(10000);

  const addToPrompt = await findAddToPromptAfterUpload(page, beforeUrls, 90000);
  await addToPrompt.click();
  console.log("Uploaded image added to prompt; waiting 3 seconds for Flow to settle.");
  await page.waitForTimeout(3000);
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
    const hasNewUpload = urls.some((candidateUrl) => !beforeUrls.has(candidateUrl));
    if (hasNewUpload) {
      const candidateAfterUpload = await findVisibleLocator([
        page.getByRole("button", { name: /add to prompt/i }),
        page.locator("button:has-text('Add to Prompt')"),
        page.locator("button:has-text('Add to prompt')"),
      ], 3000);
      if (candidateAfterUpload) return candidateAfterUpload;
    }

    await page.waitForTimeout(1000);
  }
  await dumpVisibleMenuText(page);
  throw new Error("Uploaded image did not become ready for Add to Prompt within 90 seconds.");
}

async function getPromptAttachmentUrls(page) {
  return page.evaluate(() => Array.from(document.querySelectorAll("img"))
    .map((img) => img.currentSrc || img.src || img.getAttribute("src") || "")
    .filter(Boolean)
    .map((src) => new URL(src, window.location.href).href)
  ).catch(() => []);
}
function safeFilename(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80) || "flow-output";
}

function uniqueFilename(name) {
  const ext = path.extname(name);
  const base = path.basename(name, ext);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${base}-${stamp}${ext || ".png"}`;
}








