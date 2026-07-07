import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs/promises";

const args = parseArgs(process.argv.slice(2));
const projectDir = path.dirname(new URL(import.meta.url).pathname).replace(/^\/(.:\/)/, "$1");
const url = args.url || "https://chatgpt.com/images";
const prompt = normalizePrompt(args.prompt || "create an image");
const outputDir = path.resolve(projectDir, args.out || "chatgpt_outputs");
const timeout = Number(args.timeout || 360000);
const cdpUrl = args.cdp || args["connect-cdp"] || "";
const noNavigate = Boolean(args["no-navigate"]);
const imagePaths = normalizeList(args.image).map((imagePath) => path.resolve(imagePath));
const profileDir = path.resolve(projectDir, "chatgpt-browser-profile");

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
await ensureComposerReady(page);

if (imagePaths.length > 0) {
  console.log(`Attaching ${imagePaths.length} image(s).`);
  for (let index = 0; index < imagePaths.length; index += 1) {
    const imagePath = imagePaths[index];
    console.log(`Attaching image ${index + 1}/${imagePaths.length}: ${imagePath}`);
    await attachImage(page, imagePath);
  }
}

const beforeKeys = new Set(await getCandidateImageKeys(page));
console.log(`Images before submit: ${beforeKeys.size}`);

console.log("Entering prompt...");
const composer = await findComposer(page);
if (!composer) throw new Error("Could not find the ChatGPT Images prompt box.");
await enterPrompt(page, composer, prompt);

console.log("Submitting prompt...");
const submitted = await clickFirstVisible(sendLocators(page));
if (!submitted) await page.keyboard.press("Enter");

console.log("Waiting for generated image...");
const image = await waitForNewImage(page, beforeKeys, timeout);
console.log("Waiting for ChatGPT completion marker...");
await waitForThoughtComplete(page, Math.min(timeout, 180000)).catch((error) => {
  console.log(`Completion marker was not found before timeout: ${error.message}`);
});

console.log("Trying viewer download button...");
const downloaded = await retryOpenViewerAndDownload(page, image, outputDir, prompt, 5, 10000);
if (downloaded) {
  console.log(`Saved: ${downloaded}`);
} else {
  console.log("Viewer download did not produce a file; trying direct image save.");
  const imageUrl = await newestVisibleImageUrl(page);
  const outputPath = await saveImageUrl(page, imageUrl, outputDir, prompt);
  console.log(`Saved: ${outputPath}`);
}

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
  const chatgptPage = pages.find((candidate) => candidate.url().startsWith("https://chatgpt.com/"));
  if (chatgptPage) return chatgptPage;
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
    throw new Error("Could not find ChatGPT Images prompt box. Sign into ChatGPT in the automation Chrome window, then run again.");
  }
}

async function waitForComposer(page, maxMs) {
  const started = Date.now();
  while (Date.now() - started < maxMs) {
    const composer = await findComposer(page);
    if (composer) return composer;
    await page.waitForTimeout(1000);
  }
  return null;
}

async function findComposer(page) {
  const locators = [
    page.getByPlaceholder(/describe a new image/i),
    page.getByPlaceholder(/ask anything/i),
    page.locator("textarea[placeholder*='Describe' i]"),
    page.locator("textarea[placeholder*='Ask' i]"),
    page.locator("[contenteditable='true'][data-placeholder*='Describe' i]"),
    page.locator("[contenteditable='true'][aria-label*='message' i]"),
    page.locator("[contenteditable='true']"),
    page.locator("textarea"),
  ];
  return await firstVisibleLocator(locators);
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

  const directInput = await firstExistingLocator([
    page.locator("input[type='file'][accept*='image' i]"),
    page.locator("input[type='file']"),
  ]);
  if (directInput) {
    await directInput.setInputFiles(filePath);
    console.log("Image set on file input; waiting for attachment preview.");
    await page.waitForTimeout(5000);
    return;
  }

  const chooserPromise = page.waitForEvent("filechooser", { timeout: 20000 });
  const opened = await clickFirstVisible([
    page.getByRole("button", { name: /^add photos$/i }),
    page.locator("button[aria-label*='Add photos' i]"),
    page.getByRole("button", { name: /attach|upload|add photo|add files/i }),
    page.locator("button[aria-label*='Attach' i]"),
    page.locator("button[aria-label*='Upload' i]"),
    page.locator("button[aria-label*='paperclip' i]"),
    page.locator("[data-testid*='attach' i]"),
    page.locator("[data-testid*='upload' i]"),
  ]);
  if (!opened) throw new Error("Could not find the ChatGPT paperclip/attach button.");

  const chooser = await chooserPromise;
  await chooser.setFiles(filePath);
  console.log("Image selected; waiting for attachment preview.");
  await page.waitForTimeout(5000);
}

async function firstExistingLocator(locators) {
  for (const locator of locators) {
    const count = await locator.count().catch(() => 0);
    if (count > 0) return locator.first();
  }
  return null;
}

async function enterPrompt(page, composer, text) {
  const normalized = normalizePrompt(text);
  await composer.click();
  await composer.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
  const filled = await composer.fill(normalized, { timeout: 10000 }).then(() => true).catch(() => false);
  if (filled && await promptLooksEntered(composer, normalized)) return;
  await composer.press(process.platform === "darwin" ? "Meta+A" : "Control+A").catch(() => {});
  await composer.press("Backspace").catch(() => {});
  await page.keyboard.insertText(normalized);
  await page.waitForTimeout(500);
  if (await promptLooksEntered(composer, normalized)) return;
  throw new Error("Could not enter prompt text in ChatGPT Images.");
}

async function promptLooksEntered(locator, normalizedText) {
  if (!normalizedText) return true;
  const wanted = normalizedText.slice(0, Math.min(40, normalizedText.length)).toLowerCase();
  const text = await locator.evaluate((el) => {
    return String(el.value || el.innerText || el.textContent || "").replace(/[\r\n\t]+/g, " ").replace(/\s{2,}/g, " ").trim();
  }).catch(() => "");
  return text.toLowerCase().includes(wanted);
}

function sendLocators(page) {
  return [
    page.getByRole("button", { name: /^send prompt$/i }),
    page.locator("button[aria-label*='Send prompt' i]"),
    page.getByRole("button", { name: /send prompt|send message|send/i }),
    page.locator("button[aria-label*='Send' i]"),
    page.locator("button[data-testid*='send' i]"),
  ];
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

async function waitForNewImage(page, beforeKeys, maxMs) {
  const started = Date.now();
  let lastInfo = null;
  while (Date.now() - started < maxMs) {
    const handles = await page.locator("img").elementHandles();
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
      lastInfo = info;
      if (!beforeKeys.has(info.key)) {
        console.log("Found new visible image.");
        return info;
      }
    }
    await page.waitForTimeout(3000);
  }
  if (lastInfo) {
    console.log("No brand-new image key was detected; using newest visible image fallback.");
    return lastInfo;
  }
  throw new Error("Timed out waiting for a generated ChatGPT image.");
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

    const share = controls
      .filter((item) => /^share$/i.test(item.name.trim()) || /\bshare\b/i.test(item.name))
      .sort((a, b) => b.rect.width * b.rect.height - a.rect.width * a.rect.height)[0];
    if (!share) return false;

    const topBarRightOfShare = controls
      .filter((item) => {
        const centerX = item.rect.left + item.rect.width / 2;
        const centerY = item.rect.top + item.rect.height / 2;
        const name = item.name.trim();
        return centerX > share.rect.right
          && centerY < 140
          && !/more|menu|overflow|ellipsis|^\.\.\.$/i.test(name);
      })
      .sort((a, b) => a.rect.left - b.rect.left);
    const candidate = topBarRightOfShare[0];
    if (!candidate) return false;
    candidate.el.click();
    return true;
  }).catch(() => false);
}

async function retryDownloadFromViewer(page, outputDir, promptText, attempts, delayMs) {
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    console.log(`Trying ChatGPT download button, attempt ${attempt}/${attempts}...`);
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
    await page.keyboard.press("Escape").catch(() => {});
    await page.waitForTimeout(500);
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

    console.log("Download button did not work; closing viewer and retrying.");
    await closeImageViewer(page);
    if (attempt < attempts) {
      console.log(`Waiting ${Math.round(delayMs / 1000)} seconds before reopening image...`);
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
    throw new Error(`Direct image save failed: ${result?.error || "no data"}`);
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
  return String(value || "chatgpt-image").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80) || "chatgpt-image";
}

function uniqueFilename(name) {
  const ext = path.extname(name);
  const base = path.basename(name, ext);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${base}-${stamp}${ext || ".png"}`;
}
