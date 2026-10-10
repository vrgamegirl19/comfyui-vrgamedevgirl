import path from "node:path";

export async function addUploadedFlowImageToPrompt(page, filePath, timeoutMs = 90000) {
  const filename = path.basename(filePath);
  const asset = page.getByRole("option").filter({ has: page.getByText(filename, { exact: true }) }).last();
  await asset.waitFor({ state: "visible", timeout: timeoutMs }).catch(() => {
    throw new Error(`Flow uploaded ${filename}, but its image asset did not appear in the ingredients picker.`);
  });
  const deadline = Date.now() + timeoutMs;
  let imageSrc = "";
  while (Date.now() < deadline) {
    imageSrc = await asset.locator("img").first().evaluate((img) =>
      img.complete && img.naturalWidth > 0 ? img.currentSrc || img.src : ""
    ).catch(() => "");
    if (imageSrc) break;
    await page.waitForTimeout(250);
  }
  if (!imageSrc) throw new Error(`Flow has not finished loading uploaded reference ${filename}.`);
  await asset.click();

  // Flow initially previews the account avatar. Never add that default selection.
  const add = page.getByRole("button", { name: /^add to prompt$/i }).last();
  while (Date.now() < deadline) {
    const selection = await asset.evaluate((option) => {
      const img = option.querySelector("img");
      return {
        active: option.classList.contains("asset-item-active"),
        src: img?.complete && img.naturalWidth > 0 ? img.currentSrc || img.src : "",
      };
    });
    if (selection.active && selection.src && await add.isVisible() && await add.isEnabled()) {
      const correctPreview = await add.evaluate((button, expectedSrc) => {
        const key = (src) => {
          const url = new URL(src, location.href);
          return url.origin + url.pathname.replace(/=s\d+(?:-[a-z]+)*$/i, "");
        };
        for (let parent = button.parentElement; parent && parent !== document.body; parent = parent.parentElement) {
          const previews = [...parent.querySelectorAll("img")].filter((img) => !img.closest('[role="option"]'));
          if (!previews.length) continue;
          return previews.some((img) => img.complete && img.naturalWidth > 0 && key(img.currentSrc || img.src) === key(expectedSrc));
        }
        return false;
      }, selection.src);
      if (correctPreview) {
        await add.click();
        console.log(`Added uploaded reference to prompt: ${filename}`);
        return;
      }
    }
    await page.waitForTimeout(250);
  }
  throw new Error(`Flow did not preview uploaded reference ${filename}; refusing to add a different image or avatar.`);
}
