export async function closeBrowserImageViewer(page) {
  if (page.isClosed()) return;
  // Match the image toolbar's close action, not unrelated labels containing Close.
  const controls = [
    page.getByRole("button", { name: /^close(?: (?:image(?: viewer)?|viewer|preview|dialog))?$/i }),
    page.locator("button[title='Close' i], button[title='Close image' i], button[title='Close image viewer' i]"),
  ];
  for (const locator of controls) {
    const count = await locator.count();
    for (let index = count - 1; index >= 0; index -= 1) {
      const button = locator.nth(index);
      if (!(await button.isVisible())) continue;
      const closed = await button.click({ timeout: 3000 }).then(async () => {
        await button.waitFor({ state: "hidden", timeout: 3000 });
        return true;
      }).catch(() => false);
      if (closed) return;
    }
  }
  await page.keyboard.press("Escape").catch(() => {});
}

export async function waitForBrowserChat(page, findComposer, providerLabel, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (page.isClosed()) break;
    const composer = await findComposer(page);
    // A composer behind a fullscreen image can be visible yet still blocked.
    // A trial click verifies it can receive the next request without entering text.
    if (composer && await composer.click({ trial: true, timeout: 500 }).then(() => true).catch(() => false)) {
      console.log(`${providerLabel} chat is ready for the next image request.`);
      return;
    }
    await page.waitForTimeout(250);
  }
  throw new Error(`${providerLabel} saved the image, but its image viewer did not return to a usable chat prompt.`);
}
