// Flow release announcements can cover an otherwise visible project composer.
export async function dismissFlowAnnouncement(page, waitMs = 0) {
  const deadline = Date.now() + waitMs;
  do {
    const buttons = page.getByRole("button", { name: /^get started$/i });
    const count = await buttons.count();
    for (let index = 0; index < count; index += 1) {
      const button = buttons.nth(index);
      if (!(await button.isVisible())) continue;
      const isAnnouncement = await button.evaluate((element) => {
        // Do not depend on the model name/version or generated CSS classes.
        // Stop at the page body so an unrelated Get started button is ignored.
        for (let parent = element.parentElement; parent && parent !== document.body; parent = parent.parentElement) {
          if (/view all changelogs/i.test(parent.innerText || "")) return true;
        }
        return false;
      });
      if (!isAnnouncement) continue;
      await button.click({ timeout: 5000 });
      await button.waitFor({ state: "hidden", timeout: 5000 });
      console.log("Dismissed Flow update announcement.");
      return true;
    }
    if (Date.now() >= deadline) return false;
    await page.waitForTimeout(250);
  } while (true);
}
