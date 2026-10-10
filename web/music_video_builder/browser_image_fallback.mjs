export function browserImageProviderFallbackOrder(provider, failureMode) {
  const orders = {
    gpt_image: ["gpt_image", "flow_nano_banana", "meta_ai"],
    flow_nano_banana: ["flow_nano_banana", "gpt_image", "meta_ai"],
    meta_ai: ["meta_ai", "gpt_image", "flow_nano_banana"],
  };
  const order = orders[provider] || orders.flow_nano_banana;
  return failureMode === "try_other_provider" ? [...order] : [order[0]];
}

export async function runBrowserImageWithFallback(settings, generate, {
  shouldCancel = () => false,
  onFallback = () => {},
  providerLabel = (provider) => provider,
} = {}) {
  const order = browserImageProviderFallbackOrder(settings.provider, settings.failure_mode);
  const failures = [];
  for (let index = 0; index < order.length; index += 1) {
    if (shouldCancel()) throw new Error("Stopped by user.");
    const provider = order[index];
    try {
      const images = await generate({ ...settings, provider });
      if (!Array.isArray(images) || !images.length) throw new Error("The provider returned no image output.");
      if (shouldCancel()) throw new Error("Stopped by user.");
      return { images, provider };
    } catch (error) {
      if (shouldCancel() || error?.name === "AbortError" || /stopped by user|execution_interrupted|execution interrupted/i.test(String(error?.message || error))) {
        throw error;
      }
      if (order.length === 1) throw error;
      failures.push(`${providerLabel(provider)}: ${String(error?.message || error)}`);
      const nextProvider = order[index + 1];
      if (nextProvider) onFallback(provider, nextProvider, error);
    }
  }
  throw new Error(`Browser AI failed for every provider tried:\n${failures.join("\n\n")}`);
}
