import assert from "node:assert/strict";
import { test } from "node:test";
import vm from "node:vm";
import { createRequire } from "node:module";
import { browserImageProviderFallbackOrder, runBrowserImageWithFallback } from "../web/music_video_builder/browser_image_fallback.mjs";

const orders = {
  gpt_image: ["gpt_image", "flow_nano_banana", "meta_ai"],
  flow_nano_banana: ["flow_nano_banana", "gpt_image", "meta_ai"],
  meta_ai: ["meta_ai", "gpt_image", "flow_nano_banana"],
};

for (const [provider, order] of Object.entries(orders)) {
  test(`${provider}: uses the requested fallback order and stops after success`, async () => {
    assert.deepEqual(browserImageProviderFallbackOrder(provider, "try_other_provider"), order);
    for (let successIndex = 0; successIndex < order.length; successIndex += 1) {
      const attempts = [];
      const transitions = [];
      const references = [{ path: "character.png" }, { path: "location.png" }];
      const settings = { provider, failure_mode: "try_other_provider", image_ingredients: references, max_retries: 10 };
      const image = { filename: "result.png" };
      const result = await runBrowserImageWithFallback(settings, async (attempt) => {
        attempts.push(attempt.provider);
        assert.equal(attempt.image_ingredients, references);
        if (attempt.provider !== order[successIndex]) throw new Error("Provider unavailable");
        return [image];
      }, { onFallback: (failed, next) => transitions.push([failed, next]) });
      assert.deepEqual(attempts, order.slice(0, successIndex + 1));
      assert.equal(result.provider, order[successIndex]);
      assert.deepEqual(result.images, [image]);
      assert.equal(transitions.length, successIndex);
      assert.equal(settings.provider, provider);
    }
  });
}

test("reports all provider failures and treats empty output as failure", async () => {
  const attempted = [];
  await assert.rejects(runBrowserImageWithFallback({ provider: "gpt_image", failure_mode: "try_other_provider" }, async ({ provider }) => {
    attempted.push(provider);
    if (provider === "meta_ai") return [];
    throw new Error(`${provider} unavailable`);
  }), (error) => /gpt_image unavailable/.test(error.message) && /flow_nano_banana unavailable/.test(error.message) && /meta_ai: The provider returned no image output/.test(error.message));
  assert.deepEqual(attempted, orders.gpt_image);
});

test("other failure modes do not switch providers", async () => {
  for (const failure_mode of ["stop", "last_successful_image", undefined]) {
    const attempts = [];
    const failure = new Error("Original provider failure");
    await assert.rejects(runBrowserImageWithFallback({ provider: "meta_ai", failure_mode }, async ({ provider }) => {
      attempts.push(provider);
      throw failure;
    }), (error) => error === failure);
    assert.deepEqual(attempts, ["meta_ai"]);
  }
});

test("cancellation never starts an alternate provider", async () => {
  for (const failure of [new Error("Stopped by user."), new Error("execution_interrupted"), Object.assign(new Error("Aborted"), { name: "AbortError" })]) {
    let attempts = 0;
    await assert.rejects(runBrowserImageWithFallback({ provider: "gpt_image", failure_mode: "try_other_provider" }, async () => {
      attempts += 1;
      throw failure;
    }), (error) => error === failure);
    assert.equal(attempts, 1);
  }
  let cancelled = false;
  let attempts = 0;
  await assert.rejects(runBrowserImageWithFallback({ provider: "gpt_image", failure_mode: "try_other_provider" }, async () => {
    attempts += 1;
    cancelled = true;
    throw new Error("Workflow failed");
  }, { shouldCancel: () => cancelled }), /Workflow failed/);
  assert.equal(attempts, 1);
});

const require = createRequire(import.meta.url);
const { functionSource, readBuilderModule } = require("./builder_source.cjs");

function generationFixture(failures) {
  const references = [{ path: "character.png" }, { path: "location.png" }];
  const settings = { provider: "gpt_image", failure_mode: "try_other_provider", aspect_ratio: "16:9", image_ingredients: references,
    reference_context: { has_subject_reference: true, has_location_reference: true },
    prompt: "Using the provided character reference image and location reference image, create a scene.",
    gpt_timeout_seconds: 600, flow_timeout_seconds: 420, meta_timeout_seconds: 700 };
  const segment = { id: "scene", image: { filename: "original.png" } };
  const payloads = [];
  const archives = [];
  let historyWrites = 0;
  const sandbox = {
    state: { batchCancelled: false }, runBrowserImageWithFallback,
    syncInspector() {}, render() {},
    flowGptBrowserSettingsForSegment: () => settings,
    maybeAttachPreviousSceneImage: async (value) => value,
    syncSegmentFlowGptPrompt: (_, prompt) => prompt,
    sceneDisplayName: () => "Scene 1", segmentIndexInfo: () => ({ index: 0 }),
    activeSegment: () => segment, syncPreview() {},
    buildBrowserImagePrompt: async (payload) => {
      payloads.push(payload);
      if (failures[payload.provider] === "build") throw new Error("Build failed");
      return { prompt: { provider: payload.provider }, image_count: 2 };
    },
    queueWorkflowPrompt: async ({ provider }) => {
      if (failures[provider] === "queue") throw new Error("Queue failed");
      return { prompt_id: provider };
    },
    waitForImages: async (provider, onStatus, shouldCancel) => {
      assert.equal(shouldCancel(), false);
      if (failures[provider] === "wait") throw new Error("Generation failed");
      return [{ filename: `${provider}.png` }];
    },
    pushHistory: () => { historyWrites += 1; },
    archiveGeneratedSceneImage: async (_, image) => archives.push(image),
    BROWSER_IMAGE_PROVIDERS: { GPT_IMAGE: "gpt_image", META_AI: "meta_ai", FLOW_NANO_BANANA: "flow_nano_banana" },
  };
  vm.createContext(sandbox);
  const models = readBuilderModule("model_settings.mjs");
  const generation = readBuilderModule("image_generation.mjs");
  vm.runInContext(["normalizeFlowGptBrowserProvider", "browserImageProviderLabel", "browserImageProviderTimeout"].map((name) => functionSource(models, name)).join("\n"), sandbox);
  vm.runInContext(["browserImageReferencePrompt", "createFlowGptImageForSegment"].map((name) => functionSource(generation, name)).join("\n"), sandbox);
  return { sandbox, segment, settings, references, payloads, archives, historyWrites: () => historyWrites };
}

test("scene generation falls back after build, queue, or generation failures using each provider's timeout", async () => {
  for (const stage of ["build", "queue", "wait"]) {
    const fixture = generationFixture({ gpt_image: stage, flow_nano_banana: "wait" });
    const images = await fixture.sandbox.createFlowGptImageForSegment(fixture.segment);
    assert.deepEqual(fixture.payloads.map((payload) => payload.provider), orders.gpt_image);
    assert.deepEqual(fixture.payloads.map((payload) => payload.timeout_seconds), [601, 420, 701]);
    for (const payload of fixture.payloads) {
      assert.equal(payload.image_ingredients, fixture.references);
      assert.equal(payload.prompt, fixture.settings.prompt);
      assert.equal(payload.aspect_ratio, "16:9");
    }
    assert.equal(images[0].filename, "meta_ai.png");
    assert.equal(fixture.segment.image.filename, "meta_ai.png");
    assert.equal(fixture.archives.length, 1);
    assert.equal(fixture.historyWrites(), 1);
    assert.equal(fixture.settings.provider, "gpt_image");
  }
});

test("all-provider failure leaves the scene image and history unchanged", async () => {
  const fixture = generationFixture({ gpt_image: "wait", flow_nano_banana: "wait", meta_ai: "wait" });
  await assert.rejects(fixture.sandbox.createFlowGptImageForSegment(fixture.segment), /Browser AI failed for every provider tried/);
  assert.equal(fixture.segment.image.filename, "original.png");
  assert.equal(fixture.archives.length, 0);
  assert.equal(fixture.historyWrites(), 0);
});
