import assert from "node:assert/strict";
import { test } from "node:test";

import { hasMappedStoryboardLocation, runStoryGenerationSequence } from "../web/storyboard_builder/story_workflow.mjs";

test("Story Layer requires a scene assignment, including for text-only locations", () => {
  const location = { id: "hall", name: "Hall", description: "A narrow hallway", image: {} };
  const state = { referenceBuilder: { locations: [location] }, scenes: [{ id: "one", location_ref: null }] };
  assert.equal(hasMappedStoryboardLocation(state), false);
  state.scenes[0].location_ref = location;
  assert.equal(hasMappedStoryboardLocation(state), true);
  state.scenes[0].location_ref = { id: "deleted", name: "Old location" };
  assert.equal(hasMappedStoryboardLocation(state), false);
});

test("combined story generation saves each stage before the next", async () => {
  const order = [];
  const result = await runStoryGenerationSequence({
    createArc: async () => { order.push("arc"); return "arc text"; },
    createBrief: async () => { order.push("brief"); return "brief text"; },
    createBeats: async () => { order.push("beats"); return { created: 2, failures: [] }; },
    save: async () => { order.push("save"); },
  });
  assert.deepEqual(order, ["arc", "save", "brief", "save", "beats"]);
  assert.deepEqual(result, { completed: true, created: 2 });
});

test("combined story generation stops after an empty brief or failed beats", async () => {
  const order = [];
  const steps = {
    createArc: async () => { order.push("arc"); return "arc"; },
    createBrief: async () => { order.push("brief"); return ""; },
    createBeats: async () => { order.push("beats"); return { created: 1, failures: [] }; },
    save: async () => { order.push("save"); },
  };
  assert.deepEqual(await runStoryGenerationSequence(steps), { completed: false, stoppedAt: "brief" });
  assert.deepEqual(order, ["arc", "save", "brief"]);
  steps.createBrief = async () => "brief";
  steps.createBeats = async () => ({ created: 1, failures: [{ scene: "two" }] });
  assert.deepEqual(await runStoryGenerationSequence(steps), { completed: false, stoppedAt: "beats" });
});
