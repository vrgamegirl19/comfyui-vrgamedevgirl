const assert = require("assert");
const test = require("node:test");
const { readBuilderModule } = require("./builder_source.cjs");

// batch_prompts.mjs is a factory: every dependency it uses has to be in its parameter list and passed by builder.mjs.
test("createBatchPrompts receives miniMaxH3CutPlanForSegment, which the FX guidance uses", () => {
  const source = readBuilderModule("batch_prompts.mjs");
  const parameters = source.slice(source.indexOf("function createBatchPrompts({"), source.indexOf("}) {"));
  assert.ok(source.includes("miniMaxH3CutPlanForSegment(segment)"));
  assert.ok(/\bminiMaxH3CutPlanForSegment\b/.test(parameters), "missing from the parameter list");
  const builder = readBuilderModule("builder.mjs");
  const call = builder.slice(builder.indexOf("} = createBatchPrompts({"));
  assert.ok(/\bminiMaxH3CutPlanForSegment\b/.test(call.slice(0, call.indexOf("});"))), "builder.mjs does not pass it");
});
