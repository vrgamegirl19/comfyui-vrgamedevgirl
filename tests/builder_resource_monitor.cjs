const assert = require("node:assert/strict");
const { test } = require("node:test");
const vm = require("node:vm");
const { readBuilderModule } = require("./builder_source.cjs");

const source = readBuilderModule("toolbar.mjs");
const context = vm.createContext({});
vm.runInContext(source.slice(0, source.indexOf("function wireToolbar")), context);

// Copied out of the vm context: arrays made there fail deepStrictEqual against arrays made here.
function meters(gpus) {
  return JSON.parse(vm.runInContext(`JSON.stringify(builderResourceMeters(${JSON.stringify(gpus)}))`, context));
}

test("one reading per GPU, labeled VRAM0 and VRAM1, with RAM last", () => {
  const plan = meters([{ index: "1", name: "Second", used: 1, total: 2 }, { index: "0", name: "First" }]);
  assert.deepEqual(plan.meters.map((meter) => meter.label), ["VRAM0", "VRAM1", "RAM"]);
  assert.deepEqual(plan.meters.map((meter) => meter.key), ["vram-0", "vram-1", "ram"]);
  assert.equal(plan.meters[0].textColor, "#67e8f9");
  assert.equal(plan.meters[0].barColor, "#06b6d4");
  assert.equal(plan.meters[2].textColor, "#c4b5fd");
  assert.equal(plan.meters[2].barColor, "#8b5cf6");
  assert.equal(plan.width, 375);
  assert.equal(plan.minFit, 315);
});

test("a single GPU stays VRAM, and a missing sensor keeps that same placeholder", () => {
  const single = meters([{ index: 0, name: "Only" }]);
  assert.deepEqual(single.meters.map((meter) => meter.label), ["VRAM", "RAM"]);
  assert.deepEqual(single.meters.map((meter) => meter.key), ["vram", "ram"]);
  assert.deepEqual(meters([]).meters.map((meter) => meter.label), ["VRAM", "RAM"]);
  assert.equal(meters([]).width, 250);
  assert.equal(meters([]).minFit, 210);
});

test("VRAM markup keeps the cyan meter and puts the GPU name on the column", () => {
  const plan = meters([{ index: "0", name: 'GPU "A" <0>' }]);
  const markup = vm.runInContext(`builderResourceMeterMarkup(${JSON.stringify(plan.meters[0])})`, context);
  assert.match(markup, /<span>VRAM<\/span>/);
  assert.match(markup, /color:#67e8f9/);
  assert.match(markup, /background:#06b6d4/);
  assert.match(markup, /title="GPU &quot;A&quot; &lt;0&gt;"/);
  const ram = vm.runInContext(`builderResourceMeterMarkup(${JSON.stringify(plan.meters[1])})`, context);
  assert.match(ram, /<span>RAM<\/span>/);
  assert.match(ram, /color:#c4b5fd/);
  assert.match(ram, /background:#8b5cf6/);
});
