const fs = require("fs");
const path = require("path");
const vm = require("vm");

const WEB = path.join(__dirname, "../web");
const MODULES = path.join(WEB, "music_video_builder");

// Tests evaluate builder code as plain scripts, so drop module import/export syntax.
function asScript(source) {
  return source.replace(/^import [^;]*;\r?\n/gm, "").replace(/^export /gm, "");
}

function readBuilderSource() {
  const files = fs.readdirSync(MODULES).filter((name) => name.endsWith(".mjs")).sort().map((name) => path.join(MODULES, name));
  files.push(path.join(WEB, "VRGDG_MusicVideoBuilderUI.js"));
  return asScript(files.map((file) => fs.readFileSync(file, "utf8")).join("\n"));
}

function readStoryboardSource() {
  const dir = path.join(WEB, "storyboard_builder");
  const files = fs.readdirSync(dir).filter((name) => name.endsWith(".mjs")).sort().map((name) => path.join(dir, name));
  files.push(path.join(WEB, "VRGDG_StoryboardBuilderUI.js"));
  return asScript(files.map((file) => fs.readFileSync(file, "utf8")).join("\n"));
}

function readBuilderModule(name) {
  return asScript(fs.readFileSync(path.join(MODULES, name), "utf8"));
}

// Full text of a function declaration, wherever its module placed it.
function functionSource(source, name) {
  let start = source.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`function ${name} not found`);
  if (source.slice(start - 6, start) === "async ") start -= 6;
  for (let end = source.indexOf("}", start); end !== -1; end = source.indexOf("}", end + 1)) {
    const text = source.slice(start, end + 1);
    try {
      new vm.Script(`(${text})`);
      return text;
    } catch {}
  }
  throw new Error(`function ${name} has no end`);
}

module.exports = { functionSource, readBuilderModule, readBuilderSource, readStoryboardSource };
