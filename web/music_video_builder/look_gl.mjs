// The WebGL 2 side of the effect and filter preview. The sources are the looks' GLSL ES 3.00 text (the helpers of the app's prelude and the look); the player and the
// picker run them here and the final video runs the same files through the app's own renderer. A look reads one picture with `getColor` and the page mixes its result
// into the picture by `intensity`, as the export does. A chain of looks is drawn pass by pass, each reading the one before.

import { TRANSITION_VERTEX_SHADER, compileTransitionShader, setTransitionUniform } from "./transition_gl.mjs";

const LOOK_PASS_THROUGH = "vec4 effect(vec2 uv) { return getColor(uv); }";

const wrapLookShader = (source) => `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D _from;
uniform float progress;
uniform float ratio;
uniform float intensity;
uniform float time;
in vec2 _uv;
out vec4 _color;
vec4 getColor(vec2 uv) { return texture(_from, uv); }
${source}
void main() { vec4 src = getColor(_uv); vec4 fx = effect(_uv); _color = vec4(mix(src.rgb, fx.rgb, clamp(intensity, 0.0, 1.0)), 1.0); }`;

// An effect and a filter may have the same id, so the kind is part of the key.
const lookProgramKey = (item) => (item ? `${item.kind || ""}:${item.id}` : "");

function makeLookProgram(gl, fragment) {
  const program = gl.createProgram();
  const shaders = [compileTransitionShader(gl, gl.VERTEX_SHADER, TRANSITION_VERTEX_SHADER), compileTransitionShader(gl, gl.FRAGMENT_SHADER, wrapLookShader(fragment))];
  shaders.forEach((shader) => gl.attachShader(program, shader));
  gl.bindAttribLocation(program, 0, "position");
  gl.linkProgram(program);
  shaders.forEach((shader) => gl.deleteShader(shader));
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || "unknown error";
    gl.deleteProgram(program);
    throw new Error(log);
  }
  return program;
}

// `canvas` is where the last pass is drawn. Null when the browser has no WebGL 2.
export function createLookGl(canvas) {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

  const newTexture = () => {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  };
  const picture = newTexture();
  // The two targets the passes before the last one are drawn into, made again when the canvas changes size.
  const targets = [0, 1].map(() => ({ texture: newTexture(), buffer: gl.createFramebuffer(), size: "" }));
  const sizeTarget = (target) => {
    const size = `${canvas.width}x${canvas.height}`;
    if (target.size === size) return;
    gl.bindTexture(gl.TEXTURE_2D, target.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, canvas.width, canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.buffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target.texture, 0);
    target.size = size;
  };

  const programs = new Map();
  // The program of a look, made on first use. A look that does not compile is drawn as a pass through and its error is kept.
  const programOf = (item) => {
    const key = lookProgramKey(item);
    let entry = programs.get(key);
    if (!entry) {
      entry = { program: null, params: item ? item.params || [] : [], failure: "" };
      try {
        entry.program = makeLookProgram(gl, item ? item.glsl3 : LOOK_PASS_THROUGH);
      } catch (error) {
        entry.failure = String(error.message || error);
      }
      programs.set(key, entry);
    }
    return entry.program || !item ? entry : programOf(null);
  };

  return {
    setFrame(source) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, picture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    },
    // `steps` is the chain: `{item, progress, time, intensity, params}`, each item a library look. An empty chain draws the picture as it is.
    draw(steps) {
      const chain = steps.length ? steps : [{ item: null, progress: 0, time: 0, intensity: 0, params: {} }];
      let input = picture;
      chain.forEach((step, index) => {
        const last = index === chain.length - 1;
        const target = targets[index % 2];
        if (last) gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        else {
          sizeTarget(target);
          gl.bindFramebuffer(gl.FRAMEBUFFER, target.buffer);
        }
        const entry = programOf(step.item);
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.useProgram(entry.program);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, input);
        gl.uniform1i(gl.getUniformLocation(entry.program, "_from"), 0);
        gl.uniform1f(gl.getUniformLocation(entry.program, "progress"), step.progress);
        gl.uniform1f(gl.getUniformLocation(entry.program, "ratio"), canvas.width / canvas.height);
        gl.uniform1f(gl.getUniformLocation(entry.program, "intensity"), step.intensity);
        gl.uniform1f(gl.getUniformLocation(entry.program, "time"), step.time);
        for (const param of entry.params) setTransitionUniform(gl, entry.program, param, step.params?.[param.name] ?? param.default);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        input = target.texture;
      });
    },
    // The error of a look that did not compile ("" when it did, or was not used yet).
    failure: (item) => programs.get(lookProgramKey(item))?.failure || "",
    dispose() {
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
}
