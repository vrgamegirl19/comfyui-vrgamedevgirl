// The WebGL 2 side of the transition preview. The sources are the transitions' GLSL ES 3.00 text; the player and the picker run them here, and the final
// video draws the same transitions from their WGSL sources.

const TRANSITION_HARD_CUT = "vec4 transition(vec2 uv) { return progress < 0.5 ? getFromColor(uv) : getToColor(uv); }";

export const TRANSITION_VERTEX_SHADER = `#version 300 es
in vec2 position;
out vec2 _uv;
void main() { _uv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }`;

// The sources are bare transitions: `vec4 transition(vec2 uv)` and its own `uniform` lines. The page supplies the rest.
const wrapTransitionShader = (source) => `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D _from;
uniform sampler2D _to;
uniform float progress;
uniform float ratio;
in vec2 _uv;
out vec4 _color;
vec4 getFromColor(vec2 uv) { return texture(_from, uv); }
vec4 getToColor(vec2 uv) { return texture(_to, uv); }
${source}
void main() { _color = transition(_uv); }`;

export function compileTransitionShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || "unknown error";
    gl.deleteShader(shader);
    throw new Error(log);
  }
  return shader;
}

function makeTransitionProgram(gl, fragment) {
  const program = gl.createProgram();
  const shaders = [compileTransitionShader(gl, gl.VERTEX_SHADER, TRANSITION_VERTEX_SHADER), compileTransitionShader(gl, gl.FRAGMENT_SHADER, wrapTransitionShader(fragment))];
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

// The two pictures a few transitions sample. They are the same ones the final render makes (h3-transitions/src/render.rs), so the preview shows what the video will.
function transitionLattice(x, y) {
  let h = (Math.imul(x, 0x9e3779b1) ^ Math.imul(y, 0x85ebca6b)) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  h = Math.imul(h, 0x2c1b3c6d) >>> 0;
  h = (h ^ (h >>> 12)) >>> 0;
  return (h & 0xffff) / 65535;
}

function transitionSmoothNoise(x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const sx = tx * tx * (3 - 2 * tx);
  const sy = ty * ty * (3 - 2 * ty);
  const top = transitionLattice(x0, y0) * (1 - sx) + transitionLattice(x0 + 1, y0) * sx;
  const bottom = transitionLattice(x0, y0 + 1) * (1 - sx) + transitionLattice(x0 + 1, y0 + 1) * sx;
  return top * (1 - sy) + bottom * sy;
}

function transitionExtraPicture(name) {
  const size = 256;
  const data = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const fx = x / (size - 1);
      const fy = y / (size - 1);
      const value = name === "luma" ? (fx + fy) * 0.5 : transitionSmoothNoise(fx * 8, fy * 8);
      const byte = Math.round(Math.min(1, Math.max(0, value)) * 255);
      data.set([byte, byte, byte, 255], (y * size + x) * 4);
    }
  }
  return new ImageData(data, size, size);
}

export function setTransitionUniform(gl, program, param, value) {
  const location = gl.getUniformLocation(program, param.name);
  if (!location || value === undefined || value === null) return;
  switch (param.type) {
    case "float": gl.uniform1f(location, Number(value)); break;
    case "int": gl.uniform1i(location, Math.round(Number(value))); break;
    case "bool": gl.uniform1i(location, value ? 1 : 0); break;
    case "vec2": gl.uniform2fv(location, value); break;
    case "vec3": gl.uniform3fv(location, value); break;
    case "vec4": gl.uniform4fv(location, value); break;
    case "ivec2": gl.uniform2iv(location, value); break;
    case "ivec3": gl.uniform3iv(location, value); break;
    case "ivec4": gl.uniform4iv(location, value); break;
    default: break;
  }
}

export function createTransitionGl(canvas) {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

  const newTexture = (unit, source) => {
    const texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (source) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    return texture;
  };
  const upload = (unit, texture, source) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  };
  const from = newTexture(0);
  const to = newTexture(1);
  newTexture(2, transitionExtraPicture("luma"));
  newTexture(3, transitionExtraPicture("displacementMap"));
  const extraUnits = { luma: 2, displacementMap: 3 };

  let program = null;
  let params = [];
  let failure = "";
  return {
    setFrom: (source) => upload(0, from, source),
    setTo: (source) => upload(1, to, source),
    setFrames(a, b) {
      upload(0, from, a);
      upload(1, to, b);
    },
    setTransition(entry) {
      if (program) gl.deleteProgram(program);
      program = null;
      failure = "";
      try {
        program = makeTransitionProgram(gl, entry ? entry.glsl3 : TRANSITION_HARD_CUT);
        params = entry ? entry.params : [];
      } catch (error) {
        failure = String(error.message || error);
      }
    },
    failure: () => failure,
    draw(progress, values) {
      if (!program) return;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(program);
      gl.uniform1i(gl.getUniformLocation(program, "_from"), 0);
      gl.uniform1i(gl.getUniformLocation(program, "_to"), 1);
      gl.uniform1f(gl.getUniformLocation(program, "progress"), progress);
      gl.uniform1f(gl.getUniformLocation(program, "ratio"), canvas.width / canvas.height);
      for (const [name, unit] of Object.entries(extraUnits)) {
        const location = gl.getUniformLocation(program, name);
        if (location) gl.uniform1i(location, unit);
      }
      for (const param of params) setTransitionUniform(gl, program, param, values?.[param.name] ?? param.default);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
    dispose() {
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
}
