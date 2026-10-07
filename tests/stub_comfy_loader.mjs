// Node loader hook: ComfyUI's scripts/api.js and scripts/app.js only exist in the browser, so stub them for tests.
export async function resolve(specifier, context, nextResolve) {
  if (/scripts\/(api|app)\.js$/.test(specifier)) {
    return { url: 'data:text/javascript,export const api = {}; export const app = {};', shortCircuit: true };
  }
  return nextResolve(specifier, context);
}
