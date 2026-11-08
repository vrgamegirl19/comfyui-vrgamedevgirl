import { readFileAsDataUrl } from "./media_import.mjs";

// Keep browser File objects out of the JSON session while retaining their
// contents and identity, including the fingerprint used to resume Auto Build.
const encodedFiles = new WeakMap();
export async function encodeAutoBuildFile(file) {
  if (!file) return null;
  if (!encodedFiles.has(file)) {
    const pending = readFileAsDataUrl(file).then((data) => ({
      name: file.name, type: file.type, lastModified: file.lastModified, data,
    })).catch((error) => { encodedFiles.delete(file); throw error; });
    encodedFiles.set(file, pending);
  }
  return encodedFiles.get(file);
}

export function restoreAutoBuildFile(saved) {
  if (!saved) return null;
  const bytes = atob(String(saved.data || "").split(",")[1] || "");
  const file = new File([Uint8Array.from(bytes, (c) => c.charCodeAt(0))], saved.name, {
    type: saved.type, lastModified: saved.lastModified,
  });
  encodedFiles.set(file, Promise.resolve(saved));
  return file;
}

export function autoBuildDraftForProject(draft, folder) {
  const key = (value) => String(value || "").replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
  return draft && key(draft.project_folder) === key(folder) ? draft : null;
}
