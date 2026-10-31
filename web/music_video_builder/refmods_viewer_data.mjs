// Pure helpers for the RefMods Viewer: group the saved RefMods by type, filter and sort them, and format their details.
// No DOM and no ComfyUI imports, so they run under node --test.

export const VIEWER_SORTS = [
  { value: "name", label: "Name" },
  { value: "tokens", label: "Most tokens" },
  { value: "size", label: "Largest file" },
];

const baseName = (name) => String(name || "").split("/").pop();

// "darrel_noclothes" -> "Darrel Noclothes"
export function prettyRefmodName(name) {
  return baseName(name).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().replace(/\b\w/g, (char) => char.toUpperCase());
}

// "clothing_men" -> "Clothing Men", used when a type has no label of its own.
export function prettyTypeName(type) {
  return String(type || "generic").replace(/[_-]+/g, " ").trim().replace(/\b\w/g, (char) => char.toUpperCase());
}

export function typeLabel(type, typeLabels = {}) {
  return typeLabels[type] || prettyTypeName(type);
}

// The categories present in the library, in the order of `typeOrder` first and then alphabetically, with a count each.
export function groupByType(refmods, typeOrder = []) {
  const counts = new Map();
  for (const item of refmods || []) {
    const type = String(item?.type || "generic");
    counts.set(type, (counts.get(type) || 0) + 1);
  }
  const rank = (type) => {
    const index = typeOrder.indexOf(type);
    return index === -1 ? typeOrder.length : index;
  };
  return [...counts.entries()]
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => rank(a.type) - rank(b.type) || a.type.localeCompare(b.type));
}

const searchText = (item) => [
  item?.name, prettyRefmodName(item?.name), item?.type, item?.description, item?.uploader, ...(Array.isArray(item?.tags) ? item.tags : []),
].join(" ").toLowerCase();

// `type` "" means every category. `query` matches name, type, description and tags; every word has to match.
export function filterRefmods(refmods, { type = "", query = "", sort = "name" } = {}) {
  const words = String(query || "").toLowerCase().split(/\s+/).filter(Boolean);
  const list = (refmods || []).filter((item) => {
    if (type && String(item?.type || "generic") !== type) return false;
    if (!words.length) return true;
    const text = searchText(item);
    return words.every((word) => text.includes(word));
  });
  const byName = (a, b) => String(a.name).toLowerCase().localeCompare(String(b.name).toLowerCase());
  if (sort === "tokens") return list.sort((a, b) => (b.tokens || 0) - (a.tokens || 0) || byName(a, b));
  if (sort === "size") return list.sort((a, b) => (b.size || 0) - (a.size || 0) || byName(a, b));
  return list.sort(byName);
}

export function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// One picture is a <Picture n> RefMod, several stacked pictures are a <Video n> RefMod.
export function refmodKindLabel(item) {
  return item?.kind === "video" ? `Video (${item.frames || 0} images)` : "Picture";
}

export function formatCanvas(canvas) {
  const [width, height] = Array.isArray(canvas) ? canvas : [0, 0];
  return width && height ? `${width} x ${height}` : "";
}

export function formatTokens(tokens) {
  return Number(tokens || 0).toLocaleString("en-US");
}

// "https://aidigitalcreations.com/" -> "aidigitalcreations.com", for the line under the title.
export function serverHost(url) {
  return String(url || "").replace(/^https?:\/\//i, "").replace(/\/.*$/, "");
}

// What to tell the user after sending a RefMod to the server. `result` is the answer of POST /vrgdg/refmod/server/upload.
// A new account's RefMods wait for the owner of the server to approve them before anyone else sees them.
export function sharedMessage(result, name) {
  const label = prettyRefmodName(name);
  if (result?.duplicate) return `${label} is already on the server.`;
  const waiting = result?.status === "pending" ? " Others see it once the owner of the server approves it." : "";
  if (result?.replaced) return `Replaced your earlier ${label} on the server.${waiting}`;
  return `Shared ${label} with the server.${waiting}`;
}

// The status a shared RefMod can be in for its own uploader: approved ones need no note.
export function statusLabel(status) {
  if (status === "pending") return "Waiting for approval";
  if (status === "hidden") return "Hidden";
  return "";
}

// The name of a way to sign in, as the server lists it.
export function providerLabel(provider) {
  const labels = { github: "GitHub", discord: "Discord", dev: "Test sign-in (this computer only)" };
  return labels[provider] || prettyTypeName(provider);
}

// A sign-in address from the server is opened in a browser, so only a web page is accepted (never javascript: or file:).
export function isWebAddress(text) {
  return /^https?:\/\/[^\s]+$/i.test(String(text || ""));
}

// The server wants a sign-in (the Video Builder's route passes on its `code`).
export const isSignInRequired = (error) => error?.code === "signin_required";

// What to tell the user after adding a shared RefMod to the library. `result` is the answer of POST /vrgdg/refmod/server/download.
export function addedMessage(result) {
  const label = prettyRefmodName(result?.name);
  if (result?.already) return `${label} is already in your library.`;
  if (result?.renamed) return `Added to your library as ${label}, because you already have one with that name.`;
  return `Added ${label} to your library.`;
}

// The detail lines of the card, label and value, without the empty ones.
export function createdCardRows(info) {
  const kind = info?.kind === "video" ? `Video (${info.imageCount || info.latent_frames || 0} images)` : "Picture";
  return [
    ["Category", info?.typeLabel || prettyTypeName(info?.folder)],
    ["Kind", info?.kind ? kind : ""],
    ["Images used", info?.imageCount ? String(info.imageCount) : ""],
    ["Quality", info?.quality ? prettyTypeName(info.quality) : ""],
    ["Canvas", formatCanvas(info?.canvas)],
    ["Tokens", info?.tokens ? formatTokens(info.tokens) : ""],
    ["Mode", info?.mode || ""],
    ["File size", info?.size ? formatBytes(info.size) : ""],
  ].filter(([, value]) => value);
}
