export function createMusicVideoBuilderLuts(options = {}) {
  const api = options.api;
  const toast = options.toast || (() => {});
  const getSelectedScene = options.getSelectedScene || (() => null);
  const applyLutToScene = options.applyLutToScene || null;
  const updateScene = options.updateScene || (() => {});
  const refresh = options.refresh || (() => {});
  const autoSave = options.autoSave || (() => Promise.resolve());

  const state = {
    luts: [],
    loaded: false,
    loading: false,
    error: "",
    selectedName: "",
  };

  const root = document.createElement("div");
  root.className = "vrgdg-luts-pane";
  root.style.cssText = "display:none;overflow:auto;padding:10px;min-height:0;flex:1 1 auto;";

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function lutLabel(lut) {
    return String(lut?.label || lut?.name || "").replace(/\.cube$/i, "").replace(/_/g, " ");
  }

  async function loadLuts(force = false) {
    if (state.loading) return;
    if (state.loaded && !force) return;
    state.loading = true;
    state.error = "";
    render();
    try {
      const response = await api.fetchApi("/vrgdg/music_builder/luts");
      const data = await response.json();
      if (!response.ok || data?.ok === false) {
        throw new Error(data?.error || `LUT list failed (${response.status})`);
      }
      state.luts = Array.isArray(data.luts) ? data.luts : [];
      state.loaded = true;
    } catch (error) {
      state.error = String(error?.message || error || "Could not load LUTs.");
      toast(state.error, true);
    } finally {
      state.loading = false;
      render();
    }
  }

  function setSceneLut(lut, event = null) {
    const scene = getSelectedScene();
    if (!scene) {
      toast("Select a scene first, then choose a LUT.", true);
      return;
    }
    if (applyLutToScene) {
      applyLutToScene(lut, scene);
      state.selectedName = lut.name;
      event?.preventDefault?.();
      return;
    }
    scene.lut = {
      name: lut.name,
      label: lutLabel(lut),
      strength: Number(scene.lut?.strength ?? 10) || 10,
      enabled: true,
    };
    state.selectedName = lut.name;
    updateScene(scene);
    refresh();
    autoSave("scene LUT").catch(() => null);
    toast(`Applied LUT to ${scene.label || "selected scene"}.`);
    event?.preventDefault?.();
  }

  function render() {
    root.textContent = "";

    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;gap:8px;margin:2px 0 8px;";
    const title = document.createElement("div");
    title.textContent = "LUTS";
    title.style.cssText = "font-size:12px;font-weight:900;color:#cffafe;letter-spacing:.04em;";
    const refreshButton = document.createElement("button");
    refreshButton.textContent = "Refresh";
    refreshButton.style.cssText = "margin-left:auto;border:1px solid #3f3f46;border-radius:5px;background:#18181b;color:#fafafa;padding:4px 8px;font-size:11px;font-weight:900;cursor:pointer;";
    refreshButton.onclick = () => loadLuts(true);
    header.append(title, refreshButton);
    root.append(header);

    if (state.loading) {
      const loading = document.createElement("div");
      loading.textContent = "Loading LUTs...";
      loading.style.cssText = "color:#a1a1aa;font-size:12px;padding:10px;border:1px solid #3f3f46;border-radius:7px;background:#18181b;";
      root.append(loading);
      return;
    }

    if (state.error) {
      const error = document.createElement("div");
      error.textContent = state.error;
      error.style.cssText = "color:#fecaca;font-size:12px;padding:10px;border:1px solid #7f1d1d;border-radius:7px;background:#2f1212;";
      root.append(error);
      return;
    }

    if (!state.luts.length) {
      const empty = document.createElement("div");
      empty.textContent = "No LUTs found.";
      empty.style.cssText = "color:#a1a1aa;font-size:12px;padding:10px;border:1px solid #3f3f46;border-radius:7px;background:#18181b;";
      root.append(empty);
      return;
    }

    const hint = document.createElement("div");
    hint.textContent = "Click a LUT to apply it to the selected scene.";
    hint.style.cssText = "color:#a1a1aa;font-size:11px;line-height:1.35;margin:0 0 8px;";
    root.append(hint);

    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:8px;align-items:start;";
    for (const lut of state.luts) {
      const card = document.createElement("button");
      card.type = "button";
      card.draggable = true;
      card.dataset.vrgdgLutName = lut.name;
      const active = state.selectedName === lut.name || getSelectedScene()?.lut?.name === lut.name;
      card.style.cssText = `display:flex;flex-direction:column;gap:5px;min-width:0;text-align:left;border:${active ? "2px" : "1px"} solid ${active ? "#22d3ee" : "#3f3f46"};border-radius:7px;background:${active ? "#083344" : "#18181b"};color:#f8fafc;padding:6px;cursor:pointer;`;
      card.title = lut.name;
      const image = lut.example_url
        ? `<img src="${escapeHtml(lut.example_url)}" alt="" draggable="false" style="width:100%;aspect-ratio:16/10;object-fit:cover;border-radius:4px;background:#09090b;border:1px solid #27272a;">`
        : `<div style="width:100%;aspect-ratio:16/10;border-radius:4px;background:#09090b;border:1px solid #27272a;"></div>`;
      card.innerHTML = `${image}<span style="font-size:10px;line-height:1.2;font-weight:900;color:#e0f2fe;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;">${escapeHtml(lutLabel(lut))}</span>`;
      card.onclick = (event) => setSceneLut(lut, event);
      card.ondragstart = (event) => {
        event.dataTransfer.setData("application/x-vrgdg-lut-name", lut.name);
        event.dataTransfer.setData("text/plain", lut.name);
        event.dataTransfer.effectAllowed = "copy";
      };
      grid.append(card);
    }
    root.append(grid);
  }

  return {
    element: root,
    loadLuts,
    render,
    state,
  };
}
