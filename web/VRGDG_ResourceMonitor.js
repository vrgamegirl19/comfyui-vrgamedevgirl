import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PREFIX = "VRGDG.ResourceMonitor.";
let panel, timer, controller, resizeObserver, toolbarObserver, toolbar;
let history = [];
let lastGpu = null;
let floatingPosition = null;
let drag = null;
const get = (key) => app.extensionManager.setting.get(PREFIX + key);
const valid = (value) => typeof value === "number" && Number.isFinite(value);
const display = (value, unit) => valid(value) ? `${Math.round(value)}${unit}` : "—";
const percent = (used, total) => valid(used) && valid(total) && total > 0
    ? Math.max(0, Math.min(100, used / total * 100)) : null;

function positionPanel() {
    if (!panel) return;
    const bounds = app.canvas?.canvas?.getBoundingClientRect();
    if (!bounds || bounds.width < 100 || bounds.height < 100) {
        panel.hidden = true;
        return;
    }
    if (floatingPosition) {
        const width = Math.min(get("Details") ? 980 : 740, bounds.width - 16);
        const left = Math.max(bounds.left + 8, Math.min(floatingPosition.x, bounds.right - width - 8));
        const top = Math.max(bounds.top + 8, Math.min(floatingPosition.y, bounds.bottom - 54));
        panel.hidden = !get("Enabled") || document.hidden;
        Object.assign(panel.style, { position: "fixed", right: "auto", left: `${left}px`, top: `${top}px`, width: `${width}px` });
        return;
    }
    Object.assign(panel.style, { position: "absolute", right: "calc(100% + 8px)", left: "auto", top: "0" });
    const bar = toolbar?.getBoundingClientRect();
    panel.hidden = !get("Enabled") || document.hidden || !bar || bar.width === 0;
    if (!bar) return;
    const available = Math.max(0, bar.left - bounds.left - 20);
    panel.hidden ||= available < 80;
    panel.style.width = `${Math.min(get("Details") ? 980 : 740, available)}px`;
}

function attachToolbar() {
    const target = document.querySelector('[data-testid="action-bar-card"]')
        || document.querySelector(".actionbar-container");
    const parent = floatingPosition ? document.body : target;
    if (target === toolbar && panel.parentElement === parent) return;
    if (toolbar) resizeObserver.unobserve(toolbar);
    toolbar = target;
    if (parent) parent.append(panel);
    if (toolbar) resizeObserver.observe(toolbar);
    positionPanel();
}

function restorePosition(value) {
    try {
        const saved = JSON.parse(value || "null");
        floatingPosition = saved && valid(saved.x) && valid(saved.y) ? saved : null;
    } catch {
        floatingPosition = null;
    }
    if (panel && resizeObserver) {
        attachToolbar();
        positionPanel();
    }
}

function savePosition() {
    return app.extensionManager.setting.set(PREFIX + "FloatingPosition", JSON.stringify(floatingPosition));
}

function dockPanel() {
    floatingPosition = null;
    attachToolbar();
    positionPanel();
    void savePosition();
}

function enableDragging() {
    const handle = panel.querySelector(".vrgdg-rm-drag");
    handle.onpointerdown = (event) => {
        if (event.button !== 0 || drag) return;
        event.preventDefault();
        event.stopPropagation();
        const bounds = panel.getBoundingClientRect();
        drag = { pointerId: event.pointerId, dx: event.clientX - bounds.left, dy: event.clientY - bounds.top, original: floatingPosition };
        floatingPosition = { x: bounds.left, y: bounds.top };
        attachToolbar();
        positionPanel();
        // Reparent before capturing so capture survives leaving the toolbar.
        handle.setPointerCapture(event.pointerId);
        handle.style.cursor = "grabbing";
    };
    handle.onpointermove = (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        event.preventDefault();
        event.stopPropagation();
        floatingPosition = { x: event.clientX - drag.dx, y: event.clientY - drag.dy };
        positionPanel();
    };
    const finish = (event) => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        event.stopPropagation();
        const original = drag.original;
        drag = null;
        handle.style.cursor = "";
        if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
        if (event.type !== "pointerup") {
            floatingPosition = original;
            attachToolbar();
            positionPanel();
            return;
        }
        const bounds = panel.getBoundingClientRect();
        floatingPosition = { x: bounds.left, y: bounds.top };
        void savePosition();
    };
    handle.onpointerup = finish;
    handle.onpointercancel = finish;
    handle.onlostpointercapture = finish;
    handle.ondblclick = dockPanel;
}

function setText(key, text) {
    panel.querySelector(`[data-value="${key}"]`).textContent = text;
}

function memoryRow(key, memory) {
    const value = percent(memory?.used, memory?.total);
    setText(key, value === null ? "—" : `${(memory.used / 2 ** 30).toFixed(1)} / ${(memory.total / 2 ** 30).toFixed(1)} GiB`);
    panel.querySelector(`[data-value="${key}"]`).title = value === null ? "Unavailable" : `${value.toFixed(1)}% used`;
    panel.querySelector(`[data-bar="${key}"]`).style.width = `${value ?? 0}%`;
}

function render(data) {
    const gpu = data.gpus.find((item) => item.index === String(get("GPU")));
    const identity = gpu ? `${gpu.index}:${gpu.name}` : null;
    if (identity !== lastGpu) history = [];
    lastGpu = identity;
    setText("name", gpu ? `GPU ${gpu.index} · ${gpu.name}` : "GPU readings unavailable");
    panel.title = gpu ? `GPU ${gpu.index} · ${gpu.name} · Resources on the machine running ComfyUI`
        : "RAM is available. GPU sensors require NVIDIA nvidia-smi and the selected GPU index.";
    setText("load", display(gpu?.load, "%"));
    setText("temperature", display(gpu?.temperature, "°C"));
    setText("fan", display(gpu?.fan, "%"));
    setText("clock", display(gpu?.clock, " MHz"));
    setText("power", `${display(gpu?.power, " W")} / ${display(gpu?.power_limit, " W")}`);
    memoryRow("vram", gpu);
    memoryRow("ram", data.ram);
    history.push(valid(gpu?.load) ? gpu.load : null);
    history = history.slice(-60);
    let path = "", connected = false;
    history.forEach((value, index) => {
        if (value === null) { connected = false; return; }
        const x = (60 - history.length + index) * 400 / 59;
        const y = 30 - Math.max(0, Math.min(100, value)) * 0.28;
        path += `${connected ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)} `;
        connected = true;
    });
    panel.querySelector("path").setAttribute("d", path);
}

async function poll() {
    if (!get("Enabled") || document.hidden || controller) return;
    positionPanel();
    const request = new AbortController();
    controller = request;
    const timeout = setTimeout(() => request.abort(), 5000);
    try {
        const response = await api.fetchApi("/vrgdg/resource-monitor", { signal: request.signal, cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!request.signal.aborted) render(data);
    } catch (error) {
        if (get("Enabled") && !document.hidden) {
            render({ gpus: [], ram: null });
            setText("name", "Monitor offline · retrying…");
            panel.title = "Restart ComfyUI after installing the resource monitor.";
            history = [];
        }
    } finally {
        clearTimeout(timeout);
        controller = null;
        if (get("Enabled") && !document.hidden) timer = setTimeout(poll, 2000);
    }
}

function refresh() {
    if (!panel) return;
    clearTimeout(timer);
    panel.querySelector(".vrgdg-rm-details").hidden = !get("Details");
    positionPanel();
    if (!get("Enabled") || document.hidden) {
        controller?.abort();
        history = [];
        return;
    }
    if (!controller) void poll();
}

function setting(key, name, type, defaultValue, extra = {}) {
    return { id: PREFIX + key, name, type, defaultValue,
        category: ["VRGDG", "Resource Monitor", name], onChange: refresh, ...extra };
}

async function clearMemory() {
    const button = panel.querySelector(".vrgdg-rm-clear");
    if (button.disabled) return;
    button.disabled = true;
    button.textContent = "Requesting…";
    try {
        const response = await api.fetchApi("/vrgdg/resource-monitor/clear-memory", { method: "POST" });
        if (!response.ok) {
            const message = response.status === 409 ? "Wait for running and queued jobs to finish."
                : response.status === 404 ? "Restart ComfyUI to load the Clear memory button."
                : `Memory cleanup request failed (HTTP ${response.status}).`;
            button.textContent = response.status === 409 ? "Queue busy" : "Failed";
            button.title = message;
            app.extensionManager.toast.add({ severity: "warn", summary: "Clear memory", detail: message, life: 6000 });
            return;
        }
        button.textContent = "Requested ✓";
        button.title = "Cleanup requested. Watch RAM and VRAM readings update; models reload on the next run.";
    } catch (error) {
        button.textContent = "Failed";
        button.title = "Could not reach ComfyUI. Check the server connection and try again.";
        app.extensionManager.toast.add({ severity: "error", summary: "Clear memory", detail: button.title, life: 6000 });
    } finally {
        setTimeout(() => { button.disabled = false; button.textContent = "Clear memory"; }, 3000);
    }
}

app.registerExtension({
    name: "VRGDG.ResourceMonitor",
    settings: [
        setting("Enabled", "Show resource monitor", "boolean", false),
        setting("Details", "Show fan, clock and power", "boolean", false),
        setting("GPU", "GPU index (nvidia-smi)", "number", 0, { attrs: { min: 0, step: 1, maxFractionDigits: 0 } }),
        setting("FloatingPosition", "Monitor position", "hidden", "", { onChange: restorePosition }),
    ],
    setup() {
        const style = document.createElement("style");
        style.textContent = `
            #vrgdg-resource-monitor { position:absolute; right:calc(100% + 8px); top:0; z-index:20;
                display:flex; align-items:center; gap:12px; height:46px; box-sizing:border-box; padding:5px 10px;
                overflow-x:auto; overflow-y:hidden; scrollbar-width:thin; white-space:nowrap;
                background:rgba(22,24,28,.95); color:#e5e7eb; border:1px solid #3a3d43; border-radius:10px;
                font:11px/1.4 system-ui,sans-serif; box-shadow:0 4px 18px #0004; pointer-events:auto; }
            #vrgdg-resource-monitor [hidden], #vrgdg-resource-monitor[hidden] { display:none !important; }
            #vrgdg-resource-monitor [data-value="name"] { display:none; }
            #vrgdg-resource-monitor .vrgdg-rm-metric { flex:0 0 auto; }
            #vrgdg-resource-monitor .vrgdg-rm-memory { min-width:132px; }
            #vrgdg-resource-monitor button { border:0; background:transparent; color:#b5bcc7; cursor:pointer; padding:0 4px; font-size:18px; }
            #vrgdg-resource-monitor .vrgdg-rm-controls { display:flex; align-items:center; flex:0 0 auto; gap:2px; position:sticky; left:0; background:#16181c; }
            #vrgdg-resource-monitor .vrgdg-rm-drag { cursor:grab; touch-action:none; user-select:none; }
            #vrgdg-resource-monitor .vrgdg-rm-clear { font-size:11px; padding:5px 7px; border:1px solid #4b5563; border-radius:5px; }
            #vrgdg-resource-monitor .vrgdg-rm-clear:disabled { opacity:.65; cursor:wait; }
            #vrgdg-resource-monitor .vrgdg-rm-line { display:flex; justify-content:space-between; gap:10px; }
            #vrgdg-resource-monitor .vrgdg-rm-track { height:3px; background:#363940; border-radius:4px; margin:4px 0 0; overflow:hidden; }
            #vrgdg-resource-monitor .vrgdg-rm-track i { display:block; height:100%; width:0; background:#4295ff; }
            #vrgdg-resource-monitor [data-bar="ram"] { background:#b78aff; }
            #vrgdg-resource-monitor svg { display:block; width:70px; height:15px; margin-top:2px; }
            #vrgdg-resource-monitor .vrgdg-rm-details { display:flex; flex:0 0 auto; gap:12px; border-left:1px solid #363940; padding-left:12px; color:#b5bcc7; }
            #vrgdg-resource-monitor .vrgdg-rm-details span span { display:block; }
            #vrgdg-resource-monitor strong { font-weight:500; color:#4ddbb1; }
        `;
        document.head.append(style);
        panel = document.createElement("section");
        panel.id = "vrgdg-resource-monitor";
        panel.setAttribute("aria-label", "ComfyUI host resource monitor");
        panel.hidden = true;
        panel.innerHTML = `
            <span data-value="name">Connecting to resource monitor…</span>
            <div class="vrgdg-rm-controls">
                <button class="vrgdg-rm-drag" type="button" title="Drag to move. Double-click to return to toolbar." aria-label="Drag resource monitor">⠿</button>
                <button class="vrgdg-rm-dock" type="button" title="Return to toolbar" aria-label="Return resource monitor to toolbar">↥</button>
                <button class="vrgdg-rm-close" type="button" title="Hide resource monitor (enable again in Settings → VRGDG)" aria-label="Hide resource monitor">×</button>
                <button class="vrgdg-rm-clear" type="button" title="Release ComfyUI model and execution caches, unused RAM/VRAM, and cached Gemma/GGUF models. Requires an idle queue; models reload next run.">Clear memory</button>
            </div>
            <div class="vrgdg-rm-metric"><span>GPU <strong data-value="load">—</strong></span>
            <svg viewBox="0 0 400 32" preserveAspectRatio="none" role="img" aria-label="GPU load history, up to 60 readings, scale 0 to 100 percent"><path fill="none" stroke="#4ddbb1" stroke-width="1.6" vector-effect="non-scaling-stroke" /></svg>
            </div>
            <div class="vrgdg-rm-metric">Temp<br><strong data-value="temperature">—</strong></div>
            <div class="vrgdg-rm-metric vrgdg-rm-memory"><div class="vrgdg-rm-line"><span>VRAM</span><span data-value="vram">—</span></div><div class="vrgdg-rm-track"><i data-bar="vram"></i></div></div>
            <div class="vrgdg-rm-metric vrgdg-rm-memory"><div class="vrgdg-rm-line"><span>RAM</span><span data-value="ram">—</span></div><div class="vrgdg-rm-track"><i data-bar="ram"></i></div></div>
            <div class="vrgdg-rm-details" hidden><span>Fan <span data-value="fan">—</span></span><span>Clock <span data-value="clock">—</span></span><span>Power <span data-value="power">—</span></span></div>
        `;
        panel.querySelector(".vrgdg-rm-close").onclick = () => app.extensionManager.setting.set(PREFIX + "Enabled", false);
        panel.querySelector(".vrgdg-rm-dock").onclick = dockPanel;
        panel.querySelector(".vrgdg-rm-clear").onclick = clearMemory;
        enableDragging();
        for (const event of ["pointerdown", "wheel", "dblclick"]) panel.addEventListener(event, (e) => e.stopPropagation());
        resizeObserver = new ResizeObserver(positionPanel);
        if (app.canvas?.canvas) resizeObserver.observe(app.canvas.canvas);
        toolbarObserver = new MutationObserver((mutations) => {
            if (mutations.some((mutation) => !panel.contains(mutation.target))) attachToolbar();
        });
        toolbarObserver.observe(document.body, { childList: true, subtree: true });
        restorePosition(get("FloatingPosition"));
        attachToolbar();
        window.addEventListener("resize", positionPanel);
        document.addEventListener("visibilitychange", refresh);
        refresh();
    },
});
