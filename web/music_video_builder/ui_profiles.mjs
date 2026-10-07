import { postJson } from "./comfy_api.mjs";
import { confirmDestructiveAction, promptForText } from "./confirm_dialog.mjs";

// UI layout profiles: named snapshots of how the Builder looks (left panel hidden or shown, panel widths, timeline
// height), shared by every project and stored by the server. They are independent of the video profiles. The
// profile chosen last is loaded when the Builder starts, and while one is selected every layout change is written
// back to it, so the Builder comes back the way it was left.

const SAVE_DELAY_MS = 500;

export function layoutFromState(state) {
  return {
    left_collapsed: Boolean(state.leftPanelCollapsed),
    right_collapsed: Boolean(state.rightPanelCollapsed),
    llm_popout_open: Boolean(state.llmPopoutOpen),
    llm_popout_width: Math.round(Number(state.llmPopoutWidth) || 460),
    llm_popout_height: Math.round(Number(state.llmPopoutHeight) || 460),
    llm_popout_x: Number.isFinite(state.llmPopoutX) ? Math.round(state.llmPopoutX) : null,
    llm_popout_y: Number.isFinite(state.llmPopoutY) ? Math.round(state.llmPopoutY) : null,
    left_panel_width: Math.round(Number(state.leftPanelWidth) || 260),
    right_panel_width: Math.round(Number(state.rightPanelWidth) || 360),
    timeline_panel_height: Math.round(Number(state.timelinePanelHeight) || 300),
  };
}

export function applyLayoutToState(state, layout) {
  const source = layout && typeof layout === "object" ? layout : {};
  state.leftPanelCollapsed = Boolean(source.left_collapsed);
  state.rightPanelCollapsed = Boolean(source.right_collapsed);
  state.llmPopoutOpen = Boolean(source.llm_popout_open);
  if (Number.isFinite(Number(source.llm_popout_width))) state.llmPopoutWidth = Number(source.llm_popout_width);
  if (Number.isFinite(Number(source.llm_popout_height))) state.llmPopoutHeight = Number(source.llm_popout_height);
  state.llmPopoutX = source.llm_popout_x !== null && Number.isFinite(Number(source.llm_popout_x)) ? Number(source.llm_popout_x) : null;
  state.llmPopoutY = source.llm_popout_y !== null && Number.isFinite(Number(source.llm_popout_y)) ? Number(source.llm_popout_y) : null;
  if (Number.isFinite(Number(source.left_panel_width))) state.leftPanelWidth = Number(source.left_panel_width);
  if (Number.isFinite(Number(source.right_panel_width))) state.rightPanelWidth = Number(source.right_panel_width);
  if (Number.isFinite(Number(source.timeline_panel_height))) state.timelinePanelHeight = Number(source.timeline_panel_height);
}

export function createUiProfileActions({ controls, state, toast, applyLayoutSizes, autoSaveSessionQuiet }) {
  const { select, addButton, removeButton } = controls;
  let profiles = [];
  let busy = false;
  let activeLayout = null;
  let saveTimer = null;

  function selectedName() {
    return String(select.value || "");
  }

  function syncButtons() {
    removeButton.disabled = busy || !selectedName();
    addButton.disabled = busy;
  }

  function renderOptions(preferredName = "") {
    select.textContent = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "Default layout";
    select.append(none);
    for (const profile of profiles) {
      const option = document.createElement("option");
      option.value = profile.name;
      option.textContent = profile.name;
      select.append(option);
    }
    const wanted = profiles.find((profile) => profile.name.toLowerCase() === String(preferredName).toLowerCase());
    select.value = wanted ? wanted.name : "";
    state.uiProfile = select.value;
    syncButtons();
  }

  function applyActiveLayout() {
    if (!state.uiProfile || !activeLayout) return;
    applyLayoutToState(state, activeLayout);
    state.syncLlmPopout?.();
    applyLayoutSizes();
  }

  async function activate(name) {
    state.uiProfile = name;
    if (!name) {
      activeLayout = null;
      await postJson("/vrgdg/music_builder/set_last_ui_profile", { name: "" });
      return;
    }
    const data = await postJson("/vrgdg/music_builder/load_ui_profile", { name });
    activeLayout = data?.profile?.layout || null;
    applyActiveLayout();
  }

  // Starts with the profile chosen last, unless a name is given.
  async function refresh(preferredName = "") {
    const data = await postJson("/vrgdg/music_builder/list_ui_profiles", {});
    profiles = Array.isArray(data?.profiles) ? data.profiles : [];
    renderOptions(preferredName || data?.last || "");
    await activate(selectedName());
    return profiles;
  }

  function queueLayoutSave() {
    if (!state.uiProfile) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      const name = state.uiProfile;
      if (!name) return;
      activeLayout = layoutFromState(state);
      try {
        await postJson("/vrgdg/music_builder/update_ui_profile_layout", { name, layout: activeLayout });
      } catch (error) {
        console.warn("[VRGDG Music Builder] Could not update the UI layout profile:", error);
      }
    }, SAVE_DELAY_MS);
  }

  async function saveCurrentAsProfile() {
    const name = await promptForText({
      title: "Save UI layout",
      message: "Saves how the Builder looks now: the left panel (hidden or shown), the panel widths and the timeline height. Later layout changes update the selected profile automatically.",
      label: "Layout name",
      placeholder: "For example: Wide timeline",
      confirmLabel: "Save",
    });
    if (!name) return;
    busy = true;
    syncButtons();
    try {
      const layout = layoutFromState(state);
      let overwrite = false;
      for (;;) {
        try {
          const data = await postJson("/vrgdg/music_builder/save_ui_profile", { name, layout, overwrite });
          const savedName = data?.profile?.name || name;
          profiles = (await postJson("/vrgdg/music_builder/list_ui_profiles", {}))?.profiles || profiles;
          renderOptions(savedName);
          activeLayout = data?.profile?.layout || layout;
          toast(`Saved UI layout "${savedName}".`);
          return;
        } catch (error) {
          if (!error?.data?.exists || overwrite) throw error;
          const answer = await confirmDestructiveAction({
            title: `Replace layout "${error.data.name || name}"?`,
            message: "A UI layout with this name already exists. Replacing it overwrites its saved layout with your current one.",
            confirmLabel: "OK",
          });
          if (!answer.confirmed) return;
          overwrite = true;
        }
      }
    } catch (error) {
      toast(`Could not save the UI layout: ${error?.message || error}`, true);
    } finally {
      busy = false;
      syncButtons();
    }
  }

  async function removeSelectedProfile() {
    const name = selectedName();
    if (!name) return;
    const answer = await confirmDestructiveAction({
      title: `Delete layout "${name}"?`,
      message: "This deletes the saved UI layout. The Builder keeps its current layout.",
      confirmLabel: "Delete",
    });
    if (!answer.confirmed) return;
    busy = true;
    syncButtons();
    try {
      await postJson("/vrgdg/music_builder/delete_ui_profile", { name });
      profiles = (await postJson("/vrgdg/music_builder/list_ui_profiles", {}))?.profiles || [];
      renderOptions("");
      await activate("");
      toast(`Deleted UI layout "${name}".`);
    } catch (error) {
      toast(`Could not delete the UI layout: ${error?.message || error}`, true);
    } finally {
      busy = false;
      syncButtons();
    }
  }

  function wire() {
    select.addEventListener("change", async () => {
      try {
        await activate(selectedName());
        syncButtons();
        await autoSaveSessionQuiet("UI layout profile selected");
      } catch (error) {
        toast(`Could not load the UI layout: ${error?.message || error}`, true);
      }
    });
    addButton.addEventListener("click", saveCurrentAsProfile);
    removeButton.addEventListener("click", removeSelectedProfile);
    // The timeline drag and the side panel tab call these.
    state.onLayoutChanged = queueLayoutSave;
    // A project's own saved layout is applied when it loads. The selected profile wins over it.
    state.reapplyUiProfileLayout = applyActiveLayout;
  }

  return { wire, refresh };
}
