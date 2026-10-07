import { cloneMiniMaxH3Settings, miniMaxH3ModeLabel } from "./minimax_h3.mjs";

// Video profiles: named snapshots of the MiniMax H3 video settings (video type, render pass and everything
// that belongs to them), shared by every project and stored by the server. The server leaves out the audio
// mode, the between-scene continuity settings and internal bookkeeping, so a profile never carries them.

const PASS_LABELS = { single: "Single pass", two_pass: "2 pass", three_pass: "2 pass advanced" };
const REFERENCE_MODES = ["reference_to_video", "image_reference_to_video"];

export function videoProfileLabel(profile) {
  const mode = String(profile?.video_mode || "");
  if (!mode) return String(profile?.name || "");
  const pass = REFERENCE_MODES.includes(mode) ? PASS_LABELS[String(profile?.render_pass || "")] : "";
  return `${profile.name} (${miniMaxH3ModeLabel(mode)}${pass ? `, ${pass}` : ""})`;
}

// Settings after applying a profile. Keys the profile does not carry (audio, continuity, internal markers)
// keep their current values.
export function applyVideoProfileSettings(current, profileSettings) {
  return cloneMiniMaxH3Settings({ ...(current || {}), ...(profileSettings || {}) });
}

export function createVideoProfileActions({
  controls,
  state,
  postJson,
  toast,
  confirmDestructiveAction,
  promptForText,
  requireActiveSegment,
  wizardVideoSettings,
  pushHistory,
  saveMiniMaxH3SettingsFromPanel,
  clearMiniMaxImageReferenceStartFrameOnModeSwitch,
  setMiniMaxH3RenderPassForSegment,
  setMiniMaxH3ModeForSegment,
  syncMiniMaxH3Panel,
  autoSaveSessionQuiet,
}) {
  const { select, addButton, removeButton } = controls;
  let profiles = [];
  let busy = false;

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
    none.textContent = "No profile";
    select.append(none);
    for (const profile of profiles) {
      const option = document.createElement("option");
      option.value = profile.name;
      option.textContent = videoProfileLabel(profile);
      select.append(option);
    }
    const wanted = profiles.find((profile) => profile.name.toLowerCase() === String(preferredName).toLowerCase());
    select.value = wanted ? wanted.name : "";
    state.miniMaxVideoProfile = select.value;
    syncButtons();
  }

  async function refresh(preferredName = state.miniMaxVideoProfile || "") {
    const data = await postJson("/vrgdg/music_builder/list_video_profiles", {});
    profiles = Array.isArray(data?.profiles) ? data.profiles : [];
    // Without a choice in this session, start with the profile chosen last so it does not have to be picked again.
    renderOptions(preferredName || data?.last || "");
    return profiles;
  }

  async function saveCurrentAsProfile() {
    const segment = wizardVideoSettings.global ? null : requireActiveSegment();
    if (!segment && !wizardVideoSettings.global) return;
    const name = await promptForText({
      title: "Save video profile",
      message: "Saves your current video type, render pass and every setting that goes with them. Audio and between-scene continuity are not included.",
      label: "Profile name",
      placeholder: "For example: 24 GB 2 pass advanced",
      confirmLabel: "Save",
    });
    if (!name) return;
    const settings = saveMiniMaxH3SettingsFromPanel(segment);
    busy = true;
    syncButtons();
    try {
      let overwrite = false;
      for (;;) {
        try {
          const data = await postJson("/vrgdg/music_builder/save_video_profile", { name, settings, overwrite });
          await refresh(data?.profile?.name || name);
          toast(`Saved video profile "${data?.profile?.name || name}".`);
          return;
        } catch (error) {
          if (!error?.data?.exists || overwrite) throw error;
          const answer = await confirmDestructiveAction({
            title: `Replace profile "${error.data.name || name}"?`,
            message: "A video profile with this name already exists. Replacing it overwrites its saved settings with your current ones.",
            confirmLabel: "OK",
          });
          if (!answer.confirmed) return;
          overwrite = true;
        }
      }
    } catch (error) {
      toast(`Could not save the video profile: ${error?.message || error}`, true);
    } finally {
      busy = false;
      syncButtons();
    }
  }

  async function applySelectedProfile() {
    const name = selectedName();
    state.miniMaxVideoProfile = name;
    syncButtons();
    if (!name) {
      // Choosing no profile is remembered too, so the next start does not bring one back.
      postJson("/vrgdg/music_builder/set_last_video_profile", { name: "" }).catch(() => null);
      return;
    }
    const segment = wizardVideoSettings.global ? null : requireActiveSegment();
    if (!segment && !wizardVideoSettings.global) return;
    try {
      const data = await postJson("/vrgdg/music_builder/load_video_profile", { name });
      const profile = data?.profile;
      if (!profile?.settings) throw new Error("The profile has no settings.");
      pushHistory();
      const current = saveMiniMaxH3SettingsFromPanel(segment);
      const merged = applyVideoProfileSettings(current, profile.settings);
      clearMiniMaxImageReferenceStartFrameOnModeSwitch(segment, merged.video_mode);
      if (segment?.use_scene_minimax_h3_settings) segment.minimax_h3_settings = merged;
      else state.miniMaxH3Settings = merged;
      // Same steps as clicking the video type and pass buttons, so scene flags and panels stay in step.
      setMiniMaxH3RenderPassForSegment(segment, merged.render_pass);
      setMiniMaxH3ModeForSegment(segment, merged.video_mode);
      syncMiniMaxH3Panel();
      await autoSaveSessionQuiet("MiniMax H3 video profile applied");
      toast(`Applied video profile "${profile.name || name}".`);
    } catch (error) {
      toast(`Could not apply the video profile: ${error?.message || error}`, true);
    }
  }

  // A new project starts with the selected profile's video settings. Existing projects keep their own.
  async function applyToNewProject() {
    const name = selectedName();
    if (!name) return;
    try {
      const data = await postJson("/vrgdg/music_builder/load_video_profile", { name });
      const profile = data?.profile;
      if (!profile?.settings) throw new Error("The profile has no settings.");
      const current = saveMiniMaxH3SettingsFromPanel(null);
      const merged = applyVideoProfileSettings(current, profile.settings);
      clearMiniMaxImageReferenceStartFrameOnModeSwitch(null, merged.video_mode);
      state.miniMaxH3Settings = merged;
      setMiniMaxH3RenderPassForSegment(null, merged.render_pass);
      setMiniMaxH3ModeForSegment(null, merged.video_mode);
      syncMiniMaxH3Panel();
      toast(`New project started with video profile "${profile.name || name}".`);
    } catch (error) {
      toast(`Could not apply the video profile to the new project: ${error?.message || error}`, true);
    }
  }

  async function deleteSelectedProfile() {
    const name = selectedName();
    if (!name) return;
    const answer = await confirmDestructiveAction({
      title: `Delete profile "${name}"?`,
      message: [
        "You are about to delete this video profile.",
        "Projects that already use its settings keep them. Only the saved profile is removed, for every project. This cannot be undone.",
      ],
    });
    if (!answer.confirmed) return;
    busy = true;
    syncButtons();
    try {
      await postJson("/vrgdg/music_builder/delete_video_profile", { name });
      await refresh("");
      toast(`Deleted video profile "${name}".`);
    } catch (error) {
      toast(`Could not delete the video profile: ${error?.message || error}`, true);
    } finally {
      busy = false;
      syncButtons();
    }
  }

  function wire() {
    select.addEventListener("change", () => { applySelectedProfile(); });
    addButton.onclick = () => saveCurrentAsProfile();
    removeButton.onclick = () => deleteSelectedProfile();
    state.applyVideoProfileToNewProject = applyToNewProject;
    syncButtons();
  }

  return { refresh, saveCurrentAsProfile, applySelectedProfile, deleteSelectedProfile, wire, profiles: () => profiles };
}
