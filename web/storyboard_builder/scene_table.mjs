import {
  copyTextToClipboard,
  createStoryboardProgressWindow,
  createToast,
  escapeHtml,
  makeButton,
  truncate,
} from "./controls.mjs";
import { openStoryboardGptUrl, storyboardGptPayload } from "./gpt_payload.mjs";
import { videoPromptTypeLabel } from "./prompt_generation.mjs";
import {
  settingRefHtml,
  storyboardReferenceImageSrc,
  storyboardSubjectNamesFromRefs,
  subjectRefsHtml,
} from "./references.mjs";
import { normalizeScene } from "./scenes.mjs";

export function createSceneTable({
  addStoryboardReferenceFromFile, createScenePromptForActiveMode, currentRows, openSceneEditor,
  promptRunnerName, refreshActionButtons, refreshSetupPanelSummaries, state, stats,
  syncReferenceMappingsToVideoCreator, tableWrap,
}) {
  function openStoryboardSubjectPicker(scene) {
    if (!scene) return;
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed;inset:0;z-index:100050;background:rgba(0,0,0,.72);display:flex;align-items:center;justify-content:center;padding:22px;";
    const panel = document.createElement("div");
    panel.style.cssText = "width:min(980px,calc(100vw - 44px));max-height:calc(100vh - 48px);overflow:auto;border:1px solid #155e75;border-radius:10px;background:#0b1220;color:#f8fafc;padding:14px;display:flex;flex-direction:column;gap:12px;box-shadow:0 24px 80px rgba(0,0,0,.62);";
    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;";
    const heading = document.createElement("div");
    heading.textContent = `${scene.scene_number || 1}. ${scene.label || `Scene ${scene.scene_number || 1}`} — Characters Present`;
    heading.style.cssText = "font-size:16px;font-weight:900;color:#cffafe;";
    const close = makeButton("Cancel");
    header.append(heading, close);

    const choices = document.createElement("div");
    choices.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;";
    const selected = new Set(
      (Array.isArray(scene.subject_refs) ? scene.subject_refs : [])
        .map((subject) => String(subject?.id || ""))
        .filter(Boolean),
    );
    const availableSubjects = Array.isArray(state.referenceBuilder.subjects)
      ? state.referenceBuilder.subjects
      : [];

    const renderChoices = () => {
      choices.replaceChildren();
      const clearSelection = document.createElement("button");
      clearSelection.type = "button";
      clearSelection.textContent = "Clear selection";
      clearSelection.style.cssText = `min-height:132px;border:2px dashed ${selected.size ? "#475569" : "#22d3ee"};border-radius:8px;background:${selected.size ? "#111827" : "#083344"};color:#94a3b8;cursor:pointer;font-weight:900;`;
      clearSelection.onclick = () => {
        selected.clear();
        renderChoices();
      };
      choices.append(clearSelection);

      availableSubjects.forEach((subject) => {
        const subjectId = String(subject.id || "");
        if (!subjectId) return;
        const active = selected.has(subjectId);
        const card = document.createElement("button");
        card.type = "button";
        card.style.cssText = `min-height:132px;border:2px solid ${active ? "#22d3ee" : "#334155"};border-radius:8px;background:${active ? "#083344" : "#111827"};color:#f8fafc;padding:8px;display:flex;flex-direction:column;gap:7px;align-items:center;cursor:pointer;`;
        const preview = document.createElement("div");
        preview.style.cssText = "width:96px;height:72px;border:1px solid #155e75;border-radius:6px;background:#061620;overflow:hidden;display:flex;align-items:center;justify-content:center;flex:0 0 auto;";
        const imageSource = storyboardReferenceImageSrc(subject.image || {});
        if (imageSource) {
          const image = document.createElement("img");
          image.src = imageSource;
          image.alt = subject.name || "Subject reference";
          image.draggable = false;
          image.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
          preview.append(image);
        } else {
          const empty = document.createElement("span");
          empty.textContent = "No image";
          empty.style.cssText = "font-size:11px;font-weight:900;color:#67e8f9;";
          preview.append(empty);
        }
        const name = document.createElement("div");
        name.textContent = subject.name || "Subject";
        name.style.cssText = "font-size:12px;font-weight:900;text-align:center;line-height:1.25;";
        card.append(preview, name);
        card.onclick = () => {
          if (selected.has(subjectId)) selected.delete(subjectId);
          else selected.add(subjectId);
          renderChoices();
        };
        choices.append(card);
      });

      if (!availableSubjects.length) {
        const empty = document.createElement("div");
        empty.textContent = "No subjects are in Reference Builder yet. Use Upload New Subject below to add the first one.";
        empty.style.cssText = "grid-column:1/-1;border:1px dashed #334155;border-radius:8px;padding:18px;color:#94a3b8;text-align:center;font-size:12px;";
        choices.append(empty);
      }
    };

    const footer = document.createElement("div");
    footer.style.cssText = "display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;";
    const upload = makeButton("Upload New Subject");
    const cancel = makeButton("Cancel");
    const apply = makeButton("Apply Selection", "primary");
    const dismiss = () => backdrop.remove();
    close.onclick = dismiss;
    cancel.onclick = dismiss;
    upload.onclick = async () => {
      dismiss();
      await addStoryboardReferenceFromFile("subject", scene);
    };
    apply.onclick = () => {
      const selectedSubjects = availableSubjects.filter((subject) => selected.has(String(subject.id || "")));
      scene.subject_refs = selectedSubjects;
      scene.subjects = storyboardSubjectNamesFromRefs(selectedSubjects);
      if (selectedSubjects.length) scene.no_character_present = false;
      syncReferenceMappingsToVideoCreator();
      dismiss();
      renderTable();
      createToast(selectedSubjects.length
        ? `${selectedSubjects.length} subject${selectedSubjects.length === 1 ? "" : "s"} mapped to ${scene.label || `Scene ${scene.scene_number}`}.`
        : `Subject mapping cleared for ${scene.label || `Scene ${scene.scene_number}`}.`);
    };
    backdrop.addEventListener("pointerdown", (event) => {
      if (event.target === backdrop) dismiss();
    });
    footer.append(upload, cancel, apply);
    panel.append(header, choices, footer);
    backdrop.append(panel);
    document.body.append(backdrop);
    renderChoices();
  }

  function renderTable() {
    const rows = currentRows();
    const mode = state.mode;
    const head = mode === "image_to_video_prep"
      ? ["", "#", "Image", "Scene / Lyrics", "Motion Notes", "Video Prompt", "Subjects", "Setting", "Shot Type", "Status", "Actions"]
      : ["", "#", "Reference", "Scene / Lyrics", "Prompt Summary", "Subjects", "Setting", "Shot Type", "Prompt Status", "Actions"];
    const table = document.createElement("table");
    table.style.cssText = mode === "image_to_video_prep"
      ? "width:100%;border-collapse:collapse;table-layout:fixed;min-width:1567px;font-size:13px;"
      : "width:100%;border-collapse:collapse;min-width:1250px;font-size:13px;";
    if (mode === "image_to_video_prep") {
      const colgroup = document.createElement("colgroup");
      [36, 50, 166, 190, 205, 185, 205, 150, 120, 54, 206].forEach((width) => {
        const col = document.createElement("col");
        col.style.width = `${width}px`;
        colgroup.appendChild(col);
      });
      table.appendChild(colgroup);
    }
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr>${head.map((item) => `<th style="position:sticky;top:0;background:#111827;border-bottom:1px solid #334155;color:#cffafe;text-align:${item === "Status" || item === "" ? "center" : "left"};padding:${mode === "image_to_video_prep" ? "11px 9px" : "13px"};font-weight:900;">${item === "" ? `<input type="checkbox" data-action="select-all" title="Select or deselect all visible scenes" ${rows.length && rows.every((r) => state.selected.has(r.id)) ? "checked" : ""}>` : escapeHtml(item)}</th>`).join("")}</tr>`;
    const tbody = document.createElement("tbody");
    for (const scene of rows) {
      const tr = document.createElement("tr");
      tr.style.borderBottom = "1px solid #1e293b";
      tr.style.background = "#0b1220";
      const sceneImageSource = storyboardReferenceImageSrc({ path: scene.image_path, data: scene.image_data || scene.image_reference_data });
      const imageWidth = mode === "image_to_video_prep" ? 148 : 170;
      const imageCell = sceneImageSource
        ? `<div style="width:${imageWidth}px;height:78px;border-radius:6px;background:#0f172a url('${escapeHtml(sceneImageSource)}') center/cover no-repeat;"></div>`
        : `<div style="width:${imageWidth}px;height:78px;border:1px dashed #334155;border-radius:6px;display:grid;place-items:center;color:#94a3b8;font-size:12px;text-align:center;background:#07111f;">No image in storyboard<br>Optional reference</div>`;
      const sceneActionStyle = mode === "image_to_video_prep"
        ? "border:1px solid #155e75;border-radius:6px;background:#0f172a;color:#a5f3fc;width:76px;min-height:54px;padding:6px 7px;line-height:1.15;white-space:normal;font-weight:800;cursor:pointer;"
        : "border:1px solid #155e75;border-radius:6px;background:#0f172a;color:#a5f3fc;padding:8px 10px;font-weight:800;cursor:pointer;";
      const sceneGptStyle = "border:1px solid #06b6d4;border-radius:6px;background:#0e7490;color:#f8fafc;padding:7px 8px;font-weight:900;cursor:pointer;";
      const sceneGemmaStyle = "border:1px solid #22c55e;border-radius:6px;background:#166534;color:#f0fdf4;padding:7px 8px;font-weight:900;cursor:pointer;";
      const runnerName = promptRunnerName();
      const gemmaTitle = mode === "image_to_video_prep"
        ? `Create this scene's video prompt with ${runnerName}. If the scene has an image, local vision uses it as guidance.`
        : `Create this scene's text-to-image prompt with ${runnerName}.`;
      const actionHtml = `
        <div style="display:${mode === "image_to_video_prep" ? "grid" : "flex"};grid-template-columns:${mode === "image_to_video_prep" ? "76px minmax(62px, 1fr) 44px" : "none"};align-items:stretch;gap:6px;white-space:nowrap;">
          <button data-action="edit" style="${sceneActionStyle}">${mode === "image_to_video_prep" ? "Open Scene<br>Card" : "Open Scene Card"}</button>
          <button data-action="gemma" style="${sceneGemmaStyle}" title="${escapeHtml(gemmaTitle)}">${escapeHtml(runnerName)}</button>
          <button data-action="gpt" style="${sceneGptStyle}" title="Copy only this scene card as GPT JSON.">GPT</button>
        </div>`;
      const promptReady = Boolean(String(mode === "image_to_video_prep" ? scene.video_prompt : scene.image_prompt).trim());
      const promptStatusLabel = promptReady ? "Prompt ready" : "Prompt missing";
      const promptStatusColor = promptReady ? "#22c55e" : "#ef4444";
      const status = `<span role="img" aria-label="${promptStatusLabel}" title="${promptStatusLabel}" style="display:flex;align-items:center;justify-content:center;width:100%;"><span style="width:12px;height:12px;border-radius:999px;background:${promptStatusColor};box-shadow:0 0 0 2px ${promptReady ? "rgba(34,197,94,.16)" : "rgba(239,68,68,.16)"};display:inline-block;"></span></span>`;
      const miniRefButtonStyle = "margin-top:7px;border:1px dashed #155e75;border-radius:6px;background:#07111f;color:#a5f3fc;padding:5px 7px;font-size:11px;font-weight:900;cursor:pointer;";
      const subjectCell = `<div>${subjectRefsHtml(scene, state.referenceBuilder?.subjects || [])}</div><button data-action="load-subject-ref" title="Choose subjects from Reference Builder or upload a new subject image" style="${miniRefButtonStyle}">+ Subject</button>`;
      const settingCell = `<div>${settingRefHtml(scene)}</div><button data-action="load-location-ref" title="Load a location image for this scene" style="${miniRefButtonStyle}">+ Location</button>`;
      const videoType = videoPromptTypeLabel(state.projectVideoEngine === "minimax_h3" ? scene.minimax_h3_mode : (scene.video_prompt_type || "i2v"));
      const shotCell = `<div style="display:flex;flex-direction:column;gap:4px;"><span style="align-self:flex-start;border:1px solid #155e75;border-radius:999px;background:#0f172a;color:#a5f3fc;font-size:11px;font-weight:900;padding:2px 7px;">${escapeHtml(videoType)}</span><strong style="color:#f8fafc;">${escapeHtml(scene.shot_type || "-")}</strong></div>`;
      const storyPreview = `${scene.lyric_section ? `<div style="margin-top:5px;color:#67e8f9;font-size:11px;font-weight:900;">${escapeHtml(scene.lyric_section)}</div>` : ""}${scene.story_beat ? `<div style="margin-top:5px;color:#94a3b8;font-size:11px;">Beat: ${escapeHtml(truncate(scene.story_beat, 90))}</div>` : ""}`;
      if (mode === "image_to_video_prep") {
        const motionNotes = `<textarea data-action="motion-notes" aria-label="Motion notes for ${escapeHtml(scene.label || `Scene ${scene.scene_number}`)}" placeholder="Custom motion or LLM direction..." style="display:block;width:100%;height:80px;box-sizing:border-box;resize:vertical;border:1px solid #334155;border-radius:6px;background:#07111f;color:#e2e8f0;padding:8px;font:inherit;line-height:1.35;outline:none;">${escapeHtml(scene.motion_summary || "")}</textarea>`;
        const videoPrompt = scene.video_prompt
          ? `<div title="${escapeHtml(scene.video_prompt)}" style="color:#d4d4d8;line-height:1.38;overflow-wrap:anywhere;">${escapeHtml(truncate(scene.video_prompt, 115))}</div>`
          : `<span style="color:#64748b;font-style:italic;">No video prompt yet.</span>`;
        tr.innerHTML = `
          <td style="padding:9px;text-align:center;"><input type="checkbox" data-action="select" ${state.selected.has(scene.id) ? "checked" : ""}></td>
          <td style="padding:9px;font-weight:900;font-size:17px;">${String(scene.scene_number).padStart(2, "0")}</td>
          <td style="padding:9px;">${imageCell}</td>
          <td style="padding:9px;overflow:hidden;"><strong style="color:#f8fafc;">${escapeHtml(scene.label)}</strong><br><span style="color:#cbd5e1;">${escapeHtml(truncate(scene.lyrics, 70))}</span>${storyPreview}</td>
          <td style="padding:9px;vertical-align:middle;">${motionNotes}</td>
          <td style="padding:9px;vertical-align:middle;">${videoPrompt}</td>
          <td style="padding:9px;overflow:hidden;">${subjectCell}</td>
          <td style="padding:9px;color:#d4d4d8;overflow:hidden;">${settingCell}</td>
          <td style="padding:9px;overflow:hidden;">${shotCell}</td>
          <td style="padding:9px;text-align:center;">${status}</td>
          <td style="padding:9px;white-space:nowrap;">${actionHtml}</td>
        `;
      } else {
        tr.innerHTML = `
          <td style="padding:13px;text-align:center;"><input type="checkbox" data-action="select" ${state.selected.has(scene.id) ? "checked" : ""}></td>
          <td style="padding:13px;font-weight:900;font-size:17px;">${String(scene.scene_number).padStart(2, "0")}</td>
          <td style="padding:13px;">${imageCell}</td>
          <td style="padding:13px;max-width:220px;"><strong style="color:#f8fafc;">${escapeHtml(scene.label)}</strong><br><span style="color:#cbd5e1;">${escapeHtml(truncate(scene.lyrics, 95))}</span>${storyPreview}</td>
          <td style="padding:13px;max-width:280px;color:#d4d4d8;">${escapeHtml(truncate(scene.prompt_summary || scene.image_prompt, 150))}</td>
          <td style="padding:13px;max-width:230px;">${subjectCell}</td>
          <td style="padding:13px;color:#d4d4d8;max-width:210px;">${settingCell}</td>
          <td style="padding:13px;">${shotCell}</td>
          <td style="padding:13px;">${status}</td>
          <td style="padding:13px;white-space:nowrap;">${actionHtml}</td>
        `;
      }
      tr.querySelector('[data-action="edit"]')?.addEventListener("click", () => openSceneEditor(scene));
      const motionNotesInput = tr.querySelector('[data-action="motion-notes"]');
      if (motionNotesInput) {
        motionNotesInput.addEventListener("input", () => {
          scene.motion_summary = motionNotesInput.value;
        });
        motionNotesInput.addEventListener("change", () => {
          scene.motion_summary = motionNotesInput.value.trim();
        });
        motionNotesInput.addEventListener("keydown", (event) => event.stopPropagation());
      }
      tr.querySelector('[data-action="load-subject-ref"]')?.addEventListener("click", () => openStoryboardSubjectPicker(scene));
      tr.querySelector('[data-action="load-location-ref"]')?.addEventListener("click", () => addStoryboardReferenceFromFile("location", scene));
      tr.querySelector('[data-action="gemma"]')?.addEventListener("click", async () => {
        const runnerName = promptRunnerName();
        const progress = createStoryboardProgressWindow(`Storyboard ${runnerName}`);
        try {
          progress.set(`Preparing ${scene.label || "scene"} for ${runnerName}...`, 12);
          await createScenePromptForActiveMode(scene, { progress, progressPercent: 32 });
          progress.set(state.mode === "image_to_video_prep" ? "Storyboard video prompt ready." : "Storyboard image prompt ready.", 100);
          progress.close(1200);
        } catch (error) {
          progress.set(`Error:\n${String(error?.message || error)}`, 100);
        }
      });
      tr.querySelector('[data-action="gpt"]')?.addEventListener("click", () => copySceneForGpt(scene));
      tr.querySelector('[data-action="select"]')?.addEventListener("change", (event) => {
        if (event.target.checked) state.selected.add(scene.id);
        else state.selected.delete(scene.id);
        renderTable();
      });
      tbody.append(tr);
    }
    thead.querySelector('[data-action="select-all"]')?.addEventListener("change", (event) => {
      if (event.target.checked) {
        for (const row of rows) state.selected.add(row.id);
      } else {
        for (const row of rows) state.selected.delete(row.id);
      }
      renderTable();
    });
    table.append(thead, tbody);
    tableWrap.replaceChildren(table);
    const readyCount = state.scenes.filter((scene) => String(scene.image_prompt || scene.video_prompt || "").trim()).length;
    const imageCount = state.scenes.filter((scene) => String(scene.image_path || "").trim()).length;
    stats.textContent = `${state.scenes.length} scenes  |  ${imageCount} images linked  |  ${readyCount} scenes with prompts  |  ${state.selected.size} selected`;
    refreshSetupPanelSummaries();
    refreshActionButtons();
  }

  async function copySceneForGpt(scene) {
    try {
      const normalized = normalizeScene(scene, 0);
      const payload = storyboardGptPayload(state, [scene]);
      const text = JSON.stringify(payload, null, 2);
      await copyTextToClipboard(text);
      openStoryboardGptUrl(payload);
      createToast(`Copied GPT JSON for ${normalized.label || `Scene ${normalized.scene_number}`} and opened GPT.`);
    } catch (error) {
      createToast(`Could not copy scene GPT JSON:\n${String(error?.message || error)}`, true);
    }
  }

  return { renderTable };
}
