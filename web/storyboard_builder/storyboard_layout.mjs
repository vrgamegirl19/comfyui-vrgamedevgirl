import { makeButton, makeInput } from "./controls.mjs";

export function buildStoryboardShell({ focusedTitle, promptAllButtonText, promptRunnerName, state }) {
  const backdrop = document.createElement("div");
  backdrop.style.cssText = "position:fixed;inset:0;z-index:100010;background:rgba(0,0,0,.62);display:flex;align-items:stretch;justify-content:center;padding:18px;box-sizing:border-box;";
  const shell = document.createElement("div");
  shell.className = "vrgdg-storyboard-shell";
  shell.style.cssText = "width:min(1820px,calc(100vw - 36px));max-width:100%;min-width:0;height:calc(100vh - 36px);box-sizing:border-box;border:1px solid #155e75;border-radius:10px;background:#111827;color:#e5e7eb;box-shadow:0 28px 90px rgba(0,0,0,.62);display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;overflow:hidden;font-family:system-ui,-apple-system,Segoe UI,sans-serif;";

  if (!document.getElementById("vrgdg-storyboard-responsive-styles")) {
    const responsiveStyles = document.createElement("style");
    responsiveStyles.id = "vrgdg-storyboard-responsive-styles";
    responsiveStyles.textContent = `
      @media (max-width: 1100px) {
        .vrgdg-storyboard-header {
          grid-template-columns:minmax(0,1fr) !important;
          grid-template-areas:"title" "steps" "actions" !important;
        }
        .vrgdg-storyboard-defaults-grid,
        .vrgdg-storyboard-story-grid {
          grid-template-columns:minmax(0,1fr) !important;
        }
      }
      @media (max-width: 700px) {
        .vrgdg-storyboard-header { padding:14px !important; }
        .vrgdg-storyboard-panel { margin-left:8px !important; margin-right:8px !important; }
        .vrgdg-storyboard-note { margin-left:8px !important; margin-right:8px !important; }
        .vrgdg-storyboard-footer { padding:12px !important; }
      }
    `;
    document.head.append(responsiveStyles);
  }

  const header = document.createElement("div");
  header.className = "vrgdg-storyboard-header";
  header.style.cssText = "display:grid;grid-template-columns:minmax(280px,.8fr) minmax(0,2.2fr);grid-template-areas:'title steps' 'title actions';gap:14px 22px;align-items:center;padding:20px 24px;border-bottom:1px solid #1f3b46;background:linear-gradient(180deg,#083344,#111827);min-width:0;";
  const titleBlock = document.createElement("div");
  titleBlock.style.cssText = "grid-area:title;min-width:0;overflow-wrap:anywhere;";
  titleBlock.innerHTML = `
    <div style="display:flex;gap:14px;align-items:center;min-width:0;">
      <div style="width:52px;height:52px;border-radius:12px;background:#164e63;color:#67e8f9;display:grid;place-items:center;font-size:28px;">▣</div>
      <div style="min-width:0;">
        <div style="font-size:26px;font-weight:900;color:#cffafe;">${focusedTitle || "Storyboard Builder"} <span id="vrgdg-storyboard-mode-pill" style="font-size:13px;border-radius:999px;background:#164e63;color:#a5f3fc;padding:5px 9px;vertical-align:middle;">Planning</span></div>
        <div id="vrgdg-storyboard-subtitle" style="color:#cbd5e1;font-size:14px;margin-top:3px;">Write scene cards, image prompts, and video prompts before sending them to the AI Video Builder.</div>
      </div>
    </div>
  `;
  const steps = document.createElement("div");
  steps.style.cssText = "grid-area:steps;display:flex;flex-wrap:wrap;gap:10px;align-items:center;min-width:0;width:100%;";
  const stepPrompts = makeButton("Image Prep", "purple");
  const stepPrep = makeButton("Video Prep");
  stepPrompts.style.cssText += "flex:1 1 180px;min-width:0;";
  stepPrep.style.cssText += "flex:1 1 160px;min-width:0;";
  steps.append(stepPrompts, stepPrep);
  const headerActions = document.createElement("div");
  headerActions.style.cssText = "grid-area:actions;display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:flex-end;min-width:0;width:100%;";
  const search = makeInput("", "Search scenes...");
  search.style.cssText += "flex:1 1 190px;width:auto;min-width:160px;max-width:260px;";
  const gptButton = makeButton("GPT All", "purple");
  gptButton.title = "Copy all Storyboard scene-card inputs as JSON for your custom GPT.";
  const importImagePromptsButton = makeButton("Import prompts from GPT", "purple");
  importImagePromptsButton.title = "Paste JSON from the Text to Image Prompt Builder GPT and update Image Prep prompts.";
  const gemmaAllButton = makeButton(promptAllButtonText(), "primary");
  gemmaAllButton.title = "Use the selected LLM runner to create prompts for every storyboard scene.";
  const clearPromptsButton = makeButton("Clear Prompts");
  clearPromptsButton.title = "Clear Storyboard scene-card prompt summaries, generated prompts, and extra notes without changing subjects, locations, camera, motion, or lyrics.";
  clearPromptsButton.style.borderColor = "#991b1b";
  clearPromptsButton.style.background = "#3f0808";
  const clearStoryBeatsButton = makeButton("Clear All Story Beats");
  clearStoryBeatsButton.title = "Clear the story beat from every Storyboard scene without changing lyrics, prompts, images, subjects, locations, or shot settings.";
  clearStoryBeatsButton.style.borderColor = "#991b1b";
  clearStoryBeatsButton.style.background = "#3f0808";
  const keepGemmaLoadedLabel = document.createElement("label");
  keepGemmaLoadedLabel.style.cssText = "display:flex;align-items:center;gap:6px;color:#cbd5e1;font-size:12px;font-weight:800;white-space:nowrap;";
  const keepGemmaLoadedInput = document.createElement("input");
  keepGemmaLoadedInput.type = "checkbox";
  keepGemmaLoadedInput.checked = Boolean(state.gemmaSettings?.keep_loaded_for_storyboard_all);
  keepGemmaLoadedLabel.append(keepGemmaLoadedInput, document.createTextNode("Keep local LLM loaded"));
  keepGemmaLoadedLabel.title = `When checked, ${promptRunnerName()} keeps a local text model loaded until the batch finishes. This has no effect on external runners.`;
  const add = makeButton("+ Add Scene", "purple");
  const close = makeButton("Close");
  headerActions.append(gptButton, importImagePromptsButton, gemmaAllButton, clearPromptsButton, clearStoryBeatsButton, keepGemmaLoadedLabel, search, add, close);
  for (const control of headerActions.children) {
    control.style.maxWidth = "100%";
    if (control.tagName === "BUTTON") control.style.whiteSpace = "normal";
  }

  return {
    add, backdrop, clearPromptsButton, clearStoryBeatsButton, close, gemmaAllButton, gptButton, header,
    headerActions, importImagePromptsButton, keepGemmaLoadedInput, search, shell, stepPrep, stepPrompts,
    steps, titleBlock,
  };
}
