import { escapeHtml, makeButton, toast } from "./controls.mjs";
import { hasReferenceImage } from "./llm_runner.mjs";
import { isExtraSubjectReference, subjectExtraTargetId } from "./reference_data.mjs";

function resolveImageTarget(target) {
  if (!target) return null;
  if (target.owner && target.key) {
    if (!target.owner[target.key]) target.owner[target.key] = { path: "", data: "", name: "" };
    return target.owner[target.key];
  }
  if (!target.image) target.image = { path: "", data: "", name: "" };
  return target.image;
}

function imageNameBase(name = "", fallback = "Reference") {
  const text = String(name || "").split(/[\\/]/).pop().replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
  return text || fallback;
}

function subjectLooksEmpty(subject) {
  if (!subject) return false;
  if (hasReferenceImage(subject.image || {})) return false;
  if (String(subject.description || "").trim()) return false;
  if (subjectExtraTargetId(subject)) return false;
  const name = String(subject.name || "").trim();
  return !name || /^character\s+\d+$/i.test(name) || /^subject\s+\d+$/i.test(name) || /^the performer$/i.test(name);
}

function imageFilesFromDrop(event) {
  const files = Array.from(event.dataTransfer?.files || []);
  const fromFiles = files.filter((item) => /^image\//i.test(item.type) || /\.(png|jpe?g|webp|gif|bmp|tiff?|avif)$/i.test(item.name || ""));
  if (fromFiles.length) return fromFiles;
  const items = Array.from(event.dataTransfer?.items || []);
  return items
    .filter((item) => item.kind === "file")
    .map((item) => item.getAsFile?.())
    .filter((file) => file && (/^image\//i.test(file.type) || /\.(png|jpe?g|webp|gif|bmp|tiff?|avif)$/i.test(file.name || "")));
}

function droppedImageFile(event) {
  return imageFilesFromDrop(event)[0] || null;
}

function droppedImageText(event) {
  return String(
    event.dataTransfer?.getData("text/uri-list")
    || event.dataTransfer?.getData("URL")
    || event.dataTransfer?.getData("text/plain")
    || ""
  ).split(/\r?\n/).map((line) => line.trim()).find((line) => line && !line.startsWith("#")) || "";
}

export function createReferenceImages({
  createLocation, createSubject, droppedSceneImageSource, ensureSubjectCount, fileInput, imageLabel, imageSrc,
  pendingImage, refs, renderAll, subjectCountInput, syncSingleSubjectInputsFromFirstSubject, useLocations,
  useSubject,
}) {
  function openReferenceImagePreview(image = {}, label = "Reference image") {
    const src = imageSrc(image);
    if (!src) return;
    const previewBackdrop = document.createElement("div");
    previewBackdrop.style.cssText = "position:fixed;inset:0;z-index:100020;background:rgba(0,0,0,.82);display:flex;align-items:center;justify-content:center;padding:24px;";
    const previewBox = document.createElement("div");
    previewBox.style.cssText = "width:min(1180px,calc(100vw - 48px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:8px;background:#07111f;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.65);display:flex;flex-direction:column;overflow:hidden;";
    const previewHeader = document.createElement("div");
    previewHeader.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border-bottom:1px solid #155e75;background:#0f172a;";
    const previewTitle = document.createElement("div");
    previewTitle.textContent = imageLabel(image) || label || "Reference image";
    previewTitle.style.cssText = "font-size:13px;font-weight:900;color:#cffafe;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
    const closePreview = makeButton("Close");
    previewHeader.append(previewTitle, closePreview);
    const imageStage = document.createElement("div");
    imageStage.style.cssText = "min-height:180px;max-height:calc(100vh - 118px);background:#020617;display:flex;align-items:center;justify-content:center;padding:12px;overflow:auto;";
    const img = document.createElement("img");
    img.src = src;
    img.alt = label || "Reference image preview";
    img.draggable = false;
    img.style.cssText = "display:block;max-width:100%;max-height:calc(100vh - 150px);object-fit:contain;border-radius:4px;background:#020617;";
    imageStage.append(img);
    previewBox.append(previewHeader, imageStage);
    previewBackdrop.append(previewBox);
    const closeDialog = () => {
      previewBackdrop.remove();
      document.removeEventListener("keydown", onPreviewKeydown, true);
    };
    const onPreviewKeydown = (event) => {
      if (event.key === "Escape") closeDialog();
    };
    closePreview.onclick = closeDialog;
    previewBackdrop.addEventListener("pointerdown", (event) => {
      if (event.target === previewBackdrop) closeDialog();
    });
    document.addEventListener("keydown", onPreviewKeydown, true);
    document.body.append(previewBackdrop);
  }
  function renderDrop(drop, image, emptyText) {
    const src = imageSrc(image);
    drop.style.flexDirection = "column";
    drop.style.gap = "6px";
    drop.innerHTML = "";
    if (src) {
      const img = document.createElement("img");
      img.src = src;
      img.draggable = false;
      img.alt = imageLabel(image) || "Reference image";
      img.title = "Click to preview larger";
      img.style.cssText = "width:96px;height:72px;max-width:100%;object-fit:cover;border:1px solid #155e75;border-radius:6px;background:#020617;cursor:zoom-in;";
      img.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openReferenceImagePreview(image, emptyText);
      });
      const label = document.createElement("div");
      label.textContent = imageLabel(image);
      label.style.cssText = "font-size:11px;color:#a5f3fc;overflow-wrap:anywhere;word-break:break-word;max-width:100%;";
      drop.append(img, label);
    } else {
      drop.innerHTML = `<div><strong>${emptyText}</strong><br><span style="color:#94a3b8;font-size:12px;">Drop an image here or upload one.</span></div>`;
    }
  }
  function syncPrimarySubjectImage() {
    ensureSubjectCount();
    const primary = refs.subjects[0];
    if (!primary) return;
    const globalImage = refs.subject?.image || {};
    const primaryImage = primary.image || {};
    if (hasReferenceImage(globalImage) && !hasReferenceImage(primaryImage)) {
      primary.image = { ...globalImage };
    } else if (!hasReferenceImage(globalImage) && hasReferenceImage(primaryImage)) {
      refs.subject.image = { ...primaryImage };
    }
  }
  function subjectPreviewImages(subject) {
    const images = [];
    if (hasReferenceImage(subject?.image || {})) images.push({ image: subject.image, label: subject.name || "Reference" });
    if (!isExtraSubjectReference(subject)) {
      for (const extra of refs.subjects.filter((item) => subjectExtraTargetId(item) === subject.id)) {
        if (hasReferenceImage(extra?.image || {})) images.push({ image: extra.image, label: extra.name || "Extra ref" });
      }
    }
    return images;
  }
  function renderSubjectThumbnailStrip(subject, dropTarget = null) {
    const images = subjectPreviewImages(subject);
    if (!images.length) return null;
    const strip = document.createElement("div");
    strip.style.cssText = "display:flex;gap:6px;align-items:stretch;overflow-x:auto;padding:6px;border:1px solid #155e75;border-radius:6px;background:#061923;scrollbar-width:thin;min-height:132px;";
    images.slice(0, 6).forEach(({ image, label }, index) => {
      const item = document.createElement("div");
      item.style.cssText = "flex:0 0 108px;display:flex;flex-direction:column;gap:4px;min-width:0;";
      const img = document.createElement("img");
      img.src = imageSrc(image);
      img.alt = label || `Reference ${index + 1}`;
      img.title = "Click to preview larger";
      img.draggable = false;
      img.style.cssText = "width:100%;height:104px;object-fit:cover;border:1px solid #0891b2;border-radius:5px;background:#020617;display:block;cursor:zoom-in;";
      img.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openReferenceImagePreview(image, label || `Reference ${index + 1}`);
      });
      const caption = document.createElement("div");
      caption.textContent = index === 0 ? "Primary" : `Extra ${index}`;
      caption.style.cssText = "font-size:9px;color:#a5f3fc;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
      item.append(img, caption);
      strip.append(item);
    });
    if (images.length > 6) {
      const more = document.createElement("div");
      more.textContent = `+${images.length - 6}`;
      more.style.cssText = "flex:0 0 58px;min-height:104px;display:flex;align-items:center;justify-content:center;border:1px solid #334155;border-radius:5px;background:#0f172a;color:#a5f3fc;font-size:11px;font-weight:900;";
      strip.append(more);
    }
    if (dropTarget) wireDrop(strip, dropTarget);
    return strip;
  }
  function imageTargetKind(target) {
    if (target?.kind) return target.kind;
    if (refs.locations.includes(target?.owner)) return "location";
    if (refs.subjects.includes(target?.owner) || target?.owner === refs.subject || target === refs.subject) return "subject";
    return "";
  }
  function applySubjectImageSource(subject, source = {}) {
    if (!subject) return;
    if (subjectLooksEmpty(subject)) subject.name = imageNameBase(source.name, subject.name || `Character ${refs.subjects.indexOf(subject) + 1}`);
    subject.image = { ...source };
    if (subject === refs.subjects[0] || refs.subjects.length === 1) {
      refs.subject.name = subject.name || refs.subject.name || "Character 1";
      refs.subject.description = subject.description || refs.subject.description || "";
      refs.subject.reference_type = subject.reference_type || refs.subject.reference_type || "character";
      refs.subject.image = { ...source };
    }
  }
  function setImageTargetFromSource(target, source = {}) {
    const image = resolveImageTarget(target);
    if (!image) return;
    image.path = source.path || "";
    image.data = source.data || "";
    image.name = source.name || source.path?.split?.(/[\\/]/)?.pop?.() || "reference.png";
    renderAll();
  }
  const readImageFileSource = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({
      path: "",
      data: String(reader.result || ""),
      name: file.name || "reference.png",
    });
    reader.onerror = () => reject(new Error(`Failed to read ${file.name || "reference image"}.`));
    reader.readAsDataURL(file);
  });
  function setImageTargetFromSources(target, sources = []) {
    const cleanSources = sources.filter((source) => source && (source.path || source.data));
    if (!cleanSources.length) return;
    const kind = imageTargetKind(target);
    if (target?.bulk) {
      if (kind === "subject") {
        const sourcesToCreate = [...cleanSources];
        const emptySubject = refs.subjects.find(subjectLooksEmpty);
        if (emptySubject && sourcesToCreate.length) {
          applySubjectImageSource(emptySubject, sourcesToCreate.shift());
        }
        for (const source of sourcesToCreate) {
          const subject = createSubject(imageNameBase(source.name, `Character ${refs.subjects.length + 1}`));
          subject.image = { ...source };
        }
        refs.use_subject_reference = true;
        refs.cleared = false;
        refs.subject_count = refs.subjects.length;
        subjectCountInput.value = String(refs.subject_count);
        useSubject.input.checked = true;
      } else if (kind === "location") {
        for (const source of cleanSources) {
          const location = createLocation(imageNameBase(source.name, `Location ${refs.locations.length + 1}`));
          location.image = { ...source };
        }
        refs.use_location_references = true;
        refs.locations_cleared = false;
        useLocations.input.checked = true;
      }
      renderAll();
      toast(`Loaded ${cleanSources.length} ${kind === "location" ? "location" : "subject"} reference image${cleanSources.length === 1 ? "" : "s"}.`);
      return;
    }
    setImageTargetFromSource(target, cleanSources[0]);
    if (kind === "subject") {
      const targetSubject = target?.owner && refs.subjects.includes(target.owner)
        ? target.owner
        : target?.owner === refs.subject
          ? refs.subjects[0]
          : null;
      if (targetSubject) applySubjectImageSource(targetSubject, cleanSources[0]);
      for (const source of cleanSources.slice(1)) {
        const subject = createSubject(imageNameBase(source.name, `Character ${refs.subjects.length + 1}`));
        subject.image = { ...source };
      }
      refs.use_subject_reference = true;
      refs.cleared = false;
      refs.subject_count = refs.subjects.length;
      subjectCountInput.value = String(refs.subject_count);
      useSubject.input.checked = true;
    } else if (kind === "location") {
      for (const source of cleanSources.slice(1)) {
        const location = createLocation(imageNameBase(source.name, `Location ${refs.locations.length + 1}`));
        location.image = { ...source };
      }
      refs.use_location_references = true;
      refs.locations_cleared = false;
      useLocations.input.checked = true;
    }
    renderAll();
    const label = kind === "location" ? "location" : "subject";
    toast(cleanSources.length === 1
      ? `Loaded ${label} reference image:\n${cleanSources[0].name || cleanSources[0].path || "reference.png"}`
      : `Loaded ${cleanSources.length} ${label} reference images.`);
  }
  function setImageTarget(target, file) {
    if (!file) return;
    readImageFileSource(file)
      .then((source) => setImageTargetFromSources(target, [source]))
      .catch((error) => toast(String(error?.message || error), true));
  }
  async function setImageTargetFromDroppedUrl(target, urlText) {
    const url = String(urlText || "").trim();
    if (!url) return false;
    if (/^data:image\//i.test(url)) {
      setImageTargetFromSource(target, { path: "", data: url, name: "reference.png" });
      return true;
    }
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      if (!/^image\//i.test(blob.type)) return false;
      const cleanName = url.split(/[?#]/)[0].split("/").pop() || "reference.png";
      setImageTarget(target, new File([blob], cleanName, { type: blob.type || "image/png" }));
      return true;
    } catch (error) {
      console.warn("[VRGDG Music Builder] Could not read dropped reference image URL:", error);
      return false;
    }
  }
  function setImageTargetFromFiles(target, files = []) {
    const allFiles = Array.from(files || []);
    const imageFiles = allFiles.filter((file) => /^image\//i.test(file.type) || /\.(png|jpe?g|webp|gif|bmp|tiff?|avif)$/i.test(file.name || ""));
    if (!imageFiles.length) {
      toast("No readable image files were found in that drop/selection.", true);
      return;
    }
    if (imageFiles.length < allFiles.length) {
      toast(`Skipped ${allFiles.length - imageFiles.length} non-image file${allFiles.length - imageFiles.length === 1 ? "" : "s"}.`, true);
    }
    Promise.all(imageFiles.map(readImageFileSource))
      .then((sources) => setImageTargetFromSources(target, sources))
      .catch((error) => toast(String(error?.message || error), true));
  }
  function wireDrop(drop, target) {
    drop.dataset.vrgdgFileDropZone = "true";
    for (const eventName of ["dragenter", "dragover", "dragleave"]) {
      drop.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      }, true);
    }
    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      drop.style.borderColor = "#22d3ee";
    });
    drop.addEventListener("dragleave", (event) => {
      event.preventDefault();
      event.stopPropagation();
        event.stopImmediatePropagation?.();
        drop.style.borderColor = "#0891b2";
    });
    const handleDrop = (event) => {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
      drop.style.borderColor = "#0891b2";
      const sceneSource = droppedSceneImageSource(event);
      if (sceneSource) {
        setImageTargetFromSource(target, sceneSource);
        return;
      }
      const files = imageFilesFromDrop(event);
      if (files.length) {
        setImageTargetFromFiles(target, files);
        return;
      }
      const urlText = droppedImageText(event);
      if (urlText) {
        setImageTargetFromDroppedUrl(target, urlText).then((ok) => {
          if (!ok) toast("That drop did not contain a readable image. Use Upload Image or drop a PNG/JPG/WebP file.", true);
        });
        return;
      }
      toast("That drop did not contain a readable image. Use Upload Image or drop a PNG/JPG/WebP file.", true);
    };
    drop.addEventListener("drop", handleDrop, true);
    drop.addEventListener("drop", handleDrop);
  }

  function openArrangeReferenceDialog(kind = "subject") {
    const isLocation = kind === "location";
    const titleText = isLocation ? "Arrange Locations" : "Arrange Subjects";
    const list = () => isLocation ? refs.locations : refs.subjects;
    const backdropArrange = document.createElement("div");
    backdropArrange.style.cssText = "position:fixed;inset:0;z-index:100009;background:rgba(0,0,0,.68);display:flex;align-items:center;justify-content:center;padding:22px;box-sizing:border-box;";
    const panel = document.createElement("div");
    panel.style.cssText = "width:min(760px,calc(100vw - 42px));max-height:calc(100vh - 48px);border:1px solid #155e75;border-radius:8px;background:#111827;color:#f8fafc;box-shadow:0 20px 70px rgba(0,0,0,.58);display:flex;flex-direction:column;overflow:hidden;";
    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:10px;background:#083f4f;border-bottom:1px solid #155e75;padding:12px 14px;";
    const title = document.createElement("div");
    title.innerHTML = `<div style="font-size:16px;font-weight:900;color:#cffafe;">${escapeHtml(titleText)}</div><div style="font-size:12px;color:#cbd5e1;margin-top:3px;">Move references into the order you want. Existing scene mappings stay attached to the same reference.</div>`;
    const done = makeButton("Done", "primary");
    head.append(title, done);
    const rows = document.createElement("div");
    rows.style.cssText = "display:flex;flex-direction:column;gap:8px;padding:12px;overflow:auto;";
    let draggedArrangeIndex = -1;
    const moveArrangeItem = (fromIndex, toIndex) => {
      const arr = list();
      if (!arr.length) return;
      const from = Math.max(0, Math.min(arr.length - 1, Number(fromIndex)));
      const boundedTo = Math.max(0, Math.min(arr.length, Number(toIndex)));
      const adjustedTo = boundedTo > from ? boundedTo - 1 : boundedTo;
      if (from === adjustedTo) return;
      const [moved] = arr.splice(from, 1);
      arr.splice(adjustedTo, 0, moved);
      if (!isLocation) {
        refs.subject_count = refs.subjects.length;
        subjectCountInput.value = String(refs.subject_count);
        if (refs.subjects.length <= 1) syncSingleSubjectInputsFromFirstSubject();
      }
      renderArrangeRows();
      renderAll();
    };
    const renderArrangeRows = () => {
      const items = list();
      rows.innerHTML = "";
      if (!items.length) {
        const empty = document.createElement("div");
        empty.textContent = isLocation ? "No locations to arrange yet." : "No subjects to arrange yet.";
        empty.style.cssText = "border:1px solid #334155;border-radius:7px;background:#0f172a;color:#94a3b8;padding:12px;font-size:12px;";
        rows.append(empty);
        return;
      }
      items.forEach((item, index) => {
        const row = document.createElement("div");
        row.draggable = true;
        row.title = "Drag this row above or below another reference to reorder it.";
        row.style.cssText = "display:grid;grid-template-columns:54px minmax(0,1fr) auto;gap:10px;align-items:center;border:1px solid #334155;border-radius:7px;background:#0f172a;padding:8px;cursor:grab;";
        row.addEventListener("dragstart", (event) => {
          draggedArrangeIndex = index;
          row.style.opacity = ".55";
          row.style.cursor = "grabbing";
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", String(index));
        });
        row.addEventListener("dragend", () => {
          draggedArrangeIndex = -1;
          row.style.opacity = "";
          row.style.cursor = "grab";
          row.style.borderColor = "#334155";
        });
        row.addEventListener("dragover", (event) => {
          if (draggedArrangeIndex < 0 || draggedArrangeIndex === index) return;
          event.preventDefault();
          event.dataTransfer.dropEffect = "move";
          const rect = row.getBoundingClientRect();
          const before = event.clientY < rect.top + rect.height / 2;
          row.style.borderColor = before ? "#22d3ee" : "#a3e635";
        });
        row.addEventListener("dragleave", () => {
          row.style.borderColor = "#334155";
        });
        row.addEventListener("drop", (event) => {
          if (draggedArrangeIndex < 0) return;
          event.preventDefault();
          event.stopPropagation();
          row.style.borderColor = "#334155";
          const rect = row.getBoundingClientRect();
          const before = event.clientY < rect.top + rect.height / 2;
          moveArrangeItem(draggedArrangeIndex, before ? index : index + 1);
          draggedArrangeIndex = -1;
        });
        const thumb = document.createElement("div");
        thumb.style.cssText = "width:54px;height:42px;border:1px solid #155e75;border-radius:6px;background:#020617;display:flex;align-items:center;justify-content:center;overflow:hidden;color:#67e8f9;font-size:10px;font-weight:900;";
        const src = imageSrc(item.image || {});
        if (src) {
          const img = document.createElement("img");
          img.src = src;
          img.alt = item.name || "Reference";
          img.draggable = false;
          img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
          thumb.append(img);
        } else {
          thumb.textContent = "NO IMG";
        }
        const info = document.createElement("div");
        const extraLabel = !isLocation && subjectExtraTargetId(item) ? "Extra reference" : isLocation ? "Location" : "Subject";
        info.innerHTML = `<div style="font-size:12px;font-weight:900;color:#e0f2fe;overflow-wrap:anywhere;">${index + 1}. ${escapeHtml(item.name || extraLabel)}</div><div style="font-size:11px;color:#94a3b8;margin-top:3px;overflow-wrap:anywhere;">${escapeHtml(extraLabel)}${item.description ? ` - ${String(item.description).slice(0, 90)}` : ""}</div>`;
        const controls = document.createElement("div");
        controls.style.cssText = "display:grid;grid-template-columns:repeat(4,32px);gap:6px;";
        const top = makeButton("Top");
        const up = makeButton("Up");
        const down = makeButton("Dn");
        const bottom = makeButton("Bot");
        for (const button of [top, up, down, bottom]) {
          button.style.minWidth = "32px";
          button.style.padding = "6px 4px";
        }
        const move = (toIndex) => moveArrangeItem(index, toIndex);
        top.onclick = () => move(0);
        up.onclick = () => move(index - 1);
        down.onclick = () => move(index + 2);
        bottom.onclick = () => move(items.length);
        top.disabled = up.disabled = index === 0;
        down.disabled = bottom.disabled = index === items.length - 1;
        controls.append(top, up, down, bottom);
        row.append(thumb, info, controls);
        rows.append(row);
      });
    };
    panel.append(head, rows);
    backdropArrange.append(panel);
    document.body.append(backdropArrange);
    done.onclick = () => backdropArrange.remove();
    backdropArrange.addEventListener("pointerdown", (event) => {
      if (event.target === backdropArrange) backdropArrange.remove();
    });
    renderArrangeRows();
  }
  function uploadFor(target) {
    pendingImage.target = target;
    fileInput.click();
  }

  return {
    openArrangeReferenceDialog, openReferenceImagePreview, renderDrop, renderSubjectThumbnailStrip,
    setImageTargetFromFiles, subjectPreviewImages, syncPrimarySubjectImage, uploadFor, wireDrop,
  };
}
