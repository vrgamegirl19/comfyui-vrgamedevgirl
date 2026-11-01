// The pictures the transition preview draws between: a scene's last or first frame (else its thumbnail), a placeholder and the cover scaling.

import { makeEditorThumbnailUrl, makeEditorVideoUrl } from "./comfy_api.mjs";
import { selectedSegmentVideoPath, selectedSegmentVideoThumbnailPath } from "./selection_preview.mjs";

function transitionVideoFrame(url, atEnd) {
  return new Promise((resolve, reject) => {
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    const fail = () => reject(new Error("The video did not load."));
    const timer = setTimeout(fail, 8000);
    video.onerror = fail;
    video.onloadedmetadata = () => {
      video.currentTime = atEnd ? Math.max(0, video.duration - 0.015) : 0.001;
    };
    video.onseeked = () => {
      clearTimeout(timer);
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      video.removeAttribute("src");
      video.load();
      resolve(canvas);
    };
    video.src = url;
  });
}

function transitionImageFrame(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      canvas.getContext("2d").drawImage(image, 0, 0);
      resolve(canvas);
    };
    image.onerror = () => reject(new Error("The picture did not load."));
    image.src = url;
  });
}

// The scene's last (or first) frame, else its thumbnail, else nothing.
export async function transitionSceneFrame(segment, atEnd) {
  const videoPath = selectedSegmentVideoPath(segment);
  if (videoPath) {
    try {
      return await transitionVideoFrame(makeEditorVideoUrl(videoPath, segment.video_cache_bust), atEnd);
    } catch {
      /* the thumbnail below is next best */
    }
  }
  const thumbnail = selectedSegmentVideoThumbnailPath(segment);
  if (thumbnail) {
    try {
      return await transitionImageFrame(makeEditorThumbnailUrl(thumbnail));
    } catch {
      /* a placeholder is drawn instead */
    }
  }
  return null;
}

export function transitionPlaceholder(label, hue, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  const gradient = context.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, `hsl(${hue},70%,38%)`);
  gradient.addColorStop(1, `hsl(${hue + 50},70%,16%)`);
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  context.fillStyle = "rgba(255,255,255,.9)";
  context.font = `900 ${Math.round(height / 8)}px sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(label, width / 2, height / 2);
  return canvas;
}

// The picture scaled to cover width x height.
export function transitionCover(source, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const scale = Math.max(width / source.width, height / source.height);
  const w = source.width * scale;
  const h = source.height * scale;
  canvas.getContext("2d").drawImage(source, (width - w) / 2, (height - h) / 2, w, h);
  return canvas;
}
