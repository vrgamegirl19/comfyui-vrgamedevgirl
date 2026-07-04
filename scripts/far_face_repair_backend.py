#!/usr/bin/env python
"""Backend prototype for targeted far-face repair experiments.

This intentionally stays outside ComfyUI node registration. The flow is:

1. prepare: extract marked video frames, detect faces, save crops + manifest.
2. enhance: run the crops through any image-to-image/upscale workflow manually.
3. composite: paste repaired crops back into the original extracted frames.

Example:
    python scripts/far_face_repair_backend.py prepare --video input.mp4 --ranges 120-160,300-318 --out temp/far_faces
    python scripts/far_face_repair_backend.py composite --manifest temp/far_faces/manifest.json --repaired-dir temp/far_faces/crops
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    def contains(self, frame_index: int) -> bool:
        return self.start <= frame_index <= self.end


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float


def parse_ranges(value: str) -> list[FrameRange]:
    ranges: list[FrameRange] = []
    for raw_part in (value or "").replace("\n", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
        else:
            start = end = int(part)
        if start < 0 or end < 0:
            raise ValueError(f"Frame ranges must be non-negative: {part}")
        if end < start:
            start, end = end, start
        ranges.append(FrameRange(start, end))
    if not ranges:
        raise ValueError("No frame ranges were provided.")
    return ranges


def parse_box(value: str) -> tuple[int, int, int, int] | None:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    parts = [int(float(item.strip())) for item in cleaned.replace("x", ",").split(",") if item.strip()]
    if len(parts) != 4:
        raise ValueError("--manual-box must be x,y,w,h or x1,y1,x2,y2")
    x1, y1, a, b = parts
    if a > x1 and b > y1:
        return x1, y1, a, b
    return x1, y1, x1 + max(1, a), y1 + max(1, b)


def selected_frame_set(ranges: Iterable[FrameRange]) -> set[int]:
    selected: set[int] = set()
    for item in ranges:
        selected.update(range(item.start, item.end + 1))
    return selected


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def pil_from_bgr(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def detect_faces_mediapipe(frame_bgr: np.ndarray, min_confidence: float) -> list[FaceBox]:
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    results: list[FaceBox] = []
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=float(min_confidence),
    ) as detector:
        detection_result = detector.process(rgb)

    for detection in detection_result.detections or []:
        bbox = detection.location_data.relative_bounding_box
        x = int(round(bbox.xmin * width))
        y = int(round(bbox.ymin * height))
        w = int(round(bbox.width * width))
        h = int(round(bbox.height * height))
        score = float(detection.score[0]) if detection.score else 0.0
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(1, min(width - x, w))
        h = max(1, min(height - y, h))
        results.append(FaceBox(x, y, w, h, score))
    return results


def detect_faces_opencv(frame_bgr: np.ndarray) -> list[FaceBox]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(12, 12))
    return [FaceBox(int(x), int(y), int(w), int(h), 1.0) for x, y, w, h in faces]


def detect_faces(frame_bgr: np.ndarray, detector: str, min_confidence: float) -> list[FaceBox]:
    if detector == "mediapipe":
        try:
            faces = detect_faces_mediapipe(frame_bgr, min_confidence)
        except Exception as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
            print(f"[far-face] mediapipe failed, falling back to opencv: {first_line}")
            faces = []
        return faces or detect_faces_opencv(frame_bgr)
    if detector == "opencv":
        return detect_faces_opencv(frame_bgr)
    raise ValueError(f"Unknown detector: {detector}")


def choose_face(faces: list[FaceBox], width: int, height: int, mode: str) -> FaceBox | None:
    if not faces:
        return None
    center_x = width / 2.0
    center_y = height / 2.0

    def score(face: FaceBox) -> float:
        area = face.w * face.h
        face_cx = face.x + face.w / 2.0
        face_cy = face.y + face.h / 2.0
        dist = math.hypot((face_cx - center_x) / width, (face_cy - center_y) / height)
        if mode == "center":
            return -dist
        return area - dist * area * 0.15

    return max(faces, key=score)


def expanded_square_crop(face: FaceBox, image_width: int, image_height: int, padding: float) -> tuple[int, int, int, int]:
    cx = face.x + face.w / 2.0
    cy = face.y + face.h / 2.0
    side = max(face.w, face.h) * float(padding)
    side = max(side, 32.0)
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, max(left + 1, right))
    bottom = min(image_height, max(top + 1, bottom))
    return left, top, right, bottom


def soft_face_mask(size: tuple[int, int], feather: int, shrink: float = 0.12) -> Image.Image:
    width, height = size
    inset_x = int(round(width * shrink))
    inset_y = int(round(height * shrink))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((inset_x, inset_y, width - inset_x, height - inset_y), fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return mask


def color_match_repaired(original: Image.Image, repaired: Image.Image, mask: Image.Image) -> Image.Image:
    orig = np.asarray(original.convert("RGB")).astype(np.float32)
    rep = np.asarray(repaired.convert("RGB")).astype(np.float32)
    alpha = np.asarray(mask).astype(np.float32) / 255.0
    selected = alpha > 0.25
    if selected.sum() < 16:
        return repaired
    orig_mean = orig[selected].mean(axis=0)
    rep_mean = rep[selected].mean(axis=0)
    adjusted = np.clip(rep + (orig_mean - rep_mean) * 0.65, 0, 255).astype(np.uint8)
    return Image.fromarray(adjusted, "RGB")


def prepare(args: argparse.Namespace) -> None:
    video_path = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    ranges = parse_ranges(args.ranges)
    manual_box = parse_box(args.manual_box)
    selected = selected_frame_set(ranges)
    ensure_empty_dir(out_dir, args.overwrite)
    originals_dir = out_dir / "original_frames"
    crops_dir = out_dir / "crops"
    masks_dir = out_dir / "masks"
    debug_dir = out_dir / "debug"
    for folder in (originals_dir, crops_dir, masks_dir, debug_dir):
        folder.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    entries = []
    misses = []

    max_selected = max(selected)
    frame_index = 0
    while frame_index <= max_selected:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index not in selected:
            frame_index += 1
            continue

        frame_name = f"frame_{frame_index:06d}.png"
        original_path = originals_dir / frame_name
        image = pil_from_bgr(frame)
        image.save(original_path)

        if manual_box:
            mx1, my1, mx2, my2 = manual_box
            mx1 = max(0, min(width - 1, mx1))
            my1 = max(0, min(height - 1, my1))
            mx2 = max(mx1 + 1, min(width, mx2))
            my2 = max(my1 + 1, min(height, my2))
            face = FaceBox(mx1, my1, mx2 - mx1, my2 - my1, 1.0)
        else:
            faces = detect_faces(frame, args.detector, args.min_confidence)
            face = choose_face(faces, width, height, args.face_choice)
        if face is None:
            misses.append(frame_index)
            print(f"[far-face] no face found on frame {frame_index}")
            frame_index += 1
            continue

        left, top, right, bottom = expanded_square_crop(face, width, height, args.padding)
        crop = image.crop((left, top, right, bottom))
        crop_name = f"frame_{frame_index:06d}_face_00.png"
        crop_path = crops_dir / crop_name
        crop.save(crop_path)

        mask = soft_face_mask(crop.size, int(args.feather))
        mask_path = masks_dir / crop_name
        mask.save(mask_path)

        debug = image.copy()
        draw = ImageDraw.Draw(debug)
        draw.rectangle((face.x, face.y, face.x + face.w, face.y + face.h), outline=(255, 220, 0), width=2)
        draw.rectangle((left, top, right, bottom), outline=(0, 255, 120), width=2)
        debug.save(debug_dir / frame_name)

        entries.append(
            {
                "frame": frame_index,
                "original_frame": str(original_path),
                "crop": str(crop_path),
                "mask": str(mask_path),
                "crop_box": [left, top, right, bottom],
                "face_box": [face.x, face.y, face.x + face.w, face.y + face.h],
                "face_score": face.score,
                "repaired_name": crop_name,
            }
        )
        frame_index += 1

    capture.release()
    manifest = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "ranges": [{"start": item.start, "end": item.end} for item in ranges],
        "detector": args.detector,
        "manual_box": list(manual_box) if manual_box else None,
        "padding": args.padding,
        "feather": args.feather,
        "entries": entries,
        "missed_frames": misses,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[far-face] wrote {len(entries)} crops to {crops_dir}")
    print(f"[far-face] manifest: {manifest_path}")
    if misses:
        print(f"[far-face] missed {len(misses)} frames: {misses[:20]}{'...' if len(misses) > 20 else ''}")


def composite(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    repaired_dir = Path(args.repaired_dir).expanduser().resolve() if args.repaired_dir else base_dir / "crops"
    out_dir = Path(args.out).expanduser().resolve() if args.out else base_dir / "composited_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for entry in manifest.get("entries", []):
        original = Image.open(entry["original_frame"]).convert("RGB")
        repaired_path = repaired_dir / entry["repaired_name"]
        if not repaired_path.is_file():
            print(f"[far-face] missing repaired crop: {repaired_path}")
            continue

        left, top, right, bottom = [int(v) for v in entry["crop_box"]]
        target_size = (right - left, bottom - top)
        repaired = Image.open(repaired_path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        mask = Image.open(entry["mask"]).convert("L").resize(target_size, Image.Resampling.LANCZOS)
        if int(args.feather) >= 0:
            mask = soft_face_mask(target_size, int(args.feather))
        if args.color_match:
            original_crop = original.crop((left, top, right, bottom))
            repaired = color_match_repaired(original_crop, repaired, mask)

        output = original.copy()
        output.paste(repaired, (left, top), mask)
        output_path = out_dir / f"frame_{int(entry['frame']):06d}.png"
        output.save(output_path)
        written += 1

    print(f"[far-face] wrote {written} composited frames to {out_dir}")


def contact_sheet(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    repaired_dir = Path(args.repaired_dir).expanduser().resolve() if args.repaired_dir else base_dir / "composited_frames"
    out_path = Path(args.out).expanduser().resolve() if args.out else base_dir / "contact_sheet.jpg"
    thumbs = []

    for entry in manifest.get("entries", [])[: int(args.limit)]:
        original = Image.open(entry["original_frame"]).convert("RGB")
        fixed = repaired_dir / f"frame_{int(entry['frame']):06d}.png"
        if fixed.is_file():
            fixed_img = Image.open(fixed).convert("RGB")
        else:
            fixed_img = original
        pair = Image.new("RGB", (original.width * 2, original.height), (0, 0, 0))
        pair.paste(original, (0, 0))
        pair.paste(fixed_img, (original.width, 0))
        pair.thumbnail((int(args.thumb_width), int(args.thumb_width * pair.height / pair.width)))
        thumbs.append(pair.copy())

    if not thumbs:
        raise RuntimeError("No frames were available for the contact sheet.")

    cols = max(1, int(args.columns))
    rows = math.ceil(len(thumbs) / cols)
    cell_w = max(item.width for item in thumbs)
    cell_h = max(item.height for item in thumbs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 24, 24))
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        sheet.paste(thumb, (x, y))
    sheet.save(out_path, quality=92)
    print(f"[far-face] contact sheet: {out_path}")


def rebuild_video(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixed_dir = Path(args.fixed_dir).expanduser().resolve() if args.fixed_dir else manifest_path.parent / "composited_frames"
    out_path = Path(args.out).expanduser().resolve()
    video_path = Path(manifest["video"]).expanduser().resolve()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(manifest.get("fps") or capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or manifest.get("width") or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or manifest.get("height") or 0)
    selected = {int(entry["frame"]) for entry in manifest.get("entries", [])}
    selected.update(selected_frame_set(FrameRange(int(r["start"]), int(r["end"])) for r in manifest.get("ranges", [])))
    max_selected = max(selected) if selected else -1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not write video: {out_path}")

    frame_index = 0
    written = 0
    replaced = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.only_ranges and frame_index > max_selected:
            break
        if args.only_ranges and frame_index not in selected:
            frame_index += 1
            continue
        fixed_path = fixed_dir / f"frame_{frame_index:06d}.png"
        if fixed_path.is_file():
            fixed = cv2.imread(str(fixed_path), cv2.IMREAD_COLOR)
            if fixed is not None:
                if fixed.shape[1] != width or fixed.shape[0] != height:
                    fixed = cv2.resize(fixed, (width, height), interpolation=cv2.INTER_LANCZOS4)
                frame = fixed
                replaced += 1
        writer.write(frame)
        written += 1
        frame_index += 1

    capture.release()
    writer.release()
    print(f"[far-face] wrote preview video: {out_path}")
    print(f"[far-face] frames written: {written}, frames replaced: {replaced}")


def ffprobe_fps(video_path: Path) -> str:
    command = [
        "ffprobe",
        "-v",
        "0",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout or "").strip() or "30"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prototype far-face repair prep/composite backend.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare", help="Extract marked frames and face crops.")
    prepare_parser.add_argument("--video", required=True)
    prepare_parser.add_argument("--ranges", required=True, help="Example: 120-160,300-318")
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument("--detector", choices=["opencv", "mediapipe"], default="opencv")
    prepare_parser.add_argument("--face-choice", choices=["largest", "center"], default="largest")
    prepare_parser.add_argument("--manual-box", default="", help="Optional forced face box: x,y,w,h or x1,y1,x2,y2")
    prepare_parser.add_argument("--min-confidence", type=float, default=0.35)
    prepare_parser.add_argument("--padding", type=float, default=2.35)
    prepare_parser.add_argument("--feather", type=int, default=18)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    composite_parser = sub.add_parser("composite", help="Composite repaired crops back onto extracted frames.")
    composite_parser.add_argument("--manifest", required=True)
    composite_parser.add_argument("--repaired-dir", default="")
    composite_parser.add_argument("--out", default="")
    composite_parser.add_argument("--feather", type=int, default=18, help="Use -1 to keep saved masks exactly.")
    composite_parser.add_argument("--color-match", action="store_true")
    composite_parser.set_defaults(func=composite)

    sheet_parser = sub.add_parser("contact-sheet", help="Make an original/fixed review sheet.")
    sheet_parser.add_argument("--manifest", required=True)
    sheet_parser.add_argument("--repaired-dir", default="")
    sheet_parser.add_argument("--out", default="")
    sheet_parser.add_argument("--limit", type=int, default=24)
    sheet_parser.add_argument("--columns", type=int, default=3)
    sheet_parser.add_argument("--thumb-width", type=int, default=900)
    sheet_parser.set_defaults(func=contact_sheet)

    rebuild_parser = sub.add_parser("rebuild-video", help="Make an MP4 preview with composited frames replacing originals. Audio is not included.")
    rebuild_parser.add_argument("--manifest", required=True)
    rebuild_parser.add_argument("--fixed-dir", default="")
    rebuild_parser.add_argument("--out", required=True)
    rebuild_parser.add_argument("--only-ranges", action="store_true", help="Write only marked-range frames for a quick preview.")
    rebuild_parser.set_defaults(func=rebuild_video)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
