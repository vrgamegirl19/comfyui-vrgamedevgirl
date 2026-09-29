import json
import math
import os
import random
import re
from datetime import datetime

import folder_paths
import librosa
import numpy as np
import torch
import torchaudio.functional as AF
from PIL import Image
from server import PromptServer

from ..core.any_type import any_typ


class VRGDG_BuildVideoOutputPath_General_SRT:
    """
    Computes the output file path for Video Combine.
    Handles overwrite vs backup behavior.
    Does NOT save files.
    """

    RETURN_TYPES = (
        "STRING",  # output_path
    )

    RETURN_NAMES = (
        "output_path",
    )

    FUNCTION = "run"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {}),
                "chunk_index": ("INT", {}),
                "base_name": ("STRING", {
                    "default": "video"
                }),
                "overwrite_mode": ("STRING", {}),
            }
        }


    def run(self, output_folder, chunk_index, base_name, overwrite_mode):
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Avoid double-indexing if base_name already has trailing numeric groups.
        base_name = re.sub(r"(?:_\d+)+$", "", base_name)

        # Build canonical filename
        human_index = chunk_index + 1
        filename = f"{base_name}_{human_index:04d}_{chunk_index:04d}"
        output_path = os.path.join(output_folder, filename)


        # Handle backup mode (keep normal mp4 name)
        if overwrite_mode == "backup":
            backup_dir = os.path.join(output_folder, "backup")
            os.makedirs(backup_dir, exist_ok=True)

            prefix = f"{base_name}_{human_index:04d}_{chunk_index:04d}"
            for f in os.listdir(output_folder):
                if f.startswith(prefix) and f.endswith(".mp4"):
                    src = os.path.join(output_folder, f)

                    # ✅ backup keeps same filename, overwrites previous backup
                    dst = os.path.join(backup_dir, f)

                    os.replace(src, dst)


        # In overwrite mode, Video Combine will overwrite naturally
        return (output_path,)
            


# -------------------------
# AUDIO HELPER (FIXED)
# -------------------------

def extract_mono(audio):
    """
    ComfyUI-safe AUDIO extraction.
    Handles (batch, channels, samples), (channels, samples), torch or numpy.
    Returns (mono_numpy_array, sample_rate)
    """

    if audio is None:
        return None, None

    if not isinstance(audio, dict):
        return None, None

    y = audio.get("waveform")
    sr = audio.get("sample_rate")

    if y is None or sr is None:
        return None, None

    # torch -> numpy
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # --- FIX: handle 3D audio ---
    # (batch, channels, samples) -> (channels, samples)
    if y.ndim == 3:
        y = y[0]

    # (channels, samples) -> mono
    if y.ndim == 2:
        y = y.mean(axis=0)

    # Final sanity check
    if y.ndim != 1:
        raise ValueError(f"Audio must be mono after processing, got shape {y.shape}")

    return y.astype(np.float32), int(sr)


# =========================
# NODE A
# =========================

class BeatImpactAnalysisNode:
    """
    Node A: Beat & Impact Analysis
    AUDIO inputs (not file paths)

    Required: final mix
    Optional: drums, bass, vocals, other
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_mix": ("AUDIO",),
            },
            "optional": {
                "drums": ("AUDIO",),
                "bass": ("AUDIO",),
                "vocals": ("AUDIO",),
                "other": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("beat_data",)
    FUNCTION = "analyze"
    CATEGORY = "audio/rhythm"

    def analyze(self, final_mix, drums=None, bass=None, vocals=None, other=None):

        # --- Extract audio safely ---
        y_mix, sr = extract_mono(final_mix)
        if y_mix is None:
            raise ValueError("Final mix AUDIO input is invalid")

        y_drums, _ = extract_mono(drums)
        y_bass, _ = extract_mono(bass)
        y_vocals, _ = extract_mono(vocals)
        y_other, _ = extract_mono(other)

        # --- Beat source selection ---
        # Prefer drums only if they cover the full mix duration and are not silent in the tail.
        def stem_usable(y_stem, y_ref, sr):
            if y_stem is None or y_ref is None:
                return False
            # If the stem is meaningfully shorter than the mix, don't use it for beat tracking.
            if (len(y_ref) - len(y_stem)) / sr > 1.0:
                return False
            # Check tail energy vs overall energy to avoid silence-trimmed stems.
            hop = 512
            frame = 2048
            rms = librosa.feature.rms(y=y_stem, frame_length=frame, hop_length=hop)[0]
            if rms.size == 0:
                return False
            overall = float(np.median(rms))
            tail_frames = max(1, int(10.0 * sr / hop))  # last ~10 seconds
            tail = float(np.median(rms[-tail_frames:]))
            if overall <= 1e-8:
                return False
            return tail >= overall * 0.1

        mix_duration = float(len(y_mix) / sr)
        use_drums_for_beats = stem_usable(y_drums, y_mix, sr)
        use_other_for_beats = stem_usable(y_other, y_mix, sr)

        def track_beats(y_src):
            t, frames = librosa.beat.beat_track(y=y_src, sr=sr, trim=False)
            times = librosa.frames_to_time(frames, sr=sr)
            return t, times

        tempo_mix, beat_times_mix = track_beats(y_mix)
        tempo = tempo_mix
        beat_times = beat_times_mix
        source_used = "final_mix"

        mix_last = float(beat_times_mix[-1]) if len(beat_times_mix) else 0.0
        mix_cov = mix_last / max(mix_duration, 1e-6)
        drums_last = 0.0
        drums_cov = 0.0
        drums_beats = 0
        other_last = 0.0
        other_cov = 0.0
        other_beats = 0

        if use_drums_for_beats:
            tempo_drums, beat_times_drums = track_beats(y_drums)
            drums_last = float(beat_times_drums[-1]) if len(beat_times_drums) else 0.0
            drums_cov = drums_last / max(mix_duration, 1e-6)
            drums_beats = len(beat_times_drums)
            tempo = tempo_drums
            beat_times = beat_times_drums
            source_used = "drums"
        elif use_other_for_beats:
            tempo_other, beat_times_other = track_beats(y_other)
            other_last = float(beat_times_other[-1]) if len(beat_times_other) else 0.0
            other_cov = other_last / max(mix_duration, 1e-6)
            other_beats = len(beat_times_other)
            tempo = tempo_other
            beat_times = beat_times_other
            source_used = "other"

        print(
            "[BeatImpactAnalysisNode] Beat coverage: "
            f"mix_last={mix_last:.3f}s ({mix_cov:.1%}), mix_beats={len(beat_times_mix)}; "
            f"drums_usable={use_drums_for_beats}, drums_last={drums_last:.3f}s ({drums_cov:.1%}), drums_beats={drums_beats}; "
            f"other_usable={use_other_for_beats}, other_last={other_last:.3f}s ({other_cov:.1%}), other_beats={other_beats}; "
            f"selected={source_used}"
        )

        # --- Onset strength (impact signals) ---
        def onset_strength(y):
            if y is None:
                return None
            o = librosa.onset.onset_strength(y=y, sr=sr)
            return o / (np.max(o) + 1e-6)

        onset_mix = onset_strength(y_mix)
        onset_drums = onset_strength(y_drums)
        onset_bass = onset_strength(y_bass)
        onset_vocals = onset_strength(y_vocals)
        onset_other = onset_strength(y_other)

        # If mix onset is empty, fall back to beat-only impact=0.0
        if onset_mix is None or len(onset_mix) == 0:
            print("[BeatImpactAnalysisNode] onset_mix is empty; falling back to beat-only impact=0.0")
            onset_times = np.array([], dtype=np.float32)
        else:
            onset_times = librosa.frames_to_time(
                np.arange(len(onset_mix)), sr=sr
            )

        beats = []

        def safe_onset_value(onset_arr, idx, label):
            if onset_arr is None or len(onset_arr) == 0:
                return None
            if idx < 0 or idx >= len(onset_arr):
                print(f"[BeatImpactAnalysisNode] {label} onset index out of range (idx={idx}, len={len(onset_arr)}); falling back to mix onset.")
                return None
            return onset_arr[idx]

        for i, t in enumerate(beat_times):
            # If we have no onset_times, we can't index into onset arrays; default to 0 impact.
            if onset_times.size == 0:
                idx = None
            else:
                idx = int(np.argmin(np.abs(onset_times - t)))

            impact = 0.0
            weight_sum = 0.0

            if idx is not None:
                val = safe_onset_value(onset_drums, idx, "drums")
                if val is not None:
                    impact += val * 0.45
                    weight_sum += 0.45

            if idx is not None:
                val = safe_onset_value(onset_bass, idx, "bass")
                if val is not None:
                    impact += val * 0.25
                    weight_sum += 0.25

            if idx is not None:
                val = safe_onset_value(onset_vocals, idx, "vocals")
                if val is not None:
                    impact += val * 0.15
                    weight_sum += 0.15

            if idx is not None:
                val = safe_onset_value(onset_other, idx, "other")
                if val is not None:
                    impact += val * 0.15
                    weight_sum += 0.15

            if weight_sum == 0.0:
                # Fall back to mix onset if available; otherwise keep impact at 0.
                if idx is not None and onset_mix is not None and len(onset_mix) > 0:
                    if idx < len(onset_mix):
                        impact = onset_mix[idx]
                    else:
                        print(f"[BeatImpactAnalysisNode] mix onset index out of range (idx={idx}, len={len(onset_mix)}); using impact=0.0")
            else:
                impact /= weight_sum

            beats.append({
                "time": round(float(t), 4),
                "beat_index": i,
                "downbeat": (i % 4 == 0),
                "impact": round(float(impact), 4)
            })

        # librosa may return tempo as a scalar or a 1-element ndarray depending on version.
        # Keep old behavior for scalar tempos and safely fall back for ndarray tempos.
        try:
            tempo_value = float(tempo)
        except (TypeError, ValueError):
            tempo_arr = np.asarray(tempo).reshape(-1)
            tempo_value = float(tempo_arr[0]) if tempo_arr.size else 0.0

        output = {
            "bpm": round(tempo_value, 2),
            "source_used_for_beats": source_used,
            "duration": float(len(y_mix) / sr),
            "beats": beats
        }


        return (json.dumps(output),)


# =========================
# NODE B
# =========================

class BeatSceneDurationNode:
    """
    Node B: Beat-Aligned Scene Duration Generator
    Outputs a valid .srt subtitle file AND returns the text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "beat_data": ("STRING",),
                "min_duration": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "step": 0.1
                }),
                "max_duration": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.2,
                    "step": 0.1
                }),
                "bias": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
                "duration_preset": ([
                    "impact_weighted",
                    "varied_no_repeat",
                    "clustered_no_repeat"
                ], {
                    "default": "impact_weighted"
                }),
                "seed": ("INT", {
                    "default": 0
                }),
                "output_filename": ("STRING", {
                    "default": "beats_output"
                }),
        
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("srt_text", "srt_path",)
    FUNCTION = "generate"
    CATEGORY = "audio/rhythm"

    def generate(
        self,
        beat_data,
        min_duration,
        max_duration,
        bias,
        duration_preset,
        seed,
        output_filename
    ):
        data = json.loads(beat_data)
        beats = data["beats"]
        song_end = data.get("duration", beats[-1]["time"])


        rng = random.Random(seed)

        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        def to_seconds(timestamp):
            h, m, rest = timestamp.split(":")
            s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        def merge_short_first_scene_if_needed(lines, min_first_duration=1.5):
            # Short-first-scene guard:
            # Some beat maps can still create an opening cut that is only a few
            # frames long. Keep this as a final, isolated SRT cleanup so it can
            # be removed easily if we ever want to revert this behavior.
            blocks = []
            for block in "\n".join(lines).strip().split("\n\n"):
                block_lines = [line.strip() for line in block.splitlines() if line.strip()]
                if len(block_lines) < 2 or "-->" not in block_lines[1]:
                    continue
                start_txt, end_txt = [part.strip() for part in block_lines[1].split("-->")]
                blocks.append({
                    "start": to_seconds(start_txt),
                    "end": to_seconds(end_txt),
                })

            if len(blocks) < 2:
                return lines

            first_duration = blocks[0]["end"] - blocks[0]["start"]
            if first_duration >= float(min_first_duration):
                return lines

            print(
                "[BeatSceneDurationNode] Merging short first scene into scene 2: "
                f"duration={first_duration:.3f}s, threshold={float(min_first_duration):.3f}s"
            )
            blocks[1]["start"] = blocks[0]["start"]
            blocks = blocks[1:]

            rebuilt = []
            for idx, block in enumerate(blocks, 1):
                rebuilt.append(str(idx))
                rebuilt.append(f"{format_time(block['start'])} --> {format_time(block['end'])}")
                rebuilt.append(f"SCENE {idx}")
                rebuilt.append("")
            return rebuilt

        srt_lines = []
        current_time = 0.0
        scene_index = 1
        current_index = 0
        no_candidate_windows = 0
        forced_windows = 0
        beat_aligned_windows = 0
        intro_scene_added = False
        prev_duration = None

        if len(beats) == 0:
            raise ValueError("BeatSceneDurationNode received empty beat_data['beats']")

        first_beat = float(beats[0]["time"])
        last_beat = float(beats[-1]["time"])
        coverage = last_beat / max(float(song_end), 1e-6)
        print(
            "[BeatSceneDurationNode] Start: "
            f"beats={len(beats)}, first={first_beat:.3f}s, last={last_beat:.3f}s, "
            f"song_end={float(song_end):.3f}s, beat_coverage={coverage:.1%}, "
            f"min_dur={min_duration:.3f}, max_dur={max_duration:.3f}, bias={bias:.3f}, "
            f"preset={duration_preset}, seed={seed}"
        )

        # Keep SRT clock aligned with absolute beat times. If first beat starts later
        # than 0, add intro scene(s) so current_time matches beat start.
        # Long intros with no detected beats must still respect max_duration.
        if first_beat > 1e-6:
            intro_start = 0.0
            while intro_start < first_beat - 1e-6:
                intro_end = min(intro_start + max_duration, first_beat)
                duration = intro_end - intro_start
                if duration <= 1e-6:
                    break
                srt_lines.append(str(scene_index))
                srt_lines.append(
                    f"{format_time(intro_start)} --> {format_time(intro_end)}"
                )
                srt_lines.append(f"SCENE {scene_index}")
                srt_lines.append("")
                print(
                    "[BeatSceneDurationNode] Intro scene added: "
                    f"scene={scene_index}, start={intro_start:.3f}, end={intro_end:.3f}, duration={duration:.3f}"
                )
                scene_index += 1
                intro_start = intro_end
            current_time = first_beat
            intro_scene_added = True

        while current_index < len(beats) - 1:
            start_time = beats[current_index]["time"]
            min_time = start_time + min_duration
            max_time = start_time + max_duration

            candidates = []

            for i in range(current_index + 1, len(beats)):
                t = beats[i]["time"]

                if t < min_time:
                    continue
                if t > max_time:
                    break

                impact = beats[i]["impact"]
                downbeat = beats[i]["downbeat"]

                base_weight = impact * (1.2 if downbeat else 1.0)
                duration = t - start_time
                candidates.append((i, t, base_weight, duration))

            if not candidates:
                no_candidate_windows += 1
                # No beat landed in the allowed window; force a cut at max_time,
                # then continue from the nearest beat at/after that point.
                forced_end = min(max_time, song_end)
                if forced_end <= start_time:
                    print(
                        "[BeatSceneDurationNode] BREAK no-candidate invalid forced_end: "
                        f"scene={scene_index}, start_time={start_time:.3f}, forced_end={forced_end:.3f}"
                    )
                    break

                duration = forced_end - start_time
                forced_windows += 1
                print(
                    "[BeatSceneDurationNode] No candidates, forcing window: "
                    f"scene={scene_index}, beat_idx={current_index}, "
                    f"start_time={start_time:.3f}, min_time={min_time:.3f}, max_time={max_time:.3f}, "
                    f"forced_end={forced_end:.3f}, duration={duration:.3f}"
                )

                srt_lines.append(str(scene_index))
                srt_lines.append(
                    f"{format_time(current_time)} --> {format_time(current_time + duration)}"
                )
                srt_lines.append(f"SCENE {scene_index}")
                srt_lines.append("")

                current_time += duration
                scene_index += 1
                prev_duration = duration

                next_index = current_index + 1
                while next_index < len(beats) and beats[next_index]["time"] <= forced_end:
                    next_index += 1
                if next_index >= len(beats):
                    print(
                        "[BeatSceneDurationNode] BREAK no remaining beats after forced window: "
                        f"scene={scene_index - 1}, forced_end={forced_end:.3f}, last_beat={last_beat:.3f}"
                    )
                    break
                current_index = next_index
                continue

            filtered_candidates = candidates
            if prev_duration is not None:
                # Never pick nearly identical duration back-to-back.
                repeat_epsilon = 0.20
                non_repeat = [
                    c for c in candidates
                    if abs(c[3] - prev_duration) >= repeat_epsilon
                ]
                if non_repeat:
                    filtered_candidates = non_repeat
                else:
                    print(
                        "[BeatSceneDurationNode] Non-repeat constraint relaxed (all candidates too similar): "
                        f"scene={scene_index}, prev_duration={prev_duration:.3f}, candidates={len(candidates)}"
                    )

            weights = []
            for _, _, base_weight, candidate_duration in filtered_candidates:
                w = (base_weight ** bias) + 1e-6

                if prev_duration is not None:
                    delta = abs(candidate_duration - prev_duration)

                    if duration_preset == "varied_no_repeat":
                        # Strongly favor bigger jumps from previous duration.
                        w *= 0.6 + min(2.0, delta / 0.8)
                        mid = (min_duration + max_duration) * 0.5
                        switched_band = (
                            (prev_duration >= mid and candidate_duration < mid) or
                            (prev_duration < mid and candidate_duration >= mid)
                        )
                        w *= 1.20 if switched_band else 0.85

                    elif duration_preset == "clustered_no_repeat":
                        # Keep durations in a tighter cluster, but still non-repeating.
                        w *= 1.30 if delta <= 1.5 else 0.75

                weights.append(max(w, 1e-9))

            chosen_index, chosen_time, _, _ = rng.choices(
                filtered_candidates, weights=weights, k=1
            )[0]
            beat_aligned_windows += 1

            duration = chosen_time - start_time
            if scene_index <= 5 or scene_index % 10 == 0 or chosen_time > (song_end - 25.0):
                print(
                    "[BeatSceneDurationNode] Beat-aligned cut: "
                    f"scene={scene_index}, beat_idx={current_index}->{chosen_index}, "
                    f"start_time={start_time:.3f}, chosen_time={chosen_time:.3f}, "
                    f"duration={duration:.3f}, candidates={len(candidates)}"
                )

            srt_lines.append(str(scene_index))
            srt_lines.append(
                f"{format_time(current_time)} --> {format_time(current_time + duration)}"
            )
            srt_lines.append(f"SCENE {scene_index}")

            srt_lines.append("")  # blank line required between blocks


            current_time += duration
            scene_index += 1
            current_index = chosen_index
            prev_duration = duration

        # --- Clamp tail to max_duration and always reach song end ---
        if current_time < song_end:
            remaining = song_end - current_time

            # If tail is longer than max_duration, split into fixed chunks
            while remaining > max_duration:
                print(
                    "[BeatSceneDurationNode] Tail chunk fallback: "
                    f"scene={scene_index}, current_time={current_time:.3f}, "
                    f"remaining={remaining:.3f}, chunk={max_duration:.3f}"
                )
                srt_lines.append(str(scene_index))
                srt_lines.append(
                    f"{format_time(current_time)} --> {format_time(current_time + max_duration)}"
                )
                srt_lines.append(f"SCENE {scene_index}")
                srt_lines.append("")

                current_time += max_duration
                scene_index += 1
                remaining = song_end - current_time

            # Final tail (<= max_duration)
            if current_time < song_end:
                print(
                    "[BeatSceneDurationNode] Final tail chunk: "
                    f"scene={scene_index}, current_time={current_time:.3f}, song_end={float(song_end):.3f}"
                )
                srt_lines.append(str(scene_index))
                srt_lines.append(
                    f"{format_time(current_time)} --> {format_time(song_end)}"
                )
                srt_lines.append(f"SCENE {scene_index}")
                srt_lines.append("")


        ###############
        srt_lines = merge_short_first_scene_if_needed(srt_lines)
        scene_index = len([line for line in srt_lines if line.strip().isdigit()]) + 1

        # Save next to THIS custom node file, inside /srt_files.
        # Keep this lowercase because Linux treats SRT_Files and srt_files as different folders.
        node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Create folder if missing
        srt_dir = os.path.join(node_dir, "srt_files")
        os.makedirs(srt_dir, exist_ok=True)

        # Ensure .srt extension
        filename = output_filename.strip()
        if not filename.lower().endswith(".srt"):
            filename += ".srt"

        # Final path
        out_path = os.path.join(srt_dir, filename)

        # Write clean SRT format ONLY
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        print(
            "[BeatSceneDurationNode] Summary: "
            f"beat_aligned={beat_aligned_windows}, forced={forced_windows}, "
            f"no_candidate_windows={no_candidate_windows}, intro_scene={intro_scene_added}, total_scenes={scene_index}"
        )
        print(f"[BeatSceneDurationNode] Saved SRT to: {out_path}")


        return (
            "\n".join(srt_lines),
            out_path
        )


class IndexedImageFromFolder_ForRemakeMode:
    """
    Loads a single image from a folder by matching the filename number to (index + 1).
    Example: index 0 -> image number 1 -> files like images_00001_.png
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, folder_path, index):
        if not os.path.isdir(folder_path):
            raise Exception(f"Folder does not exist: {folder_path}")

        valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts)
        ]

        if not files:
            raise Exception(f"No images found in folder: {folder_path}")

        target_number = index + 1
        target_file = None

        for filename in files:
            match = re.search(r"\d+", filename)
            if not match:
                continue
            if int(match.group()) == target_number:
                target_file = filename
                break

        if target_file is None:
            raise Exception(
                f"No image found for index {index} (expected number {target_number}) in folder: {folder_path}"
            )

        image_path = os.path.join(folder_path, target_file)
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None, ...]
        return (image_tensor,)


class VRGDG_LatestSRTAutoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("INT", {
                    "default": 0,
                    "min": -2147483648,
                    "max": 2147483647,
                    "step": 1
                }),
                "refresh": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("srt_full_path", "srt_file_name")
    FUNCTION = "load_latest_srt"
    CATEGORY = "VRGDG"

    @staticmethod
    def _get_srt_dir():
        node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(node_dir, "srt_files")

    @staticmethod
    def _get_legacy_srt_dir():
        node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(node_dir, "SRT_Files")

    @classmethod
    def _get_latest_srt_info(cls, require_srt=True):
        srt_dir = cls._get_srt_dir()
        os.makedirs(srt_dir, exist_ok=True)

        srt_files = []
        for folder in (srt_dir, cls._get_legacy_srt_dir()):
            if not os.path.isdir(folder):
                continue
            for entry in os.scandir(folder):
                if entry.is_file() and entry.name.lower().endswith(".srt"):
                    srt_files.append((entry.path, entry.name, entry.stat().st_mtime))

        if not srt_files:
            if require_srt:
                raise Exception(f"No .srt files found in: {srt_dir}")
            return ("", "", 0)

        # Most recent by modified timestamp.
        srt_files.sort(key=lambda x: x[2], reverse=True)
        return srt_files[0]

    @classmethod
    def IS_CHANGED(cls, trigger, refresh):
        latest_path, _, latest_mtime = cls._get_latest_srt_info(require_srt=False)
        return f"{trigger}|{refresh}|{latest_path}|{latest_mtime}"

    def load_latest_srt(self, trigger, refresh):
        latest_path, latest_name, _ = self._get_latest_srt_info(require_srt=False)
        if not latest_path:
            print(f"[VRGDG_LatestSRTAutoLoader] No .srt files found in: {self._get_srt_dir()}; returning empty SRT path.")
        return (latest_path, latest_name)


def round_up_8n1(n: int) -> int:
    """Round up frame count to 8N+1 (required by some video models)."""
    n = max(1, int(n))
    return ((n - 1 + 7) // 8) * 8 + 1


class VRGDG_LoadAudioSplit_SRTOnly:
    """
    SRT-only audio splitter WITH:
      - output run folder creation/reuse
      - temp_state_dir creation
      - folder-based chunk_index (normal mode)
      - redo by 1-based prompt number
      - overwrite/backup of existing chunk outputs in redo mode
      - popup + instructions output
      - auto-queue (normal + redo + resume)
      - final-only resample to 44100 for LTX
    """

    RETURN_TYPES = (
        "DICT",     # meta
        "FLOAT",    # total_duration
        "INT",      # index
        "INT",      # Frames for LTX
        "STRING",   # start_time
        "STRING",   # end_time
        "STRING",   # instructions
        "INT",      # total_sets
        "INT",      # frames_per_scene
        "INT",      # preroll_frames (always 0)
        "DICT",     # audio_meta
        "STRING",   # output_folder
        "STRING",   # overwrite_mode
    ) + ("AUDIO",) + (any_typ,)

    RETURN_NAMES = (
        "meta",
        "total_duration",
        "index",
        "frames_for_ltx",
        "start_time",
        "end_time",
        "instructions",
        "total_sets",
        "frames_per_scene",
        "preroll_frames",
        "audio_meta",
        "output_folder",
        "overwrite_mode",
    ) + ("audio", "signal_out")

    FUNCTION = "run"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "trigger": (any_typ,),

                "srt_file": ("STRING", {"default": ""}),
                "fixed_duration": ("INT", {"default": 0, "min": 0}),
                "fps": ("INT", {"default": 24, "min": 1}),

                "folder_path": ("STRING", {
                    "multiline": False,
                    "default": "VRGDG_Video"
                }),

                "enable_auto_queue": ("BOOLEAN", {"default": True}),

                # 0 = disabled, 1..N = redo that prompt (1-based)
                "redo_prompt_number": ("INT", {"default": 0, "min": 0}),
                "use_remake_folder": ("BOOLEAN", {"default": False}),

                "overwrite_mode": (["overwrite", "backup"],),

                "tail_loss_frames": ("INT", {
                    "default": 5,
                    "min": 0
                }),
                "pre_frames": ("INT", {"default": 0, "min": 0}),


            }
        }

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------

    def _send_popup_notification(self, message: str, message_type: str = "info", title: str = "SRT Instructions"):
        try:
            PromptServer.instance.send_sync(
                "vrgdg_instructions_popup",
                {"message": message, "type": message_type, "title": title}
            )
        except Exception as e:
            print(f"[Popup] Failed: {e}")

    def _ensure_output_folder(self, base_name: str) -> str:
        """
        Creates (or reuses) a timestamped run folder under ComfyUI output dir:
          <output>/<base_name>_YYYY-MM-DD_HH-MM-SS
        Reuses the most recent run folder if it exists.
        """
        base_output = folder_paths.get_output_directory()
        base_name = (base_name or "").strip() or "VRGDG_Video"

        existing_runs = sorted(
            d for d in os.listdir(base_output)
            if d.startswith(base_name + "_")
            and os.path.isdir(os.path.join(base_output, d))
        )

        if existing_runs:
            output_folder = os.path.join(base_output, existing_runs[-1])
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_folder_name = f"{base_name}_{timestamp}"
            output_folder = os.path.join(base_output, run_folder_name)
            os.makedirs(output_folder, exist_ok=True)

        temp_state_dir = os.path.join(output_folder, "vrgdg_temp")
        os.makedirs(temp_state_dir, exist_ok=True)

        return output_folder

    def _count_index_from_folder(self, output_folder: str) -> int:
        """
        Determines the next chunk index by scanning for files like:
          video_0000_00001-audio.mp4
        Uses the SECOND 4-digit group as the internal 0-based chunk index.
        """
        if not os.path.isdir(output_folder):
            return 0

        indices = []
        for f in os.listdir(output_folder):
            # Match: <base>_<human:4>_<internal:4>_...
            m = re.match(r".*?_(\d{4})_(\d{4})", f)
            if m:
                indices.append(int(m.group(2)))

        # Filenames are 1-based, internal chunk_index is 0-based.
        # If the highest existing file is 0027, the next internal index is 28.
        return (max(indices) + 1) if indices else 0


    def _backup_or_remove_existing_chunk_outputs(self, output_folder: str, chunk_index: int, overwrite_mode: str):
        """
        For redo: locate existing outputs for this chunk index and either:
          - overwrite: delete them
          - backup: move them into output_folder/backup (keep filename)
        Matches the same naming family used by your pipeline: *_{####}_*-audio.mp4
        Uses the FIRST 4-digit group as the human-facing chunk index (1-based).
        """
        # redo uses internal 0-based, but filenames are 1-based (human)
        target_idx = f"{chunk_index + 1:04d}"

        hits = []
        for f in os.listdir(output_folder):
            if not f.endswith("-audio.mp4"):
                continue
            # Match the first 4-digit index group after the base name.
            m = re.match(r".*?_(\d{4})_", f)
            if not m:
                continue
            if m.group(1) == target_idx:
                hits.append(os.path.join(output_folder, f))

        hits = sorted(set(hits))
        if not hits:
            return

        if overwrite_mode == "overwrite":
            for path in hits:
                try:
                    os.remove(path)
                    print(f"[Redo] Removed existing: {os.path.basename(path)}")
                except Exception as e:
                    print(f"[Redo] Failed to remove {path}: {e}")
            return

        # backup mode: move to backup folder, keep filename
        backup_dir = os.path.join(output_folder, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        for path in hits:
            base = os.path.basename(path)
            dst = os.path.join(backup_dir, base)
            if os.path.exists(dst):
                root, ext = os.path.splitext(base)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst = os.path.join(backup_dir, f"{root}_{stamp}{ext}")

            try:
                os.replace(path, dst)
                print(f"[Redo] Backed up: {base} -> {os.path.basename(dst)}")
            except Exception as e:
                print(f"[Redo] Failed to backup {path}: {e}")

    def _scan_remake_folder_indices(self, remake_dir: str):
        if not os.path.isdir(remake_dir):
            return []

        indices = []
        for f in os.listdir(remake_dir):
            if os.path.isdir(os.path.join(remake_dir, f)):
                continue
            m = re.search(r"(\d+)", f)
            if not m:
                continue
            try:
                indices.append(int(m.group(1)))
            except ValueError:
                continue

        return sorted(set(indices))

    def _move_remake_files_to_backup(self, remake_dir: str, output_folder: str, chunk_index_1_based: int):
        if not os.path.isdir(remake_dir):
            return

        backup_dir = os.path.join(output_folder, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        for f in os.listdir(remake_dir):
            src = os.path.join(remake_dir, f)
            if not os.path.isfile(src):
                continue
            # Match the first 4-digit index group after the base name.
            m = re.match(r".*?_(\d{4})_", f)
            if not m:
                print(f"[Remake] Skip (no match): {f}")
                continue
            if m.group(1) != f"{chunk_index_1_based:04d}":
                print(f"[Remake] Skip (idx {m.group(1)} != {chunk_index_1_based:04d}): {f}")
                continue

            dst = os.path.join(backup_dir, f)
            print(f"[Remake] Move: {src} -> {dst}")
            if os.path.exists(dst):
                base, ext = os.path.splitext(f)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst = os.path.join(backup_dir, f"{base}_{stamp}{ext}")

            try:
                os.replace(src, dst)
            except Exception as e:
                print(f"[Remake] Failed to move {src} -> {dst}: {e}")

    def parse_srt(self, path: str):
        segments = []
        if not path.strip() or not os.path.exists(path):
            raise ValueError("SRT file not found")

        with open(path, "r", encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")

        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 2 or "-->" not in lines[1]:
                continue

            start_txt, end_txt = lines[1].split("-->")

            def to_sec(tc):
                h, m, rest = tc.strip().split(":")
                s, ms = rest.split(",")
                return (
                    int(h) * 3600 +
                    int(m) * 60 +
                    int(s) +
                    int(ms) / 1000.0
                )

            segments.append((to_sec(start_txt), to_sec(end_txt)))

        if not segments:
            raise ValueError("No valid SRT entries found")

        return segments

    # --------------------------------------------------
    # main
    # --------------------------------------------------

    def run(
        self,
        audio,
        trigger,
        srt_file,
        fixed_duration,
        fps,
        folder_path,
        enable_auto_queue,
        redo_prompt_number,
        use_remake_folder,
        overwrite_mode,
        tail_loss_frames,
        pre_frames,
    ):

        # ---- output folder creation/reuse (RESTORED) ----
        output_folder = self._ensure_output_folder(folder_path)
        temp_state_dir = os.path.join(output_folder, "vrgdg_temp")
        os.makedirs(temp_state_dir, exist_ok=True)
        # Always ensure remake folder exists for manual drop-ins.
        remake_dir = os.path.join(output_folder, "remake")
        os.makedirs(remake_dir, exist_ok=True)

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        total_duration = total_samples / sample_rate

        if fixed_duration and fixed_duration > 0:
            # Build fixed-length segments across full audio duration
            segments = []
            dur = float(fixed_duration)
            num_segments = int(math.ceil(total_duration / dur))
            for i in range(num_segments):
                start = i * dur
                end = min((i + 1) * dur, total_duration)
                segments.append((start, end))
            srt_segments = segments
        else:
            raw_segments = self.parse_srt(srt_file)

            # ✅ Force final scene to end at audio end (ignore last SRT cutoff)
            last_start, last_end = raw_segments[-1]

            if last_end < total_duration:
                raw_segments[-1] = (last_start, total_duration)

            # Build continuous timeline: (start_i, start_{i+1}), last ends at audio end
            srt_segments = raw_segments
        total_sets = len(srt_segments)

        # --------------------------------------------------
        # index selection
        # --------------------------------------------------
        remake_mode = bool(use_remake_folder)
        redo_mode = (not remake_mode) and (redo_prompt_number > 0)
        remake_indices = []
        remake_remaining_to_queue = 0

        if remake_mode:
            remake_indices = self._scan_remake_folder_indices(remake_dir)
            if not remake_indices:
                raise ValueError(
                    f"Remake folder is empty: {remake_dir}"
                )

            # filenames are 1-based, so convert to 0-based internal index
            ui_index = remake_indices[0]
            chunk_index = ui_index - 1
            if chunk_index >= total_sets:
                raise ValueError(
                    f"Remake index {ui_index} out of range (total prompts: {total_sets})"
                )

            # Remake mode should not touch existing main outputs here.
            self._move_remake_files_to_backup(remake_dir, output_folder, ui_index)

            remake_remaining_to_queue = max(0, len(remake_indices) - 1)
            remake_total = remake_remaining_to_queue + 1
            remake_pos = 1

            instructions = (
                f"🛠️ REMAKE MODE\n"
                f"Remake item {remake_pos} / {remake_total}\n"
                f"Prompt index: {ui_index} (of {total_sets})\n"
                f"Remake folder: {os.path.basename(remake_dir)}\n"
                f"Overwrite mode: {overwrite_mode}"
            )

        elif redo_mode:
            chunk_index = redo_prompt_number - 1  # 1-based -> 0-based
            if chunk_index >= total_sets:
                raise ValueError(
                    f"Redo prompt {redo_prompt_number} out of range (total prompts: {total_sets})"
                )

            # ---- redo file handling (RESTORED) ----
            self._backup_or_remove_existing_chunk_outputs(output_folder, chunk_index, overwrite_mode)

            instructions = (
                f"🔁 REDO MODE\n"
                f"Redo item 1 / 1\n"
                f"Prompt index: {redo_prompt_number} (of {total_sets})\n"
                f"(SRT-driven, frame-locked)\n"
                f"Overwrite mode: {overwrite_mode}"
            )
        else:
            chunk_index = self._count_index_from_folder(output_folder)
            ui_index = chunk_index + 1

            # normal mode should never overwrite existing chunks
            overwrite_mode = "overwrite"

            if fixed_duration and fixed_duration > 0:
                instructions = (
                    f"⏱️ Fixed duration mode\n"
                    f"{fixed_duration} seconds per group\n"
                    f"Rendering chunk {chunk_index + 1} / {total_sets}\n"
                    f"Output folder: {os.path.basename(output_folder)}"
                )
            else:
                instructions = (
                    f"🎬 SRT MODE\n"
                    f"Rendering chunk {chunk_index + 1} / {total_sets}\n"
                    f"Output folder: {os.path.basename(output_folder)}"
                )
            remake_remaining_to_queue = 0

        # --------------------------------------------------
        # popup UI (RESTORED)
        # --------------------------------------------------
        is_fixed_mode = bool(fixed_duration and fixed_duration > 0)
        if remake_mode:
            self._send_popup_notification(instructions, "pink", "🛠️ SRT REMAKE")
        elif redo_mode:
            self._send_popup_notification(instructions, "pink", "🔁 SRT REDO")
        elif chunk_index == 0:
            title = "⏱️ STARTING FIXED RENDER" if is_fixed_mode else "🎬 STARTING SRT RENDER"
            self._send_popup_notification(instructions, "info", title)
        elif chunk_index + 1 < total_sets:
            title = "⏳ FIXED CHUNK IN PROGRESS" if is_fixed_mode else "⏳ SRT CHUNK IN PROGRESS"
            self._send_popup_notification(instructions, "pink", title)
        else:
            title = "🏁 FINAL FIXED CHUNK" if is_fixed_mode else "🏁 FINAL SRT CHUNK"
            self._send_popup_notification(instructions, "green", title)

        # --------------------------------------------------
        # TIMING (FRAME-LOCKED + PRE + TAIL FRAMES)
        # --------------------------------------------------
        start_sec, end_sec = srt_segments[chunk_index]

        # ✅ Snap by frame index (avoids float rounding issues)
        start_frame = int(round(start_sec * fps))
        end_frame   = int(round(end_sec   * fps))

        # ✅ Recompute exact snapped seconds
        start_sec = start_frame / fps
        end_sec   = end_frame   / fps

        # --- exact timing diagnostics ---
        frame_ms = 1000.0 / fps

        print("\n[SPLIT-TIME] ==================")
        print(f"[SPLIT-TIME] fps               = {fps}")
        print(f"[SPLIT-TIME] 1 frame           = {frame_ms:.3f} ms")

        print(f"[SPLIT-TIME] raw SRT start_sec = {start_sec:.6f}")
        print(f"[SPLIT-TIME] raw SRT end_sec   = {end_sec:.6f}")

        print(f"[SPLIT-TIME] snapped start_fr  = {start_frame}")
        print(f"[SPLIT-TIME] snapped end_fr    = {end_frame}")

        print(f"[SPLIT-TIME] snapped start_sec = {start_frame/fps:.6f}")
        print(f"[SPLIT-TIME] snapped end_sec   = {end_frame/fps:.6f}")

        print(f"[SPLIT-TIME] start snap error  = {(start_frame/fps - start_sec)*1000:.3f} ms")
        print(f"[SPLIT-TIME] end snap error    = {(end_frame/fps - end_sec)*1000:.3f} ms")
        print("[SPLIT-TIME] ==================\n")

        frames_per_scene = max(1, end_frame - start_frame)

        PRE_FRAMES  = pre_frames
        TAIL_FRAMES = tail_loss_frames

        # First chunk can only use preroll when the SRT starts after 0.
        # Single-scene UI renders trim audio with preroll before the SRT start.
        if chunk_index == 0 and start_frame <= 0:
            PRE_FRAMES = 0

        truth_frames = frames_per_scene

        # ✅ LTX requires padded frame count (8N+1)
        base_frames_for_ltx = truth_frames + PRE_FRAMES + TAIL_FRAMES
        frames_for_ltx = round_up_8n1(base_frames_for_ltx)

        print("\n[SPLIT] ----------")
        print(f"[SPLIT] chunk_index        = {chunk_index}")
        print(f"[SPLIT] start_sec/end_sec  = {start_sec:.3f} → {end_sec:.3f}")
        print(f"[SPLIT] start_frame/end    = {start_frame} → {end_frame}")
        print(f"[SPLIT] truth_frames       = {truth_frames}")
        print(f"[SPLIT] PRE_FRAMES         = {PRE_FRAMES}")
        print(f"[SPLIT] TAIL_FRAMES        = {TAIL_FRAMES}")
        print(f"[SPLIT] base_frames_for_ltx= {base_frames_for_ltx}")
        print(f"[SPLIT] frames_for_ltx(8N+1)= {frames_for_ltx}")
        print("[SPLIT] ----------")
        samples_per_frame = sample_rate / fps

        pre_samples  = int(round(PRE_FRAMES  * samples_per_frame))

        # ✅ Start sample includes preroll offset
        start_samp = max(
            0,
            int(round(start_frame * samples_per_frame)) - pre_samples
        )

        # ✅ Slice only the natural window (truth + pre + tail)
        # Padding to exact 8N+1 is done AFTER resample
        end_samp = min(
            total_samples,
            start_samp + int(round(base_frames_for_ltx * samples_per_frame))
        )
        seg = waveform[..., start_samp:end_samp].contiguous().clone()

        # --------------------------------------------------
        # final-only resample for LTX
        # --------------------------------------------------
        target_sr = 44100
        if sample_rate != target_sr:
            B, C, T = seg.shape
            seg = seg.reshape(B * C, T)
            seg = AF.resample(seg, sample_rate, target_sr)
            seg = seg.reshape(B, C, -1)
            sample_rate = target_sr
        # --------------------------------------------------

        # Force audio length to match frames_for_ltx exactly
        # (so LTX's 8N+1 padding cannot create drift)
        # --------------------------------------------------
        desired_samples = int(round(frames_for_ltx * sample_rate / fps))

        cur_samples = seg.shape[-1]
        if cur_samples < desired_samples:
            seg = torch.nn.functional.pad(seg, (0, desired_samples - cur_samples))
        elif cur_samples > desired_samples:
            seg = seg[..., :desired_samples]

        print(f"[SPLIT] sample_rate           = {sample_rate}")
        print(f"[SPLIT] desired_samples       = {desired_samples}")
        print(f"[SPLIT] actual_samples_out    = {seg.shape[-1]}")
        print("[SPLIT] ----------\n")


        # --- audio duration error diagnostics ---
        actual_sec = seg.shape[-1] / sample_rate
        expected_sec = frames_for_ltx / fps

        print("[SPLIT-AUDIO] ==================")
        print(f"[SPLIT-AUDIO] frames_for_ltx   = {frames_for_ltx}")
        print(f"[SPLIT-AUDIO] expected_sec     = {expected_sec:.6f}")
        print(f"[SPLIT-AUDIO] actual_sec       = {actual_sec:.6f}")
        print(f"[SPLIT-AUDIO] error_ms         = {(actual_sec - expected_sec)*1000:.3f} ms")
        print("[SPLIT-AUDIO] ==================\n")

        # # ✅ Only apply sync delay after chunk 0
        # if chunk_index > 0:
        #     delay_samples = int(round(sample_rate / fps))  # 1 frame

        #     seg = torch.nn.functional.pad(seg, (delay_samples, 0))
        #     seg = seg[..., :desired_samples]

        audio_out = {
                    "waveform": seg,
                    "sample_rate": sample_rate
                }

        # --------------------------------------------------
        # auto-queue
        # --------------------------------------------------
        remaining_to_queue = 0
        if enable_auto_queue:
            if remake_mode:
                # In remake mode, auto-queue ONLY the remaining remake items.
                autoqueue_state = os.path.join(temp_state_dir, "srt_remake_autoqueue.json")
                should_queue = True
                stored_indices = []
                if os.path.exists(autoqueue_state):
                    try:
                        with open(autoqueue_state, "r", encoding="utf-8") as f:
                            state = json.load(f)
                        stored_indices = state.get("indices") or []
                    except Exception as e:
                        print(f"[Remake] Failed to read auto-queue state: {e}")

                # Only re-queue if there are new indices not seen before.
                if stored_indices:
                    new_indices = [i for i in remake_indices if i not in stored_indices]
                    if not new_indices:
                        should_queue = False
                    else:
                        stored_indices = sorted(set(stored_indices + new_indices))

                if should_queue and remake_remaining_to_queue > 0:
                    remaining_to_queue = remake_remaining_to_queue
                    try:
                        with open(autoqueue_state, "w", encoding="utf-8") as f:
                            json.dump({"indices": stored_indices or remake_indices}, f, indent=2)
                    except Exception as e:
                        print(f"[Remake] Failed to write auto-queue state: {e}")
                elif remake_remaining_to_queue == 0 and os.path.exists(autoqueue_state):
                    try:
                        os.remove(autoqueue_state)
                    except Exception as e:
                        print(f"[Remake] Failed to clear auto-queue state: {e}")
            elif redo_mode:
                # In redo mode, auto-queue whatever is left after this prompt.
                autoqueue_state = os.path.join(temp_state_dir, "srt_redo_autoqueue.json")
                should_queue = True
                if os.path.exists(autoqueue_state):
                    try:
                        with open(autoqueue_state, "r", encoding="utf-8") as f:
                            state = json.load(f)
                        if (
                            state.get("start_index") == chunk_index
                            and state.get("total_sets") == total_sets
                        ):
                            should_queue = False
                    except Exception as e:
                        print(f"[Redo] Failed to read auto-queue state: {e}")

                if should_queue:
                    remaining_to_queue = max(0, total_sets - (chunk_index + 1))
                    if remaining_to_queue > 0:
                        try:
                            with open(autoqueue_state, "w", encoding="utf-8") as f:
                                json.dump(
                                    {"start_index": chunk_index, "total_sets": total_sets},
                                    f,
                                    indent=2
                                )
                        except Exception as e:
                            print(f"[Redo] Failed to write auto-queue state: {e}")
                    elif os.path.exists(autoqueue_state):
                        try:
                            os.remove(autoqueue_state)
                        except Exception as e:
                            print(f"[Redo] Failed to clear auto-queue state: {e}")
            else:
                # Normal mode (including resume) queues whatever is left after this prompt.
                autoqueue_state = os.path.join(temp_state_dir, "srt_autoqueue.json")
                should_queue = True
                current_run = os.path.basename(output_folder)
                if os.path.exists(autoqueue_state):
                    try:
                        with open(autoqueue_state, "r", encoding="utf-8") as f:
                            state = json.load(f)
                        # If this run already auto-queued once for this output folder,
                        # do NOT auto-queue again (prevents re-queue on restart).
                        if (
                            state.get("queued_once") is True
                            and state.get("total_sets") == total_sets
                            and state.get("run_folder") == current_run
                        ):
                            should_queue = False
                        # Legacy behavior: avoid re-queue if same start/total.
                        elif (
                            state.get("start_index") == chunk_index
                            and state.get("total_sets") == total_sets
                        ):
                            should_queue = False
                    except Exception as e:
                        print(f"[AutoQueue] Failed to read auto-queue state: {e}")

                if should_queue:
                    remaining_to_queue = max(0, total_sets - (chunk_index + 1))
                    if remaining_to_queue > 0:
                        try:
                            with open(autoqueue_state, "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "start_index": chunk_index,
                                        "total_sets": total_sets,
                                        "run_folder": current_run,
                                        "queued_once": True,
                                    },
                                    f,
                                    indent=2
                                )
                        except Exception as e:
                            print(f"[AutoQueue] Failed to write auto-queue state: {e}")

        if remaining_to_queue > 0:
            for _ in range(remaining_to_queue):
                PromptServer.instance.send_sync("impact-add-queue", {})

        def fmt(sec: float) -> str:
            m = int(sec // 60)
            s = sec % 60
            return f"{m}:{s:06.3f}"

        meta = {
            "offset_seconds": start_sec,
            "sample_rate": sample_rate,
            "audio_total_duration": total_duration,
            "output_folder": output_folder,
            "chunk_index": chunk_index,
            "ui_chunk_index": chunk_index + 1,

            "total_sets": total_sets,
        }

        audio_meta = {"durations_frames": [frames_per_scene]}
        print("\n[RETURN] ==================")
        print(f"[RETURN] chunk_index      = {chunk_index}")
        print(f"[RETURN] frames_per_scene = {frames_per_scene}")
        print(f"[RETURN] PRE_FRAMES       = {PRE_FRAMES}")
        print(f"[RETURN] frames_for_ltx   = {frames_for_ltx}")
        print("[RETURN] ==================\n")

        return (
            meta,
            total_duration,
            chunk_index,
            frames_for_ltx,      # ✅ OVER-REQUESTED for LTX (includes tail)
            fmt(start_sec),
            fmt(end_sec),
            instructions,
            total_sets,
            frames_per_scene,   # ✅ TRUTH length for trim
            PRE_FRAMES,
            audio_meta,
            output_folder,
            overwrite_mode,
            audio_out,
            any_typ
        )

class VRGDG_TrimImageBatch_SRTOnly:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {}),
                "frames_per_scene": ("INT", {}),
                "pre_frames": ("INT", {}),
                "chunk_index": ("INT", {}),
                "fps": ("INT", {"default": 25, "min": 1}),  # ✅ added
            }
        }

    def run(self, images, frames_per_scene, pre_frames, chunk_index, fps):  # ✅ added fps
        total_frames = images.shape[0]

        print("[TRIM] ----------")
        print(f"[TRIM] chunk_index      = {chunk_index}")
        print(f"[TRIM] total_frames    = {total_frames}")
        print(f"[TRIM] frames_per_scene= {frames_per_scene}")
        print(f"[TRIM] pre_frames      = {pre_frames}")

        expected_min = pre_frames + frames_per_scene
        print(f"[TRIM] expected_min_frames_from_LTX = {expected_min}")

        extra = total_frames - expected_min
        print(f"[TRIM] extra_frames_after_truth = {extra}")

        if total_frames < expected_min:
            print("[TRIM] ❌ LTX returned TOO FEW frames!")

        if chunk_index == 0 and pre_frames <= 0:
            end = min(frames_per_scene, total_frames)
            print(f"[TRIM] FIRST CHUNK WITHOUT PREROLL -> slicing [0:{end}]")
            out = images[:end]
            print(f"[TRIM] output_frames = {out.shape[0]}")
            return (out,)

        start = min(pre_frames, total_frames)
        end = min(start + frames_per_scene, total_frames)

        print(f"[TRIM] slicing [{start}:{end}]")

        if end <= start:
            print("[TRIM] ⚠ EMPTY SLICE — forcing fallback")
            start = 0
            end = min(frames_per_scene, total_frames)

        out = images[start:end]

        # --- final output duration diagnostics ---
        out_frames = out.shape[0]
        out_sec = out_frames / float(fps)  # ✅ uses fps now

        print("[TRIM-TIME] ==================")
        print(f"[TRIM-TIME] output_frames = {out_frames}")
        print(f"[TRIM-TIME] output_sec    = {out_sec:.6f}")
        print(f"[TRIM-TIME] output_ms     = {out_sec*1000:.3f}")
        print("[TRIM-TIME] ==================\n")

        print(f"[TRIM] output_frames = {out.shape[0]}")
        print("[TRIM] ----------")

        return (out,)


NODE_CLASS_MAPPINGS = {
    "VRGDG_BuildVideoOutputPath_General_SRT": VRGDG_BuildVideoOutputPath_General_SRT,
    "BeatImpactAnalysisNode": BeatImpactAnalysisNode,
    "BeatSceneDurationNode": BeatSceneDurationNode,
    "IndexedImageFromFolder_ForRemakeMode": IndexedImageFromFolder_ForRemakeMode,
    "VRGDG_LatestSRTAutoLoader": VRGDG_LatestSRTAutoLoader,
    "VRGDG_LoadAudioSplit_SRTOnly": VRGDG_LoadAudioSplit_SRTOnly,
    "VRGDG_TrimImageBatch_SRTOnly": VRGDG_TrimImageBatch_SRTOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_BuildVideoOutputPath_General_SRT": "VRGDG Build Video Output Path (General_SRT)",
    "BeatImpactAnalysisNode": "BeatImpactAnalysisNode",
    "BeatSceneDurationNode": "Beat-Aligned Scene Durations",
    "IndexedImageFromFolder_ForRemakeMode": "Image From Folder (Index For Remake Mode)",
    "VRGDG_LatestSRTAutoLoader": "VRGDG Latest SRT Auto Loader",
    "VRGDG_LoadAudioSplit_SRTOnly": "VRGDG_LoadAudioSplit_SRTOnly",
    "VRGDG_TrimImageBatch_SRTOnly": "VRGDG_TrimImageBatch_SRTOnly",
}
