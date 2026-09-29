import re

import os

from datetime import datetime

from server import PromptServer

import folder_paths

import torch

import math

import subprocess

import json

import numpy as np

import json


import tempfile

from .video_preroll import add_preroll_frames

from .VRGDG_AnyType import any_typ


class VRGDG_LoadAudioSplit_General:
    # UPDATED: lyrics/context/transcription removed, auto-queue + audio split kept

    RETURN_TYPES = (
        "DICT",     # meta
        "FLOAT",    # total_duration
        "INT",      # index
        "INT",      #Frames for LTX
        "STRING",   # start_time
        "STRING",   # end_time
        "STRING",   # instructions
        "INT",      # total_sets
        "INT",      # frames_per_scene
        "INT",     #pre roll frames
        "DICT",     # audio_meta
        "STRING",   # output_folder
        "STRING",    #overwrite
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
        # UPDATED: removed context + play buttons + all lyric/transcription controls
        return {
            "required": {
                "audio": ("AUDIO",),
                "trigger": (any_typ,),
                "scene_duration_seconds": ("FLOAT",),

                "fps": ("INT", {
                    "default": 24,
                    "min": 1
                }),
                
                
                "folder_path": ("STRING", {
                    "multiline": False,
                    "default": "VRGDG_Video"
                }),
                "enable_auto_queue": ("BOOLEAN", {
                    "default": True
                }),

                "override_chunk_index": ("INT", {
                    "default": -1,
                    "min": -1
                }),
                "overwrite_mode": (["overwrite", "backup"],),

                "List_of_Scene_durations": ("FLOAT", {
                    "default": 0.0
                }),
                "manual_total_sets": ("INT", {
                    "default": 0,
                    "min": 0
                }),

            }
        }


    # ---------- helpers (single-chunk model) ----------


    def _count_index_from_folder(self, folder_path: str) -> int:
        try:
            if not os.path.isdir(folder_path):
                return 0

            indices = []

            for f in os.listdir(folder_path):
                # Match: video_0000_00001-audio.mp4 → captures FIRST 0000
                m = re.match(r".*?_(\d{4})_\d+-audio\.mp4$", f)
                if m:
                    indices.append(int(m.group(1)))

            if not indices:
                return 0

            return max(indices) + 1

        except Exception as e:
            print(f"[Index] Failed to scan folder '{folder_path}': {e}")
            return 0


    def _calculate_sets(self, audio, scene_duration_seconds, fps, enable_auto_queue=True):
        """
        Calculate total chunks and generate instructions
        for the single-chunk-per-run model.
        """

        end_time_str = "0:00"
        total_sets = 0

        try:


            if audio is None:
                return (
                    "❌ No audio provided.",
                    "0:00",
                    0,
                    0,
                    {"durations_frames": []}
                )
            
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except Exception:
            return (
                "❌ Expected audio to be a dict with 'waveform' and 'sample_rate'.",
                "0:00",
                0,
                0,
                {"durations_frames": []}
            )


        # -------------------------------------------------
        # Frame calculation (per chunk)
        # -------------------------------------------------
        frames_per_scene_raw = int(round(fps * scene_duration_seconds))
        frames_per_scene = self._adjust_frames(frames_per_scene_raw)

        print(
            f"[Frames] fps={fps}, "
            f"scene_duration={scene_duration_seconds}s, "
            f"raw_frames={frames_per_scene_raw}, "
            f"final_frames={frames_per_scene}"
        )


        # -------------------------------------------------
        # Audio duration
        # -------------------------------------------------
        num_samples = waveform.shape[-1]
        audio_duration = num_samples / sample_rate if sample_rate else 0.0

        print(
            f"[Audio] samples={num_samples}, "
            f"sample_rate={sample_rate}, "
            f"duration={audio_duration:.2f}s"
        )

        # -------------------------------------------------
        # Total chunks for entire job (CRITICAL for auto-queue)
        # Use REAL padded duration, not UI duration
        # -------------------------------------------------
        real_scene_duration = frames_per_scene / fps
        total_sets = max(1, math.ceil(audio_duration / real_scene_duration))


        print(
            f"[Chunks] total_sets={total_sets} "
            f"(audio_duration={audio_duration:.2f}s / "
            f"scene_duration={scene_duration_seconds}s)"
        )

        # -------------------------------------------------
        # End time string (informational)
        # -------------------------------------------------
        minutes = int(audio_duration // 60)
        seconds = int(audio_duration % 60)
        end_time_str = f"{minutes}:{seconds:02d}"

        # -------------------------------------------------
        # Base instructions (job-level, not per-run)
        # -------------------------------------------------
        if total_sets <= 0:
            instructions = "❌ Audio too short. No chunks required."

        elif total_sets == 1:
            instructions = (
                "✅ 1 chunk required\n"
                "🎬 Rendering single chunk"
            )

        else:
            if enable_auto_queue:
                instructions = (
                    f"⚠️  {total_sets} chunks required\n"
                    f"✅ Auto-queue enabled — remaining chunks will be queued automatically"
                )
            else:
                instructions = (
                    f"⚠️  {total_sets} chunks required\n"
                    f"🔴 Auto-queue is DISABLED\n"
                    f"❗ Manually run each chunk"
                )

        # -------------------------------------------------
        # Audio metadata (single-chunk model by design)
        # -------------------------------------------------
        audio_meta = {
            "durations_frames": [frames_per_scene]
        }

        return (
            instructions,
            end_time_str,
            total_sets,
            frames_per_scene,
            audio_meta,
        )


    def _maybe_auto_queue(self, total_sets: int, index: int, enable: bool):
        """
        Auto-queue remaining chunks.
        Single-chunk-per-run model.
        Only triggers on the very first run.
        """
        if not enable:
            return

        # Only auto-queue on the very first chunk
        if index != 0:
            return

        if total_sets <= 1:
            return

        runs = total_sets - 1
        print(f"[AutoQueue] Queuing {runs} additional chunks")

        for _ in range(runs):
            PromptServer.instance.send_sync("impact-add-queue", {})


    def _send_popup_notification(self, message: str, message_type: str = "info", title: str = "Audio Split Instructions"):
        """Same popup mechanism you already had."""
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("vrgdg_instructions_popup", {
                "message": message,
                "type": message_type,
                "title": title
            })
            print(f"[Popup] Sent {message_type} notification to UI")
        except Exception as e:
            print(f"[Popup] Could not send notification: {e}")

    def _adjust_frames(self, frames: int) -> int:
        adjusted = ((frames + 8) // 9) * 9
        if adjusted != frames:
            print(f"[Frame Align] PAD (8n+1): {frames} → {adjusted}")
        return adjusted

     

        # General video models (no alignment)
        return frames


    # --------------- main ---------------
    def run(
        self,
        audio,
        trigger,
        scene_duration_seconds,
        fps,
        List_of_Scene_durations,
        manual_total_sets,
        folder_path,
        enable_auto_queue,
        override_chunk_index,
        overwrite_mode,
    ):
        

    
        try:
            if audio is None:
                raise ValueError("audio is None")

            waveform = audio["waveform"]
            sample_rate = int(audio["sample_rate"])
        except Exception as e:
            raise ValueError(f"Invalid audio input: {e}")

        print(f"[Audio] original sample_rate: {sample_rate}")

        # Ensure batch dimension
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        # --- FORCE RESAMPLE TO 44.1kHz ---
        target_sr = 44100
        if sample_rate != target_sr:
            print(f"[Audio] Resampling {sample_rate} → {target_sr}")

            # waveform shape: [B, C, T]
            waveform = torch.nn.functional.interpolate(
                waveform,
                scale_factor=target_sr / sample_rate,
                mode="linear",
                align_corners=False
            )

            sample_rate = target_sr

        print(f"[Audio] final sample_rate: {sample_rate}")


        audio = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }        

        total_samples = waveform.shape[-1]
        total_duration = float(total_samples) / float(sample_rate)

        # -------------------------------------------------
        # Mode selection
        # -------------------------------------------------
        using_custom_durations = List_of_Scene_durations > 0

        if not using_custom_durations:
            # ---------------- FIXED-DURATION MODE ----------------
            instructions, end_time_str_hr, total_sets, frames_per_scene, audio_meta = \
                self._calculate_sets(
                    audio,
                    scene_duration_seconds,
                    fps,
                    enable_auto_queue
                )

            active_duration = scene_duration_seconds
            reported_duration = frames_per_scene / fps


        else:
            # ---------------- CUSTOM-DURATION MODE ----------------
            if manual_total_sets <= 0:
                raise ValueError(
                    "manual_total_sets must be provided when using List_of_Scene_durations"
                )

            total_sets = manual_total_sets

            # placeholder values — will be overridden later from JSON
            frames_per_scene = 0
            samples_per_scene = 0
            reported_duration = 0.0

            audio_meta = {
                "durations_frames": []
            }

            instructions = (
                f"⚠️  {total_sets} chunks required\n"
                f"🧮 Custom scene durations enabled"
            )

            end_time_str_hr = ""
            


        # -------------------------------------------------
        # Output folder creation + reuse (FINAL, CORRECT)
        # -------------------------------------------------
        from datetime import datetime

        base_output = folder_paths.get_output_directory()

        # Resolve base folder name
        base_name = folder_path.strip() if folder_path.strip() else "VRGDG_Video"

        # Reuse most recent timestamped run folder if it exists
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


        # FINAL index resolution
        if override_chunk_index >= 0:
            set_index = override_chunk_index
            enable_auto_queue = False
        else:
            set_index = self._count_index_from_folder(output_folder)
            overwrite_mode = "overwrite"   # 🔒 FORCE overwrite during normal runs


        print(f"[Index] Detected set_index={set_index} from folder: {output_folder}")
        chunk_index = set_index


        # split parameters
        samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)
        # 🔧 FIX: update reported duration PER CHUNK
        reported_duration = frames_per_scene / fps
        using_custom_durations = List_of_Scene_durations > 0

        if using_custom_durations:
            # Read full duration timeline written by VRGDG_DurationIndexFloat
            durations_path = os.path.join(
                tempfile.gettempdir(),
                "vrgdg_scene_durations.json"
            )

            if not os.path.exists(durations_path):
                raise ValueError(
                    "Custom-duration mode requires duration timeline file, "
                    "but it was not found."
                )

            with open(durations_path, "r") as f:
                durations_sec = json.load(f)

            # ✅ per-chunk duration comes ONLY from the timeline
            current_duration_sec = durations_sec[chunk_index]

            frames_per_scene_raw = int(round(fps * current_duration_sec))
            frames_per_scene = self._adjust_frames(frames_per_scene_raw)

            samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)

            # ✅ FIX: metadata MUST match the per-chunk frames
            reported_duration = frames_per_scene / fps
            audio_meta = {
                "durations_frames": [frames_per_scene]
            }

            # ✅ correct cumulative offset
            offset_sec = sum(durations_sec[:chunk_index])
            offset_samples = int(offset_sec * sample_rate + 0.5)

        else:
            # fixed-duration mode (unchanged)
            offset_samples = samples_per_scene * chunk_index


        # ---- PREROLL ADDITION (MUST BE HERE) ----

        frames_with_preroll, preroll_frames = add_preroll_frames(
            frames_per_scene,
            chunk_index,
            preroll_frames=6
        )

        # -------------------------------------------------
        # LTX tail-loss compensation (CRITICAL)
        # -------------------------------------------------
        TAIL_LOSS_FRAMES = 8  # empirically 7–8 frames per clip

        frames_for_ltx = frames_with_preroll + TAIL_LOSS_FRAMES

        # FIX: compensate audio for preroll frames
        samples_per_frame = sample_rate / fps
        preroll_samples = int(preroll_frames * samples_per_frame + 0.5)

        start_samp = max(0, offset_samples - preroll_samples)
        end_samp = start_samp + samples_per_scene


        if start_samp >= total_samples:
            seg = torch.zeros(
                (1, 2, samples_per_scene),
                dtype=waveform.dtype,
                device=waveform.device
            )
        else:
            end_samp = min(total_samples, end_samp)
            seg = waveform[..., start_samp:end_samp].contiguous().clone()

            cur_len = seg.shape[-1]
            if cur_len < samples_per_scene:
                pad = samples_per_scene - cur_len
                seg = torch.nn.functional.pad(seg, (0, pad))

        audio = {
            "waveform": seg,
            "sample_rate": sample_rate
        }


        # meta (kept)
        meta = {
            "durations": [reported_duration],
            "offset_seconds": offset_samples / sample_rate,
            "starts": [offset_samples],
            "sample_rate": sample_rate,
            "audio_total_duration": total_duration,
            "outputs_count": 1,
            "output_folder": output_folder,
        }


        # -------------------------------------------------
        # OVERRIDE SAFETY CHECK (must be AFTER total_sets is known)
        # -------------------------------------------------
        if override_chunk_index >= 0 and total_sets > 0:
            if override_chunk_index >= total_sets:
                raise ValueError(
                    f"override_chunk_index {override_chunk_index} "
                    f"is out of range (total chunks: {total_sets})"
                )

        chunk_index = set_index

        # -------------------------------------------------
        # Popup behavior + override clarity + auto-queue safety
        # (single-chunk-per-run model)
        # -------------------------------------------------

        # Prefix instructions with clear chunk context
        if override_chunk_index >= 0:
            prefix = (
                f"🔁 Re-rendering chunk {chunk_index + 1} / {total_sets}\n"
                f"⚠️ OVERRIDE MODE — manual re-render\n\n"
            )
        else:
            prefix = f"🎬 Rendering chunk {chunk_index + 1} / {total_sets}\n\n"

        instructions = prefix + instructions
        popup_message = instructions

        # --- popup notifications (no group logic, no special cases) ---
        if set_index == 0:
            self._send_popup_notification(
                popup_message,
                "info",
                "🎬 STARTING AUDIO SPLIT"
            )

        elif set_index + 1 < total_sets:
            self._send_popup_notification(
                popup_message,
                "yellow",
                "⏳ CHUNK IN PROGRESS"
            )

        elif set_index + 1 == total_sets:
            self._send_popup_notification(
                popup_message,
                "green",
                "🏁 FINAL CHUNK"
            )

        # --- HARD SAFETY RULE ---
        # Auto-queue ONLY in normal mode (never during override)
        if override_chunk_index < 0:
            self._maybe_auto_queue(total_sets, set_index, enable_auto_queue)
        else:
            print("[AutoQueue] Override mode active — auto-queue suppressed.")


        
        # start/end time strings for this set
        actual_scene_duration = frames_per_scene / fps
        start_sec = offset_samples / sample_rate
        end_sec = start_sec + actual_scene_duration

        # default duration (all non-final chunks)
        reported_duration = actual_scene_duration 
        

        # Clamp ONLY the final chunk
        if set_index == total_sets - 1:
            end_sec = min(end_sec, total_duration)
            reported_duration = end_sec - start_sec


        def fmt_time(sec):
            m = int(sec // 60)
            s = sec % 60
            return f"{m}:{s:06.3f}"

        start_time_str = fmt_time(start_sec)
        end_time_str = fmt_time(end_sec)

        # outputs (lyrics removed!)
        return (
            meta,
            total_duration,
            set_index,
            frames_for_ltx,       # <-- LTX gets OVER-GENERATED frames
            start_time_str,
            end_time_str,
            instructions,
            total_sets,
            frames_per_scene,     # <-- audio + timing truth
            preroll_frames,            
            audio_meta,
            output_folder,
            overwrite_mode,
            audio,
            any_typ
        )


class VRGDG_BuildVideoOutputPath_General:
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

        # Build canonical filename
        filename = f"{base_name}_{chunk_index:04d}"
        output_path = os.path.join(output_folder, filename)


        # Handle backup mode
        if overwrite_mode == "backup":
            backup_dir = os.path.join(output_folder, "backup")
            os.makedirs(backup_dir, exist_ok=True)

            prefix = f"{base_name}_{chunk_index:04d}"
            for f in os.listdir(output_folder):
                if f.startswith(prefix) and f.endswith(".mp4"):
                    src = os.path.join(output_folder, f)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dst = os.path.join(backup_dir, f"{f}.{timestamp}.bak")
                    os.replace(src, dst)


        # In overwrite mode, Video Combine will overwrite naturally
        return (output_path,)


class VRGDG_TrimFinalClip:
    """
    Conditionally trims the final padded video clip.
    Runs ONLY when index == total_sets - 1.
    Triggered by VHS_FILENAMES to ensure Video Combine finished.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_clip_path",)
    FUNCTION = "run"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("VHS_FILENAMES", {}),   # execution gate ONLY
                "output_folder": ("STRING", {}),
                "base_name": ("STRING", {"default": "video"}),
                "frames_per_scene": ("INT", {}),
                "audio_total_duration": ("FLOAT", {}),
                "index": ("INT", {}),
                "total_sets": ("INT", {}),
                "fps": ("INT", {"default": 24}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    def run(
        self,
        trigger,                # unused, gates execution timing
        output_folder,
        base_name,
        frames_per_scene,
        audio_total_duration,
        index,
        total_sets,
        fps,
        overwrite,
    ):
        # -------------------------------------------------
        # CONDITIONAL: only run on final chunk
        # -------------------------------------------------
        if index != total_sets - 1:
            return ("",)

        # -------------------------------------------------
        # Find last chunk file (Video Combine already ran)
        # -------------------------------------------------
        files = [
            f for f in os.listdir(output_folder)
            if f.startswith(base_name + "_") and f.endswith(".mp4")
        ]

        if not files:
            return ("",)

        last_clip = max(
            files,
            key=lambda f: int(re.search(rf"{re.escape(base_name)}_(\d{{4}})", f).group(1))
        )
        last_clip = os.path.join(output_folder, last_clip)

        # -------------------------------------------------
        # Trim math (USE LOGICAL INDEX, NOT FILENAME INDEX)
        # -------------------------------------------------
        scene_duration_seconds = frames_per_scene / fps
        expected_start = index * scene_duration_seconds
        remaining_duration = audio_total_duration - expected_start

        if remaining_duration <= 0:
            return (last_clip,)

        trim_seconds = remaining_duration

        # -------------------------------------------------
        # FFmpeg safe trim (no in-place overwrite)
        # -------------------------------------------------
        final_path = last_clip
        if not overwrite:
            final_path = os.path.join(
                output_folder,
                f"{base_name}_{index:04d}_trimmed.mp4"
            )

        temp_path = final_path + ".tmp.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", last_clip,
            "-t", f"{trim_seconds:.6f}",
            "-c", "copy",
            temp_path,
        ]

        subprocess.run(cmd, check=True)
        os.replace(temp_path, final_path)

        return (final_path,)


class VRGDG_PromptSplitter_General:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "[]"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    def split_prompt(self, json_string, index, **kwargs):
        try:
            data = json.loads(json_string)

            # Extract prompts in order
            prompts = []
            if isinstance(data, dict):
                sorted_keys = sorted(
                    data.keys(),
                    key=lambda x: int(''.join(filter(str.isdigit, x)))
                    if any(c.isdigit() for c in x) else 0
                )
                prompts = [data[key] for key in sorted_keys]
            elif isinstance(data, list):
                prompts = data

            if not prompts:
                return ("",)

            # Cycle through prompts
            selected_prompt = prompts[index % len(prompts)]

            return (selected_prompt,)

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON - {str(e)}")
            return ("",)
        except Exception as e:
            print(f"Error loading prompts: {str(e)}")
            return ("",)


class VRGDG_PadVideoWithLastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pad_frames": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1000,
                    "step": 1
                }),
                "pad_front": ("BOOLEAN", {   # ← ADD
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "pad_video"
    CATEGORY = "video/utils"

    def pad_video(self, images, pad_frames, pad_front):

        if images.shape[0] == 0 or pad_frames <= 0:
            return (images,)

        if pad_front:
            # Use FIRST frame for preroll
            frame = images[:1].clone()
        else:
            # Use LAST frame for tail padding
            frame = images[-1:].clone()

        padded_frames = frame.repeat(pad_frames, 1, 1, 1)

        if pad_front:
            output = torch.cat([padded_frames, images], dim=0)
        else:
            output = torch.cat([images, padded_frames], dim=0)

        return (output,)


import tempfile

class VRGDG_DurationIndexFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "durations_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    # FIX THESE LINES
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("duration", "num_scenes")
    FUNCTION = "run"
    CATEGORY = "audio"

    def run(self, durations_text, index):
        # accept commas, newlines, or spaces
        raw = durations_text.replace("\n", ",").replace(" ", ",")
        parts = [p for p in raw.split(",") if p.strip()]

        if not parts:
            return (0.0, 0)

        idx = max(0, min(index, len(parts) - 1))

        try:
            value = float(parts[idx])
        except ValueError:
            value = 0.0

        # FIX: persist full duration list so downstream audio node
        # can compute correct cumulative offsets
        durations_sec = []
        for p in parts:
            try:
                durations_sec.append(float(p))
            except ValueError:
                durations_sec.append(0.0)

        temp_path = os.path.join(
            tempfile.gettempdir(),
            "vrgdg_scene_durations.json"
        )

        with open(temp_path, "w") as f:
            json.dump(durations_sec, f, indent=2)

        return (value, len(durations_sec))


class VRGDG_TrimImageBatch:
    """
    Trims an IMAGE batch to an exact frame count.
    Removes:
      - preroll frames at the FRONT (only when chunk_index > 0)
      - LTX tail-loss frames at the BACK (always)
    Designed to run BEFORE Video Combine.
    """

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
                "preroll_frames": ("INT", {}),
                "chunk_index": ("INT", {}),
            }
        }

    def run(self, images, frames_per_scene, preroll_frames, chunk_index):
        # images shape: [frames, H, W, C]
        total_frames = images.shape[0]

        # MUST match the generator's TAIL_LOSS_FRAMES
        TAIL_LOSS_FRAMES = 6

        # Trim preroll ONLY for non-first chunks
        start = preroll_frames if chunk_index > 0 else 0

        # Only trim tail-loss if preroll was added
        effective_tail_loss = TAIL_LOSS_FRAMES if chunk_index > 0 else 0


        # Keep exactly frames_per_scene frames for the "real" scene
        desired_end = start + frames_per_scene

        # Always remove tail-loss frames from the back
        max_end = total_frames - effective_tail_loss
        if max_end < 0:
            max_end = 0

        end = min(desired_end, max_end)

        # Safety clamps
        if start < 0:
            start = 0
        if end < start:
            end = start
        if start > total_frames:
            start = total_frames
        if end > total_frames:
            end = total_frames

        trimmed = images[start:end]
        return (trimmed,)


from PIL import Image

class IndexedImageFromFolder:
    """
    Loads a single image from a folder based on an index.
    Images are sorted numerically by the numbers found in filenames.
    Loops safely, with optional random mode after the end.
    Random mode prevents reuse until 2 other images are shown.
    """

    # Persistent random history (class-level)
    random_history = []

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
                "random_after_end": ("BOOLEAN", {
                    "default": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, folder_path, index, random_after_end):

        # Validate folder
        if not os.path.isdir(folder_path):
            raise Exception(f"Folder does not exist: {folder_path}")

        # Supported image extensions
        valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

        # Collect image files
        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts)
        ]

        if not files:
            raise Exception(f"No images found in folder: {folder_path}")

        # Extract first number found in filename (used for sorting)
        def extract_number(filename):
            match = re.search(r"\d+", filename)
            return int(match.group()) if match else float("inf")

        # Sort files numerically
        files.sort(key=extract_number)

        # Random mode after reaching the end
        if random_after_end and index >= len(files):
            import random

            choices = list(range(len(files)))

            # Remove last 2 used images
            for prev in self.__class__.random_history:
                if prev in choices and len(choices) > 2:
                    choices.remove(prev)

            index = random.choice(choices)

            # Store index in history
            self.__class__.random_history.append(index)

            # Keep only last 2 picks
            if len(self.__class__.random_history) > 2:
                self.__class__.random_history.pop(0)

        else:
            # Normal looping
            index = index % len(files)

        # Load selected image
        image_path = os.path.join(folder_path, files[index])
        image = Image.open(image_path).convert("RGB")

        # Convert to ComfyUI IMAGE format
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None, ...]

        return (image_tensor,)


class VRGDG_PromptSplitterWithIndex:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_output", "image_index")
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "[]"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    def _normalize_image_index(self, value):
        if value is None:
            return "0"
        if isinstance(value, list):
            parts = []
            for v in value:
                try:
                    parts.append(str(int(v)))
                except Exception:
                    continue
            return ",".join(parts) if parts else "0"
        try:
            return str(int(value))
        except Exception:
            s = str(value).strip()
            return s if s else "0"

    def split_prompt(self, json_string, index, **kwargs):
        try:
            data = json.loads(json_string)

            prompts = []
            if isinstance(data, dict):
                sorted_keys = sorted(
                    data.keys(),
                    key=lambda x: int(''.join(filter(str.isdigit, x)))
                    if any(c.isdigit() for c in x) else 0
                )
                prompts = [data[key] for key in sorted_keys]
            elif isinstance(data, list):
                prompts = data

            if not prompts:
                return ("", "0")

            selected_prompt = prompts[index % len(prompts)]

            # New format: {"text": "...", "imageIndex": [1,2]}
            if isinstance(selected_prompt, dict):
                text = selected_prompt.get("text", "")
                image_index = self._normalize_image_index(selected_prompt.get("imageIndex"))
                return (text, image_index)

            # Old format: plain string or other scalar
            return (str(selected_prompt), "0")

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON - {str(e)}")
            return ("", "0")
        except Exception as e:
            print(f"Error loading prompts: {str(e)}")
            return ("", "0")


NODE_CLASS_MAPPINGS = {
    "VRGDG_LoadAudioSplit_General": VRGDG_LoadAudioSplit_General,
    "VRGDG_BuildVideoOutputPath_General": VRGDG_BuildVideoOutputPath_General,
    "VRGDG_TrimFinalClip": VRGDG_TrimFinalClip,
    "VRGDG_PromptSplitter_General": VRGDG_PromptSplitter_General,
    "VRGDG_PadVideoWithLastFrame": VRGDG_PadVideoWithLastFrame,
    "VRGDG_DurationIndexFloat": VRGDG_DurationIndexFloat,
    "VRGDG_TrimImageBatch": VRGDG_TrimImageBatch,
    "IndexedImageFromFolder": IndexedImageFromFolder,
    "VRGDG_PromptSpitterWithIndex": VRGDG_PromptSplitterWithIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LoadAudioSplit_General": "VRGDG Load Audio Split (General)",
    "VRGDG_BuildVideoOutputPath_General": "VRGDG Build Video Output Path (General)",
    "VRGDG_TrimFinalClip": "VRGDG_TrimFinalClip",
    "VRGDG_PromptSplitter_General": "VRGDG_PromptSplitter_General",
    "VRGDG_PadVideoWithLastFrame": "VRGDG_PadVideoWithLastFrame",
    "VRGDG_DurationIndexFloat": "VRGDG_DurationIndexFloat",
    "VRGDG_TrimImageBatch": "VRGDG_TrimImageBatch",
    "IndexedImageFromFolder": "Image From Folder (Index)",
}
