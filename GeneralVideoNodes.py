import re
import os
from datetime import datetime
from server import PromptServer
import folder_paths
import torch
import math
import subprocess
import json
import comfy.model_management
import numpy as np
import comfy.samplers
import cv2
import librosa
import json
import random
import tempfile
from .video_preroll import add_preroll_frames





# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


# class VRGDG_LoadAudioSplit_General:
#     # UPDATED: lyrics/context/transcription removed, auto-queue + audio split kept

#     RETURN_TYPES = (
#         "DICT",     # meta
#         "FLOAT",    # total_duration
#         "INT",      # index
#         "STRING",   # start_time
#         "STRING",   # end_time
#         "STRING",   # instructions
#         "INT",      # total_sets
#         "INT",      # frames_per_scene
#         "DICT",     # audio_meta
#         "STRING",   # output_folder
#         "STRING",    #overwrite
#     ) + ("AUDIO",) + (any_typ,)
    
#     RETURN_NAMES = (
#         "meta",
#         "total_duration",
#         "index",
#         "start_time",
#         "end_time",
#         "instructions",
#         "total_sets",
#         "frames_per_scene",
#         "audio_meta",
#         "output_folder",
#         "overwrite_mode",
#     ) + ("audio", "signal_out")

#     FUNCTION = "run"
#     CATEGORY = "VRGDG"

#     @classmethod
#     def INPUT_TYPES(cls):
#         # UPDATED: removed context + play buttons + all lyric/transcription controls
#         return {
#             "required": {
#                 "audio": ("AUDIO",),
#                 "trigger": (any_typ,),
#                 "scene_duration_seconds": ("FLOAT",),
                
#                 "folder_path": ("STRING", {
#                     "multiline": False,
#                     "default": "VRGDG_Video"
#                 }),
#                 "enable_auto_queue": ("BOOLEAN", {
#                     "default": True
#                 }),

#                 # 🆕 STEP 2 additions
#                 "override_chunk_index": ("INT", {
#                     "default": -1,
#                     "min": -1
#                 }),
#                 "overwrite_mode": (["overwrite", "backup"],),

#                 "use_humo_alignment": ("BOOLEAN", {
#                     "default": False
#                 }),

#             }
#         }


#     # ---------- helpers (single-chunk model) ----------


#     def _count_index_from_folder(self, folder_path: str) -> int:
#         try:
#             if not os.path.isdir(folder_path):
#                 return 0

#             indices = []

#             for f in os.listdir(folder_path):
#                 # Match: video_0000_00001-audio.mp4 → captures FIRST 0000
#                 m = re.match(r".*?_(\d{4})_\d+-audio\.mp4$", f)
#                 if m:
#                     indices.append(int(m.group(1)))

#             if not indices:
#                 return 0

#             return max(indices) + 1

#         except Exception as e:
#             print(f"[Index] Failed to scan folder '{folder_path}': {e}")
#             return 0





#     def _calculate_sets(self, audio, scene_duration_seconds, enable_auto_queue=True,use_humo_alignment=False,):
#         """
#         Calculate total chunks and generate instructions
#         for the single-chunk-per-run model.
#         """

#         instructions = ""
#         end_time_str = "0:00"
#         total_sets = 0

#         try:


#             if audio is None:
#                 return (
#                     "❌ No audio provided.",
#                     "0:00",
#                     0,
#                     0,
#                     {"durations_frames": []}
#                 )
            
#             waveform = audio["waveform"]
#             sample_rate = audio["sample_rate"]
#         except Exception:
#             return (
#                 "❌ Expected audio to be a dict with 'waveform' and 'sample_rate'.",
#                 "0:00",
#                 0,
#                 0,
#                 {"durations_frames": []}
#             )

#         fps = 24  ##### this needs to be a input paramter!!! not hardcoded.

#         # -------------------------------------------------
#         # Frame calculation (per chunk)
#         # -------------------------------------------------
#         frames_per_scene_raw = int(round(fps * scene_duration_seconds))
#         frames_per_scene = self._adjust_frames(frames_per_scene_raw, use_humo_alignment)

#         print(
#             f"[Frames] fps={fps}, "
#             f"scene_duration={scene_duration_seconds}s, "
#             f"raw_frames={frames_per_scene_raw}, "
#             f"final_frames={frames_per_scene}, "
#             f"humo_alignment={use_humo_alignment}"
#         )

#         # -------------------------------------------------
#         # Audio duration
#         # -------------------------------------------------
#         num_samples = waveform.shape[-1]
#         audio_duration = num_samples / sample_rate if sample_rate else 0.0

#         print(
#             f"[Audio] samples={num_samples}, "
#             f"sample_rate={sample_rate}, "
#             f"duration={audio_duration:.2f}s"
#         )

#         # -------------------------------------------------
#         # Total chunks for entire job (CRITICAL for auto-queue)
#         # -------------------------------------------------
#         total_sets = max(1, math.ceil(audio_duration / scene_duration_seconds))

#         print(
#             f"[Chunks] total_sets={total_sets} "
#             f"(audio_duration={audio_duration:.2f}s / "
#             f"scene_duration={scene_duration_seconds}s)"
#         )

#         # -------------------------------------------------
#         # End time string (informational)
#         # -------------------------------------------------
#         minutes = int(audio_duration // 60)
#         seconds = int(audio_duration % 60)
#         end_time_str = f"{minutes}:{seconds:02d}"

#         # -------------------------------------------------
#         # Base instructions (job-level, not per-run)
#         # -------------------------------------------------
#         if total_sets <= 0:
#             instructions = "❌ Audio too short. No chunks required."

#         elif total_sets == 1:
#             instructions = (
#                 "✅ 1 chunk required\n"
#                 "🎬 Rendering single chunk"
#             )

#         else:
#             if enable_auto_queue:
#                 instructions = (
#                     f"⚠️  {total_sets} chunks required\n"
#                     f"✅ Auto-queue enabled — remaining chunks will be queued automatically"
#                 )
#             else:
#                 instructions = (
#                     f"⚠️  {total_sets} chunks required\n"
#                     f"🔴 Auto-queue is DISABLED\n"
#                     f"❗ Manually run each chunk"
#                 )

#         # -------------------------------------------------
#         # Audio metadata (single-chunk model by design)
#         # -------------------------------------------------
#         audio_meta = {
#             "durations_frames": [frames_per_scene]
#         }

#         return (
#             instructions,
#             end_time_str,
#             total_sets,
#             frames_per_scene,
#             audio_meta,
#         )


#     def _maybe_auto_queue(self, total_sets: int, index: int, enable: bool):
#         """
#         Auto-queue remaining chunks.
#         Single-chunk-per-run model.
#         Only triggers on the very first run.
#         """
#         if not enable:
#             return

#         # Only auto-queue on the very first chunk
#         if index != 0:
#             return

#         if total_sets <= 1:
#             return

#         runs = total_sets - 1
#         print(f"[AutoQueue] Queuing {runs} additional chunks")

#         for _ in range(runs):
#             PromptServer.instance.send_sync("impact-add-queue", {})


#     def _send_popup_notification(self, message: str, message_type: str = "info", title: str = "Audio Split Instructions"):
#         """Same popup mechanism you already had."""
#         try:
#             from server import PromptServer
#             PromptServer.instance.send_sync("vrgdg_instructions_popup", {
#                 "message": message,
#                 "type": message_type,
#                 "title": title
#             })
#             print(f"[Popup] Sent {message_type} notification to UI")
#         except Exception as e:
#             print(f"[Popup] Could not send notification: {e}")

#     def _adjust_frames(self, frames: int, use_humo_alignment: bool) -> int:
#         """
#         Optionally apply HuMo frame alignment (4n + 1).
#         """
#         if use_humo_alignment:
#             adjusted = 4 * ((frames + 2) // 4) + 1
#             if adjusted != frames:
#                 actual_duration = adjusted / 25
#                 print(f"[HuMo Adjust] {frames} → {adjusted} frames ({actual_duration:.2f}s)")
#             return adjusted

#         # General video models (no alignment)
#         return frames

#     # --------------- main ---------------
#     def run(
#         self,
#         audio,
#         trigger,
#         folder_path,
#         enable_auto_queue,
#         override_chunk_index,
#         overwrite_mode,    # NOTE: overwrite_mode is handled by the save/encode node, not this split node
#         use_humo_alignment,
#         scene_duration_seconds=4.0,
#     ):
#         try:
#             if audio is None:
#                 raise ValueError("audio is None")

#             waveform = audio["waveform"]
#             sample_rate = int(audio["sample_rate"])
#         except Exception as e:
#             raise ValueError(f"Invalid audio input: {e}")

#         print(f"[Audio] sample_rate: {sample_rate}")


#         if waveform.ndim == 2:
#             waveform = waveform.unsqueeze(0)

            

#         total_samples = waveform.shape[-1]
#         total_duration = float(total_samples) / float(sample_rate)

#         # -------------------------------------------------
#         # Output folder creation + reuse (FINAL, CORRECT)
#         # -------------------------------------------------
#         from datetime import datetime

#         base_output = folder_paths.get_output_directory()

#         # Resolve base folder name
#         base_name = folder_path.strip() if folder_path.strip() else "VRGDG_Video"

#         # Reuse most recent timestamped run folder if it exists
#         existing_runs = sorted(
#             d for d in os.listdir(base_output)
#             if d.startswith(base_name + "_")
#             and os.path.isdir(os.path.join(base_output, d))
#         )

#         if existing_runs:
#             output_folder = os.path.join(base_output, existing_runs[-1])
#         else:
#             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             run_folder_name = f"{base_name}_{timestamp}"
#             output_folder = os.path.join(base_output, run_folder_name)
#             os.makedirs(output_folder, exist_ok=True)


#         # FINAL index resolution
#         if override_chunk_index >= 0:
#             set_index = override_chunk_index
#             enable_auto_queue = False
#         else:
#             set_index = self._count_index_from_folder(output_folder)
#             overwrite_mode = "overwrite"   # 🔒 FORCE overwrite during normal runs


#         print(f"[Index] Detected set_index={set_index} from folder: {output_folder}")
#         chunk_index = set_index



#         # calculate sets (gets frames_per_scene)
#         instructions, end_time_str_hr, total_sets, frames_per_scene, audio_meta = \
#             self._calculate_sets(audio, scene_duration_seconds, enable_auto_queue, use_humo_alignment)

#         # split parameters (use frames_per_scene from _calculate_sets)
#         fps = 25
#         samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)
#         offset_samples = chunk_index * samples_per_scene



#         start_samp = offset_samples
#         end_samp = start_samp + samples_per_scene

#         if start_samp >= total_samples:
#             seg = torch.zeros(
#                 (1, 2, samples_per_scene),
#                 dtype=waveform.dtype,
#                 device=waveform.device
#             )
#         else:
#             end_samp = min(total_samples, end_samp)
#             seg = waveform[..., start_samp:end_samp].contiguous().clone()

#             cur_len = seg.shape[-1]
#             if cur_len < samples_per_scene:
#                 pad = samples_per_scene - cur_len
#                 seg = torch.nn.functional.pad(seg, (0, pad))

#         audio = {
#             "waveform": seg,
#             "sample_rate": sample_rate
#         }


#         # meta (kept)
#         meta = {
#             "durations": [frames_per_scene / fps],
#             "offset_seconds": chunk_index * (frames_per_scene / fps),
#             "starts": [offset_samples],
#             "sample_rate": sample_rate,
#             "audio_total_duration": total_duration,
#             "outputs_count": 1,
#             "output_folder": output_folder,
#         }


#         # -------------------------------------------------
#         # OVERRIDE SAFETY CHECK (must be AFTER total_sets is known)
#         # -------------------------------------------------
#         if override_chunk_index >= 0 and total_sets > 0:
#             if override_chunk_index >= total_sets:
#                 raise ValueError(
#                     f"override_chunk_index {override_chunk_index} "
#                     f"is out of range (total chunks: {total_sets})"
#                 )

#         chunk_index = set_index

#         # -------------------------------------------------
#         # Popup behavior + override clarity + auto-queue safety
#         # (single-chunk-per-run model)
#         # -------------------------------------------------

#         # Prefix instructions with clear chunk context
#         if override_chunk_index >= 0:
#             prefix = (
#                 f"🔁 Re-rendering chunk {chunk_index + 1} / {total_sets}\n"
#                 f"⚠️ OVERRIDE MODE — manual re-render\n\n"
#             )
#         else:
#             prefix = f"🎬 Rendering chunk {chunk_index + 1} / {total_sets}\n\n"

#         instructions = prefix + instructions
#         popup_message = instructions

#         # --- popup notifications (no group logic, no special cases) ---
#         if set_index == 0:
#             self._send_popup_notification(
#                 popup_message,
#                 "info",
#                 "🎬 STARTING AUDIO SPLIT"
#             )

#         elif set_index + 1 < total_sets:
#             self._send_popup_notification(
#                 popup_message,
#                 "yellow",
#                 "⏳ CHUNK IN PROGRESS"
#             )

#         elif set_index + 1 == total_sets:
#             self._send_popup_notification(
#                 popup_message,
#                 "green",
#                 "🏁 FINAL CHUNK"
#             )

#         # --- HARD SAFETY RULE ---
#         # Auto-queue ONLY in normal mode (never during override)
#         if override_chunk_index < 0:
#             self._maybe_auto_queue(total_sets, set_index, enable_auto_queue)
#         else:
#             print("[AutoQueue] Override mode active — auto-queue suppressed.")


        

#         # start/end time strings for this set
#         actual_scene_duration = frames_per_scene / fps
#         set_duration_sec = actual_scene_duration
#         start_sec = set_index * set_duration_sec
#         end_sec = min(start_sec + set_duration_sec, total_duration)

#         def fmt_time(sec):
#             m = int(sec // 60)
#             s = sec % 60
#             return f"{m}:{s:06.3f}"

#         start_time_str = fmt_time(start_sec)
#         end_time_str = fmt_time(end_sec)

#         # outputs (lyrics removed!)
#         return (
#             meta,
#             total_duration,
#             set_index,
#             start_time_str,
#             end_time_str,
#             instructions,
#             total_sets,
#             frames_per_scene,
#             audio_meta,
#             output_folder,
#             overwrite_mode,
#             audio,
#             any_typ
#         )
    

# class VRGDG_LoadAudioSplit_General:
#     # UPDATED: lyrics/context/transcription removed, auto-queue + audio split kept

#     RETURN_TYPES = (
#         "DICT",     # meta
#         "FLOAT",    # total_duration
#         "INT",      # index
#         "STRING",   # start_time
#         "STRING",   # end_time
#         "STRING",   # instructions
#         "INT",      # total_sets
#         "INT",      # frames_per_scene
#         "DICT",     # audio_meta
#         "STRING",   # output_folder
#         "STRING",    #overwrite
#     ) + ("AUDIO",) + (any_typ,)
    
#     RETURN_NAMES = (
#         "meta",
#         "total_duration",
#         "index",
#         "start_time",
#         "end_time",
#         "instructions",
#         "total_sets",
#         "frames_per_scene",
#         "audio_meta",
#         "output_folder",
#         "overwrite_mode",
#     ) + ("audio", "signal_out")

#     FUNCTION = "run"
#     CATEGORY = "VRGDG"

#     @classmethod
#     def INPUT_TYPES(cls):
#         # UPDATED: removed context + play buttons + all lyric/transcription controls
#         return {
#             "required": {
#                 "audio": ("AUDIO",),
#                 "trigger": (any_typ,),
#                 "scene_duration_seconds": ("FLOAT",),

#                 "fps": ("INT", {
#                     "default": 24,
#                     "min": 1
#                 }),
                
                
#                 "folder_path": ("STRING", {
#                     "multiline": False,
#                     "default": "VRGDG_Video"
#                 }),
#                 "enable_auto_queue": ("BOOLEAN", {
#                     "default": True
#                 }),

#                 # 🆕 STEP 2 additions
#                 "override_chunk_index": ("INT", {
#                     "default": -1,
#                     "min": -1
#                 }),
#                 "overwrite_mode": (["overwrite", "backup"],),

#                 "use_humo_alignment": ("BOOLEAN", {
#                     "default": False
#                 }),

#             }
#         }


#     # ---------- helpers (single-chunk model) ----------


#     def _count_index_from_folder(self, folder_path: str) -> int:
#         try:
#             if not os.path.isdir(folder_path):
#                 return 0

#             indices = []

#             for f in os.listdir(folder_path):
#                 # Match: video_0000_00001-audio.mp4 → captures FIRST 0000
#                 m = re.match(r".*?_(\d{4})_\d+-audio\.mp4$", f)
#                 if m:
#                     indices.append(int(m.group(1)))

#             if not indices:
#                 return 0

#             return max(indices) + 1

#         except Exception as e:
#             print(f"[Index] Failed to scan folder '{folder_path}': {e}")
#             return 0





#     def _calculate_sets(self, audio, scene_duration_seconds, fps, enable_auto_queue=True, use_humo_alignment=False):
#         """
#         Calculate total chunks and generate instructions
#         for the single-chunk-per-run model.
#         """

#         instructions = ""
#         end_time_str = "0:00"
#         total_sets = 0

#         try:


#             if audio is None:
#                 return (
#                     "❌ No audio provided.",
#                     "0:00",
#                     0,
#                     0,
#                     {"durations_frames": []}
#                 )
            
#             waveform = audio["waveform"]
#             sample_rate = audio["sample_rate"]
#         except Exception:
#             return (
#                 "❌ Expected audio to be a dict with 'waveform' and 'sample_rate'.",
#                 "0:00",
#                 0,
#                 0,
#                 {"durations_frames": []}
#             )



#         # -------------------------------------------------
#         # Frame calculation (per chunk)
#         # -------------------------------------------------
#         frames_per_scene_raw = int(round(fps * scene_duration_seconds))
#         frames_per_scene = self._adjust_frames(frames_per_scene_raw, fps, use_humo_alignment)

#         print(
#             f"[Frames] fps={fps}, "
#             f"scene_duration={scene_duration_seconds}s, "
#             f"raw_frames={frames_per_scene_raw}, "
#             f"final_frames={frames_per_scene}, "
#             f"humo_alignment={use_humo_alignment}"
#         )

#         # -------------------------------------------------
#         # Audio duration
#         # -------------------------------------------------
#         num_samples = waveform.shape[-1]
#         audio_duration = num_samples / sample_rate if sample_rate else 0.0

#         print(
#             f"[Audio] samples={num_samples}, "
#             f"sample_rate={sample_rate}, "
#             f"duration={audio_duration:.2f}s"
#         )

#         # -------------------------------------------------
#         # Total chunks for entire job (CRITICAL for auto-queue)
#         # -------------------------------------------------
#         total_sets = max(1, math.ceil(audio_duration / scene_duration_seconds))

#         print(
#             f"[Chunks] total_sets={total_sets} "
#             f"(audio_duration={audio_duration:.2f}s / "
#             f"scene_duration={scene_duration_seconds}s)"
#         )

#         # -------------------------------------------------
#         # End time string (informational)
#         # -------------------------------------------------
#         minutes = int(audio_duration // 60)
#         seconds = int(audio_duration % 60)
#         end_time_str = f"{minutes}:{seconds:02d}"

#         # -------------------------------------------------
#         # Base instructions (job-level, not per-run)
#         # -------------------------------------------------
#         if total_sets <= 0:
#             instructions = "❌ Audio too short. No chunks required."

#         elif total_sets == 1:
#             instructions = (
#                 "✅ 1 chunk required\n"
#                 "🎬 Rendering single chunk"
#             )

#         else:
#             if enable_auto_queue:
#                 instructions = (
#                     f"⚠️  {total_sets} chunks required\n"
#                     f"✅ Auto-queue enabled — remaining chunks will be queued automatically"
#                 )
#             else:
#                 instructions = (
#                     f"⚠️  {total_sets} chunks required\n"
#                     f"🔴 Auto-queue is DISABLED\n"
#                     f"❗ Manually run each chunk"
#                 )

#         # -------------------------------------------------
#         # Audio metadata (single-chunk model by design)
#         # -------------------------------------------------
#         audio_meta = {
#             "durations_frames": [frames_per_scene]
#         }

#         return (
#             instructions,
#             end_time_str,
#             total_sets,
#             frames_per_scene,
#             audio_meta,
#         )


#     def _maybe_auto_queue(self, total_sets: int, index: int, enable: bool):
#         """
#         Auto-queue remaining chunks.
#         Single-chunk-per-run model.
#         Only triggers on the very first run.
#         """
#         if not enable:
#             return

#         # Only auto-queue on the very first chunk
#         if index != 0:
#             return

#         if total_sets <= 1:
#             return

#         runs = total_sets - 1
#         print(f"[AutoQueue] Queuing {runs} additional chunks")

#         for _ in range(runs):
#             PromptServer.instance.send_sync("impact-add-queue", {})


#     def _send_popup_notification(self, message: str, message_type: str = "info", title: str = "Audio Split Instructions"):
#         """Same popup mechanism you already had."""
#         try:
#             from server import PromptServer
#             PromptServer.instance.send_sync("vrgdg_instructions_popup", {
#                 "message": message,
#                 "type": message_type,
#                 "title": title
#             })
#             print(f"[Popup] Sent {message_type} notification to UI")
#         except Exception as e:
#             print(f"[Popup] Could not send notification: {e}")

#     def _adjust_frames(self, frames: int, fps: int, use_humo_alignment: bool) -> int:
#         """
#         Optionally apply HuMo frame alignment (4n + 1).
#         HuMo REQUIRES 25 fps.
#         """
#         if use_humo_alignment:
#             if fps != 25:
#                 raise ValueError("HuMo alignment requires fps=25")

#             adjusted = 4 * ((frames + 2) // 4) + 1
#             if adjusted != frames:
#                 actual_duration = adjusted / 25
#                 print(f"[HuMo Adjust] {frames} → {adjusted} frames ({actual_duration:.2f}s)")
#             return adjusted

#         # General video models (no alignment)
#         return frames


#     # --------------- main ---------------
#     def run(
#         self,
#         audio,
#         trigger,
#         scene_duration_seconds,
#         fps,
#         folder_path,
#         enable_auto_queue,
#         override_chunk_index,
#         overwrite_mode,
#         use_humo_alignment,
#     ):
#         try:
#             if audio is None:
#                 raise ValueError("audio is None")

#             waveform = audio["waveform"]
#             sample_rate = int(audio["sample_rate"])
#         except Exception as e:
#             raise ValueError(f"Invalid audio input: {e}")

#         print(f"[Audio] original sample_rate: {sample_rate}")

#         # Ensure batch dimension
#         if waveform.ndim == 2:
#             waveform = waveform.unsqueeze(0)

#         # --- FORCE RESAMPLE TO 44.1kHz ---
#         target_sr = 44100
#         if sample_rate != target_sr:
#             print(f"[Audio] Resampling {sample_rate} → {target_sr}")

#             # waveform shape: [B, C, T]
#             waveform = torch.nn.functional.interpolate(
#                 waveform,
#                 scale_factor=target_sr / sample_rate,
#                 mode="linear",
#                 align_corners=False
#             )

#             sample_rate = target_sr

#         print(f"[Audio] final sample_rate: {sample_rate}")

#         total_samples = waveform.shape[-1]
#         total_duration = float(total_samples) / float(sample_rate)


#         # -------------------------------------------------
#         # Output folder creation + reuse (FINAL, CORRECT)
#         # -------------------------------------------------
#         from datetime import datetime

#         base_output = folder_paths.get_output_directory()

#         # Resolve base folder name
#         base_name = folder_path.strip() if folder_path.strip() else "VRGDG_Video"

#         # Reuse most recent timestamped run folder if it exists
#         existing_runs = sorted(
#             d for d in os.listdir(base_output)
#             if d.startswith(base_name + "_")
#             and os.path.isdir(os.path.join(base_output, d))
#         )

#         if existing_runs:
#             output_folder = os.path.join(base_output, existing_runs[-1])
#         else:
#             timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             run_folder_name = f"{base_name}_{timestamp}"
#             output_folder = os.path.join(base_output, run_folder_name)
#             os.makedirs(output_folder, exist_ok=True)


#         # FINAL index resolution
#         if override_chunk_index >= 0:
#             set_index = override_chunk_index
#             enable_auto_queue = False
#         else:
#             set_index = self._count_index_from_folder(output_folder)
#             overwrite_mode = "overwrite"   # 🔒 FORCE overwrite during normal runs


#         print(f"[Index] Detected set_index={set_index} from folder: {output_folder}")
#         chunk_index = set_index



#         # calculate sets (gets frames_per_scene)
#         instructions, end_time_str_hr, total_sets, frames_per_scene, audio_meta = \
#             self._calculate_sets(audio, scene_duration_seconds, fps, enable_auto_queue, use_humo_alignment)

#         # split parameters (use frames_per_scene from _calculate_sets)
        
#         samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)
#         offset_samples = chunk_index * samples_per_scene



#         start_samp = offset_samples
#         end_samp = start_samp + samples_per_scene

#         if start_samp >= total_samples:
#             seg = torch.zeros(
#                 (1, 2, samples_per_scene),
#                 dtype=waveform.dtype,
#                 device=waveform.device
#             )
#         else:
#             end_samp = min(total_samples, end_samp)
#             seg = waveform[..., start_samp:end_samp].contiguous().clone()

#             cur_len = seg.shape[-1]
#             if cur_len < samples_per_scene:
#                 pad = samples_per_scene - cur_len
#                 seg = torch.nn.functional.pad(seg, (0, pad))

#         audio = {
#             "waveform": seg,
#             "sample_rate": sample_rate
#         }


#         # meta (kept)
#         meta = {
#             "durations": [frames_per_scene / fps],
#             "offset_seconds": chunk_index * (frames_per_scene / fps),
#             "starts": [offset_samples],
#             "sample_rate": sample_rate,
#             "audio_total_duration": total_duration,
#             "outputs_count": 1,
#             "output_folder": output_folder,
#         }


#         # -------------------------------------------------
#         # OVERRIDE SAFETY CHECK (must be AFTER total_sets is known)
#         # -------------------------------------------------
#         if override_chunk_index >= 0 and total_sets > 0:
#             if override_chunk_index >= total_sets:
#                 raise ValueError(
#                     f"override_chunk_index {override_chunk_index} "
#                     f"is out of range (total chunks: {total_sets})"
#                 )

#         chunk_index = set_index

#         # -------------------------------------------------
#         # Popup behavior + override clarity + auto-queue safety
#         # (single-chunk-per-run model)
#         # -------------------------------------------------

#         # Prefix instructions with clear chunk context
#         if override_chunk_index >= 0:
#             prefix = (
#                 f"🔁 Re-rendering chunk {chunk_index + 1} / {total_sets}\n"
#                 f"⚠️ OVERRIDE MODE — manual re-render\n\n"
#             )
#         else:
#             prefix = f"🎬 Rendering chunk {chunk_index + 1} / {total_sets}\n\n"

#         instructions = prefix + instructions
#         popup_message = instructions

#         # --- popup notifications (no group logic, no special cases) ---
#         if set_index == 0:
#             self._send_popup_notification(
#                 popup_message,
#                 "info",
#                 "🎬 STARTING AUDIO SPLIT"
#             )

#         elif set_index + 1 < total_sets:
#             self._send_popup_notification(
#                 popup_message,
#                 "yellow",
#                 "⏳ CHUNK IN PROGRESS"
#             )

#         elif set_index + 1 == total_sets:
#             self._send_popup_notification(
#                 popup_message,
#                 "green",
#                 "🏁 FINAL CHUNK"
#             )

#         # --- HARD SAFETY RULE ---
#         # Auto-queue ONLY in normal mode (never during override)
#         if override_chunk_index < 0:
#             self._maybe_auto_queue(total_sets, set_index, enable_auto_queue)
#         else:
#             print("[AutoQueue] Override mode active — auto-queue suppressed.")


        

#         # start/end time strings for this set
#         actual_scene_duration = frames_per_scene / fps
#         set_duration_sec = actual_scene_duration
#         start_sec = set_index * set_duration_sec
#         end_sec = start_sec + set_duration_sec


#         def fmt_time(sec):
#             m = int(sec // 60)
#             s = sec % 60
#             return f"{m}:{s:06.3f}"

#         start_time_str = fmt_time(start_sec)
#         end_time_str = fmt_time(end_sec)

#         # outputs (lyrics removed!)
#         return (
#             meta,
#             total_duration,
#             set_index,
#             start_time_str,
#             end_time_str,
#             instructions,
#             total_sets,
#             frames_per_scene,
#             audio_meta,
#             output_folder,
#             overwrite_mode,
#             audio,
#             any_typ
#         )


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

                "use_humo_alignment": ("BOOLEAN", {
                    "default": False
                }),
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





    def _calculate_sets(self, audio, scene_duration_seconds, fps, enable_auto_queue=True, use_humo_alignment=False):
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
        frames_per_scene = self._adjust_frames(frames_per_scene_raw, fps, use_humo_alignment)

        print(
            f"[Frames] fps={fps}, "
            f"scene_duration={scene_duration_seconds}s, "
            f"raw_frames={frames_per_scene_raw}, "
            f"final_frames={frames_per_scene}, "
            f"humo_alignment={use_humo_alignment}"
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

    def _adjust_frames(self, frames: int, fps: int, use_humo_alignment: bool) -> int:
        """
        Optionally apply HuMo frame alignment (4n + 1).
        HuMo REQUIRES 25 fps.
        """
        if use_humo_alignment:
            if fps != 25:
                raise ValueError("HuMo alignment requires fps=25")

            adjusted = 4 * ((frames + 2) // 4) + 1
            if adjusted != frames:
                actual_duration = adjusted / 25
                print(f"[HuMo Adjust] {frames} → {adjusted} frames ({actual_duration:.2f}s)")
            return adjusted
        
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
        use_humo_alignment,
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
                    enable_auto_queue,
                    use_humo_alignment
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
            frames_per_scene = self._adjust_frames(
                frames_per_scene_raw,
                fps,
                use_humo_alignment
            )

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
    Optional: drums, bass, vocals
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
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("beat_data",)
    FUNCTION = "analyze"
    CATEGORY = "audio/rhythm"

    def analyze(self, final_mix, drums=None, bass=None, vocals=None):

        # --- Extract audio safely ---
        y_mix, sr = extract_mono(final_mix)
        if y_mix is None:
            raise ValueError("Final mix AUDIO input is invalid")

        y_drums, _ = extract_mono(drums)
        y_bass, _ = extract_mono(bass)
        y_vocals, _ = extract_mono(vocals)

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

        beat_source = y_drums if stem_usable(y_drums, y_mix, sr) else y_mix

        tempo, beat_frames = librosa.beat.beat_track(
            y=beat_source,
            sr=sr,
            trim=False
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

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

        onset_times = librosa.frames_to_time(
            np.arange(len(onset_mix)), sr=sr
        )

        beats = []

        for i, t in enumerate(beat_times):
            idx = np.argmin(np.abs(onset_times - t))

            impact = 0.0
            weight_sum = 0.0

            if onset_drums is not None:
                impact += onset_drums[idx] * 0.5
                weight_sum += 0.5

            if onset_bass is not None:
                impact += onset_bass[idx] * 0.3
                weight_sum += 0.3

            if onset_vocals is not None:
                impact += onset_vocals[idx] * 0.2
                weight_sum += 0.2

            if weight_sum == 0.0:
                impact = onset_mix[idx]
            else:
                impact /= weight_sum

            beats.append({
                "time": round(float(t), 4),
                "beat_index": i,
                "downbeat": (i % 4 == 0),
                "impact": round(float(impact), 4)
            })

        output = {
            "bpm": round(float(tempo), 2),
            "source_used_for_beats": "drums" if y_drums is not None else "final_mix",
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

        srt_lines = []
        current_time = 0.0
        scene_index = 1
        current_index = 0

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

                weight = impact * (1.2 if downbeat else 1.0)
                candidates.append((i, t, weight))

            if not candidates:
                break

            weights = [(w ** bias) + 1e-6 for _, _, w in candidates]
            chosen_index, chosen_time, _ = rng.choices(
                candidates, weights=weights, k=1
            )[0]

            duration = chosen_time - start_time

            srt_lines.append(str(scene_index))
            srt_lines.append(
                f"{format_time(current_time)} --> {format_time(current_time + duration)}"
            )
            srt_lines.append(f"SCENE {scene_index}")

            srt_lines.append("")  # blank line required between blocks


            current_time += duration
            scene_index += 1
            current_index = chosen_index

        # --- Force final subtitle to reach song end ---
        if current_time < song_end:
            srt_lines.append(str(scene_index))
            srt_lines.append(
                f"{format_time(current_time)} --> {format_time(song_end)}"
            )
            srt_lines.append(f"SCENE {scene_index}")

            srt_lines.append("")





        ###############
        # Save next to THIS custom node file, inside /SRT_Files
        node_dir = os.path.dirname(os.path.abspath(__file__))

        # Create folder if missing
        srt_dir = os.path.join(node_dir, "SRT_Files")
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

        print(f"[BeatSceneDurationNode] Saved SRT to: {out_path}")



        return (
            "\n".join(srt_lines),
            out_path
        )




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
    "VRGDG_TrimFinalClip":VRGDG_TrimFinalClip,
    "VRGDG_PromptSplitter_General":VRGDG_PromptSplitter_General,
    "VRGDG_PadVideoWithLastFrame":VRGDG_PadVideoWithLastFrame,
    "BeatImpactAnalysisNode":BeatImpactAnalysisNode,
    "VRGDG_DurationIndexFloat":VRGDG_DurationIndexFloat,
    "VRGDG_TrimImageBatch":VRGDG_TrimImageBatch,
    "BeatSceneDurationNode": BeatSceneDurationNode,
    "IndexedImageFromFolder": IndexedImageFromFolder,
    "VRGDG_PromptSpitterWithIndex":VRGDG_PromptSplitterWithIndex,
    
    


}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LoadAudioSplit_General": "VRGDG Load Audio Split (General)",
    "VRGDG_BuildVideoOutputPath_General": "VRGDG Build Video Output Path (General)",
    "VRGDG_TrimFinalClip":"VRGDG_TrimFinalClip",
    "VRGDG_PromptSplitter_General":"VRGDG_PromptSplitter_General",
    "VRGDG_PadVideoWithLastFrame":"VRGDG_PadVideoWithLastFrame",
    "BeatImpactAnalysisNode":"BeatImpactAnalysisNode",
    "VRGDG_DurationIndexFloat":"VRGDG_DurationIndexFloat",
    "VRGDG_TrimImageBatch":"VRGDG_TrimImageBatch",
    "BeatSceneDurationNode": "Beat-Aligned Scene Durations",
    "IndexedImageFromFolder": "Image From Folder (Index)",
    "VRGDG_PromptSplitterWithIndex":"VRGDG_PromptSplitterWithIndex",
    




}

