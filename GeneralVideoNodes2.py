import os
import torch
import torchaudio.functional as AF
from server import PromptServer
import re
import folder_paths
import math
import subprocess
import torchaudio
import imageio
import json
import shutil
from datetime import datetime


def round_up_8n1(n: int) -> int:
    """Round up frame count to 8N+1 (required by some video models)."""
    n = max(1, int(n))
    return ((n - 1 + 7) // 8) * 8 + 1


# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

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
        from datetime import datetime

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

        # chunk 0 should never have preroll
        if chunk_index == 0:
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
        tail_samples = int(round(TAIL_FRAMES * samples_per_frame))

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

        expected_min = (0 if chunk_index == 0 else pre_frames) + frames_per_scene
        print(f"[TRIM] expected_min_frames_from_LTX = {expected_min}")

        extra = total_frames - expected_min
        print(f"[TRIM] extra_frames_after_truth = {extra}")

        if total_frames < expected_min:
            print("[TRIM] ❌ LTX returned TOO FEW frames!")

        if chunk_index == 0:
            end = min(frames_per_scene, total_frames)
            print(f"[TRIM] FIRST CHUNK → slicing [0:{end}]")
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


class VRGDG_AudioDelayByIndex:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "chunk_index": ("INT", {}),
                "delay_ms": ("FLOAT", {"default": 40.0, "min": -100.0, "max": 200.0}),
            }
        }

    def run(self, audio, chunk_index, delay_ms):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        # Apply delay to all chunks except index 0
        if chunk_index != 0:
            delay_samples = int(round(delay_ms * sample_rate / 1000.0))

            if delay_samples > 0:
                waveform = torch.nn.functional.pad(waveform, (delay_samples, 0))

            elif delay_samples < 0:
                cut = min(-delay_samples, waveform.shape[-1])
                waveform = waveform[..., cut:]

            print(f"[AUDIO-DELAY] Applied {delay_ms}ms to chunk {chunk_index}")

        else:
            print(f"[AUDIO-DELAY] Skipped chunk 0")


        return ({
            "waveform": waveform,
            "sample_rate": sample_rate
        },)



def find_ffmpeg_path():
    """
    Checks if system ffmpeg is available; 
    if not, falls back to imageio-ffmpeg bundled binary.
    """
    try:
        # Try to call system ffmpeg
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"  # System ffmpeg is available
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"[VRGDG] Using fallback ffmpeg from imageio: {ffmpeg_path}")
            return ffmpeg_path
        except Exception as e:
            print(f"[VRGDG] ⚠️ No FFmpeg found. Error: {e}")
            return None
        


    

class VRGDG_RunStateLogger_SRT:
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "run"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("VHS_FILENAMES", {}),
                "index": ("INT", {"default": 0, "min": 0}),
                "total_sets": ("INT", {"default": 0, "min": 0}),
                "output_folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "note": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def _safe_json_value(self, value):
        try:
            json.dumps(value)
            return value
        except Exception:
            return repr(value)

    def run(self, trigger, index, total_sets, output_folder, note=""):
        folder = (output_folder or "").strip()
        if not folder:
            folder = folder_paths.get_output_directory()
        elif not os.path.isabs(folder):
            folder = os.path.join(folder_paths.get_output_directory(), folder)

        temp_state_dir = os.path.join(folder, "vrgdg_temp")
        os.makedirs(temp_state_dir, exist_ok=True)

        log_path = os.path.join(temp_state_dir, "srt_run_state.jsonl")
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "index": int(index),
            "total_sets": int(total_sets),
            "output_folder": folder,
            "trigger": self._safe_json_value(trigger),
        }
        if note:
            entry["note"] = note

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")
        except Exception as e:
            print(f"[RunStateLogger] Failed to write log: {e}")

        return (trigger,)


class SRTLyricsMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_text": ("STRING", {"multiline": True}),
                "lyrics_json": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_json",)
    FUNCTION = "merge"
    CATEGORY = "Text"

    def merge(self, srt_text, lyrics_json):
        # Load lyric JSON
        lyrics = json.loads(lyrics_json)

        # Extract durations from SRT
        pattern = r"(\d+)\s+(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)\s+SCENE\s+(\d+)"
        matches = re.findall(pattern, srt_text)

        def to_seconds(t):
            h, m, rest = t.split(":")
            s, ms = rest.split(",")
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

        durations = {}
        for _, start, end, seg_num in matches:
            duration = to_seconds(end) - to_seconds(start)
            durations[int(seg_num)] = f"{duration:.3f}s"

        # Merge durations into lyric keys
        merged = {}
        for key, value in lyrics.items():
            seg_match = re.search(r"lyricSegment(\d+)", key)
            if not seg_match:
                continue

            seg_num = int(seg_match.group(1))
            dur = durations.get(seg_num, "UNKNOWN")

            new_key = f"{key}_Duration_{dur}"
            merged[new_key] = value

        return (json.dumps(merged, indent=2),)

class VRGDG_StoryBoardCreator:
    """
    Storyboard Prompt Runner.
    - Tracks next index from existing output filenames.
    - Supports auto-queue, redo lists, backups, and prompt overrides.
    """

    RETURN_TYPES = ("STRING", "INT", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "index", "index_str", "total_prompts", "output_folder_name", "save_subpath")
    FUNCTION = "run"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list": ("STRING", {"multiline": True, "default": "{}"}),
                "output_folder": ("STRING", {"default": ""}),
                "trigger": ("INT", {"default": 0}),
                "use_remake_folder": ("BOOLEAN", {"default": False}),
                "auto_queue": ("BOOLEAN", {"default": True}),
                "redo_mode": ("BOOLEAN", {"default": False}),
                "redo_indexes": ("STRING", {"default": ""}),
                "redo_prompt_overrides": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def _parse_prompt_list(self, prompt_list_input):
        if prompt_list_input is None:
            return []

        data = None
        if isinstance(prompt_list_input, (dict, list)):
            data = prompt_list_input
        else:
            raw = str(prompt_list_input).strip()
            if not raw:
                return []
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[StoryBoard] Invalid JSON prompt list: {e}")
                return []

        def _extract_text(value):
            if isinstance(value, dict):
                if "text" in value:
                    return str(value.get("text", ""))
                if "prompt" in value:
                    return str(value.get("prompt", ""))
            return str(value)

        prompts = []
        if isinstance(data, dict):
            sorted_keys = sorted(
                data.keys(),
                key=lambda x: int("".join(filter(str.isdigit, x)))
                if any(c.isdigit() for c in x) else 0
            )
            prompts = [_extract_text(data[k]) for k in sorted_keys]
        elif isinstance(data, list):
            prompts = [_extract_text(p) for p in data]

        return prompts

    def _scan_next_index(self, output_folder):
        if not os.path.isdir(output_folder):
            return 1

        indices = []
        for f in os.listdir(output_folder):
            m = re.match(r"^(\d+)", f)
            if m:
                try:
                    indices.append(int(m.group(1)))
                except ValueError:
                    pass

        if not indices:
            return 1

        return max(indices) + 1

    def _parse_redo_indexes(self, redo_indexes):
        raw = str(redo_indexes).strip()
        if not raw:
            return []

        parts = re.split(r"[,\s]+", raw)
        indices = []
        for p in parts:
            if not p:
                continue
            try:
                v = int(p)
                if v > 0:
                    indices.append(v)
            except ValueError:
                continue

        # Preserve order, remove duplicates
        seen = set()
        ordered = []
        for v in indices:
            if v not in seen:
                ordered.append(v)
                seen.add(v)
        return ordered

    def _parse_override_blocks(self, override_text):
        text = str(override_text).strip()
        if not text:
            return []
        blocks = re.split(r"\n\s*\n", text)
        return [b.strip() for b in blocks if b.strip()]

    def _load_prompt_state(self, temp_dir, prompts):
        state_path = os.path.join(temp_dir, "storyboard_prompt_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if isinstance(state, list) and len(state) == len(prompts):
                    return state
            except Exception as e:
                print(f"[StoryBoard] Failed to read prompt state: {e}")

        return list(prompts)

    def _save_prompt_state(self, temp_dir, prompt_state):
        state_path = os.path.join(temp_dir, "storyboard_prompt_state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(prompt_state, f, indent=2, ensure_ascii=False)

    def _write_prompt_json(self, path, prompt_state):
        data = {f"prompt{i+1}": prompt_state[i] for i in range(len(prompt_state))}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _backup_existing_images(self, output_folder, index):
        if not os.path.isdir(output_folder):
            return

        backup_dir = os.path.join(output_folder, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        for f in os.listdir(output_folder):
            src = os.path.join(output_folder, f)
            if not os.path.isfile(src):
                continue

            m = re.match(r"^(\d+)", f)
            if not m:
                continue
            try:
                if int(m.group(1)) != index:
                    continue
            except ValueError:
                continue

            base, ext = os.path.splitext(f)
            backup_name = f"{base}_old{ext}"
            dst = os.path.join(backup_dir, backup_name)
            if os.path.exists(dst):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst = os.path.join(backup_dir, f"{base}_old_{stamp}{ext}")

            os.replace(src, dst)

    def _move_remake_files_to_backup(self, remake_dir, index):
        if not os.path.isdir(remake_dir):
            return

        backup_dir = os.path.join(remake_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        for f in os.listdir(remake_dir):
            src = os.path.join(remake_dir, f)
            if not os.path.isfile(src):
                continue

            m = re.match(r"^(\d+)", f)
            if not m:
                continue
            try:
                if int(m.group(1)) != index:
                    continue
            except ValueError:
                continue

            dst = os.path.join(backup_dir, f)
            if os.path.exists(dst):
                base, ext = os.path.splitext(f)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst = os.path.join(backup_dir, f"{base}_{stamp}{ext}")

            os.replace(src, dst)

    def _maybe_auto_queue(self, temp_dir, start_index, total_prompts, enable):
        if not enable:
            return

        if start_index > total_prompts:
            return

        queue_state_path = os.path.join(temp_dir, "storyboard_autoqueue.json")
        if os.path.exists(queue_state_path):
            try:
                with open(queue_state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if (
                    isinstance(state, dict)
                    and state.get("total_prompts") == total_prompts
                    and state.get("start_index", 0) < start_index
                ):
                    return
            except Exception:
                pass

        remaining = total_prompts - start_index
        if remaining <= 0:
            return

        with open(queue_state_path, "w", encoding="utf-8") as f:
            json.dump(
                {"start_index": start_index, "total_prompts": total_prompts},
                f,
                indent=2
            )

        print(f"[StoryBoard] Auto-queue remaining: {remaining}")
        for _ in range(remaining):
            PromptServer.instance.send_sync("impact-add-queue", {})

    def run(
        self,
        prompt_list,
        output_folder,
        trigger,
        use_remake_folder,
        auto_queue,
        redo_mode,
        redo_indexes,
        redo_prompt_overrides,
    ):
        os.makedirs(output_folder, exist_ok=True)
        temp_dir = os.path.join(output_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        remake_dir = os.path.join(output_folder, "remake")
        os.makedirs(remake_dir, exist_ok=True)

        prompts = self._parse_prompt_list(prompt_list)
        total_prompts = len(prompts)
        total_prompts_out = total_prompts
        if total_prompts == 0:
            return ("", 0, "", 0)

        prompt_state = self._load_prompt_state(temp_dir, prompts)

        override_blocks = self._parse_override_blocks(redo_prompt_overrides)
        redo_indices_full = self._parse_redo_indexes(redo_indexes)
        redo_indices_full = [i for i in redo_indices_full if 1 <= i <= total_prompts]

        current_index = 0
        remake_queue_path = os.path.join(temp_dir, "storyboard_remake_queue.json")
        remake_autoqueue_path = os.path.join(temp_dir, "storyboard_remake_autoqueue.json")
        remake_total_path = os.path.join(temp_dir, "storyboard_remake_total.json")
        redo_queue_path = os.path.join(temp_dir, "storyboard_redo_queue.json")
        redo_counter_path = os.path.join(temp_dir, "storyboard_redo_step.json")

        if use_remake_folder:
            remake_indices = []
            remake_queue = None
            remake_total = None

            if os.path.exists(remake_queue_path):
                try:
                    with open(remake_queue_path, "r", encoding="utf-8") as f:
                        remake_queue = json.load(f)
                except Exception as e:
                    print(f"[StoryBoard] Failed to load remake queue: {e}")
                    remake_queue = None
            if os.path.exists(remake_total_path):
                try:
                    with open(remake_total_path, "r", encoding="utf-8") as f:
                        remake_total = json.load(f)
                except Exception:
                    remake_total = None

            if remake_queue is None:
                for f in os.listdir(remake_dir):
                    m = re.match(r"^(\d+)", f)
                    if not m:
                        continue
                    try:
                        v = int(m.group(1))
                    except ValueError:
                        continue
                    if 1 <= v <= total_prompts:
                        remake_indices.append(v)

                remake_indices = sorted(set(remake_indices))
                remake_queue = remake_indices[:]
                remake_total = len(remake_queue)
                with open(remake_total_path, "w", encoding="utf-8") as f:
                    json.dump(remake_total, f, indent=2)

                if override_blocks:
                    for i, idx in enumerate(remake_indices):
                        if i >= len(override_blocks):
                            break
                        prompt_state[idx - 1] = override_blocks[i]

                    override_path = os.path.join(temp_dir, "prompts_override.json")
                    self._write_prompt_json(override_path, prompt_state)
                    self._save_prompt_state(temp_dir, prompt_state)

            total_prompts_out = remake_total if isinstance(remake_total, int) else len(remake_queue)
            if not remake_queue:
                if os.path.exists(remake_autoqueue_path):
                    os.remove(remake_autoqueue_path)
                if os.path.exists(remake_queue_path):
                    os.remove(remake_queue_path)
                if os.path.exists(remake_total_path):
                    os.remove(remake_total_path)
                return ("", 0, "", 0, "", "")

            current_index = remake_queue.pop(0)
            self._move_remake_files_to_backup(remake_dir, current_index)

            if remake_queue:
                with open(remake_queue_path, "w", encoding="utf-8") as f:
                    json.dump(remake_queue, f, indent=2)
            else:
                if os.path.exists(remake_queue_path):
                    os.remove(remake_queue_path)

            if auto_queue and remake_queue:
                should_queue = True
                if os.path.exists(remake_autoqueue_path):
                    try:
                        with open(remake_autoqueue_path, "r", encoding="utf-8") as f:
                            state = json.load(f)
                        if (
                            isinstance(state, dict)
                            and state.get("total_prompts_out") == total_prompts_out
                            and state.get("queued_for") == total_prompts_out
                        ):
                            should_queue = False
                    except Exception:
                        pass

                if should_queue:
                    with open(remake_autoqueue_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {"total_prompts_out": total_prompts_out, "queued_for": total_prompts_out},
                            f,
                            indent=2
                        )
                    print(f"[StoryBoard] Auto-queue remake remaining: {len(remake_queue)}")
                    for _ in range(len(remake_queue)):
                        PromptServer.instance.send_sync("impact-add-queue", {})
                else:
                    print(f"[StoryBoard] Remake auto-queue already set for {total_prompts_out}")
            else:
                print(f"[StoryBoard] Remake mode: index={current_index} remaining={len(remake_queue)} auto_queue={auto_queue}")

        elif redo_mode:
            if os.path.exists(redo_queue_path):
                try:
                    with open(redo_queue_path, "r", encoding="utf-8") as f:
                        redo_queue = json.load(f)
                    with open(redo_counter_path, "r", encoding="utf-8") as f:
                        redo_step = json.load(f)
                except Exception as e:
                    print(f"[StoryBoard] Failed to load redo queue: {e}")
                    redo_queue = redo_indices_full[:]
                    redo_step = 0
            else:
                redo_queue = redo_indices_full[:]
                redo_step = 0

            if not redo_queue:
                return ("", 0, "", total_prompts)

            current_index = redo_queue.pop(0)
            redo_step += 1

            # Only apply overrides once, at redo start
            if override_blocks and redo_step == 1:
                for i, idx in enumerate(redo_indices_full):
                    if i >= len(override_blocks):
                        break
                    prompt_state[idx - 1] = override_blocks[i]

                override_path = os.path.join(temp_dir, "prompts_override.json")
                self._write_prompt_json(override_path, prompt_state)

            self._save_prompt_state(temp_dir, prompt_state)
            self._backup_existing_images(output_folder, current_index)

            if redo_queue:
                with open(redo_queue_path, "w", encoding="utf-8") as f:
                    json.dump(redo_queue, f, indent=2)
                with open(redo_counter_path, "w", encoding="utf-8") as f:
                    json.dump(redo_step, f, indent=2)
            else:
                if os.path.exists(redo_queue_path):
                    os.remove(redo_queue_path)
                if os.path.exists(redo_counter_path):
                    os.remove(redo_counter_path)

            if auto_queue and redo_step == 1 and redo_queue:
                print(f"[StoryBoard] Auto-queue redo remaining: {len(redo_queue)}")
                for _ in range(len(redo_queue)):
                    PromptServer.instance.send_sync("impact-add-queue", {})

        else:
            current_index = self._scan_next_index(output_folder)
            if current_index > total_prompts:
                queue_state_path = os.path.join(temp_dir, "storyboard_autoqueue.json")
                if os.path.exists(queue_state_path):
                    os.remove(queue_state_path)
                return ("", total_prompts, "", total_prompts)

            self._maybe_auto_queue(temp_dir, current_index, total_prompts, auto_queue)
            self._save_prompt_state(temp_dir, prompt_state)

        prompt_text = prompt_state[current_index - 1]
        pad = max(3, len(str(total_prompts)))
        index_str = f"{current_index:0{pad}d}"

        final_path = os.path.join(output_folder, "final_prompts.json")

        # Only write final manifest when we hit the end
        if not redo_mode and current_index == total_prompts:
            self._write_prompt_json(final_path, prompt_state)

        # Or when redo mode finishes the full redo queue
        if redo_mode and not os.path.exists(redo_queue_path):
            self._write_prompt_json(final_path, prompt_state)

        # Or when remake mode finishes the full remake queue
        if use_remake_folder and not os.path.exists(remake_queue_path):
            self._write_prompt_json(final_path, prompt_state)

        folder_name = os.path.basename(output_folder.rstrip("\\/"))
        if not folder_name:
            folder_name = os.path.basename(output_folder)
        save_subpath = os.path.join(folder_name, index_str).replace("\\", "/")

        return (prompt_text, current_index, index_str, total_prompts_out, folder_name, save_subpath)


NODE_CLASS_MAPPINGS = {
    "VRGDG_LoadAudioSplit_SRTOnly": VRGDG_LoadAudioSplit_SRTOnly,
    "VRGDG_TrimImageBatch_SRTOnly": VRGDG_TrimImageBatch_SRTOnly,
    "VRGDG_AudioDelayByIndex": VRGDG_AudioDelayByIndex,
    "VRGDG_RunStateLogger_SRT": VRGDG_RunStateLogger_SRT,
    "SRTLyricsMerger": SRTLyricsMerger,
    "VRGDG_StoryBoardCreator":VRGDG_StoryBoardCreator


}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_LoadAudioSplit_SRTOnly": "VRGDG_LoadAudioSplit_SRTOnly",
    "VRGDG_TrimImageBatch_SRTOnly": "VRGDG_TrimImageBatch_SRTOnly",
    "VRGDG_AudioDelayByIndex": "VRGDG_AudioDelayByIndex",
    "VRGDG_RunStateLogger_SRT": "VRGDG_RunStateLogger_SRT",
    "SRTLyricsMerger": "SRTLyricsMerger",
    "VRGDG_StoryBoardCreator":"VRGDG_StoryBoardCreator"


}
