import os

import torch

from server import PromptServer

import re

import folder_paths

import subprocess

import torchaudio

import json

from datetime import datetime


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
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"[VRGDG] Using fallback ffmpeg from imageio: {ffmpeg_path}")
            return ffmpeg_path
        except Exception as e:
            print(f"[VRGDG] ⚠️ No FFmpeg found. Error: {e}")
            return None


class VRGDG_CreateFinalVideo_SRT:
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "create_final"
    CATEGORY = "Video"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("VHS_FILENAMES", {}),
                "audio": ("AUDIO",),
                "threshold": ("INT", {"default": 3}),
                "group_list": ("STRING", {"default": "-1"}),
                "video_folder": ("STRING", {"default": "video_output", "multiline": False}),
            }
        }

    def create_final(self, trigger, audio, threshold, group_list, video_folder):
        video_folder = video_folder.strip()

        if not os.path.isabs(video_folder):
            base_output = folder_paths.get_output_directory()
            video_folder = os.path.join(base_output, video_folder)

        print(f"[CreateFinalVideo] Looking in: {video_folder}")
        # ✅ Local temp state folder (stored with videos)
        temp_state_dir = os.path.join(video_folder, "vrgdg_temp")
        os.makedirs(temp_state_dir, exist_ok=True)

        # -------------------------------------------------
        # ✅ RERUN MODE: WAIT FOR OVERRIDE QUEUE TO FINISH
        # -------------------------------------------------
        if group_list.strip() != "-1":

            override_path = os.path.join(
                temp_state_dir,
                "vrgdg_override_queue.json"
            )

            if os.path.exists(override_path):
                with open(override_path, "r") as f:
                    remaining = json.load(f)

                if remaining:
                    print(f"[CreateFinalVideo] Waiting for override reruns: {remaining}")
                    return ()

        # -------------------------------------------------
        # Collect video files
        # -------------------------------------------------
        videos = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith(".mp4") and "-audio" in f.lower()
        ])

        video_count = len(videos)

        # -------------------------------------------------
        # Normal mode threshold check
        # -------------------------------------------------
        if group_list.strip() == "-1":
            if video_count < threshold:
                print(f"[CreateFinalVideo] Threshold not met ({video_count}/{threshold}), skipping.")
                return ()

        # -------------------------------------------------
        # Output name depends on rerun mode
        # -------------------------------------------------
        if group_list.strip() != "-1":
            final_name = "FINAL_VIDEO_REDO.mp4"
        else:
            final_name = "FINAL_VIDEO.mp4"

        final_output = os.path.join(video_folder, final_name)

        # ✅ If file already exists (or is locked), make a new numbered one
        if os.path.exists(final_output):
            base, ext = os.path.splitext(final_name)
            count = 2

            while True:
                candidate = os.path.join(video_folder, f"{base}{count}{ext}")
                if not os.path.exists(candidate):
                    final_output = candidate
                    break
                count += 1


        # -------------------------------------------------
        # Build concat list
        # -------------------------------------------------
        concat_file = os.path.join(video_folder, "concat_list.txt")
        with open(concat_file, "w") as f:
            for vid in videos:
                f.write(f"file '{os.path.join(video_folder, vid)}'\n")

        temp_video = os.path.join(video_folder, "_temp_video_no_audio.mp4")

        print(f"[CreateFinalVideo] Concatenating {video_count} videos (removing audio)...")

        ffmpeg_path = find_ffmpeg_path()
        if not ffmpeg_path:
            print("❌ [CreateFinalVideo] FFmpeg not available.")
            return ()

        cmd_concat = [
            ffmpeg_path, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-an",
            "-c:v", "copy",
            temp_video
        ]

        try:
            subprocess.run(cmd_concat, capture_output=True, text=True, errors="replace", check=True)
            print("✅ [CreateFinalVideo] Videos concatenated (no audio)")
        except subprocess.CalledProcessError as e:
            print(f"❌ [CreateFinalVideo] Concatenation failed: {e.stderr}")
            return ()

        # -------------------------------------------------
        # Save original audio
        # -------------------------------------------------
        temp_audio = os.path.join(video_folder, "_temp_original_audio.wav")
        print("[CreateFinalVideo] Saving original audio...")

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        def _is_libtorchcodec_error(err_text):
            return "libtorchcodec" in str(err_text).lower()

        try:
            torchaudio.save(temp_audio, waveform.squeeze(0).cpu(), sample_rate)
        except Exception as e:
            if _is_libtorchcodec_error(e):
                print("[CreateFinalVideo] torchaudio.save failed (libtorchcodec).")
                message = (
                    "❌ Final video creation failed due to missing FFmpeg shared libraries.\n\n"
                    "Fix:\n"
                    "1) Install the full shared build of FFmpeg into your portable directory:\n"
                    "   https://www.gyan.dev/ffmpeg/builds/\n"
                    "2) Copy the DLLs into the root folder.\n"
                    "3) Ensure your .bat includes:\n"
                    "   set \"PATH=%~dp0ffmpeg\\bin;%PATH%\"\n\n"
                    "Then run again."
                )
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync("vrgdg_instructions_popup", {
                        "message": message,
                        "type": "red",
                        "title": "FFmpeg Setup Required"
                    })
                except Exception:
                    pass
                print(message)
                return ()
            else:
                print(f"❌ [CreateFinalVideo] Failed to save audio: {e}")
                return ()

        # -------------------------------------------------
        # Combine video + audio
        # -------------------------------------------------
        print("[CreateFinalVideo] Adding original audio to video...")

        cmd_combine = [
            ffmpeg_path, "-y",
            "-i", temp_video,
            "-i", temp_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_output
        ]

        try:
            subprocess.run(cmd_combine, capture_output=True, text=True, errors="replace", check=True)
        except subprocess.CalledProcessError as e:
            if _is_libtorchcodec_error(e.stderr):
                print("[CreateFinalVideo] FFmpeg mux failed (libtorchcodec).")
                message = (
                    "❌ Final video creation failed due to missing FFmpeg shared libraries.\n\n"
                    "Fix:\n"
                    "1) Install the full shared build of FFmpeg into your portable directory:\n"
                    "   https://www.gyan.dev/ffmpeg/builds/\n"
                    "2) Copy the DLLs into the root folder.\n"
                    "3) Ensure your .bat includes:\n"
                    "   set \"PATH=%~dp0ffmpeg\\bin;%PATH%\"\n\n"
                    "Then run again."
                )
                try:
                    from server import PromptServer
                    PromptServer.instance.send_sync("vrgdg_instructions_popup", {
                        "message": message,
                        "type": "red",
                        "title": "FFmpeg Setup Required"
                    })
                except Exception:
                    pass
                print(message)
                return ()
            else:
                print(f"❌ [CreateFinalVideo] Failed to add audio: {e.stderr}")
                return ()
        except Exception as e:
            print(f"❌ [CreateFinalVideo] Failed to add audio: {e}")
            return ()

        os.remove(temp_video)
        os.remove(temp_audio)

        from server import PromptServer
        message = (
            f"🎉 Final video created!\n\n"
            f"📁 Location:\n{final_output}\n\n"
            f"✅ {video_count} sets combined\n"
            f"✅ Original clean audio added"
        )
        PromptServer.instance.send_sync("vrgdg_instructions_popup", {
            "message": message,
            "type": "green",
            "title": "✅ VIDEO COMPLETE!"
        })

        print(f"✅ [CreateFinalVideo] SUCCESS! Final video saved: {final_output}")

        return ()


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
    "VRGDG_AudioDelayByIndex": VRGDG_AudioDelayByIndex,
    "VRGDG_CreateFinalVideo_SRT": VRGDG_CreateFinalVideo_SRT,
    "VRGDG_RunStateLogger_SRT": VRGDG_RunStateLogger_SRT,
    "SRTLyricsMerger": SRTLyricsMerger,
    "VRGDG_StoryBoardCreator": VRGDG_StoryBoardCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_AudioDelayByIndex": "VRGDG_AudioDelayByIndex",
    "VRGDG_CreateFinalVideo_SRT": "VRGDG_CreateFinalVideo_SRT",
    "VRGDG_RunStateLogger_SRT": "VRGDG_RunStateLogger_SRT",
    "SRTLyricsMerger": "SRTLyricsMerger",
    "VRGDG_StoryBoardCreator": "VRGDG Legacy Storyboard Prompt Queue",
}
