import os, gc
import torch
import numpy as np
import imageio
import torchaudio
import random
import re
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import folder_paths
import json
from typing import TypedDict, Tuple
from torch import Tensor



class AUDIO(TypedDict):
    """
    Required Fields:
        waveform (torch.Tensor): Audio data [Batch, Channels, Frames]
        sample_rate (int): Sample rate of the audio.
    """
    waveform: Tensor
    sample_rate: int




class VRGDG_CombinevideosV2: 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_video_frames",)
    FUNCTION = "blend_videos"
    CATEGORY = "Video"

    @classmethod
    def INPUT_TYPES(cls):
        # Hardcode exactly 16 video inputs
        opt_videos = {f"video_{i}": ("IMAGE",) for i in range(1, 17)}

        return {
            "required": {
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0}),
                "audio_meta": ("DICT",),   # required to drive durations
            },
            "optional": {
                **opt_videos,
            },
        }

    # --- helpers -------------------------------------------------------------

    def _target_frames_for_index(self, durations, idx_zero_based, fps, current_frames):
        """durations[idx] seconds * fps -> target frames. If duration <= 0, use current_frames."""
        try:
            dur_sec = float(durations[idx_zero_based])
        except Exception:
            dur_sec = 0.0
        if dur_sec > 0.0:
            tgt = int(round(dur_sec * float(fps)))
            return max(1, tgt)
        return int(current_frames)

    def _trim_or_pad(self, video, target_frames, pad_short=False):
        if video is None:
            return None
        if video.ndim != 4:
            raise ValueError(f"Expected video tensor with 4 dims (frames,H,W,C), got {tuple(video.shape)}")
        cur = int(video.shape[0])
        if cur > target_frames:
            return video[:target_frames]
        if cur < target_frames and pad_short:
            need = target_frames - cur
            last = video[-1:].clone()
            pad = last.repeat(need, 1, 1, 1)
            return torch.cat([video, pad], dim=0)
        return video

    # --- main op -------------------------------------------------------------

    def blend_videos(self, fps, audio_meta, **kwargs):
        pad_short_videos = True
        effective_scene_count = 16  # hardcoded

        # ✅ Always use durations from audio_meta
        durations = []
        if isinstance(audio_meta, dict) and "durations" in audio_meta and isinstance(audio_meta["durations"], (list, tuple)):
            meta_durations = list(audio_meta.get("durations", []))
            if len(meta_durations) < effective_scene_count:
                meta_durations += [0.0] * (effective_scene_count - len(meta_durations))
            durations = meta_durations[:effective_scene_count]
        else:
            durations = [0.0] * effective_scene_count

        # Collect videos (up to 16, some may be None)
        vids = []
        for i in range(1, effective_scene_count + 1):
            v = kwargs.get(f"video_{i}")
            if v is not None:
                vids.append((i, v))

        if len(vids) < 1:
            raise ValueError("Provide at least one videos (e.g., video_1).")

        trimmed = []
        for slot_idx, vid in vids:
            if vid.ndim != 4:
                raise ValueError(f"video_{slot_idx} must have shape (frames,H,W,C), got {tuple(vid.shape)}")
            tgt = self._target_frames_for_index(durations, slot_idx - 1, fps, vid.shape[0])
            trimmed.append(self._trim_or_pad(vid, tgt, pad_short=pad_short_videos))

        final = torch.cat([t.to(dtype=torch.float32) for t in trimmed], dim=0).cpu()
        return (final,)





class VRGDG_PromptSplitter:
    RETURN_TYPES = tuple(["STRING"] * 50)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 51)])
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "scene_count": ("INT", {"default": 2, "min": 1, "max": 50}),
            }
        }

    @classmethod
    def IS_DYNAMIC(cls):
        return True

    @classmethod
    def get_output_types(cls, **kwargs):
        count = int(kwargs.get("scene_count", 2))
        count = max(1, min(50, count))
        return tuple(["STRING"] * count)

    @classmethod
    def get_output_names(cls, **kwargs):
        count = int(kwargs.get("scene_count", 2))
        count = max(1, min(50, count))
        return [f"text_output_{i+1}" for i in range(count)]

    def split_prompt(self, prompt_text, scene_count=2, **kwargs):
        scene_count = max(1, min(50, scene_count))
        parts = [p.strip() for p in prompt_text.strip().split("|") if p.strip()]
        outputs = [parts[i] if i < len(parts) else "" for i in range(scene_count)]
        return tuple(outputs)
    


class VRGDG_TimecodeFromIndex:
    # keep duration only in backend
    _DURATION_SECONDS = 62.0  

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("start_time",)

    FUNCTION = "format_timecode"

    CATEGORY = "utils"

    def format_timecode(self, index):
        start_seconds = int(index * self._DURATION_SECONDS)
        start_time_str = f"{start_seconds // 60}:{start_seconds % 60:02d}"
        return (start_time_str,)







class VRGDG_ConditionalLoadVideos:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load_videos"
    CATEGORY = "Video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("VHS_FILENAMES", {}),
                "threshold": ("INT", {"default": 3}),
                "video_folder": ("STRING", {"default": "./videos", "multiline": False}),
                "batch_size": ("INT", {"default": 100, "min": 1, "max": 1000}),  # 🔧 new option
            }
        }

    def load_videos(self, trigger, threshold, video_folder, batch_size=100):
        video_folder = video_folder.strip()

        # ensure folder exists
        if not os.path.exists(video_folder):
            os.makedirs(video_folder, exist_ok=True)
            print(f"[VRGDG_ConditionalLoadVideos] Created folder: {video_folder}")
            print(f"[VRGDG_ConditionalLoadVideos] No videos yet, skipping.")
            return (None,)

        # collect only -audio.mp4 (final videos)
        videos = sorted(
            [f for f in os.listdir(video_folder)
             if f.lower().endswith(".mp4") and "-audio" in f.lower()]
        )
        video_count = len(videos)

        print(f"[VRGDG_ConditionalLoadVideos] Found {video_count} -audio.mp4 videos in {video_folder}")

        if video_count < threshold:
            print(f"[VRGDG_ConditionalLoadVideos] Threshold not met ({video_count}/{threshold}), skipping.")
            return (None,)

        all_frames = []
        for vid in videos:
            path = os.path.join(video_folder, vid)
            print(f"[VRGDG_ConditionalLoadVideos] Loading {path}")
            reader = imageio.get_reader(path, "ffmpeg")

            frames = []
            batch = []
            frame_count = 0

            for frame in reader:
                # normalize to float32 RGB tensor
                frame = frame.astype(np.float32) / 255.0
                tensor = torch.from_numpy(frame).unsqueeze(0)  # (1,H,W,C)
                batch.append(tensor)
                frame_count += 1

                # flush in batches
                if len(batch) >= batch_size:
                    frames.append(torch.cat(batch, dim=0))
                    batch = []

                if frame_count % 500 == 0:
                    print(f"[VRGDG_ConditionalLoadVideos] {vid}: loaded {frame_count} frames...")

            # add any leftovers
            if batch:
                frames.append(torch.cat(batch, dim=0))

            reader.close()

            if not frames:
                print(f"[VRGDG_ConditionalLoadVideos] Skipped {vid}, no frames read.")
                continue

            video_tensor = torch.cat(frames, dim=0)  # (F,H,W,C)
            all_frames.append(video_tensor)

            print(f"[VRGDG_ConditionalLoadVideos] Finished {vid} with {video_tensor.shape[0]} frames")

            # cleanup per video
            del frames, batch
            gc.collect()

        if not all_frames:
            print("[VRGDG_ConditionalLoadVideos] No valid videos loaded, returning None.")
            return (None,)

        # final concat + cleanup
        final_video = torch.cat(all_frames, dim=0)  # (ΣF,H,W,C)
        print(f"[VRGDG_ConditionalLoadVideos] Returning tensor with {final_video.shape[0]} frames")

        del all_frames
        gc.collect()

        # move to CPU (important for downstream save nodes)
        final_video = final_video.cpu()

        return (final_video,)



class VRGDG_CalculateSetsFromAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT",)
    RETURN_NAMES = ("instructions", "end_time", "total_sets",)
    FUNCTION = "calculate"

    CATEGORY = "utils/audio"

    def calculate(self, audio, index):
        set_duration = 62.0
        group_duration = 3.88
        groups_per_set = 16

        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except Exception:
            return ("❌ Expected audio to be a dict with 'waveform' and 'sample_rate'.", "0:00", 0)

        try:
            num_samples = waveform.shape[-1]
            audio_duration = num_samples / sample_rate
        except Exception:
            return ("❌ Failed to compute audio duration from waveform.", "0:00", 0)

        # Convert duration to MM:SS
        minutes = int(audio_duration // 60)
        seconds = int(audio_duration % 60)
        end_time_str = f"{minutes}:{seconds:02d}"

        # Compute sets
        full_sets = int(audio_duration // set_duration)
        remainder = audio_duration - (full_sets * set_duration)

        if remainder > 0:
            total_sets = full_sets + 1
            groups_in_last_set = int(min(remainder // group_duration, groups_per_set))
        else:
            total_sets = full_sets
            groups_in_last_set = groups_per_set

        # ---------------------
        # Instructions section
        # ---------------------
        if index == 0:
            run_num = 1
            header = f"▶️ Run {run_num} of {total_sets} in progress…\n"

            if audio_duration < set_duration:
                instructions = (
                    header +
                    f"Audio is shorter than one set (62s). "
                    f"Cancel this run and disable groups {groups_in_last_set+1}–16 "
                    f"so only groups 1–{groups_in_last_set} are enabled then run again."
                )

            elif total_sets == 1:
                instructions = (
                    header +
                    "Audio is exactly one full set (62s) so you’re good to go! "
                    "You don’t need to run again."
                )

            elif remainder > 0:
                middle_runs = max(total_sets - 2, 0)

                if groups_in_last_set == 0:
                    # Remainder doesn’t even cover one group
                    instructions = (
                        header +
                        f"This audio requires {total_sets-1} full runs in total.\n"
                        "You don’t need to run again after the last full set."
                    )

                elif middle_runs > 0:
                    instructions = (
                        header +
                        f"This audio requires {total_sets} runs in total.\n"
                        f"➡️ Click 'Run' {middle_runs} more times with all 16 groups enabled.\n"
                        f"➡️ Then, disable groups {groups_in_last_set+1}–16 so only groups 1–{groups_in_last_set} are enabled, "
                        f"➡️ and click 'Run' once more."
                    )
                else:
                    # Only 2 runs total — no "Then"
                    instructions = (
                        header +
                        f"This audio requires {total_sets} runs in total.\n"
                        f"➡️ Disable groups {groups_in_last_set+1}–16 so only groups 1–{groups_in_last_set} are enabled, "
                        f"➡️ and click 'Run' once more."
                    )

            else:
                instructions = (
                    header +
                    f"This audio requires {total_sets} runs in total.\n"
                    f"Click 'Run' {total_sets - 1} more times. "
                    "Keep all 16 groups enabled for every run."
                )

        elif index < total_sets - 1:
            # 🎬 Middle runs
            run_num = index + 1
            instructions = (
                f"🎬 Video creation in progress…\n"
                f"➡️ Run {run_num} of {total_sets}"
            )

        else:
            # 🏁 Final run
            run_num = index + 1
            instructions = (
                f"🏁 Final run in progress…\n"
                f"➡️ Run {run_num} of {total_sets}"
            )

        return (instructions, end_time_str, total_sets)



class VRGDG_GetFilenamePrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Filename_Prefix",)
    FUNCTION = "get_prefix"

    CATEGORY = "utils/files"

    def get_prefix(self, folder_path):
        # Clean up any accidental whitespace or newline characters
        folder_path = folder_path.strip()

        # Ensure the folder exists (create it if missing)
        os.makedirs(folder_path, exist_ok=True)

        # Normalize and extract last folder name
        base_name = os.path.basename(os.path.normpath(folder_path))

        # Build result in an OS-safe way
        result = os.path.join(base_name, "video")

        return (result,)


class VRGDG_TriggerCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {"id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)
    FUNCTION = "generate"

    CATEGORY = "utils/control"

    def generate(self, seed, id=None):
        # Just return the seed, ComfyUI handles incrementing via control_after_generate
        return (seed,)
    


class VRGDG_LoadAudioSplit_HUMO_TranscribeV2:

    RETURN_TYPES = ("DICT", "FLOAT", "STRING") + tuple(["AUDIO"] * 16)
    RETURN_NAMES = ("meta", "total_duration", "lyrics_string") + tuple([f"audio_{i}" for i in range(1, 17)])
    FUNCTION = "split_audio"
    CATEGORY = "VRGDG"

    fallback_words = ["standing", "sitting", "laying", "resting", "waiting",
                      "walking", "dancing", "looking", "thinking"]

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        hidden = {}
        for i in range(1, 17):
            optional[f"context_{i}"] = ("STRING", {"default": "", "multiline": True})
            hidden[f"play_{i}"] = ("BUTTON", {"label": f"▶️ Play {i}"})

        return {
            "required": {
                "audio": ("AUDIO",),
                "language": (
                    [
                        "auto", "english", "chinese", "german", "spanish", "russian", "korean", "french",
                        "japanese", "portuguese", "turkish", "polish", "catalan", "dutch", "arabic", "swedish",
                        "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew", "ukrainian", "greek",
                        "malay", "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", "thai", "urdu",
                        "croatian", "bulgarian", "lithuanian", "latin", "maori", "malayalam", "welsh", "slovak",
                        "telugu", "persian", "latvian", "bengali", "serbian", "azerbaijani", "slovenian", "kannada",
                        "estonian", "macedonian", "breton", "basque", "icelandic", "armenian", "nepali", "mongolian",
                        "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi", "punjabi", "sinhala",
                        "khmer", "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", "belarusian",
                        "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese", "haitian creole",
                        "pashto", "turkmen", "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan",
                        "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", "hausa", "bashkir",
                        "javanese", "sundanese", "cantonese", "burmese", "valencian", "flemish", "haitian",
                        "letzeburgesch", "pushto", "panjabi", "moldavian", "moldovan", "sinhalese", "castilian", "mandarin"
                    ],
                    {"default": "english"}
                ),
                "enable_lyrics": ("BOOLEAN", {"default": True}),
                "overlap_lyric_seconds": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "fallback_words": ("STRING", {"default": "thinking,walking,sitting"}),
            },
            "optional": optional,
            "hidden": hidden,
        }

    def split_audio(self, audio, language="english", enable_lyrics=True, overlap_lyric_seconds=0.0, fallback_words="", **kwargs):
        waveform = audio["waveform"]
        src_sample_rate = int(audio.get("sample_rate", 44100))
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        total_duration = float(total_samples) / float(src_sample_rate)

        scene_count = 16
        durations = [3.88] * scene_count
        starts = [i * 3.88 for i in range(scene_count)]

        segments = []
        transcriptions = []

        if enable_lyrics:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device).eval()
        else:
            processor = model = device = None

        output_dir = folder_paths.get_input_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Clear out old audio files in the audiochunks folder
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if filename.startswith("audio_") and filename.endswith(".wav"):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


        fb_words = [w.strip() for w in fallback_words.split(",") if w.strip()] or self.fallback_words

        overlap_samples = int(overlap_lyric_seconds * src_sample_rate)

        for idx, start_time in enumerate(starts):
            dur = durations[idx]
            start_samp = max(0, int(start_time * src_sample_rate))
            end_samp = min(total_samples, int(start_samp + dur * src_sample_rate))

            # --- segment for saving (always 3.88s) ---
            seg = waveform[..., start_samp:end_samp].contiguous().clone()
            expected_len = int(dur * src_sample_rate)
            if seg.shape[-1] < expected_len:
                seg = torch.nn.functional.pad(seg, (0, expected_len - seg.shape[-1]))
            elif seg.shape[-1] > expected_len:
                seg = seg[..., :expected_len]
            segments.append({"waveform": seg, "sample_rate": src_sample_rate})

            filepath = os.path.join(output_dir, f"audio_{idx+1}.wav")
            torchaudio.save(filepath, seg.squeeze(0).cpu(), src_sample_rate)

            if not enable_lyrics:
                transcriptions.append("")
                continue

            # --- segment for transcription (extended with overlap) ---
            trans_start = max(0, start_samp - overlap_samples)
            trans_end = min(total_samples, end_samp + overlap_samples)
            seg_for_transcribe = waveform[..., trans_start:trans_end].contiguous().clone()

            try:
                flat_seg = seg_for_transcribe.mean(dim=1).squeeze()
                if src_sample_rate != 16000:
                    flat_seg = torchaudio.functional.resample(flat_seg, src_sample_rate, 16000)

                inputs = processor(flat_seg, sampling_rate=16000, return_tensors="pt", padding="longest")
                input_features = inputs["input_features"].to(device)
                if language == "auto":
                    generated_ids = model.generate(input_features)
                else:
                    decoder_ids = processor.get_decoder_prompt_ids(language=language)
                    generated_ids = model.generate(input_features, forced_decoder_ids=decoder_ids)

                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if not text:
                    text = random.choice(fb_words)
            except Exception:
                text = random.choice(fb_words)

            transcriptions.append(text)

        safe_transcriptions = [t if t else random.choice(fb_words) for t in transcriptions]

        enriched = []
        for i in range(scene_count):
            lyric_line = safe_transcriptions[i]
            context = kwargs.get(f"context_{i+1}", "").strip()
            if context:
                lyric_line = f"{context}, {lyric_line}"
            enriched.append(lyric_line)

        lyrics_text = " | ".join(enriched)

        meta = {
            "durations": durations,
            "offset_seconds": 0.0,
            "starts": starts,
            "sample_rate": src_sample_rate,
            "audio_total_duration": total_duration,
            "outputs_count": len(segments),
            "used_padding": False,
        }

        return (meta, total_duration, lyrics_text, *tuple(segments))



class VRGDG_StringConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "song_theme_style": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "pipe_separated_lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated_string",)

    FUNCTION = "concat_strings"
    CATEGORY = "VRGDG/Prompt Tools"

    def concat_strings(self, instructions, song_theme_style, pipe_separated_lyrics):
        full_string = (
            "Instructions:\n" + instructions.strip() + "\n\n"
            "Song theme/style:\n" + song_theme_style.strip() + "\n\n"
            "Pipe separated lyrics:\n" + pipe_separated_lyrics.strip()
        )
        return (full_string,)



class VRGDG_AudioCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": (
                    "STRING",
                    {
                        "default": "0:00",
                    },
                ),
                "end_time": (
                    "STRING",
                    {
                        "default": "1:00",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"
    DESCRIPTION = "Crop (trim) audio to a specific start and end time."

    def main(
        self,
        audio: AUDIO,
        start_time: str = "0:00",
        end_time: str = "1:00",
    ) -> Tuple[AUDIO]:

        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        if ":" not in start_time:
            start_time = f"00:{start_time}"
        if ":" not in end_time:
            end_time = f"00:{end_time}"

        start_seconds_time = 60 * int(start_time.split(":")[0]) + int(
            start_time.split(":")[1]
        )
        start_frame = start_seconds_time * sample_rate
        if start_frame >= waveform.shape[-1]:
            start_frame = waveform.shape[-1] - 1

        end_seconds_time = 60 * int(end_time.split(":")[0]) + int(
            end_time.split(":")[1]
        )
        end_frame = end_seconds_time * sample_rate
        if end_frame >= waveform.shape[-1]:
            end_frame = waveform.shape[-1] - 1
        if start_frame < 0:
            start_frame = 0
        if end_frame < 0:
            end_frame = 0

        if start_frame > end_frame:
            total_duration_sec = waveform.shape[-1] / sample_rate
            raise ValueError(
                f"Invalid crop range:\n"
                f"- Start time: {start_seconds_time} sec\n"
                f"- End time: {end_seconds_time} sec\n"
                f"- Total duration: {total_duration_sec:.2f} sec\n"
                f"Start time must come before end time, and both must be within the audio duration.\n"
                f"If this is your first run, double-check that the index or batch position is set to 0 or not set higher than the total number of sets in the read-me note."
            )


        return (
            {
                "waveform": waveform[..., start_frame:end_frame],
                "sample_rate": sample_rate,
            },
        )


class VRGDG_GetIndexNumber:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("INT",),  # ✅ forces execution each run
                "folder_path": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)

    FUNCTION = "count_videos"

    CATEGORY = "utils"

    def count_videos(self, trigger, folder_path):
        import os
        # 🔹 Always run — trigger is just here to ensure ComfyUI re-executes on queue
        if not os.path.isdir(folder_path):
            return (0,)

        mp4_count = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(".mp4") and "-audio" in f.lower()
        ])

        return (mp4_count,)





class VRGDG_DisplayIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    # We output a STRING so ComfyUI will actually render it in the node UI
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("index_display",)

    FUNCTION = "show"

    OUTPUT_NODE = True   # makes this a "display-only" node

    CATEGORY = "utils"

    def show(self, index):
        # Convert int to string for UI preview
        return (f"Current index: {index}",)



NODE_CLASS_MAPPINGS = {

     "VRGDG_CombinevideosV2": VRGDG_CombinevideosV2,
     "VRGDG_PromptSplitter":VRGDG_PromptSplitter,
     "VRGDG_TimecodeFromIndex":VRGDG_TimecodeFromIndex,
     "VRGDG_ConditionalLoadVideos":VRGDG_ConditionalLoadVideos,
     "VRGDG_CalculateSetsFromAudio":VRGDG_CalculateSetsFromAudio,
     "VRGDG_GetFilenamePrefix":VRGDG_GetFilenamePrefix,
     "VRGDG_TriggerCounter":VRGDG_TriggerCounter,
     "VRGDG_LoadAudioSplit_HUMO_TranscribeV2":VRGDG_LoadAudioSplit_HUMO_TranscribeV2,
     "VRGDG_StringConcat":VRGDG_StringConcat,
     "VRGDG_AudioCrop":VRGDG_AudioCrop,
     "VRGDG_GetIndexNumber":VRGDG_GetIndexNumber,
     "VRGDG_DisplayIndex":VRGDG_DisplayIndex



}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_CombinevideosV2": "🌀 VRGDG_CombinevideosV2",
    "VRGDG_PromptSplitter": "✂️ VRGDG_PromptSplitter",
    "VRGDG_TimecodeFromIndex": "⏱️ VRGDG_TimecodeFromIndex",
    "VRGDG_ConditionalLoadVideos": "🎞️ VRGDG_ConditionalLoadVideos",
    "VRGDG_CalculateSetsFromAudio": "🎧 VRGDG_CalculateSetsFromAudio",
    "VRGDG_GetFilenamePrefix": "📂 VRGDG_GetFilenamePrefix",
    "VRGDG_TriggerCounter": "🎯 VRGDG_TriggerCounter",
    "VRGDG_LoadAudioSplit_HUMO_TranscribeV2": "🗣️ VRGDG_LoadAudioSplit_HUMO_TranscribeV2",
    "VRGDG_StringConcat":"VRGDG_StringConcat",
    "VRGDG_AudioCrop":"VRGDG_AudioCrop",
    "VRGDG_GetIndexNumber":"VRGDG_GetIndexNumber",
    "VRGDG_DisplayIndex":"VRGDG_DisplayIndex"
}
