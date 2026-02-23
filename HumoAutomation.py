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
from server import PromptServer
from torchaudio.transforms import Vad

#########################
import subprocess

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

####################

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
    # one set = 16 groups × 97 frames @ 25 fps
    _FRAMES_PER_GROUP = 97
    _FPS = 25
    _GROUPS_PER_SET = 16

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
        # compute exact duration per set
        set_duration = (self._FRAMES_PER_GROUP * self._GROUPS_PER_SET) / self._FPS  # 62.08
        start_seconds = index * set_duration
        minutes = int(start_seconds // 60)
        seconds = start_seconds % 60
        return (f"{minutes}:{seconds:05.2f}",)






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

        import math

        if remainder > 0:
            total_sets = full_sets + 1
            groups_in_last_set = min(math.ceil(remainder / group_duration), groups_per_set)
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
                "set_index": ("INT", {"default": 0, "min": 0}),
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

   
    def split_audio(
        self,
        audio,
        set_index=0,              
        language="english",
        enable_lyrics=True,
        overlap_lyric_seconds=0.0,
        fallback_words="",
        **kwargs
    ):
        waveform = audio["waveform"]
        src_sample_rate = int(audio.get("sample_rate", 44100))
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        total_duration = float(total_samples) / float(src_sample_rate)

        scene_count = 16
        fps = 25
        frames_per_scene = 97
        samples_per_frame = int(round(src_sample_rate / fps))
        samples_per_scene = frames_per_scene * samples_per_frame

        # NEW: offset per set
        offset_samples = set_index * scene_count * samples_per_scene
        print(f"[Split] set_index={set_index}, offset_samples={offset_samples}")

        durations = [frames_per_scene / fps] * scene_count
        starts = [offset_samples + i * samples_per_scene for i in range(scene_count)]  # integer sample indices
     
# ---- Split precisely on integer sample boundaries ----
        segments = []
        total_len_samples = scene_count * samples_per_scene

        # Always try to make `scene_count` groups, even if audio runs short
        for idx in range(scene_count):
            start_samp = offset_samples + idx * samples_per_scene
            end_samp = start_samp + samples_per_scene

            # If we're entirely past end of file, fill full silence
            if start_samp >= total_samples:
                seg = torch.zeros((1, 1, samples_per_scene), dtype=waveform.dtype)
                print(f"[Split] idx={idx} filled with silence (no audio left)")
            else:
                # Clamp end within the file
                end_samp = min(total_samples, end_samp)
                seg = waveform[..., start_samp:end_samp].contiguous().clone()
                cur_len = seg.shape[-1]

                # Pad with silence if short
                if cur_len < samples_per_scene:
                    pad = samples_per_scene - cur_len
                    seg = torch.nn.functional.pad(seg, (0, pad))
                    print(f"[Split] idx={idx} padded {pad} samples")

            segments.append({"waveform": seg, "sample_rate": src_sample_rate})

        print(f"[Split] Created {len(segments)} segments (scene_count={scene_count})")

        # Never return None — always at least silence
        if not segments:
            empty_audio = {"waveform": torch.zeros((1, 1, samples_per_scene)), "sample_rate": src_sample_rate}
            return (empty_audio, total_duration, "")
        # -----------------------------------------------------------
        # The rest of your existing transcription and saving logic
        # -----------------------------------------------------------
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

        transcriptions = []
        for idx, start_samp in enumerate(starts):
            end_samp = min(total_samples, start_samp + samples_per_scene)
            filepath = os.path.join(output_dir, f"audio_{idx+1}.wav")
            torchaudio.save(filepath, segments[idx]["waveform"].squeeze(0).cpu(), src_sample_rate)

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
        audio: "AUDIO",
        start_time: str = "0:00",
        end_time: str = "1:00",
    ):
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]

        if ":" not in start_time:
            start_time = f"00:{start_time}"
        if ":" not in end_time:
            end_time = f"00:{end_time}"

        # --- UPDATED PARSING (accepts mm:ss or mm:ss.xx) ---
        start_min, start_sec = start_time.split(":")
        start_seconds_time = 60 * int(start_min) + float(start_sec)
        start_frame = int(start_seconds_time * sample_rate)
        if start_frame >= waveform.shape[-1]:
            start_frame = waveform.shape[-1] - 1

        end_min, end_sec = end_time.split(":")
        end_seconds_time = 60 * int(end_min) + float(end_sec)
        end_frame = int(end_seconds_time * sample_rate)
        if end_frame >= waveform.shape[-1]:
            end_frame = waveform.shape[-1] - 1
        # --- END UPDATED SECTION ---

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


class VRGDG_PromptSplitterV2:
    RETURN_TYPES = tuple(["STRING"] * 16)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 17)])
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def split_prompt(self, prompt_text, **kwargs):
        parts = [p.strip() for p in prompt_text.strip().split("|") if p.strip()]
        outputs = [parts[i] if i < len(parts) else "" for i in range(16)]
        return tuple(outputs)



class VRGDG_CombinevideosV3: 
    VERSION = "v3.1_fixed"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_video_frames",)
    FUNCTION = "blend_videos"
    CATEGORY = "Video"

    @classmethod
    def INPUT_TYPES(cls):
        # fixed 16 video inputs
        opt_videos = {f"video_{i}": ("IMAGE",) for i in range(1, 17)}
        return {
            "required": {
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0}),
                "duration": ("FLOAT", {"default": 4.0, "min": 0.01}),  # 👈 per-group duration in seconds
                "audio_meta": ("DICT",),  # always use meta for durations
                "index": ("INT", {"default": 0, "min": 0}),
                "total_sets": ("INT", {"default": 1, "min": 1}),
                "groups_in_last_set": ("INT", {"default": 16, "min": 0, "max": 16}),
            },
            "optional": {**opt_videos},
        }

    # --- helpers -------------------------------------------------------------

    def _target_frames_for_index(self, durations, idx_zero_based, fps, is_frames=False):
        """
        Decide target frame count from audio_meta.
        durations can be in seconds OR frames depending on the key.
        """
        value = float(durations[idx_zero_based])
        
        if is_frames:
            # Already in frames
            return max(1, int(round(value)))
        else:
            # In seconds, convert to frames
            frames_per_scene = int(round(fps * value))
            return max(1, frames_per_scene)


    def _trim_or_pad(self, video, target_frames):
        if video is None:
            return None
        if video.ndim != 4:
            raise ValueError(f"Expected video tensor with 4 dims (frames,H,W,C), got {tuple(video.shape)}")
        
        cur = int(video.shape[0])
        
        if cur > target_frames:
            # Trim if too long
            return video[:target_frames]
        
        if cur < target_frames:
            # Warn if too short (video generation issue)
            #print(f"⚠️ WARNING: Video has {cur} frames but needs {target_frames} (short by {target_frames - cur})")
            return video  # Return as-is, don't pad
        
        return video
    # --- main op -------------------------------------------------------------

    def blend_videos(self, fps, duration, audio_meta=None, index=0, total_sets=1, groups_in_last_set=16, **kwargs):
        print(f"[CombineV3 {self.VERSION}] index={index}, ...")  # Update this line

        effective_scene_count = 16
        is_last_run = (index == total_sets - 1)

        print(f"[CombineV3] index={index}, total_sets={total_sets}, last_run={is_last_run}")

        # ✅ Always use durations from audio_meta
        if not isinstance(audio_meta, dict):
            raise ValueError("[CombineV3] audio_meta must be a dict")

        # Accept either key: "durations" (seconds) or "durations_frames" (frames)
        durations_seconds = audio_meta.get("durations")
        durations_frames = audio_meta.get("durations_frames")

        if durations_frames is not None:
            durations = list(durations_frames)
            is_frames = True
            print(f"[CombineV3] Using durations_frames (already in frames)")
        elif durations_seconds is not None:
            durations = list(durations_seconds)
            is_frames = False
            print(f"[CombineV3] Using durations (in seconds)")
        else:
            raise ValueError("[CombineV3] audio_meta missing 'durations' or 'durations_frames' list")

        # Pad or trim durations list
        if len(durations) < effective_scene_count:
            durations += [0.0] * (effective_scene_count - len(durations))
        else:
            durations = durations[:effective_scene_count]

        # Determine how many videos to process this run
        limit_videos = effective_scene_count
        if is_last_run:
            limit_videos = max(1, min(groups_in_last_set, effective_scene_count))

        # Collect videos
        vids = []
        for i in range(1, limit_videos + 1):
            v = kwargs.get(f"video_{i}")
            if v is not None:
                vids.append((i, v))

        if len(vids) < 1:
            raise ValueError("[CombineV3] No video inputs detected. Connect at least one video_x input.")

        print(f"[CombineV3] combining {len(vids)} video(s) for this run (limit={limit_videos})")

        trimmed = []
        for slot_idx, vid in vids:
            if vid.ndim != 4:
                raise ValueError(
                    f"video_{slot_idx} must have shape (frames,H,W,C), got {tuple(vid.shape)}"
                )

            # --- Decide target frames ---
            tgt = self._target_frames_for_index(durations, slot_idx - 1, fps, is_frames=is_frames)

            # --- If last run and slot beyond available groups, skip it ---
            if is_last_run and slot_idx > groups_in_last_set:
                print(f"[CombineV3] video_{slot_idx} skipped (beyond groups_in_last_set)")
                continue

            # --- Normal trim ---
            trimmed_vid = self._trim_or_pad(vid, tgt)

            print(
                f"[CombineV3] video_{slot_idx}: actual={vid.shape[0]} -> target={tgt}, final={trimmed_vid.shape[0]}"
            )
            trimmed.append(trimmed_vid)

        # Concatenate along frame dimension
        final = torch.cat([t.to(dtype=torch.float32) for t in trimmed], dim=0).cpu()

        print(f"[CombineV3] final concatenated frames: {final.shape[0]}")
        return (final,)


####################################added on 10-5


   
# wildcard trick is taken from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

from server import PromptServer

class VRGDG_QueueTriggerFromAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "signal": (any_typ,),
                    "mode": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
                    "total_sets": ("INT", {"default": 1, "min": 1}),
                    "groups_in_last_set": ("INT", {"default": 16, "min": 0, "max": 16}),
                    "index": ("INT", {"default": 0, "min": 0}),  # ✅ added index input
                    }
                }

    FUNCTION = "doit"

    CATEGORY = "Utilities"
    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("signal_opt",)
    OUTPUT_NODE = True

    def doit(self, signal, mode, total_sets, groups_in_last_set, index=0):
        # Start with 0 extra runs
        runs = 0

        # ✅ Only trigger auto queue on the very first run
        if mode and index == 0:
            if total_sets > 0:
                if groups_in_last_set == 16:
                    # Last set is full → first click counts as run #1
                    runs = max(0, total_sets - 1)
                else:
                    # Last set is partial → first click counts as run #1, leave final run manual
                    runs = max(0, total_sets - 2)

            print(f"[VRGDG_QueueTriggerFromAudio] total_sets={total_sets}, groups_in_last_set={groups_in_last_set}, index={index}")
            print(f"[VRGDG_QueueTriggerFromAudio] Queuing {runs} runs (first Run already counts as one)")

            for _ in range(runs):
                PromptServer.instance.send_sync("impact-add-queue", {})

        else:
            # ✅ Skip auto-queue if index > 0 (already inside queued runs)
            print(f"[VRGDG_QueueTriggerFromAudio] Skipping auto-queue (index={index})")

        return (signal,)

    
import re

class VRGDG_ThemeSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context_block": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING", "STRING"
    )
    RETURN_NAMES = (
        "character_description",
        "song_theme_style",
        "environment",
        "lighting",
        "camera_motion",
        "physical_interaction",
        "facial_expression",
        "shots",
        "outfit_rules",
        "character_visibility"
    )

    FUNCTION = "split_context"
    CATEGORY = "VRGDG/Prompt Tools"

    def normalize_key(self, text):
        """Normalize keys by lowercasing and stripping spaces/underscores."""
        return re.sub(r'[^a-z]', '', text.strip().lower())

    def split_context(self, context_block):
        # Initialize empty slots
        sections = {
            "character_description": "",
            "song_theme_style": "",
            "environment": "",
            "lighting": "",
            "camera_motion": "",
            "physical_interaction": "",
            "facial_expression": "",
            "shots": "",
            "outfit_rules": "",
            "character_visibility": ""
        }

        # Map normalized keys to real keys
        key_map = {self.normalize_key(k): k for k in sections}

        lines = context_block.splitlines()
        current_key = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            norm = self.normalize_key(line)

            if norm in key_map:
                # Found a new section header
                current_key = key_map[norm]
            elif current_key:
                # Append line text (skip the header itself)
                if sections[current_key]:
                    sections[current_key] += " " + line
                else:
                    sections[current_key] = line

        # Return values in fixed order, headers stripped
        return tuple(sections[k] for k in sections)


import math


class VRGDG_CalculateSetsFromAudio_Queue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "index": ("INT", {"default": 0, "min": 0}),
                # "scene_duration": ("FLOAT", {"default": 5.0, "min": 0.01, "step": 0.01}),
                # 🔸 Scene duration input commented out — now hardcoded to 3.88 seconds below
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "DICT")
    RETURN_NAMES = (
        "instructions",
        "end_time",
        "total_sets",
        "groups_in_last_set",
        "frames_per_scene",
        "audio_meta"
    )
    FUNCTION = "calculate"
    CATEGORY = "utils/audio"

    def calculate(self, audio, index):
        fps = 25
        groups_per_set = 16
        scene_duration = 3.88  # 🔸 Hardcoded scene duration (was user input before)

        # 🔍 Debug print to check precise float math alignment
        print(f"[CalcSets] scene_duration={scene_duration}, fps={fps}, product={scene_duration*fps}")

        # ---- Get audio duration ----
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except Exception:
            return ("❌ Expected audio to be a dict with 'waveform' and 'sample_rate'.",
                    "0:00", 0, 0, 0, {})

        try:
            num_samples = waveform.shape[-1]
            audio_duration = num_samples / sample_rate
        except Exception:
            return ("❌ Failed to compute audio duration from waveform.",
                    "0:00", 0, 0, 0, {})

        # End time
        minutes = int(audio_duration // 60)
        seconds = int(audio_duration % 60)
        end_time_str = f"{minutes}:{seconds:02d}"

        # ---- Frame-based durations ----
        frames_per_scene = int(round(scene_duration * fps))
        total_audio_frames = int(round(audio_duration * fps))

        durations_frames_full = []
        if total_audio_frames > 0:
            full_groups = total_audio_frames // frames_per_scene
            leftover_frames = total_audio_frames - full_groups * frames_per_scene
            if full_groups > 0:
                durations_frames_full.extend([frames_per_scene] * full_groups)
            if leftover_frames > 0:
                durations_frames_full.append(leftover_frames)

        total_groups = len(durations_frames_full)
        total_sets = math.ceil(total_groups / groups_per_set) if total_groups > 0 else 0
        groups_in_last_set = (total_groups % groups_per_set) if total_groups % groups_per_set != 0 else (groups_per_set if total_groups > 0 else 0)

        fps = 25
        frames_per_scene = 97

        # exact integer samples per frame at 48 kHz = 1920
        samples_per_frame = round(sample_rate / fps)          # 1920 at 48 kHz
        total_audio_frames = num_samples // samples_per_frame  # <-- floor, not round

        durations_frames_full = []
        if total_audio_frames > 0:
            full_groups = total_audio_frames // frames_per_scene
            leftover_frames = total_audio_frames % frames_per_scene
            if full_groups > 0:
                durations_frames_full.extend([frames_per_scene] * full_groups)
            if leftover_frames > 0:
                durations_frames_full.append(leftover_frames)
        # ---- Instructions (your original style) ----
        if total_sets == 0:
            instructions = "❌ Audio too short. No runs required."
        elif groups_in_last_set == 16:
            instructions = (
                f"✅ {total_sets} runs needed.\n"
                f"All runs are already queued automatically.\n"
                f"No more action required."
            )
        elif total_sets == 1:
            if groups_in_last_set == 15:
                disable_text = "disable 16"
            else:
                disable_text = f"disable {groups_in_last_set+1}–16"
            instructions = (
                f"⚠️ 1 run needed.\n"
                f"No runs were queued automatically.\n"
                f"Set groups 1–{groups_in_last_set} ON and {disable_text},\n"
                f"then press Run ONE time."
            )
        else:
            if groups_in_last_set == 15:
                disable_text = "disable 16"
            else:
                disable_text = f"disable {groups_in_last_set+1}–16"
            instructions = (
                f"⚠️ {total_sets} runs needed.\n"
                f"{total_sets - 1} full runs are already queued automatically.\n"
                f"Set groups 1–{groups_in_last_set} ON and {disable_text},\n"
                f"then press Run ONE more time for the last set."
            )


        return (
            instructions,
            end_time_str,
            total_sets,
            groups_in_last_set,
            frames_per_scene,
            {"durations_frames": durations_frames_full}   # ✅ always full list
        )



class VRGDG_MusicVideoPromptCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": "The Women.",
                }),
                "song_theme_style": ("STRING", {
                    "multiline": True,
                    "default": "Cinematic, dramatic, vibrant, and edgy ",
                }),
                "pipe_separated_lyrics": ("STRING", {
                    "multiline": True,
                    "default": "line1 | line2 | line3",
                }),
                "word_count_min": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 200,
                }),
                "word_count_max": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                }),
                "environment": ("STRING", {
                    "multiline": True,
                    "default": " rain-slicked city street, night, minimalist, industrial warehouse.",
                }),
                "lighting": ("STRING", {
                    "multiline": True,
                    "default": "High-contrast, dramatically moody, hard shadows, single, intense practical light sources, cool blues, sharp whites.",
                }),
                "camera_motion": ("STRING", {
                    "multiline": True,
                    "default": "dolly movements, track alongside, swift, sudden whip pans, emphasize dramatic shifts, dynamic motion.",
                }),
                "physical_interaction": ("STRING", {
                    "multiline": True,
                    "default": "walks through environments while touching walls or objects. Touches hair, gesture toward the camera to connect.",
                }),
                "facial_expression": ("STRING", {
                    "multiline": True,
                    "default": "Intense raw emotion and Brief moments of calm break the tension.",
                }),
                "shots": ("STRING", {
                    "multiline": True,
                    "default": "Use a mix of close-ups and medium shots for intimacy and Wide moving shots follow the performer through spaces.",
                }),
                "outfit_rules": ("STRING", {
                    "multiline": True,
                    "default": "a white dress",
                }),
                "character_visibility": ("STRING", {
                    "multiline": True,
                    "default": "Fully present for a majority of shots",
                }),
                "signal": (any_typ,),  # 👈 added universal passthrough at the very end
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated_string",)

    FUNCTION = "build_prompt_instructions"
    CATEGORY = "VRGDG/Prompt Tools"

    def build_prompt_instructions(
        self,
        character_description,
        song_theme_style,
        pipe_separated_lyrics,
        word_count_min,
        word_count_max,
        environment,
        lighting,
        camera_motion,
        physical_interaction,
        facial_expression,
        shots,
        outfit_rules,
        character_visibility,
        signal,  # 👈 new passthrough input
    ):
        full_string = f"""
AI Music Video Prompt Creator

User Input:
Character: {character_description.strip()}
Style/Theme: {song_theme_style.strip()}
Lyrics: {pipe_separated_lyrics.strip()}

Core Rules:

1. Structure (this order must always be followed):
   [Shot Type] → [Character + Outfit] → [Physical Interaction] → [Environment] → [Lighting] → [Camera Motion] → [Cinematic Detail] → [Facial Expression]
2. Lyric Mapping:
   - Each lyric fragment = one complete prompt.
   - Exactly one prompt per lyric fragment.
   - Prompts must connect smoothly to the final visual detail of the previous prompt.
3. Visual Requirements:
   Every prompt must include:
   - Character + Outfit
   - Physical Interaction
   - Environment
   - Lighting
   - Camera Motion
   - Facial Expression
4. Language Rules:
   - Clear, direct, natural wording only.
   - No abstract or poetic terms, no sound descriptions, no static shots.
   - Do not use quotation marks, colons, semicolons, or special characters.
   - The ONLY allowed special character is the "|" PIPE separator BETWEEN prompts.
   - Never use "|" inside a prompt itself.
5. Word Count:
   - Every prompt must be between {word_count_min} and {word_count_max} words.
6. Endings:
   - End each prompt on a strong visual detail.
   - Never end with mood labels or trailing phrases like “captivated gaze,” “vulnerable,” or “conveying emotion.”
   - Mood must be shown through visuals, not named.
7. Continuity:
   - Camera motion must only use movements listed in {camera_motion}.
   - Do not invent new ones.
Environment: {environment.strip()}
Lighting: {lighting.strip()}
Camera Motion/Angles: {camera_motion.strip()}
Physical Interaction: {physical_interaction.strip()}
Facial Expression: {facial_expression.strip()}
Shots: {shots.strip()}
Outfit Rules: {outfit_rules.strip()}
Character Visibility: {character_visibility.strip()}

Prompt Structure (for every lyric fragment, {word_count_min}–{word_count_max} words):

-Start with the Shot Type
-Then add in the Character and Outfit if any
-Then add their Physical Interaction
-Then add the Environment
-Then add the Lighting
-Then add the Camera Motion
-Then provide the Cinematic Detail
-Then mention the Facial Expression / Emotion

Formatting Rules:
- Input lyrics are split by "|"
- Output prompts MUST be joined with "|" (one prompt per lyric)
- Do NOT insert "|" anywhere inside a prompt
- Use simple everyday words

Example prompt using this Structure:
Close up of a woman in a white tank top and brown cargo shorts as she touches a broad jungle leaf, in a vibrant jungle under a sun-dappled canopy, slow tracking reveals textured leaves. Her face shows a pondering expression

"""
        return (full_string.strip(),)

class VRGDG_MusicVideoPromptCreatorV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": "The Women.",
                }),
                "song_theme_style": ("STRING", {
                    "multiline": True,
                    "default": "cinematic realism, emotional storytelling, soft surrealism, naturalistic tone, dreamlike nostalgia, modern drama, poetic symbolism, intimate atmosphere",
                }),
                "pipe_separated_lyrics": ("STRING", {
                    "multiline": True,
                    "default": "line1 | line2 | line3",
                }),
                "word_count_min": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 200,
                }),
                "word_count_max": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                }),
                
                # ✅ NEW: List handling mode
                "list_handling_mode": ([
                    "Strict Cycle (use each once, then repeat)",
                    "Reference Guide (LLM creates variations inspired by list)",
                    "Random Selection (pick randomly from list)",
                    "Free Interpretation (LLM can ignore or combine items)"
                ], {
                    "default": "Reference Guide (LLM creates variations inspired by list)"
                }),
                
                "environment": ("STRING", {
                    "multiline": True,
                    "default": "open field at dusk, dimly lit bedroom, empty city street at night, forest clearing with morning fog, seaside cliff at golden hour, rainy urban alley, sunlit living room, desert road at sunrise",
                }),
                "lighting": ("STRING", {
                    "multiline": True,
                    "default": "warm amber glow, cool window light, neon reflections, diffused morning light, soft backlight haze, flickering streetlights, gentle afternoon sun, pink-orange dawn light",
                }),
                "camera_motion": ("STRING", {
                    "multiline": True,
                    "default": "zoom in, zoom out, tilt down, rotate around, tilt up, pan, track",
                }),
                "physical_interaction": ("STRING", {
                    "multiline": True,
                    "default": "walking through tall grass, lying on bed staring upward, leaning against a wall in stillness, reaching toward sunlight, hair moving in wind, footsteps in puddles, brushing hand across furniture, standing motionless in breeze",
                }),
                "facial_expression": ("STRING", {
                    "multiline": True,
                    "default": "Intense raw emotion",
                }),
                "shots": ("STRING", {
                    "multiline": True,
                    "default": "Close up, medium, wide angle, over the shoulder, point of view, overhead, ground level",
                }),
                "outfit_rules": ("STRING", {
                    "multiline": True,
                    "default": "a white dress",
                }),
                "character_visibility": ("STRING", {
                    "multiline": True,
                    "default": "mostly visible, half-shadowed, silhouetted, reflected or obscured, seen from behind, partially out of frame, emerging from light, fading into darkness",
                }),
                "signal": (any_typ,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated_string",)

    FUNCTION = "build_prompt_instructions"
    CATEGORY = "VRGDG/Prompt Tools"

    def build_prompt_instructions(
        self,
        character_description,
        song_theme_style,
        pipe_separated_lyrics,
        word_count_min,
        word_count_max,
        list_handling_mode,  # ✅ New parameter
        environment,
        lighting,
        camera_motion,
        physical_interaction,
        facial_expression,
        shots,
        outfit_rules,
        character_visibility,
        signal,
    ):
        # ✅ Generate list handling instructions based on mode
        if "Strict Cycle" in list_handling_mode:
            list_instructions = """8. List Handling:
- If multiple options are provided for any of the below categories, treat them as a list.
- Cycle through list items across prompts in order.
- Do not repeat an item until all others have been used.
- Once all have been used, restart the cycle.
- Each prompt must use exactly one item from each category."""
        
        elif "Reference Guide" in list_handling_mode:
            list_instructions = """8. List Handling:
- The categories below are INSPIRATION and REFERENCE GUIDES.
- Use them as starting points to create variations and similar ideas.
- Feel free to combine elements or create new options in the same style.
- Prioritize what works best for each lyric fragment and the overall narrative flow.
- Maintain variety across prompts - avoid repeating the exact same choices.
- Stay true to the overall aesthetic and mood of the provided examples."""
        
        elif "Random Selection" in list_handling_mode:
            list_instructions = """8. List Handling:
- If multiple options are provided for any category, select randomly from the list.
- Items can repeat across prompts - there is no cycling requirement.
- Prioritize what works best for each lyric fragment and the overall narrative flow.
- Ensure overall variety across the full sequence of prompts.
- Each prompt should feel fresh even if some elements repeat."""
        
        else:  # Free Interpretation
            list_instructions = """8. List Handling:
- The categories below are LOOSE GUIDELINES ONLY.
- You may use them as-is, combine them, modify them, or create entirely new options.
- Prioritize what works best for each lyric fragment and the overall narrative flow.
- Feel free to ignore any category if it doesn't serve the visual storytelling.
- Creativity and coherence are more important than strict adherence to the lists."""
        
        full_string = f"""
AI Music Video Prompt Creator

User Input:
Character: {character_description.strip()}
Style/Theme: {song_theme_style.strip()}
Lyrics: {pipe_separated_lyrics.strip()}

Core Rules:

1. Lyric-Driven Prompts (MOST IMPORTANT):
   - The lyrics provided above are pipe-separated (|).
   - There are exaclty 16 lyric fragments and each one corresponds to ONE video prompt.
   - FIRST, read through ALL the lyrics to understand the full narrative arc and emotional journey of the song.
   - Understand the overall story, themes, and progression before creating any individual prompts.
   - Then, create one prompt per lyric fragment that reflects both:
     a) The specific meaning/mood of THAT lyric fragment
     b) How it fits into the larger narrative and aesthetic of the FULL song
   - The NUMBER of prompts MUST MATCH the NUMBER of lyric fragments exactly this will always be 16.
   - Each prompt's visual content should be INSPIRED BY and REFLECT the meaning, mood, and imagery of its corresponding lyric fragment.
   - The visuals should enhance and complement what the lyric is expressing.

2. Structure (this order must always be followed):
   [Shot Type] → [Character + Outfit] → [Physical Interaction] → [Environment] → [Lighting] → [Camera Motion] → [Cinematic Detail] → [Facial Expression]

3. Continuity Between Prompts:
   - Each prompt should flow naturally from the previous one.
   - Connect the ending visual detail of one prompt to the beginning of the next.
   - Create a cohesive visual narrative that follows the lyrical journey.

4. Visual Requirements:
   Every prompt must include:
   - Character + Outfit
   - Physical Interaction
   - Environment
   - Lighting
   - Camera Motion
   - Facial Expression

5. Language Rules:
   - Clear, direct, natural wording only.
   - No abstract or poetic terms, no sound descriptions, no static shots.
   - Do not use quotation marks, colons, semicolons, or special characters.
   - The ONLY allowed special character is the "|" PIPE separator BETWEEN prompts.
   - Never use "|" inside a prompt itself.

6. Word Count:
   - Every prompt must be between {word_count_min} and {word_count_max} words.

7. Endings:
   - End each prompt on a strong visual detail.
   - Never end with mood labels or trailing phrases like "captivated gaze," "vulnerable," or "conveying emotion."
   - Mood must be shown through visuals, not named.

{list_instructions}

Environment: {environment.strip()}
Lighting: {lighting.strip()}
Camera Motion/Angles: {camera_motion.strip()}
Physical Interaction: {physical_interaction.strip()}
Facial Expression: {facial_expression.strip()}
Shots: {shots.strip()}
Outfit Rules: {outfit_rules.strip()}
Character Visibility: {character_visibility.strip()}

Prompt Structure (for every lyric fragment, {word_count_min}–{word_count_max} words):

-Start with the Shot Type
-Then add in the Character and Outfit if any
-Then add their Physical Interaction
-Then add the Environment
-Then add the Lighting
-Then add the Camera Motion
-Then provide the Cinematic Detail
-Then mention the Facial Expression / Emotion

Formatting Rules:
- Input lyrics are split by "|"
- Output prompts MUST be joined with "|" (one prompt per lyric)
- Do NOT insert "|" anywhere inside a prompt
- Use simple everyday words.
- There should be exaclty 16 prompts that are PIPE separated. 
- Remember that the prompts should be lyric driven while taking into account user input.

Example prompt using this Structure:
Close up of a woman in a white dress as she touches a broad jungle leaf, in a vibrant jungle under a sun-dappled canopy, slow tracking reveals textured leaves. Intense raw emotion

"""
        return (full_string.strip(),)

class VRGDG_PromptSplitterV3:
    RETURN_TYPES = tuple(["STRING"] * 16)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 17)])
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    def split_prompt(self, prompt_text, **kwargs):
        text = prompt_text.strip()

        if "|" in text:
            # Split only on pipes
            parts = [p.strip() for p in text.split("|") if p.strip()]
        else:
            # Otherwise split on paragraph breaks (double newlines) or single newlines
            parts = re.split(r'\n\s*\n', text)
            parts = [p.strip() for p in parts if p.strip()]

        # Pad / trim to 16 outputs
        outputs = [parts[i] if i < len(parts) else "" for i in range(16)]
        return tuple(outputs)


#########################updated on 10/18
import hashlib
class VRGDG_LoadAudioSplit_HUMO_TranscribeV3:
    # meta, total_duration, lyrics_string, index, instructions, total_sets, groups_in_last_set,
    # frames_per_scene, audio_meta, and 16 AUDIO outputs
    RETURN_TYPES = (
        "DICT", "FLOAT", "STRING", "INT", "STRING", "STRING", "STRING",
        "INT", "INT", "INT", "DICT","STRING"
    ) + tuple(["AUDIO"] * 16) + (any_typ,)

    RETURN_NAMES = (
        "meta", "total_duration", "lyrics_string", "index",
        "start_time", "end_time", "instructions",
        "total_sets", "groups_in_last_set", "frames_per_scene", "audio_meta",
        "output_folder" 
    ) + tuple([f"audio_{i}" for i in range(1, 17)]) + ("signal_out",)

    FUNCTION = "run"
    CATEGORY = "VRGDG"

    fallback_words = [
        "standing", "sitting", "laying", "resting", "waiting",
        "walking", "dancing", "looking", "thinking"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        hidden = {}

        for i in range(1, 17):  # show play button above each context field
            optional[f"context_{i}"] = ("STRING", {"default": "", "multiline": True})
            hidden[f"play_{i}"] = ("BUTTON", {"label": f"▶️ Play {i}"})

        return {
            "required": {
                "audio": ("AUDIO",),
                "trigger": (any_typ,),
                "scene_duration_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 5.0}),
                # ▼ updated as requested
                "folder_path": ("STRING", {"multiline": False, "default": "video_output"}),
                "enable_auto_queue": ("BOOLEAN", {"default": True}),
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
                "use_context_only": ("BOOLEAN", {"default": False}),  # ✅ added 10/13
                "overlap_lyric_seconds": ("FLOAT", {"default": 0.0, "min": 0.0}),
                "fallback_words": ("STRING", {"default": "thinking,walking,sitting"}),
            },
            "optional": optional,
            "hidden": hidden,
        }

    # ---------- helpers (from your nodes) ----------
    def _count_index_from_folder(self, folder_path: str) -> int:
        """Matches VRGDG_GetIndexNumber: count *-audio.mp4 as sets already done."""
        try:
            if not os.path.isdir(folder_path):
                return 0
            return len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(".mp4") and "-audio" in f.lower()
            ])
        except Exception as e:
            print(f"[Index] Failed to scan folder '{folder_path}': {e}")
            return 0

    def _calculate_sets(self, audio, index, scene_duration_seconds,enable_auto_queue=True):
        """Inlined VRGDG_CalculateSetsFromAudio_Queue (cleaned and drift-free)."""
        # ---- initialize safe defaults ----
        instructions = ""
        end_time_str = "0:00"
        total_sets = 0
        groups_in_last_set = 0
        durations_frames_full = []

        # ---- read audio duration ----
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except Exception:
            return (
                "❌ Expected audio to be a dict with 'waveform' and 'sample_rate'.",
                "0:00", 0, 0, 0, {"durations_frames": []}
            )

        fps = 25
        frames_per_scene = int(round(fps * scene_duration_seconds))
        frames_per_scene = self._adjust_frames_for_humo(frames_per_scene)  # ✅ Round UP for HuMo

        groups_per_set = 16

        # --- compute sample-based timing ---
        samples_per_frame = sample_rate / fps
        samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)

        # ---- compute total audio frames and duration ----
        try:
            num_samples = waveform.shape[-1]
            audio_duration = num_samples / sample_rate
        except Exception:
            return (
                "❌ Failed to compute audio duration from waveform.",
                "0:00", 0, 0, frames_per_scene, {"durations_frames": []}
            )

        # --- build list of frame counts per group ---
        total_audio_frames = int(num_samples / samples_per_frame + 0.5) if num_samples > 0 else 0

        if total_audio_frames > 0:
            full_groups = math.floor(total_audio_frames / frames_per_scene)
            leftover_frames = total_audio_frames - full_groups * frames_per_scene

            if full_groups > 0:
                durations_frames_full.extend([frames_per_scene] * full_groups)
            if leftover_frames > 0:
                durations_frames_full.append(leftover_frames)

            if durations_frames_full and durations_frames_full[0] != frames_per_scene:
                print(f"[Fixup] Forcing first group {durations_frames_full[0]} → {frames_per_scene}")
                durations_frames_full[0] = frames_per_scene

            total_groups = len(durations_frames_full)
            total_sets = math.ceil(total_groups / groups_per_set) if total_groups > 0 else 0
            groups_in_last_set = (
                total_groups % groups_per_set
                if (total_groups % groups_per_set) != 0
                else (groups_per_set if total_groups > 0 else 0)
            )

        ########################################instructions  for read me note
        # --- human-readable time strings ---
        minutes = int(audio_duration // 60)
        seconds = int(audio_duration % 60)
        end_time_str = f"{minutes}:{seconds:02d}"

        # --- generate instructions ---
        if total_sets == 0:
            instructions = "❌ Audio too short. No runs required."
            
        elif total_sets == 1:
            # Single run needed
            disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"
            if groups_in_last_set == 16:
                # Full single set - no muting needed
                instructions = (
                    f"⚠️  1 run needed\n"
                    f"✅ Using all 16 groups"
                )
            else:
                # Partial single set - muting required BEFORE running
                instructions = (
                    f"⚠️  Audio is less than 16 groups ({groups_in_last_set} groups detected)\n"
                    f"❗ Mute {disable_text} on 'Fast Groups Muter' node\n"
                    f"🔴 Cancel this run and re-run after muting"
                )
            
        elif groups_in_last_set != 16:
            # Multiple runs with partial last set
            disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"
            
            if enable_auto_queue:
                # Calculate what's currently in queue (this run + auto-queued)
                queued_now = 1 + max(0, total_sets - 2)  # This run + auto-queued middle runs
                instructions = (
                    f"⚠️  {total_sets} runs needed\n"
                    f"✅ {queued_now} run(s) currently in queue\n"
                    f"❗ Mute {disable_text} on 'Fast Groups Muter', then hit RUN one more time"
                )
            else:
                # Auto-queue disabled
                instructions = (
                    f"⚠️  {total_sets} runs needed\n"
                    f"🔴 Auto-queue is DISABLED\n"
                    f"❗ Manually run each set and mute {disable_text} on final run"
                )
            
        else:
            # All runs are full sets
            if enable_auto_queue:
                instructions = (
                    f"⚠️  {total_sets} runs needed\n"
                    f"✅ All {total_sets} runs are auto-queued"
                )
            else:
                instructions = (
                    f"⚠️  {total_sets} runs needed\n"
                    f"🔴 Auto-queue is DISABLED\n"
                    f"❗ Manually run all {total_sets} sets"
                )

        # Progress messages during runs
        if total_sets > 1 and index > 0:
            if index + 1 == total_sets:
                # Final run
                if groups_in_last_set != 16:
                    disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"
                    instructions = (
                        f"🏁 Final run ({index + 1} of {total_sets})\n"
                        f"✅ Make sure {disable_text} are muted!"
                    )
                else:
                    instructions = f"🏁 Final run ({index + 1} of {total_sets}) in progress..."
            else:
                # Middle runs
                if groups_in_last_set != 16:
                    disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"
                    instructions = (
                        f"⏳ Run {index + 1} of {total_sets} in progress\n"
                        f"📝 Reminder: {disable_text} need to be muted on last run"
                    )
                else:
                    instructions = f"⏳ Run {index + 1} of {total_sets} in progress..."

        # --- slice durations_frames for THIS set only ---
        start_group = index * 16
        end_group = min(start_group + 16, len(durations_frames_full))
        durations_frames_this_set = durations_frames_full[start_group:end_group] if durations_frames_full else []

        return (
            instructions,
            end_time_str,
            total_sets,
            groups_in_last_set,
            frames_per_scene,
            {"durations_frames": durations_frames_this_set},
        )

    def _maybe_auto_queue(self, total_sets: int, groups_in_last_set: int, index: int, enable: bool):
        """Inlined VRGDG_QueueTriggerFromAudio behavior."""
        if not enable:
            print("[AutoQueue] Disabled by user toggle.")
            return

        runs = 0
        if index == 0:
            if total_sets > 0:
                if groups_in_last_set == 16:
                    runs = max(0, total_sets - 1)  # full final set → queue all
                else:
                    runs = max(0, total_sets - 2)  # partial final set → leave last manual
                print(f"[AutoQueue] Queuing {runs} extra runs (total_sets={total_sets}, last={groups_in_last_set})")
                for _ in range(runs):
                    PromptServer.instance.send_sync("impact-add-queue", {})
        else:
            print(f"[AutoQueue] Skipping (index={index} > 0)")

    # ---------- NEW: Project metadata helpers ----------
    def _get_or_create_project_metadata(self, folder_path: str, audio_duration: float, scene_duration: float, audio_waveform) -> tuple:
        """
        Creates/reads a project metadata file to track project state.
        Returns (project_info_dict, is_new_project_bool)
        """
        metadata_file = os.path.join(folder_path, ".project_metadata.json")

        # Calculate audio hash (first 1 second to be fast)
        try:
            sample_data = audio_waveform[..., :48000].cpu().numpy().tobytes()
            audio_hash = hashlib.md5(sample_data).hexdigest()[:16]
        except Exception:
            audio_hash = "unknown"

        # Calculate expected project specs
        total_groups = math.ceil(audio_duration / scene_duration)
        expected_sets = math.ceil(total_groups / 16)

        current_project = {
            "audio_duration": audio_duration,
            "scene_duration": scene_duration,
            "audio_hash": audio_hash,
            "expected_sets": expected_sets,
            "total_groups": total_groups,
        }

        # Check if metadata file exists
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)

                # Compare current audio with metadata
                is_same_audio = (
                    abs(existing_metadata.get("audio_duration", 0) - audio_duration) < 1.0 and
                    existing_metadata.get("audio_hash") == audio_hash and
                    abs(existing_metadata.get("scene_duration", 0) - scene_duration) < 0.1
                )

                if is_same_audio:
                    print(f"✅ [Metadata] Continuing existing project — Audio {audio_duration:.1f}s; {expected_sets} sets expected")
                    return existing_metadata, False  # Not a new project
                else:
                    print(f"⚠️ [Metadata] DIFFERENT audio/settings detected — starting new project")
                    print(f"   Old: {existing_metadata.get('audio_duration', 0):.1f}s @ {existing_metadata.get('scene_duration', 0):.1f}s/scene")
                    print(f"   New: {audio_duration:.1f}s @ {scene_duration:.1f}s/scene")
                    return current_project, True  # New project needed

            except Exception as e:
                print(f"⚠️ [Metadata] Could not read existing metadata: {e}")
                return current_project, True
        else:
            # No metadata file exists - new project
            print(f"🆕 [Metadata] New project starting — Audio {audio_duration:.1f}s; {expected_sets} sets needed")
            return current_project, True

    def _save_project_metadata(self, folder_path: str, metadata: dict):
        """Save project metadata to JSON file."""
        metadata_file = os.path.join(folder_path, ".project_metadata.json")

        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"💾 [Metadata] Saved project metadata")
        except Exception as e:
            print(f"⚠️ [Metadata] Could not save metadata: {e}")

    def _get_smart_output_folder(self, folder_name: str, audio_duration: float, scene_duration: float, audio_waveform) -> tuple:
        """
        Creates output folder intelligently using metadata.
        Returns (folder_path, project_metadata_dict)
        """
        # Clean the folder name
        folder_name = folder_name.strip()
        if not folder_name:
            folder_name = "video_output"

        # Remove invalid characters
        folder_name = re.sub(r'[<>:"|?*]', '_', folder_name)
        folder_name = folder_name.replace('..', '').replace('/', '_').replace('\\', '_')

        # Get ComfyUI output directory
        base_output = folder_paths.get_output_directory()
        target_folder = os.path.join(base_output, folder_name)

        # Create folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Check metadata
        metadata, is_new_project = self._get_or_create_project_metadata(
            target_folder, audio_duration, scene_duration, audio_waveform
        )

        # If it's a new project but folder has files, create versioned folder
        if is_new_project:
            existing_files = os.listdir(target_folder)
            existing_files = [f for f in existing_files if f != ".project_metadata.json"]
            # Force version bump if FINAL_VIDEO.mp4 exists
            if os.path.exists(os.path.join(target_folder, "FINAL_VIDEO.mp4")):
                print("🎬 [SmartFolder] FINAL_VIDEO.mp4 detected — forcing new version folder.")
                is_new_project = True


            if existing_files:
                version = 2
                while os.path.exists(os.path.join(base_output, f"{folder_name}_v{version}")):
                    version += 1

                new_folder_name = f"{folder_name}_v{version}"
                target_folder = os.path.join(base_output, new_folder_name)
                os.makedirs(target_folder, exist_ok=True)

                print(f"🔄 [Auto-Version] Different audio detected → creating '{new_folder_name}'")
                print(f"🔄 [Auto-Version] Old project preserved in '{folder_name}'")
                print(f"🔄 [Auto-Version] Full path: {target_folder}")

        # Save/update metadata
        self._save_project_metadata(target_folder, metadata)

        print(f"📁 [Folder] Using output folder: {target_folder}")
        return target_folder, metadata
    

    def _send_popup_notification(self, message: str, message_type: str = "info", title: str = "Audio Split Instructions"):
        """
        Sends a non-blocking popup notification to the ComfyUI frontend.
        message_type: "red", "yellow"/"warning", "green"/"success", "info", "error"
        """
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

    def _adjust_frames_for_humo(self, frames: int) -> int:
        """
        Rounds UP to the nearest HuMo-compatible frame count (4n + 1).
        This ensures videos are at least as long as requested.
        Examples: 100→101, 75→77, 50→53
        """
        # Round UP: if frames = 100, we want 101 (not 97)
        adjusted = 4 * ((frames + 2) // 4) + 1
        
        if adjusted != frames:
            actual_duration = adjusted / 25  # assuming 25fps
            print(f"[HuMo Adjust] {frames} frames → {adjusted} frames ({actual_duration:.2f}s) - rounded up for HuMo compatibility")
        
        return adjusted
    # --------------- main ---------------
    def run(
        self,
        audio,
        trigger,
        folder_path,
        enable_auto_queue=True,
        language="english",
        enable_lyrics=True,
        use_context_only=False,  # ✅ added 10/13
        overlap_lyric_seconds=0.0,
        fallback_words="",
        scene_duration_seconds=4.0,
        **kwargs
    ):
        # ---- prep audio / dimensions FIRST (need for metadata) ----
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        print(f"[Audio] sample_rate: {sample_rate}")

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)  # ensure (1, C, T) shape if needed

        total_samples = waveform.shape[-1]
        total_duration = float(total_samples) / float(sample_rate)

        # ---- Check if metadata exists BEFORE creating folder/metadata ----
        base_output = folder_paths.get_output_directory()
        temp_folder = os.path.join(base_output, folder_path.strip() if folder_path.strip() else "video_output")
        metadata_existed_before = os.path.exists(os.path.join(temp_folder, ".project_metadata.json"))

        # ---- Get smart output folder with metadata ----
        output_folder, project_metadata = self._get_smart_output_folder(
            folder_path, 
            total_duration, 
            scene_duration_seconds,
            waveform
        )

        # ---- index from folder (replaces TriggerCounter + GetIndexNumber) ----
        set_index = self._count_index_from_folder(output_folder)
        print(f"[Index] Detected set_index={set_index} from folder: {output_folder}")

        # ---- split parameters ----
        scene_count = 16
        fps = 25

        samples_per_frame = sample_rate / fps
        frames_per_scene = int(round(fps * scene_duration_seconds))
        frames_per_scene = self._adjust_frames_for_humo(frames_per_scene)  # ✅ Round UP for HuMo

        samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)

        offset_samples = int(round(set_index * scene_count * samples_per_scene))

        # durations (seconds) used for meta only
        durations_sec = [frames_per_scene / fps] * scene_count

        # ✅ integer sample starts for each scene (must round)
        starts = [int(round(offset_samples + i * samples_per_scene)) for i in range(scene_count)]

        # ---- split on exact sample boundaries (pad with silence if needed) ----
        segments = []
        for idx in range(scene_count):
            start_samp = starts[idx]
            end_samp = start_samp + samples_per_scene

            if start_samp >= total_samples:
                seg = torch.zeros((1, 2, samples_per_scene), dtype=waveform.dtype)
            else:
                end_samp = min(total_samples, end_samp)
                seg = waveform[..., start_samp:end_samp].contiguous().clone()

                cur_len = seg.shape[-1]
                if cur_len < samples_per_scene:
                    pad = samples_per_scene - cur_len
                    seg = torch.nn.functional.pad(seg, (0, pad))

            segments.append({"waveform": seg, "sample_rate": sample_rate})

        # ✅ SYNC VERIFICATION (concise)
        total_segment_samples = sum(seg["waveform"].shape[-1] for seg in segments)
        expected_samples = scene_count * samples_per_scene
        diff_samples = total_segment_samples - expected_samples
        if abs(diff_samples) > 10:
            print(f"⚠️ [Sync] Mismatch: expected {expected_samples}, got {total_segment_samples} ({diff_samples} samples)")

        # ---- optional transcription ----
        if use_context_only:
            processor = model = device = None
            transcriptions = [""] * scene_count
            print("[TranscribeV3] Context-only mode (transcription disabled)")
        elif enable_lyrics:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3"
            ).to(device).eval()

            fb_words = [w.strip() for w in fallback_words.split(",") if w.strip()] or self.fallback_words
            overlap_samples = int(overlap_lyric_seconds * sample_rate)
            transcriptions = []

            for idx, start_samp in enumerate(starts):
                trans_start = int(round(max(0, start_samp - overlap_samples)))
                trans_end = int(round(min(total_samples, start_samp + samples_per_scene + overlap_samples)))
                seg_for_transcribe = waveform[..., trans_start:trans_end].contiguous().clone()

                try:
                    flat_seg = seg_for_transcribe.mean(dim=1).squeeze()
                    if sample_rate != 16000:
                        flat_seg = torchaudio.functional.resample(flat_seg, sample_rate, 16000)

                    inputs = processor(
                        flat_seg,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding="longest"
                    )
                    input_features = inputs["input_features"].to(device)

                    if language == "auto":
                        generated_ids = model.generate(input_features)
                    else:
                        decoder_ids = processor.get_decoder_prompt_ids(language=language)
                        generated_ids = model.generate(input_features, forced_decoder_ids=decoder_ids)

                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    #print(f"[Transcribe DEBUG] Segment {idx}: '{text[:60]}...'")

                except Exception:
                    text = random.choice(fb_words)

                transcriptions.append(text)
        else:
            processor = model = device = None
            transcriptions = [""] * scene_count
            print("[TranscribeV3] Transcription disabled")

        fb_words = [w.strip() for w in fallback_words.split(",") if w.strip()] or self.fallback_words
        safe_transcriptions = [t if t else random.choice(fb_words) for t in transcriptions]

        # Build final prompts
        enriched = []
        for i in range(scene_count):
            ctx = kwargs.get(f"context_{i+1}", "").strip()

            if use_context_only:
                lyric = ctx if ctx else random.choice(fb_words)
            else:
                lyric = safe_transcriptions[i]
                if ctx:
                    lyric = f"{ctx}, {lyric}"

            # cleanup step
            if lyric:
                lyric = re.sub(r'(.)\1{3,}', r'\1' * 3, lyric)
                lyric = re.sub(r'[-–—_,]+', ' ', lyric)
                words = lyric.split()
                cleaned = []
                repeat_limit = 3
                for w in words:
                    if len(cleaned) < repeat_limit or not all(
                        w.lower() == cleaned[-j-1].lower()
                        for j in range(min(repeat_limit, len(cleaned)))
                    ):
                        cleaned.append(w)
                lyric = " ".join(cleaned)

            if len(lyric) > 200:

                lyric = lyric[:200].rstrip() + "…"


            enriched.append(lyric)

        # --- Conditional merge based on overlap setting ---
        if overlap_lyric_seconds > 0:
            # Merge with overlap to remove duplicate words at boundaries
            def merge_with_overlap(prev, curr, max_check=5):
                prev_words = prev.split()
                curr_words = curr.split()
                for k in range(min(max_check, len(prev_words), len(curr_words)), 0, -1):
                    if prev_words[-k:] == curr_words[:k]:
                        return " ".join(prev_words + curr_words[k:])
                return prev + " " + curr
            
            merged = []
            for i, lyric in enumerate(enriched):
                if i == 0:
                    merged.append(lyric)
                else:
                    merged.append(merge_with_overlap(enriched[i-1], lyric))
            
            lyrics_text = " | ".join(merged)
            print(f"[TranscribeV3] Merged lyrics with overlap removal")
        else:
            # No overlap - use lyrics as-is with clean separation
            lyrics_text = " | ".join(enriched)
            print(f"[TranscribeV3] Using lyrics without merge")       

        meta = {
            "durations": durations_sec,
            "offset_seconds": 0.0,
            "starts": starts,  # sample indices
            "sample_rate": sample_rate,
            "audio_total_duration": total_duration,
            "outputs_count": len(segments),
            "used_padding": False,
            "output_folder": output_folder,
            "project_metadata": project_metadata,
        }

        # ---- inlined CalculateSets + AutoQueue ----
        # Pass beat segments if available for accurate group counting
        instructions, end_time_str_hr, total_sets, groups_in_last_set, calc_frames_per_scene, audio_meta = \
            self._calculate_sets(audio, set_index, scene_duration_seconds, enable_auto_queue)

        # 🎨 Send color-coded popup notifications
        if set_index == 0:
            # Use the flag we captured BEFORE creating metadata
            is_rerun = metadata_existed_before
            
            # Verify it's the same audio if metadata existed
            if is_rerun:
                metadata_file = os.path.join(output_folder, ".project_metadata.json")
                try:
                    with open(metadata_file, 'r') as f:
                        existing_meta = json.load(f)
                    # Check if it's actually the same audio
                    is_same_audio = (
                        abs(existing_meta.get("audio_duration", 0) - total_duration) < 1.0 and
                        abs(existing_meta.get("scene_duration", 0) - scene_duration_seconds) < 0.1
                    )
                    is_rerun = is_same_audio
                except:
                    is_rerun = False
            
            # First run
            if total_sets == 1 and groups_in_last_set != 16:
                # Single partial run
                if is_rerun:
                    # Re-run of same audio after seeing warning
                    disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"
                    rerun_instructions = (
                        f"⏳ Run 1 of 1 in progress\n"
                        f"📝 Reminder: {disable_text} should be muted"
                    )
                    instructions = rerun_instructions  # ← Override the README note too!
                    self._send_popup_notification(rerun_instructions, "warning", "⏳ RUN IN PROGRESS")
                else:
                    # First time seeing this audio
                    self._send_popup_notification(instructions, "red", "🚨 CANCEL & RECONFIGURE REQUIRED")
            elif total_sets > 1 and groups_in_last_set != 16:
                # Multiple runs with partial last
                self._send_popup_notification(instructions, "red", "🎬 STARTING WORKFLOW")
            else:
                # All full sets or single full set
                self._send_popup_notification(instructions, "info", "🎬 STARTING WORKFLOW")
                
        elif set_index > 0 and set_index + 1 < total_sets:
            # Middle runs - YELLOW/ORANGE (reminder)
            if groups_in_last_set != 16:
                self._send_popup_notification(instructions, "yellow", "⏳ PROGRESS UPDATE")
            # Don't show popup for middle runs if all sets are full
            
        elif set_index + 1 == total_sets:
            # Final run - GREEN (if full) or RED (if partial and needs action)
            if groups_in_last_set != 16:
                self._send_popup_notification(instructions, "red", "🏁 FINAL RUN - ACTION REQUIRED!")
            else:
                self._send_popup_notification(instructions, "green", "🏁 FINAL RUN")

        # Keep a concise mismatch check
        if calc_frames_per_scene != frames_per_scene:
            print(f"[Note] frames_per_scene(calc)={calc_frames_per_scene} != frames_per_scene(main)={frames_per_scene}")


        self._maybe_auto_queue(total_sets, groups_in_last_set, set_index, enable_auto_queue)

        # ---- calculate start/end times for this set ---
        actual_scene_duration = frames_per_scene / fps  # Use adjusted frames (e.g., 101/25 = 4.04)
        set_duration_sec = 16 * actual_scene_duration  # 16 scenes per set
        start_sec = set_index * set_duration_sec
        end_sec = min(start_sec + set_duration_sec, total_duration)

        def fmt_time(sec):
            m = int(sec // 60)
            s = sec % 60
            return f"{m}:{s:06.3f}"

        start_time_str = fmt_time(start_sec)
        end_time_str = fmt_time(end_sec)

        print(f"[Run] Set {set_index+1}/{max(1,total_sets)} • {frames_per_scene} frames/scene • {scene_duration_seconds:.2f}s/scene")
        print(f"[Run] Window {start_time_str} → {end_time_str} (of {int(total_duration//60)}:{int(total_duration%60):02d})")
        print(f"[Run] Output folder: {output_folder}")

        # outputs
        return (
            meta,
            total_duration,
            lyrics_text,
            set_index,
            start_time_str,
            end_time_str,
            instructions,
            total_sets,
            groups_in_last_set,
            frames_per_scene,
            audio_meta,
            output_folder,
            *tuple(segments),
            any_typ
        )

class VRGDG_HumoReminderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (any_typ, {"default": None}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "run"
    CATEGORY = "utils"

    def run(self, trigger, enabled):
        print("🔄 Reminder node is executing!")

        if enabled:
            raise ValueError(
                "🛑 Humo Workflow Reminder:\n"
                "- Please update your file paths before proceeding.\n"
                "- Once you're ready, disable this node to continue."
            )

        return (trigger,)
    

class VRGDG_CleanAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "format_audio"
    CATEGORY = "VRGDG/Audio"

    def format_audio(self, audio):
        """
        Converts audio to:
        - 48 kHz sample rate
        - Stereo (2 channels)
        - 16-bit PCM range
        - Snaps total samples to nearest 25 fps frame boundary (1920 samples)
        """

        # --- Unpack AUDIO object and normalize shape ---
        if isinstance(audio, dict):
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
        else:
            waveform, sr = audio

        # Flatten [1, 2, N] → [2, N]
        if waveform.dim() == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        target_sr = 48000

        # --- Resample to 48 kHz ---
        if sr != target_sr:
            print(f"[FormatAudio] Resampling {sr} → {target_sr}")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr

            # snap resample length to theoretical exact
            target_len = int(round(waveform.shape[1] * target_sr / sr))
            waveform = waveform[..., :target_len]

        # --- Force stereo (2 channels) ---
        if waveform.shape[0] == 1:            # mono → stereo
            waveform = waveform.expand(2, -1)
        elif waveform.shape[0] > 2:           # down-mix >2 → stereo
            waveform = torch.mean(waveform, dim=0, keepdim=True).expand(2, -1)

        # --- Convert to 16-bit PCM range ---
        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = (waveform * 32767.0).short().float() / 32767.0

        # --- Align total samples to frame boundary (25 fps @ 48 kHz = 1920 samples) ---
        samples_per_frame = 48000 // 25  # = 1920
        remainder = waveform.shape[1] % samples_per_frame
        if remainder:
            trim = samples_per_frame - remainder
            print(f"[FormatAudio] Padding {trim} samples to align with 25 fps frame grid")
            pad = torch.zeros((waveform.shape[0], trim), dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=1)

        print(f"[FormatAudio] Output: {waveform.shape[1]} samples @ {sr} Hz, "
              f"{waveform.shape[0]} channels (16-bit, frame-aligned)")

        # Final shape guarantee: [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return ({"waveform": waveform, "sample_rate": sr},)




import tempfile
class VRGDG_CreateFinalVideo:
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
                "video_folder": ("STRING", {"default": "video_output", "multiline": False}),
            }
        }

    def create_final(self, trigger, audio, threshold, video_folder):
        video_folder = video_folder.strip()

        if not os.path.isabs(video_folder):
            base_output = folder_paths.get_output_directory()
            video_folder = os.path.join(base_output, video_folder)

        print(f"[Video] Looking in: {video_folder}")

        # --- Collect video files ---
        videos = sorted([
            f for f in os.listdir(video_folder)
            if f.lower().endswith(".mp4") and "-audio" in f.lower()
        ])

        video_count = len(videos)

        if video_count < threshold:
            print(f"[Video] Threshold not met ({video_count}/{threshold}), skipping.")
            return ()

        concat_file = os.path.join(video_folder, "concat_list.txt")
        with open(concat_file, 'w') as f:
            for vid in videos:
                f.write(f"file '{os.path.join(video_folder, vid)}'\n")

        temp_video = os.path.join(video_folder, "_temp_video_no_audio.mp4")

        print(f"[Video] Concatenating {video_count} videos (removing audio)...")

        ffmpeg_path = find_ffmpeg_path()
        if not ffmpeg_path:
            print("❌ [Video] FFmpeg not available. Cannot continue.")
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
            subprocess.run(cmd_concat, capture_output=True, text=True, check=True)
            print(f"✅ [Video] Videos concatenated (no audio)")
        except subprocess.CalledProcessError as e:
            print(f"❌ [Video] Concatenation failed: {e.stderr}")
            return ()

        # --- Save original audio ---
        temp_audio = os.path.join(video_folder, "_temp_original_audio.wav")
        print(f"[Video] Saving original audio...")

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        torchaudio.save(temp_audio, waveform.squeeze(0).cpu(), sample_rate)

        # --- Combine video + audio ---
        final_output = os.path.join(video_folder, "FINAL_VIDEO.mp4")
        if os.path.exists(final_output):
            os.remove(final_output)

        print(f"[Video] Adding original audio to video...")

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
            subprocess.run(cmd_combine, capture_output=True, text=True, check=True)
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

            print(f"✅ [Video] SUCCESS! Final video saved: {final_output}")

        except subprocess.CalledProcessError as e:
            print(f"❌ [Video] Failed to add audio: {e.stderr}")

        return ()




import tempfile

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
            subprocess.run(cmd_concat, capture_output=True, text=True, check=True)
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
            subprocess.run(cmd_combine, capture_output=True, text=True, check=True)
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



#######################################added on 12/27
class VRGDG_LoadAudioSplit_Wan22HumoFMML:
    # UPDATED: lyrics/context/transcription removed, auto-queue + audio split kept

    RETURN_TYPES = (
        "DICT",     # meta
        "FLOAT",    # total_duration
        "INT",      # index
        "STRING",   # start_time
        "STRING",   # end_time
        "STRING",   # instructions
        "INT",      # total_sets
        "INT",      # groups_in_last_set
        "INT",      # frames_per_scene
        "DICT",     # audio_meta
        "STRING",   # output_folder
    ) + tuple(["AUDIO"] * 16) + (any_typ,)

    RETURN_NAMES = (
        "meta",
        "total_duration",
        "index",
        "start_time",
        "end_time",
        "instructions",
        "total_sets",
        "groups_in_last_set",
        "frames_per_scene",
        "audio_meta",
        "output_folder",
    ) + tuple([f"audio_{i}" for i in range(1, 17)]) + ("signal_out",)

    FUNCTION = "run"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        # UPDATED: removed context + play buttons + all lyric/transcription controls
        return {
            "required": {
                "audio": ("AUDIO",),
                "trigger": (any_typ,),
                "scene_duration_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 5.0}),
                "folder_path": ("STRING", {"multiline": False, "default": "video_output"}),
                "enable_auto_queue": ("BOOLEAN", {"default": True}),
            }
        }

    # ---------- helpers (from your node) ----------
    def _count_index_from_folder(self, folder_path: str) -> int:
        """Matches VRGDG_GetIndexNumber: count *-audio.mp4 as sets already done."""
        try:
            if not os.path.isdir(folder_path):
                return 0
            return len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(".mp4") and "-audio" in f.lower()
            ])
        except Exception as e:
            print(f"[Index] Failed to scan folder '{folder_path}': {e}")
            return 0

    def _calculate_sets(self, audio, index, scene_duration_seconds, enable_auto_queue=True):
        """Inlined VRGDG_CalculateSetsFromAudio_Queue (kept for instructions + popup)."""
        instructions = ""
        end_time_str = "0:00"
        total_sets = 0
        groups_in_last_set = 0
        durations_frames_full = []

        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except Exception:
            return (
                "❌ Expected audio to be a dict with 'waveform' and 'sample_rate'.",
                "0:00", 0, 0, 0, {"durations_frames": []}
            )

        fps = 25
        frames_per_scene = int(round(fps * scene_duration_seconds))
        frames_per_scene = self._adjust_frames_for_humo(frames_per_scene)

        groups_per_set = 16

        samples_per_frame = sample_rate / fps
        num_samples = waveform.shape[-1]
        audio_duration = num_samples / sample_rate if sample_rate else 0.0

        total_audio_frames = int(num_samples / samples_per_frame + 0.5) if num_samples > 0 else 0

        if total_audio_frames > 0:
            full_groups = math.floor(total_audio_frames / frames_per_scene)
            leftover_frames = total_audio_frames - full_groups * frames_per_scene

            if full_groups > 0:
                durations_frames_full.extend([frames_per_scene] * full_groups)
            if leftover_frames > 0:
                durations_frames_full.append(leftover_frames)

            if durations_frames_full and durations_frames_full[0] != frames_per_scene:
                print(f"[Fixup] Forcing first group {durations_frames_full[0]} → {frames_per_scene}")
                durations_frames_full[0] = frames_per_scene

            total_groups = len(durations_frames_full)
            total_sets = math.ceil(total_groups / groups_per_set) if total_groups > 0 else 0
            groups_in_last_set = (
                total_groups % groups_per_set
                if (total_groups % groups_per_set) != 0
                else (groups_per_set if total_groups > 0 else 0)
            )

        minutes = int(audio_duration // 60)
        seconds = int(audio_duration % 60)
        end_time_str = f"{minutes}:{seconds:02d}"

        # --- instructions (same structure as your original) ---
        if total_sets == 0:
            instructions = "❌ Audio too short. No runs required."

        elif total_sets == 1:
            disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"
            if groups_in_last_set == 16:
                instructions = (
                    f"⚠️  1 run needed\n"
                    f"✅ Using all 16 groups"
                )
            else:
                instructions = (
                    f"⚠️  Audio is less than 16 groups ({groups_in_last_set} groups detected)\n"
                    f"❗ Mute {disable_text} on 'Fast Groups Muter' node\n"
                    f"🔴 Cancel this run and re-run after muting"
                )

        elif groups_in_last_set != 16:
            disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"

            if enable_auto_queue:
                queued_now = 1 + max(0, total_sets - 2)
                instructions = (
                    f"⚠️  {total_sets} runs needed\n"
                    f"✅ {queued_now} run(s) currently in queue\n"
                    f"❗ Mute {disable_text} on 'Fast Groups Muter', then hit RUN one more time"
                )
            else:
                instructions = (
                    f"⚠️  {total_sets} runs needed\n"
                    f"🔴 Auto-queue is DISABLED\n"
                    f"❗ Manually run each set and mute {disable_text} on final run"
                )

        else:
            if enable_auto_queue:
                instructions = (
                    f"⚠️  {total_sets} runs needed\n"
                    f"✅ All {total_sets} runs are auto-queued"
                )
            else:
                instructions = (
                    f"⚠️  {total_sets} runs needed\n"
                    f"🔴 Auto-queue is DISABLED\n"
                    f"❗ Manually run all {total_sets} sets"
                )

        # Progress messages during runs
        if total_sets > 1 and index > 0:
            if index + 1 == total_sets:
                if groups_in_last_set != 16:
                    disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"
                    instructions = (
                        f"🏁 Final run ({index + 1} of {total_sets})\n"
                        f"✅ Make sure {disable_text} are muted!"
                    )
                else:
                    instructions = f"🏁 Final run ({index + 1} of {total_sets}) in progress..."
            else:
                if groups_in_last_set != 16:
                    disable_text = "group 16" if groups_in_last_set == 15 else f"groups {groups_in_last_set+1}–16"
                    instructions = (
                        f"⏳ Run {index + 1} of {total_sets} in progress\n"
                        f"📝 Reminder: {disable_text} need to be muted on last run"
                    )
                else:
                    instructions = f"⏳ Run {index + 1} of {total_sets} in progress..."

        start_group = index * 16
        end_group = min(start_group + 16, len(durations_frames_full))
        durations_frames_this_set = durations_frames_full[start_group:end_group] if durations_frames_full else []

        return (
            instructions,
            end_time_str,
            total_sets,
            groups_in_last_set,
            frames_per_scene,
            {"durations_frames": durations_frames_this_set},
        )

    def _maybe_auto_queue(self, total_sets: int, groups_in_last_set: int, index: int, enable: bool):
        """Inlined VRGDG_QueueTriggerFromAudio behavior."""
        if not enable:
            print("[AutoQueue] Disabled by user toggle.")
            return

        runs = 0
        if index == 0:
            if total_sets > 0:
                if groups_in_last_set == 16:
                    runs = max(0, total_sets - 1)
                else:
                    runs = max(0, total_sets - 2)
                print(f"[AutoQueue] Queuing {runs} extra runs (total_sets={total_sets}, last={groups_in_last_set})")
                for _ in range(runs):
                    PromptServer.instance.send_sync("impact-add-queue", {})
        else:
            print(f"[AutoQueue] Skipping (index={index} > 0)")

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

    def _adjust_frames_for_humo(self, frames: int) -> int:
        """Rounds UP to nearest HuMo-compatible frame count (4n + 1)."""
        adjusted = 4 * ((frames + 2) // 4) + 1
        if adjusted != frames:
            actual_duration = adjusted / 25
            print(f"[HuMo Adjust] {frames} → {adjusted} frames ({actual_duration:.2f}s)")
        return adjusted

    # --------------- main ---------------
    def run(
        self,
        audio,
        trigger,
        folder_path,
        enable_auto_queue=True,
        scene_duration_seconds=4.0,
        **kwargs
    ):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        print(f"[Audio] sample_rate: {sample_rate}")

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        total_duration = float(total_samples) / float(sample_rate)

        # output folder (simple; keep your original folder behavior)
        base_output = folder_paths.get_output_directory()
        output_folder = os.path.join(base_output, folder_path.strip() if folder_path.strip() else "video_output")
        os.makedirs(output_folder, exist_ok=True)

        # index from folder
        set_index = self._count_index_from_folder(output_folder)
        print(f"[Index] Detected set_index={set_index} from folder: {output_folder}")

        # split parameters
        scene_count = 16
        fps = 25

        frames_per_scene = int(round(fps * scene_duration_seconds))
        frames_per_scene = self._adjust_frames_for_humo(frames_per_scene)
        samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)

        offset_samples = int(round(set_index * scene_count * samples_per_scene))

        durations_sec = [frames_per_scene / fps] * scene_count
        starts = [int(round(offset_samples + i * samples_per_scene)) for i in range(scene_count)]

        # split into 16 segments
        segments = []
        for idx in range(scene_count):
            start_samp = starts[idx]
            end_samp = start_samp + samples_per_scene

            if start_samp >= total_samples:
                seg = torch.zeros((1, 2, samples_per_scene), dtype=waveform.dtype)
            else:
                end_samp = min(total_samples, end_samp)
                seg = waveform[..., start_samp:end_samp].contiguous().clone()

                cur_len = seg.shape[-1]
                if cur_len < samples_per_scene:
                    pad = samples_per_scene - cur_len
                    seg = torch.nn.functional.pad(seg, (0, pad))

            segments.append({"waveform": seg, "sample_rate": sample_rate})

        # meta (kept)
        meta = {
            "durations": durations_sec,
            "offset_seconds": 0.0,
            "starts": starts,
            "sample_rate": sample_rate,
            "audio_total_duration": total_duration,
            "outputs_count": len(segments),
            "output_folder": output_folder,
        }

        # calculate sets + popup + autoqueue
        instructions, end_time_str_hr, total_sets, groups_in_last_set, calc_frames_per_scene, audio_meta = \
            self._calculate_sets(audio, set_index, scene_duration_seconds, enable_auto_queue)

        # popup behavior (kept, simplified — no lyric dependencies)
        if set_index == 0:
            if total_sets == 1 and groups_in_last_set != 16:
                self._send_popup_notification(instructions, "red", "🚨 CANCEL & RECONFIGURE REQUIRED")
            elif total_sets > 1 and groups_in_last_set != 16:
                self._send_popup_notification(instructions, "red", "🎬 STARTING WORKFLOW")
            else:
                self._send_popup_notification(instructions, "info", "🎬 STARTING WORKFLOW")

        elif set_index > 0 and set_index + 1 < total_sets:
            if groups_in_last_set != 16:
                self._send_popup_notification(instructions, "yellow", "⏳ PROGRESS UPDATE")

        elif set_index + 1 == total_sets:
            if groups_in_last_set != 16:
                self._send_popup_notification(instructions, "red", "🏁 FINAL RUN - ACTION REQUIRED!")
            else:
                self._send_popup_notification(instructions, "green", "🏁 FINAL RUN")

        self._maybe_auto_queue(total_sets, groups_in_last_set, set_index, enable_auto_queue)

        # start/end time strings for this set
        actual_scene_duration = frames_per_scene / fps
        set_duration_sec = 16 * actual_scene_duration
        start_sec = set_index * set_duration_sec
        end_sec = min(start_sec + set_duration_sec, total_duration)

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
            start_time_str,
            end_time_str,
            instructions,
            total_sets,
            groups_in_last_set,
            frames_per_scene,
            audio_meta,
            output_folder,
            *tuple(segments),
            any_typ
        )
    
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
     "VRGDG_DisplayIndex":VRGDG_DisplayIndex,
     "VRGDG_PromptSplitterV2":VRGDG_PromptSplitterV2,
     "VRGDG_QueueTriggerFromAudio":VRGDG_QueueTriggerFromAudio,
     "VRGDG_ThemeSplitter":VRGDG_ThemeSplitter,
     "VRGDG_CalculateSetsFromAudio_Queue":VRGDG_CalculateSetsFromAudio_Queue,
     "VRGDG_MusicVideoPromptCreator":VRGDG_MusicVideoPromptCreator,
     "VRGDG_CombinevideosV3":VRGDG_CombinevideosV3,
     "VRGDG_LoadAudioSplit_HUMO_TranscribeV3":VRGDG_LoadAudioSplit_HUMO_TranscribeV3,
     "VRGDG_HumoReminderNode":VRGDG_HumoReminderNode,
     "VRGDG_CleanAudio":VRGDG_CleanAudio,
     "VRGDG_MusicVideoPromptCreatorV2":VRGDG_MusicVideoPromptCreatorV2,
     "VRGDG_CreateFinalVideo":VRGDG_CreateFinalVideo,
     "VRGDG_CreateFinalVideo_SRT":VRGDG_CreateFinalVideo_SRT,         
     "VRGDG_LoadAudioSplit_Wan22HumoFMML":VRGDG_LoadAudioSplit_Wan22HumoFMML    

 



}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_CombinevideosV2": "🌀 VRGDG_CombinevideosV2",
    "VRGDG_PromptSplitter": "✂️ VRGDG_PromptSplitter",
    "VRGDG_TimecodeFromIndex": "⏱️ VRGDG_TimecodeFromIndex",
    "VRGDG_ConditionalLoadVideos": "🎞️ VRGDG_ConditionalLoadVideos",
    "VRGDG_CalculateSetsFromAudio": "🎧 VRGDG_CalculateSetsFromAudio",
    "VRGDG_GetFilenamePrefix": "📂 VRGDG Get File name Prefix",
    "VRGDG_TriggerCounter": "🎯 VRGDG Trigger Counter",
    "VRGDG_LoadAudioSplit_HUMO_TranscribeV2": "🗣️ VRGDG_LoadAudioSplit_HUMO_TranscribeV2",
    "VRGDG_StringConcat":"VRGDG_StringConcat",
    "VRGDG_AudioCrop":"✂️ VRGDG Audio Crop",
    "VRGDG_GetIndexNumber":"🔢 VRGDG Get Index Number",
    "VRGDG_DisplayIndex":"VRGDG_DisplayIndex",
    "VRGDG_PromptSplitterV2":"VRGDG_PromptSplitterV2",
    "VRGDG_QueueTriggerFromAudio":"VRGDG_QueueTriggerFromAudio",
    "VRGDG_ThemeSplitter":"VRGDG_ThemeSplitter",
    "VRGDG_CalculateSetsFromAudio_Queue":"VRGDG_CalculateSetsFromAudio_Queue",
    "VRGDG_MusicVideoPromptCreator":"VRGDG_MusicVideoPromptCreator",
    "VRGDG_CombinevideosV3":"🎬 VRGDG Combine Videos V3",
    "VRGDG_LoadAudioSplit_HUMO_TranscribeV3":"🎙️ VRGDG Load Audio Split HUMO Transcribe V3",
    "VRGDG_HumoReminderNode":"VRGDG_HumoReminderNode",
    "VRGDG_CleanAudio":"VRGDG_CleanAudio",
    "VRGDG_MusicVideoPromptCreatorV2":"VRGDG_MusicVideoPromptCreatorV2",
    "VRGDG_CreateFinalVideo":"VRGDG_CreateFinalVideo",
    "VRGDG_CreateFinalVideo_SRT":"VRGDG_CreateFinalVideo_SRT",     
    "VRGDG_LoadAudioSplit_Wan22HumoFMML":"VRGDG_LoadAudioSplit_Wan22HumoFMML"    

}















