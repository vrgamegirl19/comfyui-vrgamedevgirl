import math
import random
import re
import torch
import torchaudio
import json
import cv2
import numpy as np
import os
from folder_paths import get_output_directory
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)


class VRGDG_ManualLyricsExtractor:
    """
    Transcribes entire audio file and outputs all lyrics as a formatted string.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("all_lyrics_combined",)

    FUNCTION = "extract_lyrics"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "scene_duration_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0}),
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
            },
        }

    def _adjust_frames_for_humo(self, frames: int) -> int:
        adjusted = 4 * ((frames + 2) // 4) + 1
        if adjusted != frames:
            actual_duration = adjusted / 25
            print(f"[HuMo Adjust] {frames} frames → {adjusted} frames ({actual_duration:.2f}s)")
        return adjusted

    def _transcribe_segment(self, waveform, sample_rate, start_sample, end_sample,
                            processor, model, device, language):
        try:
            seg_for_transcribe = waveform[..., start_sample:end_sample].contiguous().clone()
            flat_seg = seg_for_transcribe.mean(dim=1).squeeze()
            if sample_rate != 16000:
                flat_seg = torchaudio.functional.resample(flat_seg, sample_rate, 16000)

            inputs = processor(flat_seg, sampling_rate=16000, return_tensors="pt", padding="longest")
            input_features = inputs["input_features"].to(device)

            if language == "auto":
                generated_ids = model.generate(input_features)
            else:
                decoder_ids = processor.get_decoder_prompt_ids(language=language)
                generated_ids = model.generate(input_features, forced_decoder_ids=decoder_ids)

            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return text

        except Exception as e:
            print(f"[Transcribe] Error: {e}")
            return "[Error]"

    def _clean_lyric(self, lyric: str) -> str:
        lyric = re.sub(r'(.)\1{3,}', r'\1' * 3, lyric)
        lyric = re.sub(r'[-—–_,]+', ' ', lyric)

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
        return lyric[:200].rstrip() + "…" if len(lyric) > 200 else lyric

    def extract_lyrics(
        self,
        audio,
        scene_duration_seconds=4.0,
        language="english",
        **kwargs
    ):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        total_duration = float(total_samples) / float(sample_rate)
        print(f"[ManualLyrics] Processing audio: {total_duration:.2f}s @ {sample_rate}Hz")

        fps = 25
        frames_per_scene = int(round(fps * scene_duration_seconds))
        frames_per_scene = self._adjust_frames_for_humo(frames_per_scene)
        samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)

        total_segments = math.ceil(total_samples / samples_per_scene)
        all_transcriptions = []

        print("[ManualLyrics] Starting Whisper transcription...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device).eval()

        for i in range(total_segments):
            start_sample = i * samples_per_scene
            end_sample = min(start_sample + samples_per_scene, total_samples)

            text = self._transcribe_segment(
                waveform, sample_rate, start_sample, end_sample,
                processor, model, device, language
            )
            text = self._clean_lyric(text)
            all_transcriptions.append(text)

            if (i + 1) % 16 == 0 or i == total_segments - 1:
                print(f"[ManualLyrics] Transcribed {i+1}/{total_segments} segments")

        combined_lines = [f"# Lyrics to fix: ({total_segments} segments)", ""]
        for i, lyric in enumerate(all_transcriptions, 1):
            combined_lines.append(f"lyricSegment{i}={lyric}")

        all_lyrics_combined = "\n".join(combined_lines)
        print(f"[ManualLyrics] Extraction complete!")

        return (all_lyrics_combined,)





class VRGDG_PromptSplitterForManual:
    RETURN_TYPES = tuple(["STRING"] * 16)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 17)])
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
            # Parse the JSON string
            data = json.loads(json_string)
            
            # Extract prompts in order
            prompts = []
            if isinstance(data, dict):
                # Sort keys by their numeric value (for "prompt1", "prompt2", etc.)
                sorted_keys = sorted(data.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
                prompts = [data[key] for key in sorted_keys]
            elif isinstance(data, list):
                # If it's already a list, use it directly
                prompts = data
            
            # Calculate the starting position based on index
            start_idx = index * 16
            
            # Get 16 prompts starting from start_idx
            outputs = [prompts[start_idx + i] if (start_idx + i) < len(prompts) else "" for i in range(16)]
            
            return tuple(outputs)
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON - {str(e)}")
            return tuple([""] * 16)
        except Exception as e:
            print(f"Error loading prompts: {str(e)}")
            return tuple([""] * 16)




class VRGDG_CombinevideosV5:
    VERSION = "v3.9_label_safe"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_video_frames",)
    FUNCTION = "blend_videos"
    CATEGORY = "Video"

    @classmethod
    def INPUT_TYPES(cls):
        opt_videos = {f"video_{i}": ("IMAGE",) for i in range(1, 17)}
        return {
            "required": {
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0}),
                "duration": ("FLOAT", {"default": 4.0, "min": 0.01}),
                "audio_meta": ("DICT",),
                "index": ("INT", {"default": 0, "min": 0}),
                "total_sets": ("INT", {"default": 1, "min": 1}),
                "groups_in_last_set": ("INT", {"default": 16, "min": 0, "max": 16}),
                "folder_path": ("STRING", {"default": "./output_videos"}),
                "with_labels": ("BOOLEAN", {
                    "default": True,
                    "label": "Add Labels and Save Video",
                    "tooltip": "If enabled, adds label bars and saves labeled video to WithLabels/."
                }),
            },
            "optional": {**opt_videos},
        }

    # ---------------------------------------------------------------------
    # --- Helpers ---------------------------------------------------------

    def _target_frames_for_index(self, durations, idx_zero_based, fps, is_frames=False):
        value = float(durations[idx_zero_based])
        if is_frames:
            return max(1, int(round(value)))
        else:
            return max(1, int(round(fps * value)))

    def _trim_or_pad(self, video, target_frames):
        if video is None:
            return None
        if video.ndim != 4:
            raise ValueError(f"Expected (frames,H,W,C), got {tuple(video.shape)}")

        cur = int(video.shape[0])
        if cur > target_frames:
            return video[:target_frames]
        if cur < target_frames:
            return video  # shorter is fine
        return video

    def _add_label_bar(self, video_tensor, label_text):
        """Add black bar and centered white label text under each frame"""
        frames = []
        for frame in video_tensor.numpy():
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

            h, w, _ = frame_bgr.shape
            bar_height = 60
            new_frame = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
            new_frame[:h, :, :] = frame_bgr

            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h + int(bar_height * 0.7)

            cv2.putText(
                new_frame,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # convert back to RGB and normalize
            new_frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            frames.append(new_frame_rgb.astype(np.float32) / 255.0)

        return torch.from_numpy(np.stack(frames))

    def _save_video(self, frames, folder_path, filename, fps):
        """Save RGB frames to MP4 safely inside ComfyUI output folder"""
        folder_path = folder_path.strip()
        if not os.path.isabs(folder_path):
            base_output = get_output_directory()
            folder_path = os.path.join(base_output, folder_path)

        os.makedirs(folder_path, exist_ok=True)
        out_path = os.path.join(folder_path, filename)

        h, w, _ = frames[0].shape
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            frame_uint8 = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()

        print(f"[CombineV5] ✅ Saved labeled video to: {out_path}")
        return out_path

    # ---------------------------------------------------------------------
    # --- Main operation --------------------------------------------------

    def blend_videos(
        self,
        fps,
        duration,
        audio_meta=None,
        index=0,
        total_sets=1,
        groups_in_last_set=16,
        folder_path="./output_videos",
        with_labels=True,
        **kwargs,
    ):
        print(f"[CombineV5 {self.VERSION}] index={index}, with_labels={with_labels}")

        effective_scene_count = 16
        is_last_run = (index == total_sets - 1)

        if not isinstance(audio_meta, dict):
            raise ValueError("[CombineV5] audio_meta must be a dict")

        durations_seconds = audio_meta.get("durations")
        durations_frames = audio_meta.get("durations_frames")

        if durations_frames is not None:
            durations = list(durations_frames)
            is_frames = True
        elif durations_seconds is not None:
            durations = list(durations_seconds)
            is_frames = False
        else:
            raise ValueError("[CombineV5] audio_meta missing durations info")

        if len(durations) < effective_scene_count:
            durations += [0.0] * (effective_scene_count - len(durations))
        else:
            durations = durations[:effective_scene_count]

        limit_videos = effective_scene_count
        if is_last_run:
            limit_videos = max(1, min(groups_in_last_set, effective_scene_count))

        vids = []
        for i in range(1, limit_videos + 1):
            v = kwargs.get(f"video_{i}")
            if v is not None:
                vids.append((i, v))

        if not vids:
            raise ValueError("[CombineV5] No video inputs detected.")

        # --- Create the clean combined video (no labels)
        trimmed = []
        for slot_idx, vid in vids:
            tgt = self._target_frames_for_index(durations, slot_idx - 1, fps, is_frames)
            if is_last_run and slot_idx > groups_in_last_set:
                continue
            trimmed_vid = self._trim_or_pad(vid, tgt)
            trimmed.append(trimmed_vid)

        final = torch.cat(trimmed, dim=0).cpu()
        print(f"[CombineV5] Final concatenated frames: {final.shape[0]}")

        # --- Create labeled version ONLY for saving ---
        if with_labels:
            labeled_copy = []
            for slot_idx, vid in vids:
                tgt = self._target_frames_for_index(durations, slot_idx - 1, fps, is_frames)
                if is_last_run and slot_idx > groups_in_last_set:
                    continue
                trimmed_vid = self._trim_or_pad(vid, tgt)
                label_text = f"set {index+1} - group {slot_idx}"
                labeled_copy.append(self._add_label_bar(trimmed_vid, label_text))

            labeled_video = torch.cat(labeled_copy, dim=0).cpu().numpy()
            filename = f"set{index+1}_combined.mp4"
            out_dir = os.path.join(folder_path, "WithLabels")
            self._save_video(labeled_video, out_dir, filename, fps)
        else:
            print("[CombineV5] Clean mode — labels disabled, nothing saved.")

        # --- Always return clean frames for downstream use ---
        return (final,)



#####Added on 12/17
class VRGDG_PromptSplitterForFMML:
    RETURN_TYPES = tuple(["STRING"] * 16)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 17)])
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
            prompts = []
            if isinstance(data, dict):
                sorted_keys = sorted(
                    data.keys(),
                    key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0
                )
                # Join each prompt list into a single string with newlines
                prompts = [
                    "\n".join(data[key]) if isinstance(data[key], list) else str(data[key])
                    for key in sorted_keys
                ]
            elif isinstance(data, list):
                prompts = [
                    "\n".join(item) if isinstance(item, list) else str(item)
                    for item in data
                ]
            else:
                prompts = []

            start_idx = index * 16
            outputs = [prompts[start_idx + i] if (start_idx + i) < len(prompts) else "" for i in range(16)]
            return tuple(outputs)
        except json.JSONDecodeError:
            return tuple([""] * 16)
        except Exception:
            return tuple([""] * 16)

import json

import re

class VRGDG_PromptSplitter4:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text_output_1", "text_output_2", "text_output_3", "text_output_4")
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "{}"}),
            }
        }

    def clean_json(self, text):
        # Remove markdown wrappers like ```json or ```
        text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "")

        # Remove stray backticks entirely
        text = text.replace("`", "")

        # Trim whitespace
        text = text.strip()

        return text

    def split_prompt(self, json_string):
        try:
            # Clean any markdown or formatting garbage
            cleaned = self.clean_json(json_string)

            # Attempt to parse cleaned JSON
            data = json.loads(cleaned)

            if not isinstance(data, dict):
                return ("", "", "", "")

            # Extract numeric keys like Prompt#3 → 3
            items = []
            for key, value in data.items():
                num = "".join(filter(str.isdigit, key))
                if num.isdigit():
                    items.append((int(num), value))

            # Sort by numeric portion
            items.sort(key=lambda x: x[0])

            # Fill outputs (4 total)
            outputs = [items[i][1] if i < len(items) else "" for i in range(4)]

            return tuple(outputs)

        except Exception:
            # Any error → return empty outputs
            return ("", "", "", "")





class VRGDG_SpeechEmotionExtractor:
    """
    Segments audio and extracts dominant emotion per segment
    using a locally stored Whisper-based emotion classification model.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("emotion_timeline",)

    FUNCTION = "extract_emotions"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "scene_duration_seconds": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 10.0}
                ),
            },
        }

    def _adjust_frames_for_humo(self, frames: int) -> int:
        adjusted = 4 * ((frames + 2) // 4) + 1
        if adjusted != frames:
            actual_duration = adjusted / 25
            print(
                f"[HuMo Adjust] {frames} frames → {adjusted} frames "
                f"({actual_duration:.2f}s)"
            )
        return adjusted

    def _classify_segment(
        self,
        waveform,
        sample_rate,
        start_sample,
        end_sample,
        feature_extractor,
        model,
        device,
        id2label,
        max_duration=30.0,
    ):
        try:
            # Slice segment
            segment = waveform[..., start_sample:end_sample].contiguous().clone()

            # Convert to mono
            segment = segment.mean(dim=1).squeeze()

            # Resample if needed
            target_sr = feature_extractor.sampling_rate
            if sample_rate != target_sr:
                segment = torchaudio.functional.resample(
                    segment, sample_rate, target_sr
                )

            segment_np = segment.cpu().numpy()

            # Pad / truncate
            max_length = int(target_sr * max_duration)
            if len(segment_np) > max_length:
                segment_np = segment_np[:max_length]
            else:
                segment_np = np.pad(
                    segment_np, (0, max_length - len(segment_np))
                )

            inputs = feature_extractor(
                segment_np,
                sampling_rate=target_sr,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=-1).item()
            emotion = id2label[predicted_id]

            return emotion

        except Exception as e:
            print(f"[Emotion] Error: {e}")
            return "Error"

    def extract_emotions(
        self,
        audio,
        scene_duration_seconds=4.0,
        **kwargs
    ):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        total_duration = total_samples / sample_rate
        print(f"[Emotion] Processing audio: {total_duration:.2f}s @ {sample_rate}Hz")

        fps = 25
        frames_per_scene = int(round(fps * scene_duration_seconds))
        frames_per_scene = self._adjust_frames_for_humo(frames_per_scene)
        samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)

        total_segments = math.ceil(total_samples / samples_per_scene)

        print("[Emotion] Loading LOCAL emotion model (offline)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = (
            r"A:\COMFY_UI\ComfyUI_windows_portable_nvidia"
            r"\ComfyUI_windows_portable"
            r"\ComfyUI\models\audio_encoders"
            r"\speech-emotion-whisper"
        )

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_path,
            local_files_only=True
        )

        model = AutoModelForAudioClassification.from_pretrained(
            model_path,
            local_files_only=True
        ).to(device).eval()

        id2label = model.config.id2label

        emotions = []

        for i in range(total_segments):
            start_sample = i * samples_per_scene
            end_sample = min(start_sample + samples_per_scene, total_samples)

            emotion = self._classify_segment(
                waveform,
                sample_rate,
                start_sample,
                end_sample,
                feature_extractor,
                model,
                device,
                id2label,
            )

            emotions.append(emotion)

            if (i + 1) % 16 == 0 or i == total_segments - 1:
                print(f"[Emotion] Processed {i+1}/{total_segments} segments")

        lines = [f"# Emotion timeline ({total_segments} segments)", ""]
        for i, emo in enumerate(emotions, 1):
            lines.append(f"emotionSegment{i}={emo}")

        result = "\n".join(lines)
        print("[Emotion] Extraction complete!")

        return (result,)

import re


class VRGDG_LyricsEmotionMerger:
    """
    Merges lyric segments and emotion segments into a single aligned output.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_with_emotions",)

    FUNCTION = "merge"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics_text": ("STRING",),
                "emotion_text": ("STRING",),
            }
        }

    def merge(self, lyrics_text: str, emotion_text: str):
        # --- Parse emotion segments ---
        emotion_map = {}

        for line in emotion_text.splitlines():
            line = line.strip()
            if not line or not line.startswith("emotionSegment"):
                continue

            match = re.match(r"emotionSegment(\d+)\s*=\s*(.+)", line)
            if match:
                idx = int(match.group(1))
                emotion = match.group(2).strip()
                emotion_map[idx] = emotion

        # --- Parse lyric segments and merge ---
        merged_lines = []

        for line in lyrics_text.splitlines():
            line = line.strip()
            if not line or not line.startswith("lyricSegment"):
                continue

            match = re.match(r"lyricSegment(\d+)\s*=\s*(.+)", line)
            if not match:
                continue

            idx = int(match.group(1))
            lyric = match.group(2).strip()

            emotion = emotion_map.get(idx, "Unknown")

            merged_lines.append(
                f'lyricSegment{idx}-emotion={emotion} "{lyric}"'
            )

        # --- Build output ---
        header = f"# Lyrics with emotions ({len(merged_lines)} segments)"
        result = "\n".join([header, ""] + merged_lines)

        return (result,)


import json
import re

class VRGDG_PromptSplitter2:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_output_1", "text_output_2")
    FUNCTION = "split_prompt"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "{}"}),
            }
        }

    def clean_json(self, text):
        # remove markdown formatting
        text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "").replace("`", "").strip()
        return text

    def attempt_json_repair(self, text):
        """Auto-wrap broken JSON like `"Prompt1": "text"` into { "Prompt1": "text" }"""

        # if it already looks like JSON with braces, return as-is
        if text.startswith("{") and text.endswith("}"):
            return text

        # If missing braces, try wrapping
        if ":" in text and not text.startswith("{"):
            return "{ " + text.rstrip(", ") + " }"

        return text

    def split_prompt(self, json_string):
        try:
            cleaned = self.clean_json(json_string)
            cleaned = self.attempt_json_repair(cleaned)

            data = json.loads(cleaned)

            if not isinstance(data, dict):
                return ("", "")

            items = []
            has_numbered_keys = False

            # detect if key names include numbers
            for key in data.keys():
                num = "".join(filter(str.isdigit, key))
                if num.isdigit():
                    has_numbered_keys = True
                    break

            # If keys contain numbers → ordered numerically
            if has_numbered_keys:
                for key, value in data.items():
                    num = "".join(filter(str.isdigit, key))
                    if num.isdigit():
                        items.append((int(num), value))
                items.sort(key=lambda x: x[0])
                ordered_values = [v for _, v in items]

            # otherwise just use natural order
            else:
                ordered_values = list(data.values())

            # provide first two outputs
            text1 = ordered_values[0] if len(ordered_values) > 0 else ""
            text2 = ordered_values[1] if len(ordered_values) > 1 else ""

            return (text1, text2)

        except Exception:
            return ("", "")





import json

class VRGDG_PromptSplitterForFL:
    RETURN_TYPES = tuple(["STRING"] * 16)
    RETURN_NAMES = tuple([f"text_output_{i}" for i in range(1, 17)])
    FUNCTION = "split"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "{}"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    def split(self, json_string, index):
        try:
            data = json.loads(json_string)

            if not isinstance(data, dict):
                return tuple([""] * 16)

            # Sort prompt keys numerically (prompt1, Prompt#2, etc.)
            def sort_key(k):
                digits = "".join(filter(str.isdigit, k))
                return int(digits) if digits else 0

            sorted_keys = sorted(data.keys(), key=sort_key)

            prompts = []
            for key in sorted_keys:
                entry = data.get(key)
                if isinstance(entry, dict):
                    prompts.append(entry)

            start_idx = index * 16
            outputs = []

            for i in range(16):
                idx = start_idx + i
                if idx < len(prompts):
                    # IMPORTANT: dump prompt object AS-IS
                    outputs.append(json.dumps(prompts[idx], ensure_ascii=False))
                else:
                    outputs.append("")

            return tuple(outputs)

        except Exception:
            return tuple([""] * 16)




class VRGDG_SplitPrompt_T2I_I2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_json": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("t2i_prompt", "i2v_prompt")
    FUNCTION = "split"
    CATEGORY = "VRGDG"

    def split(self, prompt_json):
        if not prompt_json:
            return "", ""

        try:
            text = prompt_json.strip()

            # ONLY remove fenced code blocks if they are actually present
            if text.startswith("```"):
                lines = text.splitlines()
                # Remove first line (``` or ```json)
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            data = json.loads(text)

            if not isinstance(data, dict):
                return "", ""

            t2i = str(data.get("t2i", "")).strip()

            i2v_data = data.get("i2v", "")
            if isinstance(i2v_data, list):
                i2v = "\n".join(str(line).strip() for line in i2v_data if line)
            else:
                i2v = str(i2v_data).strip()

            return t2i, i2v

        except Exception:
            return "", ""



class VRGDG_PromptTemplateBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        section_types = [
            "Theme / Style",
            "Instructions",
            "Image to Video Prompt",
            "Text to Video Prompt",
            "Text to Image Prompt",
            "Story",
            "Lyric Segment",
            "Ideas",
            "Other Notes",
        ]

        return {
            "required": {
                "section_1_type": (section_types,),
                "section_1_text": ("STRING", {"multiline": True, "default": ""}),

                "section_2_type": (section_types,),
                "section_2_text": ("STRING", {"multiline": True, "default": ""}),

                "section_3_type": (section_types,),
                "section_3_text": ("STRING", {"multiline": True, "default": ""}),

                "section_4_type": (section_types,),
                "section_4_text": ("STRING", {"multiline": True, "default": ""}),

                "section_5_type": (section_types,),
                "section_5_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_prompt",)
    FUNCTION = "build"
    CATEGORY = "VRGDG"

    def build(
        self,
        section_1_type, section_1_text,
        section_2_type, section_2_text,
        section_3_type, section_3_text,
        section_4_type, section_4_text,
        section_5_type, section_5_text,
    ):
        sections = [
            (section_1_type, section_1_text),
            (section_2_type, section_2_text),
            (section_3_type, section_3_text),
            (section_4_type, section_4_text),
            (section_5_type, section_5_text),
        ]

        output_blocks = []

        for section_type, section_text in sections:
            if section_text and section_text.strip():
                block = f"### {section_type}\n{section_text.strip()}"
                output_blocks.append(block)

        final_prompt = "\n\n".join(output_blocks)
        return (final_prompt,)



class VRGDG_SmartSplitTextTwo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("part_1", "part_2")
    FUNCTION = "split"
    CATEGORY = "Text"

    def split(self, text):
        if not text:
            return "", ""

        # Normalize all newline variants
        normalized = text.replace("\\r\\n", "\n").replace("\\n", "\n")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # Case 1: real newline exists → split on first one
        if "\n" in normalized:
            first, second = normalized.split("\n", 1)
            return first.strip(), second.strip()

        # Case 2: no newline → sentence-based split near middle
        sentences = re.split(r'(?<=[.!?])\s+', normalized)

        if len(sentences) <= 1:
            mid = len(normalized) // 2
            return normalized[:mid].strip(), normalized[mid:].strip()

        mid_index = len(sentences) // 2
        part_1 = " ".join(sentences[:mid_index]).strip()
        part_2 = " ".join(sentences[mid_index:]).strip()

        return part_1, part_2

NODE_CLASS_MAPPINGS = {

     "VRGDG_ManualLyricsExtractor": VRGDG_ManualLyricsExtractor,
     "VRGDG_PromptSplitterForManual":VRGDG_PromptSplitterForManual,
     "VRGDG_CombinevideosV5":VRGDG_CombinevideosV5,
     "VRGDG_PromptSplitterForFMML":VRGDG_PromptSplitterForFMML,   
     "VRGDG_PromptSplitter4":VRGDG_PromptSplitter4,
     "VRGDG_SpeechEmotionExtractor":VRGDG_SpeechEmotionExtractor,
     "VRGDG_LyricsEmotionMerger":VRGDG_LyricsEmotionMerger,
     "VRGDG_PromptSplitter2":VRGDG_PromptSplitter2,
     "VRGDG_PromptSplitterForFL":VRGDG_PromptSplitterForFL,
     "VRGDG_SplitPrompt_T2I_I2V":VRGDG_SplitPrompt_T2I_I2V,
     "VRGDG_PromptTemplateBuilder":VRGDG_PromptTemplateBuilder,
     "VRGDG_SmartSplitTextTwo":VRGDG_SmartSplitTextTwo,
    
    
    
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ManualLyricsExtractor": "🌀 VRGDG_ManualLyricsExtractor",
    "VRGDG_PromptSplitterForManual":"✂️ VRGDG_PromptSplitterForManual",
    "VRGDG_CombinevideosV5":"VRGDG_CombinevideosV5",
    "VRGDG_PromptSplitterForFMML":"VRGDG_PromptSplitterForFMML",
    "VRGDG_PromptSplitter4":"VRGDG_PromptSplitter4",
    "VRGDG_SpeechEmotionExtractor":"VRGDG_SpeechEmotionExtractor",
    "VRGDG_LyricsEmotionMerger":"VRGDG_LyricsEmotionMerger",
    "VRGDG_PromptSplitter2":"VRGDG_PromptSplitter2",
    "VRGDG_PromptSplitterForFL":"VRGDG_PromptSplitterForFL",
    "VRGDG_SplitPrompt_T2I_I2V":"VRGDG_SplitPrompt_T2I_I2V",
    "VRGDG_PromptTemplateBuilder":"VRGDG_PromptTemplateBuilder",
    "VRGDG_SmartSplitTextTwo":"VRGDG_SmartSplitTextTwo",    
    
    
}





