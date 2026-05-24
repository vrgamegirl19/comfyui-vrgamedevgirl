import math
import random
import re
import torch
import torchaudio
import json
import cv2
import numpy as np
import os
import tempfile
import difflib
from folder_paths import get_output_directory
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)


def _whisper_input_features_to_model(input_features, model, fallback_device):
    encoder = getattr(getattr(model, "model", None), "encoder", None)
    conv1 = getattr(encoder, "conv1", None)
    for tensor in (
        getattr(conv1, "bias", None),
        getattr(conv1, "weight", None),
    ):
        if tensor is not None:
            return input_features.to(device=tensor.device, dtype=tensor.dtype)

    for tensor in model.parameters():
        if tensor.is_floating_point():
            return input_features.to(device=tensor.device, dtype=tensor.dtype)

    return input_features.to(device=fallback_device, dtype=torch.float32)


def _is_whisper_dtype_mismatch_error(exc):
    message = str(exc).lower()
    return (
        "input type" in message
        and "bias type" in message
        and "should be the same" in message
    )


def _is_whisper_mel_length_error(exc):
    message = str(exc).lower()
    return (
        "whisper expects the mel input features to be of length 3000" in message
        and "found" in message
    )


def _whisper_pad_input_features(input_features, target_length=3000):
    current_length = input_features.shape[-1]
    if current_length == target_length:
        return input_features
    if current_length > target_length:
        return input_features[..., :target_length].contiguous()
    return torch.nn.functional.pad(input_features, (0, target_length - current_length))


def _whisper_generate_with_dtype_fallback(model, input_features, fallback_device, **kwargs):
    try:
        return model.generate(input_features, **kwargs)
    except ValueError as exc:
        if not _is_whisper_mel_length_error(exc):
            raise

        print("[Whisper] Retrying with mel input features padded to length 3000.")
        padded_features = _whisper_pad_input_features(input_features)
        return model.generate(padded_features, **kwargs)
    except RuntimeError as exc:
        if not _is_whisper_dtype_mismatch_error(exc):
            raise

        print("[Whisper] Retrying with model dtype/device after PyTorch dtype mismatch.")
        aligned_features = _whisper_input_features_to_model(
            input_features,
            model,
            fallback_device,
        )
        return model.generate(aligned_features, **kwargs)


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
                generated_ids = _whisper_generate_with_dtype_fallback(
                    model,
                    input_features,
                    device,
                )
            else:
                decoder_ids = processor.get_decoder_prompt_ids(language=language)
                generated_ids = _whisper_generate_with_dtype_fallback(
                    model,
                    input_features,
                    device,
                    forced_decoder_ids=decoder_ids,
                )

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


class VRGDG_ManualLyricsExtractor_SRT:
    """
    Transcribes entire audio file and outputs all lyrics as a formatted string.

    ✅ Normal mode:
      - HuMo frame adjustment ON
      - 200-char truncation ON

    ✅ LTX-2 mode:
      - HuMo adjustment OFF
      - No truncation
      - Hard clamp to 30.0s per segment (Whisper limit)
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("all_lyrics_combined",)

    FUNCTION = "extract_lyrics"
    CATEGORY = "VRGDG"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_path": ("STRING", {"default": ""}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),

                "audio": ("AUDIO",),
                "scene_duration_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 60.0}),
                "use_ltx2": ("BOOLEAN", {"default": False}),
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
                        "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese",
                        "haitian creole", "pashto", "turkmen", "nynorsk", "maltese", "sanskrit",
                        "luxembourgish", "myanmar", "tibetan", "tagalog", "malagasy", "assamese",
                        "tatar", "hawaiian", "lingala", "hausa", "bashkir", "javanese", "sundanese",
                        "cantonese", "burmese", "valencian", "flemish", "haitian", "letzeburgesch",
                        "pushto", "panjabi", "moldavian", "moldovan", "sinhalese", "castilian",
                        "mandarin"
                    ],
                    {"default": "english"}
                ),
            }
        }

    # Only used for HuMo mode
    def _adjust_frames_for_humo(self, frames: int) -> int:
        adjusted = 4 * ((frames + 2) // 4) + 1
        if adjusted != frames:
            actual_duration = adjusted / 25
            print(f"[HuMo Adjust] {frames} frames → {adjusted} frames ({actual_duration:.2f}s)")
        return adjusted

    def _transcribe_segment(
        self,
        waveform,
        sample_rate,
        start_sample,
        end_sample,
        processor,
        model,
        device,
        language,
    ):
        try:
            seg = waveform[..., start_sample:end_sample].contiguous()
            flat = seg.mean(dim=1).squeeze()

            if sample_rate != 16000:
                flat = torchaudio.functional.resample(flat, sample_rate, 16000)

            inputs = processor(
                flat,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest",
                truncation=False
            )

            input_features = inputs["input_features"].to(device)

            with torch.no_grad():
                if language == "auto":
                    generated_ids = _whisper_generate_with_dtype_fallback(
                        model,
                        input_features,
                        device,
                    )
                else:
                    decoder_ids = processor.get_decoder_prompt_ids(language=language)
                    generated_ids = _whisper_generate_with_dtype_fallback(
                        model,
                        input_features,
                        device,
                        forced_decoder_ids=decoder_ids
                    )

            return processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

        except Exception:
            import traceback
            print("\n====== WHISPER TRANSCRIBE FAILED ======")
            traceback.print_exc()
            print("======================================\n")
            return "[Error]"

    def _clean_lyric(self, lyric: str, use_ltx2: bool) -> str:
        lyric = re.sub(r"(.)\1{3,}", r"\1" * 3, lyric)
        lyric = re.sub(r"[-—–_,]+", " ", lyric)

        lyric = lyric.strip()

        # ✅ LTX-2: no truncation
        if use_ltx2:
            return lyric

        # Normal mode: keep old cap
        return lyric[:200].rstrip() + "…" if len(lyric) > 200 else lyric
    def _parse_srt_segments(self, srt_path):
        segments = []
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 2:
                continue

            times = lines[1]
            start_str, end_str = times.split(" --> ")

            def to_seconds(t):
                h, m, rest = t.split(":")
                s, ms = rest.split(",")
                return int(h)*3600 + int(m)*60 + float(s) + float(ms)/1000

            start = to_seconds(start_str)
            end = to_seconds(end_str)

            segments.append((start, end))

        return segments

    def extract_lyrics(
        self,
        audio,
        scene_duration_seconds=4.0,
        fps=25,
        srt_path="",
        use_ltx2=False,
        language="english",
        **kwargs
    ):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        total_samples = waveform.shape[-1]
        total_duration = total_samples / sample_rate
        print(f"[ManualLyrics] Processing audio: {total_duration:.2f}s @ {sample_rate}Hz")

        # ✅ FPS now user-controlled
        fps = int(fps)

        # ✅ LTX-2 mode: NO HuMo adjustment (only used if no SRT)
        frames_per_scene = int(round(fps * scene_duration_seconds))
        if not use_ltx2:
            frames_per_scene = self._adjust_frames_for_humo(frames_per_scene)

        samples_per_scene = int(frames_per_scene * sample_rate / fps + 0.5)

        # ✅ Whisper hard limit: never exceed 30.0s per chunk
        max_whisper_samples = int(sample_rate * 30.0)
        samples_per_scene = min(samples_per_scene, max_whisper_samples)

        # ✅ If SRT provided, override segmentation completely
        if srt_path:
            time_segments = self._parse_srt_segments(srt_path)
            total_segments = len(time_segments)
        else:
            time_segments = None
            total_segments = math.ceil(total_samples / samples_per_scene)

        all_transcriptions = []

        print("[ManualLyrics] Loading Whisper Large V3...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3"
        ).to(device).eval()

        print("[ManualLyrics] Starting transcription...")

        for i in range(total_segments):

            # ✅ SRT-driven segment timing
            if time_segments:
                start_time, end_time = time_segments[i]

                start = int(start_time * sample_rate)
                end = int(end_time * sample_rate)

                # ✅ Whisper hard limit: truncate overly long SRT segments instead of failing
                seg_len_samples = end - start
                if seg_len_samples > max_whisper_samples:
                    truncated_end = min(start + max_whisper_samples, total_samples)
                    truncated_len = truncated_end - start
                    print(
                        f"[ManualLyrics] Truncating segment {i+1}/{total_segments}: "
                        f"{seg_len_samples / sample_rate:.2f}s exceeds 30.0s Whisper limit. "
                        f"Using {truncated_len / sample_rate:.2f}s."
                    )
                    end = truncated_end

            # ✅ Fixed-duration fallback
            else:
                start = i * samples_per_scene
                end = min(start + samples_per_scene, total_samples)

            text = self._transcribe_segment(
                waveform,
                sample_rate,
                start,
                end,
                processor,
                model,
                device,
                language
            )

            text = self._clean_lyric(text, use_ltx2)
            all_transcriptions.append(text)

            print(f"[ManualLyrics] Segment {i+1}/{total_segments} complete")

        combined_lines = [f"# Lyrics to fix: ({total_segments} segments)", ""]
        for i, lyric in enumerate(all_transcriptions, 1):
            combined_lines.append(f"lyricSegment{i}={lyric}")

        print("[ManualLyrics] Extraction complete!")
        return ("\n".join(combined_lines),)


class VRGDG_ManualLyricsExtractor_SRT_Advanced:
    """
    Advanced lyrics extractor using stable-ts (stable_whisper).

    - Transcribes the full track once with stable-ts.
    - Uses SRT timing when provided, otherwise fixed-size segmentation.
    - Outputs lyricSegmentN lines for manual cleanup workflows.
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
                "scene_duration_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 60.0}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "srt_path": ("STRING", {"default": ""}),
                "reference_lyrics": ("STRING", {"multiline": True, "default": ""}),
                "strict_reference_text": ("BOOLEAN", {"default": True}),
                "fill_aggressiveness": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "preserve_nonvocal_segments": ("BOOLEAN", {"default": True}),
                "alignment_min_words": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "model_name": ("STRING", {"default": "large-v3"}),
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
                        "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese",
                        "haitian creole", "pashto", "turkmen", "nynorsk", "maltese", "sanskrit",
                        "luxembourgish", "myanmar", "tibetan", "tagalog", "malagasy", "assamese",
                        "tatar", "hawaiian", "lingala", "hausa", "bashkir", "javanese", "sundanese",
                        "cantonese", "burmese", "valencian", "flemish", "haitian", "letzeburgesch",
                        "pushto", "panjabi", "moldavian", "moldovan", "sinhalese", "castilian",
                        "mandarin"
                    ],
                    {"default": "english"}
                ),
            }
        }

    def _parse_srt_segments(self, srt_path):
        segments = []
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.strip().split("\n\n")
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 2:
                continue

            times = lines[1]
            start_str, end_str = times.split(" --> ")

            def to_seconds(t):
                h, m, rest = t.split(":")
                s, ms = rest.split(",")
                return int(h) * 3600 + int(m) * 60 + float(s) + float(ms) / 1000.0

            start = to_seconds(start_str)
            end = to_seconds(end_str)
            segments.append((start, end))

        return segments

    def _clean_lyric(self, lyric: str) -> str:
        lyric = re.sub(r"(.)\1{3,}", r"\1" * 3, lyric)
        lyric = re.sub(r"[-—–_,]+", " ", lyric)
        lyric = re.sub(r"\s+", " ", lyric).strip()
        return lyric

    def _normalize_language(self, language: str):
        if language == "auto":
            return None

        try:
            import whisper
            lang_map = {name.lower(): code for code, name in whisper.tokenizer.LANGUAGES.items()}
            return lang_map.get(language.lower(), language)
        except Exception:
            return language

    def _collect_time_text_chunks(self, result):
        chunks = []
        for seg in getattr(result, "segments", []) or []:
            words = getattr(seg, "words", None)
            if words:
                for w in words:
                    wtxt = getattr(w, "word", "")
                    if not wtxt:
                        continue
                    start = float(getattr(w, "start", 0.0))
                    end = float(getattr(w, "end", start))
                    chunks.append((start, end, wtxt.strip()))
            else:
                stxt = getattr(seg, "text", "")
                if stxt:
                    start = float(getattr(seg, "start", 0.0))
                    end = float(getattr(seg, "end", start))
                    chunks.append((start, end, stxt.strip()))

        chunks.sort(key=lambda x: x[0])
        return chunks

    def _text_for_window(self, chunks, start_t, end_t):
        parts = [txt for st, en, txt in chunks if not (en <= start_t or st >= end_t)]
        return self._clean_lyric(" ".join(parts))

    def _normalize_for_match(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _split_reference_lyrics(self, reference_lyrics: str):
        lines = []
        for raw in reference_lyrics.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            raw = re.sub(r"\[[^\]]*\]", " ", raw)
            clean = self._clean_lyric(raw)
            if clean.lower() in {
                "lyrics",
                "full lyrics",
                "song lyrics",
                "reference lyrics",
            }:
                continue
            if clean:
                lines.append(clean)
        return lines

    def _clean_aligned_lyric_text(self, text: str) -> str:
        text = re.sub(r"\[[^\]]*\]", " ", str(text or ""))
        text = re.sub(r"\b(?:full\s+lyrics|song\s+lyrics|reference\s+lyrics|lyrics)\b", " ", text, flags=re.IGNORECASE)
        return self._clean_lyric(text)

    def _content_tokens(self, text: str):
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
            "from", "in", "into", "is", "it", "me", "my", "no", "not", "of",
            "on", "or", "so", "the", "then", "to", "up", "when", "with",
            "you", "your",
        }
        return [
            token
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if token not in stopwords
        ]

    def _strip_repeated_boundary_word(self, previous: str, current: str) -> str:
        prev_tokens = re.findall(r"[A-Za-z0-9']+", str(previous or ""))
        current_text = str(current or "").strip()
        current_tokens = re.findall(r"[A-Za-z0-9']+", current_text)
        if not prev_tokens or not current_tokens:
            return current_text

        if prev_tokens[-1].lower().strip("'") != current_tokens[0].lower().strip("'"):
            return current_text

        pattern = r"^\s*" + re.escape(current_tokens[0]) + r"\b\s*"
        return re.sub(pattern, "", current_text, count=1, flags=re.IGNORECASE).strip()

    def _cleanup_reference_segments(self, segments, reference_lines):
        if not reference_lines:
            return segments

        reference_token_set = set(self._content_tokens(" ".join(reference_lines)))
        cleaned_segments = []
        boundary_fixes = 0
        non_reference_blanks = 0
        for segment in segments:
            text = self._clean_aligned_lyric_text(segment)
            if cleaned_segments:
                before_boundary = text
                text = self._strip_repeated_boundary_word(cleaned_segments[-1], text)
                if text != before_boundary:
                    boundary_fixes += 1

            content = self._content_tokens(text)
            if content and not any(token in reference_token_set for token in content):
                text = ""
                non_reference_blanks += 1

            cleaned_segments.append(text)

        print(
            f"[ManualLyricsAdv] Reference cleanup active: "
            f"boundary_fixes={boundary_fixes}, non_reference_blanks={non_reference_blanks}"
        )
        return cleaned_segments

    def _is_alignment_meaningful_text(self, text: str, min_words: int = 2) -> bool:
        clean = self._clean_lyric(str(text or ""))
        if not clean:
            return False

        tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", clean)]
        if not tokens:
            return False

        filler_tokens = {
            "oh", "ooh", "oooh", "ooooh", "ah", "aah", "aaah", "aww",
            "yeah", "yah", "ya", "uh", "um", "hmm", "mm", "la", "na",
            "woah", "whoa", "ok", "okay", "hey", "yo",
        }
        meaningful = [t for t in tokens if t not in filler_tokens]
        return len(meaningful) >= max(1, int(min_words))

    def _nonvocal_placeholder(self, seg_index: int, asr_text: str = "") -> str:
        clean = self._clean_lyric(str(asr_text or ""))
        if clean:
            return clean

        fillers = [
            "ooohhh",
            "yeah, yeah",
            "oohh yeah",
            "ahh ahh",
            "la la",
        ]
        if seg_index < 0:
            seg_index = 0
        return fillers[seg_index % len(fillers)]

    def _align_segments_to_reference(
        self,
        asr_segments,
        reference_lines,
        strict_reference_text: bool = True,
        preserve_nonvocal_segments: bool = True,
        alignment_min_words: int = 2,
    ):
        if not reference_lines:
            return asr_segments

        aligned = []
        cursor = 0
        ref_count = len(reference_lines)
        seg_count = max(1, len(asr_segments))

        for i, asr_text in enumerate(asr_segments):
            if preserve_nonvocal_segments and (not self._is_alignment_meaningful_text(asr_text, alignment_min_words)):
                aligned.append(self._nonvocal_placeholder(i, asr_text))
                continue

            # Strict mode: enforce chronological lyric lines for each vocal segment.
            if strict_reference_text:
                if cursor < ref_count:
                    aligned.append(reference_lines[cursor])
                    cursor += 1
                else:
                    aligned.append(reference_lines[-1])
                continue

            asr_norm = self._normalize_for_match(asr_text)

            # Keep monotonic matching, but allow local search around position-based estimate.
            base = int((i / seg_count) * ref_count)
            start_idx = max(cursor, base - 3)
            end_idx = min(ref_count - 1, base + 8)

            best_idx = None
            best_score = -1.0
            if start_idx <= end_idx:
                for idx in range(start_idx, end_idx + 1):
                    ref_norm = self._normalize_for_match(reference_lines[idx])
                    score = difflib.SequenceMatcher(None, asr_norm, ref_norm).ratio()
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            if best_idx is None:
                if cursor < ref_count:
                    best_idx = cursor
                else:
                    aligned.append(self._clean_lyric(asr_text))
                    continue

            # If confidence is too weak, fall back to next chronological line.
            if best_score < 0.22 and cursor < ref_count:
                best_idx = cursor

            aligned.append(reference_lines[best_idx])
            cursor = min(ref_count, best_idx + 1)

        return aligned

    def _is_meaningful_text(self, text: str, aggressiveness: int = 1) -> bool:
        clean = self._clean_lyric(text)
        if not clean:
            return False
        tokens = re.findall(r"[A-Za-z0-9]+", clean)

        # Level 1: conservative (roughly current behavior)
        if aggressiveness <= 1:
            return any(len(tok) >= 2 for tok in tokens)
        # Level 2+: accept short words like "I", "a", "oh"
        if aggressiveness == 2:
            return len(tokens) > 0
        # Level 3+: allow any non-space content
        return len(clean) > 0

    def _merge_missing_segments(self, primary_segments, backup_segments, aggressiveness: int = 1):
        merged = []
        filled_backup = 0
        filled_neighbor = 0
        total = min(len(primary_segments), len(backup_segments))
        for i in range(total):
            p = self._clean_lyric(primary_segments[i])
            b = self._clean_lyric(backup_segments[i])

            # Prefer aligned text, but recover blank/low-signal segments from backup ASR.
            if (not self._is_meaningful_text(p, aggressiveness)) and self._is_meaningful_text(b, aggressiveness):
                merged.append(b)
                filled_backup += 1
            else:
                merged.append(p)

        if len(primary_segments) > total:
            merged.extend(primary_segments[total:])
        elif len(backup_segments) > total:
            merged.extend(backup_segments[total:])

        # Level 3+: if still weak, borrow nearest meaningful neighbor.
        if aggressiveness >= 3:
            for i in range(len(merged)):
                if self._is_meaningful_text(merged[i], aggressiveness):
                    continue

                left = None
                for j in range(i - 1, -1, -1):
                    if self._is_meaningful_text(merged[j], aggressiveness):
                        left = merged[j]
                        break

                right = None
                for j in range(i + 1, len(merged)):
                    if self._is_meaningful_text(merged[j], aggressiveness):
                        right = merged[j]
                        break

                replacement = left if left is not None else right
                if replacement is not None:
                    merged[i] = replacement
                    filled_neighbor += 1

        return merged, filled_backup, filled_neighbor

    def extract_lyrics(
        self,
        audio,
        scene_duration_seconds=4.0,
        fps=25,
        srt_path="",
        reference_lyrics="",
        strict_reference_text=True,
        fill_aggressiveness=1,
        preserve_nonvocal_segments=True,
        alignment_min_words=2,
        model_name="large-v3",
        language="english",
        **kwargs
    ):
        try:
            import stable_whisper
        except ImportError:
            raise RuntimeError(
                "stable-ts is not installed. Install with: pip install -U stable-ts"
            )

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        mono = waveform.mean(dim=1).squeeze().detach().cpu()
        total_samples = mono.shape[-1]
        total_duration = float(total_samples) / float(sample_rate)
        print("[ManualLyricsAdv] Code version: 2026-05-23-reference-cleanup-v2")
        print(f"[ManualLyricsAdv] Processing audio: {total_duration:.2f}s @ {sample_rate}Hz")

        if srt_path:
            time_segments = self._parse_srt_segments(srt_path)
        else:
            frames_per_scene = int(round(int(fps) * float(scene_duration_seconds)))
            samples_per_scene = int(frames_per_scene * sample_rate / int(fps) + 0.5)
            total_segments = math.ceil(total_samples / samples_per_scene)
            time_segments = []
            for i in range(total_segments):
                st = (i * samples_per_scene) / sample_rate
                en = min((i + 1) * samples_per_scene, total_samples) / sample_rate
                time_segments.append((st, en))

        total_segments = len(time_segments)
        print(f"[ManualLyricsAdv] Segments: {total_segments}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        lang = self._normalize_language(language)

        print(f"[ManualLyricsAdv] Loading stable-ts model: {model_name} ({device})")
        model = stable_whisper.load_model(model_name, device=device)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav_path = tmp.name

        try:
            torchaudio.save(tmp_wav_path, mono.unsqueeze(0), sample_rate)

            use_reference = bool(reference_lyrics and reference_lyrics.strip())
            reference_lines = self._split_reference_lyrics(reference_lyrics) if use_reference else []
            cleaned_reference_lyrics = "\n".join(reference_lines)

            aggressiveness = int(fill_aggressiveness)

            if use_reference and cleaned_reference_lyrics:
                print("[ManualLyricsAdv] Running native stable-ts text alignment...")
                try:
                    result = model.align(
                        tmp_wav_path,
                        cleaned_reference_lyrics,
                        language=lang,
                        verbose=False,
                    )
                except Exception as align_err:
                    print(f"[ManualLyricsAdv] Native align failed: {align_err}")
                    result = None

                if result is None:
                    print("[ManualLyricsAdv] Falling back to transcription + fuzzy reference mapping...")
                    result = model.transcribe(
                        tmp_wav_path,
                        language=lang,
                        word_timestamps=True,
                        verbose=False,
                    )
                    chunks = self._collect_time_text_chunks(result)
                    all_transcriptions = []
                    for i, (start_t, end_t) in enumerate(time_segments, 1):
                        text = self._clean_aligned_lyric_text(self._text_for_window(chunks, start_t, end_t))
                        all_transcriptions.append(text)
                        if i % 16 == 0 or i == total_segments:
                            print(f"[ManualLyricsAdv] Segment {i}/{total_segments} complete")

                    if reference_lines:
                        print(f"[ManualLyricsAdv] Aligning {len(all_transcriptions)} segments to {len(reference_lines)} reference lyric lines")
                        all_transcriptions = self._align_segments_to_reference(
                            all_transcriptions,
                            reference_lines,
                            strict_reference_text=bool(strict_reference_text),
                            preserve_nonvocal_segments=bool(preserve_nonvocal_segments),
                            alignment_min_words=int(alignment_min_words),
                        )
                else:
                    chunks = self._collect_time_text_chunks(result)
                    all_transcriptions = []
                    for i, (start_t, end_t) in enumerate(time_segments, 1):
                        text = self._clean_aligned_lyric_text(self._text_for_window(chunks, start_t, end_t))
                        all_transcriptions.append(text)
                        if i % 16 == 0 or i == total_segments:
                            print(f"[ManualLyricsAdv] Segment {i}/{total_segments} complete")

                    # Native align can miss sparse windows; recover blanks from regular ASR timing.
                    empty_count = sum(
                        1 for t in all_transcriptions if not self._is_meaningful_text(t, aggressiveness)
                    )
                    if empty_count > 0:
                        print(f"[ManualLyricsAdv] Native align produced {empty_count} low-signal segments; running backup transcription fill...")
                        backup_result = model.transcribe(
                            tmp_wav_path,
                            language=lang,
                            word_timestamps=True,
                            verbose=False,
                        )
                        backup_chunks = self._collect_time_text_chunks(backup_result)
                        backup_transcriptions = []
                        for start_t, end_t in time_segments:
                            backup_transcriptions.append(self._clean_aligned_lyric_text(self._text_for_window(backup_chunks, start_t, end_t)))

                        all_transcriptions, filled_backup, filled_neighbor = self._merge_missing_segments(
                            all_transcriptions,
                            backup_transcriptions,
                            aggressiveness=aggressiveness,
                        )
                        print(
                            f"[ManualLyricsAdv] Filled {filled_backup} segments from backup transcription "
                            f"and {filled_neighbor} from neighbor carry (aggressiveness={aggressiveness})"
                        )

                    if reference_lines:
                        print("[ManualLyricsAdv] Keeping native stable-ts reference timing windows.")
                    all_transcriptions = self._cleanup_reference_segments(
                        all_transcriptions,
                        reference_lines,
                    )
            else:
                print("[ManualLyricsAdv] Running full transcription...")
                result = model.transcribe(
                    tmp_wav_path,
                    language=lang,
                    word_timestamps=True,
                    verbose=False,
                )

                chunks = self._collect_time_text_chunks(result)
                all_transcriptions = []

                for i, (start_t, end_t) in enumerate(time_segments, 1):
                    text = self._text_for_window(chunks, start_t, end_t)
                    all_transcriptions.append(text)
                    if i % 16 == 0 or i == total_segments:
                        print(f"[ManualLyricsAdv] Segment {i}/{total_segments} complete")

            combined_lines = [f"# Lyrics to fix: ({total_segments} segments)", ""]
            for i, lyric in enumerate(all_transcriptions, 1):
                combined_lines.append(f"lyricSegment{i}={lyric}")

            print("[ManualLyricsAdv] Extraction complete!")
            return ("\n".join(combined_lines),)

        finally:
            try:
                if os.path.exists(tmp_wav_path):
                    os.remove(tmp_wav_path)
            except Exception:
                pass




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
     "VRGDG_ManualLyricsExtractor_SRT": VRGDG_ManualLyricsExtractor_SRT,
     "VRGDG_ManualLyricsExtractor_SRT_Advanced": VRGDG_ManualLyricsExtractor_SRT_Advanced
    
    
    
    
    
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
    "VRGDG_ManualLyricsExtractor_SRT": "Manual Lyrics Extractor (SRT Segments)",
    "VRGDG_ManualLyricsExtractor_SRT_Advanced": "Manual Lyrics Extractor (SRT Advanced - stable-ts)"
    
    
    
}









