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
from transformers import WhisperProcessor, WhisperForConditionalGeneration


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


NODE_CLASS_MAPPINGS = {

     "VRGDG_ManualLyricsExtractor": VRGDG_ManualLyricsExtractor,
     "VRGDG_PromptSplitterForManual":VRGDG_PromptSplitterForManual,
     "VRGDG_CombinevideosV5":VRGDG_CombinevideosV5


 



}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ManualLyricsExtractor": "🌀 VRGDG_ManualLyricsExtractor",
    "VRGDG_PromptSplitterForManual":"✂️ VRGDG_PromptSplitterForManual",
    "VRGDG_CombinevideosV5":"VRGDG_CombinevideosV5"


}
