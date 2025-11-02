import math
import random
import re
import torch
import torchaudio
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


NODE_CLASS_MAPPINGS = {

     "VRGDG_ManualLyricsExtractor": VRGDG_ManualLyricsExtractor,
     "VRGDG_PromptSplitterForManual":VRGDG_PromptSplitterForManual


 



}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_ManualLyricsExtractor": "🌀 VRGDG_ManualLyricsExtractor",
    "VRGDG_PromptSplitterForManual":"✂️ VRGDG_PromptSplitterForManual"


}
