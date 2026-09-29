

from .routes import _ensure_music_builder_routes


class VRGDG_MusicVideoBuilderUI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"default": ""}),
                "project_folder": ("STRING", {"default": ""}),
                "session_path": ("STRING", {"default": ""}),
                "srt_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("project_folder", "session_path", "srt_path")
    FUNCTION = "noop"
    CATEGORY = "VRGDG/UI"
    DESCRIPTION = "Prototype UI for building a music video from audio, timing segments, prompts, and approved ZImage previews."

    def noop(self, audio_path, project_folder, session_path, srt_path):
        return (project_folder, session_path, srt_path)


_ensure_music_builder_routes()


NODE_CLASS_MAPPINGS = {
    "VRGDG_MusicVideoBuilderUI": VRGDG_MusicVideoBuilderUI,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "VRGDG_MusicVideoBuilderUI": "VRGDG Music Video Builder UI",
}
