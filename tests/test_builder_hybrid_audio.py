import ast
import os
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_SOURCE = ROOT / "VRGDG_MusicVideoBuilderNodes.py"
UI_SOURCE = ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js"


class _CompletedProcess:
    returncode = 0
    stderr = ""
    stdout = ""


class _RecordingSubprocess:
    def __init__(self):
        self.commands = []

    def run(self, command, **_kwargs):
        self.commands.append(command)
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return _CompletedProcess()


def load_scene_audio_mixer(recording_subprocess):
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"), filename=str(NODE_SOURCE))
    names = {"_concat_file_path", "_scene_audio_mix_folder", "_prepare_scene_audio_mix"}
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "os": os,
        "shutil": shutil,
        "subprocess": recording_subprocess,
        "_find_ffmpeg_path": lambda: "ffmpeg",
        "_srt_path": lambda folder: os.path.join(folder, "builder_segments.srt"),
        "_segments_to_srt": lambda _segments: "",
        "_read_audio_peaks": lambda _path, _count: {"duration": 8.0, "peaks": []},
        "_estimate_beats_from_audio": lambda *_args, **_kwargs: ([], 0.0),
    }
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(NODE_SOURCE), "exec"), namespace)
    return namespace["_prepare_scene_audio_mix"]


class BuilderHybridAudioTests(unittest.TestCase):
    def test_global_audio_fills_scenes_without_custom_override(self):
        recorder = _RecordingSubprocess()
        prepare_mix = load_scene_audio_mixer(recorder)
        with tempfile.TemporaryDirectory() as folder:
            global_audio = Path(folder) / "song.wav"
            silent_override = Path(folder) / "silent_scene.wav"
            global_audio.touch()
            silent_override.touch()

            result = prepare_mix({
                "project_folder": folder,
                "global_audio_path": str(global_audio),
                "segments": [
                    {"start": 0, "end": 4, "custom_audio_path": ""},
                    {
                        "start": 4,
                        "end": 8,
                        "custom_audio_path": str(silent_override),
                        "custom_audio_duration": 4,
                    },
                ],
            })

        trim_commands = [command for command in recorder.commands if "-ss" in command]
        self.assertEqual(len(trim_commands), 2)
        self.assertEqual(Path(trim_commands[0][trim_commands[0].index("-i") + 1]), global_audio)
        self.assertEqual(trim_commands[0][trim_commands[0].index("-ss") + 1], "0.000000")
        self.assertEqual(Path(trim_commands[1][trim_commands[1].index("-i") + 1]), silent_override)
        self.assertTrue(result["used_scene_audio"])

    def test_ui_preserves_global_audio_and_prioritizes_it_for_stitch(self):
        source = UI_SOURCE.read_text(encoding="utf-8")
        self.assertIn("global_audio_path: currentProjectAudioPath()", source)
        self.assertIn(
            "const sceneAudioMode = !embeddedSceneAudioMode && !globalAudioPath && usingSceneAudioMode();",
            source,
        )
        self.assertIn('audio_path: embeddedSceneAudioMode ? "" : globalAudioPath', source)

        prepare_start = source.index("async function prepareSceneAudioMix(")
        prepare_end = source.index("async function renderSceneVideoWithProgress(", prepare_start)
        prepare_source = source[prepare_start:prepare_end]
        self.assertNotIn("audioInput.value = data.audio_path", prepare_source)

    def test_loaded_global_audio_controls_timeline_playback_and_waveform(self):
        source = UI_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "return !currentProjectAudioPath() && (usingSceneAudioMode() || usingRenderedSceneAudioMode());",
            source,
        )
        self.assertIn(
            "const peaks = currentProjectAudioPath() && state.peaks.length ? state.peaks : [0];",
            source,
        )
        self.assertGreaterEqual(source.count("activateGlobalTimelineAudioPlayback(0);"), 4)


if __name__ == "__main__":
    unittest.main()
