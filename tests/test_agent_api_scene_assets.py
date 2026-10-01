"""Scene views report what the Video Builder saved on the scene: its lyrics, prompts, picture, video and audio."""

import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

projects = importlib.import_module(f"{ROOT.name}.agent_api.projects")
from test_agent_api_references_llm import Base  # noqa: E402


class SceneAssetTests(Base):
    def setUp(self):
        super().setUp()
        videos = os.path.join(self.folder, "rendered_scene_videos")
        thumbs = os.path.join(self.folder, "scene_video_thumbnails")
        audio = os.path.join(self.folder, "minimax_h3_scene_audio")
        for folder in (videos, thumbs, audio):
            os.makedirs(folder)
        self.video = os.path.join(videos, "video_0001-audio.mp4")
        self.thumb = os.path.join(thumbs, "video_0001-audio.jpg")
        for path in (self.video, self.thumb, os.path.join(audio, "scene_audio_0001.wav")):
            Path(path).write_bytes(b"x")
        session = self.read_session()
        first = session["segments"][0]
        first.update({"video_path": self.video, "video_thumbnail_path": self.thumb, "minimax_h3_prompt": "detailed_description:\n[Shot 1] x.",
                      "story_beat": "Darrel paces.", "lyric_text": "line one", "notes": "a note", "approved_image_path": self.image})
        self.write_session(session)

    def test_a_rendered_scene_shows_its_video_thumbnail_prompt_and_lyrics(self):
        scene = projects.get_scene_detail("Song", "1")
        self.assertEqual(scene["rendered_video"]["filename"], "video_0001-audio.mp4")
        self.assertEqual(scene["video_thumbnail"]["filename"], "video_0001-audio.jpg")
        self.assertEqual(scene["approved_image"]["filename"], "darrel.png", "a picture outside the project folder is still reported")
        self.assertEqual(scene["scene_audio"]["filename"], "scene_audio_0001.wav")
        self.assertEqual((scene["lyrics"], scene["story_beat"]), ("line one", "Darrel paces."))
        self.assertTrue(scene["minimax_h3_prompt"].startswith("detailed_description:"))
        self.assertEqual(scene["status"], "has_video")

    def test_a_scene_with_only_a_minimax_prompt_counts_as_having_a_prompt(self):
        session = self.read_session()
        session["segments"][1]["minimax_h3_prompt"] = "detailed_description:\n[Shot 1] y."
        self.write_session(session)
        self.assertEqual(projects.get_scene_detail("Song", "2")["status"], "has_prompt")
        self.assertEqual(len(projects.get_project_scenes("Song", has_prompt=True)), 2)
        summary = projects.get_project_summary("Song")
        self.assertEqual((summary["scenes_with_prompt"], summary["scenes_with_video"]), (2, 1))

    def test_a_scene_without_media_has_none(self):
        scene = projects.get_scene_detail("Song", "3")
        self.assertIsNone(scene["rendered_video"])
        self.assertIsNone(scene["video_thumbnail"])
        self.assertEqual(scene["lyrics"], "line 3")


if __name__ == "__main__":
    unittest.main()
