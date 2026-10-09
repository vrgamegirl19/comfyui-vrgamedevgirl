"""Speech provider contracts, dialogue drafting and scene import parity."""

import base64
import copy
import importlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
speech = importlib.import_module(f"{ROOT.name}.builder.elevenlabs_speech")
llm = importlib.import_module(f"{ROOT.name}.llm.scene_dialogue")
audio = importlib.import_module(f"{ROOT.name}.builder.scene_audio_settings")
api = importlib.import_module(f"{ROOT.name}.agent_api.scene_dialogue")
audio_api = importlib.import_module(f"{ROOT.name}.agent_api.scene_audio")
instructions = importlib.import_module(f"{ROOT.name}.llm.builder_instructions")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")

DRAFT = {"speaker_id": "alice", "text": "Stay behind me.", "delivery": "Quiet and worried",
         "allow_rewrite": False, "model_id": "eleven_v4"}
VOICE = {"enabled": True, "voice_id": "voice1", "name": "Alice voice"}


def session():
    return {"video_type": "speaking", "revision": 7, "text_gemma_runner": "own_server",
            "elevenlabs_api_key_project": "saved-secret", "audio_clips": [],
            "segments": [{"id": "a", "start": 0, "end": 4}, {"id": "b", "start": 4, "end": 8}],
            "flux_reference_builder": {"subjects": [{"id": "alice", "name": "Alice", "elevenlabs_voice": VOICE}]}}


class Response(io.BytesIO):
    headers = {"Content-Type": "audio/mpeg"}


class SceneDialogueTests(unittest.TestCase):
    def test_provider_request_preserves_tags_and_returns_transient_mp3(self):
        draft = {**DRAFT, "text": "[whispers] Stay behind me."}
        with patch.object(speech, "urlopen", return_value=Response(b"mp3-audio")) as fetch:
            result = speech.generate_speech("secret", "voice1", draft)
        request = fetch.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.elevenlabs.io/v1/text-to-speech/voice1?output_format=mp3_44100_128")
        self.assertEqual(request.get_header("Xi-api-key"), "secret")
        self.assertEqual(json.loads(request.data), {"text": draft["text"], "model_id": "eleven_v4"})
        self.assertEqual(base64.b64decode(result["audio_data"].split(",")[1]), b"mp3-audio")
        self.assertNotIn("secret", json.dumps(result))
        self.assertEqual(fetch.call_count, 1)

    def test_bad_inputs_fail_before_network_and_errors_do_not_leak(self):
        with patch.object(speech, "urlopen") as fetch:
            for draft in ({**DRAFT, "text": ""}, {**DRAFT, "model_id": "other"},
                          {**DRAFT, "model_id": "eleven_v3", "text": "x" * 5001}):
                with self.assertRaises(ValueError):
                    speech.generate_speech("secret", "voice1", draft)
            fetch.assert_not_called()
        for error in (URLError("secret"), HTTPError("https://api.elevenlabs.io", 401, "secret", {}, io.BytesIO(b"secret"))):
            with patch.object(speech, "urlopen", side_effect=error) as fetch:
                with self.assertRaises(ValueError) as raised:
                    speech.generate_speech("secret", "voice1", DRAFT)
                self.assertNotIn("secret", str(raised.exception))
                self.assertEqual(fetch.call_count, 1)

    def test_llm_selected_runner_preset_and_preserved_words(self):
        payload = {"text_runner": "own_server", "dialogue": DRAFT, "project_folder": "project", "scene_id": "a"}
        with patch.object(llm, "_effective_builder_instruction", return_value="CUSTOM DIALOGUE PRESET") as preset, \
             patch.object(llm, "_run_builder_text_llm", return_value=("[whispers] Stay behind me!", {"runner": "own_server"})) as run:
            result = llm.craft_dialogue(payload)
            self.assertEqual(result["dialogue"]["text"], "[whispers] Stay behind me!")
            self.assertIn("CUSTOM DIALOGUE PRESET", run.call_args.args[1])
            self.assertEqual(run.call_args.args[0]["text_runner"], "own_server")
            self.assertEqual(preset.call_args.args[1], "elevenlabs_dialogue")
            run.return_value = ("[whispers] Run away!", {})
            with self.assertRaisesRegex(ValueError, "changed the spoken words"):
                llm.craft_dialogue(payload)
            self.assertEqual(llm.craft_dialogue({**payload, "dialogue": {**DRAFT, "allow_rewrite": True}})["dialogue"]["text"], "[whispers] Run away!")

    def test_saved_key_voice_runner_and_mode_guards(self):
        saved = session()
        with patch.object(api, "_get_active_session_and_folder", return_value=("folder", saved)), \
             patch.object(api, "generate_speech", return_value={"audio_data": "preview"}) as generate:
            request = api.dialogue_request("p", "a", DRAFT)
            self.assertEqual(request["text_runner"], "own_server")
            self.assertEqual(request["voice_id"], "voice1")
            api.generate_project_speech("p", "a", DRAFT)
            generate.assert_called_once_with("saved-secret", "voice1", DRAFT)
            saved["video_type"] = "singing"
            generate.reset_mock()
            with self.assertRaises(errors.ValidationError):
                api.generate_project_speech("p", "a", DRAFT)
            generate.assert_not_called()
        for subject in ({"id": "alice", "reference_type": "prop", "elevenlabs_voice": VOICE},
                        {"id": "alice", "elevenlabs_voice": {**VOICE, "enabled": False}}):
            with self.assertRaises(ValueError):
                speech.dialogue_speaker({"subjects": [subject]}, "alice")

    def test_description_job_does_not_put_credentials_in_llm_request(self):
        request = {"api_key": "eleven-secret", "text_runner": "lm_studio", "dialogue": DRAFT}
        job = SimpleNamespace(id="j", project_id="p", params={"scene_id": "a", "dialogue": DRAFT})
        manager = SimpleNamespace(update_progress=lambda *args, **kwargs: None)
        with patch.object(api, "dialogue_request", return_value=request), \
             patch.object(api, "prepare_llm_payload", side_effect=lambda value: value) as prepare, \
             patch.object(api, "craft_dialogue", return_value={"dialogue": DRAFT}):
            api.run_craft_dialogue_job(job, manager)
            self.assertNotIn("api_key", prepare.call_args.args[0])

    def test_import_six_second_take_updates_only_target_dialogue_and_ripples(self):
        saved = session()
        original = copy.deepcopy(saved)
        attachment = {"saved_path": "original-source.mp3", "duration": 6, "audio_name": "speech.mp3"}
        with patch.object(audio_api, "_get_active_session_and_folder", return_value=("folder", saved)), \
             patch.object(audio_api, "_save_scene_audio", return_value=attachment), \
             patch.object(audio_api, "_persist_session", return_value={"revision": 8}) as persist:
            result = audio_api.patch_audio_settings("p", {}, "a", 7, "data:audio/mpeg;base64,YQ==", "speech.mp3", dialogue={**DRAFT, "audio_data": "must not persist"})
            updated = persist.call_args.args[1]
        self.assertEqual(saved, original)
        self.assertEqual(updated["segments"][0]["end"], 6)
        self.assertEqual(updated["segments"][1]["start"], 6)
        self.assertEqual(updated["audio_clips"][0]["scene_id"], "a")
        self.assertEqual(updated["segments"][0]["scene_dialogue"], DRAFT)
        self.assertNotIn("scene_dialogue", updated["segments"][1])
        self.assertEqual(result["settings"]["dialogue"], DRAFT)

    def test_instruction_key_uses_atomic_scene_and_preset_writes(self):
        self.assertIn("elevenlabs_dialogue", instructions._BUILDER_INSTRUCTION_DEFAULTS)
        with patch.object(instructions, "_project_folder_from_builder_payload", return_value="folder"), \
             patch.object(instructions, "atomic_write_text") as write, \
             patch.object(instructions, "_get_builder_instruction", return_value={}), \
             patch.object(instructions.os, "makedirs"):
            instructions._save_builder_instruction({"key": "elevenlabs_dialogue", "scene_id": "a", "text": "custom"})
            self.assertEqual(write.call_args.args[1], "custom\n")

    def test_browser_dialogue_generation_review_and_import(self):
        result = subprocess.run(["node", str(ROOT / "tests/scene_dialogue.cjs")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
