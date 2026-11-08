"""The Thumbnail dialog's API image engine: the frame edited by Gemini, Grok or OpenAI with the LLM Runner's key. No real calls are made."""

import base64
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
api_image = importlib.import_module(f"{pkg_name}.llm.api_image")
engine = importlib.import_module(f"{pkg_name}.builder.thumbnail_api_image")
thumbnail = importlib.import_module(f"{pkg_name}.builder.thumbnail")

KEY = "sk-test-secret-key"


def png(width: int = 640, height: int = 360, color=(200, 40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class Reply:
    """What ``urlopen_verified`` returns: a context manager with ``read``."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self) -> bytes:
        return self.body


def http_error(request, code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(request.full_url, code, "error", {}, io.BytesIO(body.encode("utf-8")))


class Wire:
    """Stands in for the network: answers each request with the next item and keeps the requests for the assertions."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, tuple):
            raise http_error(request, *answer)
        return Reply(answer if isinstance(answer, bytes) else json.dumps(answer).encode("utf-8"))

    def body(self, index: int = 0):
        return json.loads(self.requests[index].data.decode("utf-8"))

    def header(self, index: int, name: str):
        return {key.lower(): value for key, value in self.requests[index].header_items()}.get(name.lower())


def gemini_reply(image: bytes) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": "Here it is."}, {"inlineData": {"mimeType": "image/png", "data": b64(image)}}]}}]}


def edit(provider: str, wire: Wire, source: Image.Image = None, model: str = "m", key: str = KEY):
    with patch.object(api_image, "urlopen_verified", wire):
        return api_image.edit_with_api(provider, key, model, "Add the title HEAT in graffiti.", source or Image.new("RGB", (640, 360)))


class ProviderTests(unittest.TestCase):
    def test_the_llm_runner_provider_names_map_to_the_image_providers(self):
        self.assertEqual(api_image.api_image_provider("google"), "gemini")
        self.assertEqual(api_image.api_image_provider(" Grok "), "grok")
        self.assertEqual(api_image.api_image_provider("xai"), "grok")
        self.assertEqual(api_image.api_image_provider("openai"), "openai")

    def test_a_provider_that_cannot_draw_is_refused_with_what_to_pick(self):
        with self.assertRaisesRegex(ValueError, "Gemini \\(google\\), Grok or OpenAI"):
            api_image.api_image_provider("anthropic")
        with self.assertRaisesRegex(ValueError, "no API provider"):
            api_image.api_image_provider("")

    def test_the_capabilities_list_every_provider_with_its_models_and_a_default(self):
        providers = api_image.api_image_capabilities()["providers"]
        self.assertEqual(set(providers), {"gemini", "grok", "openai"})
        for info in providers.values():
            self.assertTrue(info["models"])
            self.assertEqual(info["default"], info["models"][0])
            self.assertTrue(info["label"])

    def test_the_grok_image_models_are_not_added_to_the_chat_model_lists(self):
        text = (ROOT / "llm" / "api.py").read_text(encoding="utf-8")
        self.assertNotIn("grok-imagine", text)


class GeminiTests(unittest.TestCase):
    def test_the_frame_and_the_prompt_go_out_and_the_key_stays_in_a_header(self):
        wire = Wire(gemini_reply(png()))
        edit("gemini", wire, model="gemini-3-pro-image-preview")
        request = wire.requests[0]
        self.assertIn("/models/gemini-3-pro-image-preview:generateContent", request.full_url)
        self.assertNotIn(KEY, request.full_url)
        self.assertEqual(wire.header(0, "x-goog-api-key"), KEY)
        parts = wire.body()["contents"][0]["parts"]
        self.assertEqual(parts[0]["text"], "Add the title HEAT in graffiti.")
        self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/png")
        self.assertEqual(wire.body()["generationConfig"]["imageConfig"]["aspectRatio"], "16:9")

    def test_the_picture_is_taken_from_the_inline_data(self):
        result = edit("gemini", Wire(gemini_reply(png(color=(10, 200, 30)))))
        self.assertEqual(result.getpixel((5, 5)), (10, 200, 30))

    def test_an_option_the_model_does_not_know_is_retried_once_without_it(self):
        wire = Wire((400, "imageConfig is not supported"), gemini_reply(png()))
        edit("gemini", wire)
        self.assertEqual(len(wire.requests), 2)
        self.assertIn("generationConfig", wire.body(0))
        self.assertNotIn("generationConfig", wire.body(1))

    def test_an_answer_with_words_only_says_what_the_model_said(self):
        wire = Wire({"candidates": [{"content": {"parts": [{"text": "I cannot draw that."}]}}]})
        with self.assertRaisesRegex(api_image.ApiImageError, "Gemini returned no picture.*I cannot draw that"):
            edit("gemini", wire)

    def test_a_blocked_prompt_says_why(self):
        wire = Wire({"promptFeedback": {"blockReason": "SAFETY"}})
        with self.assertRaisesRegex(api_image.ApiImageError, "Blocked: SAFETY"):
            edit("gemini", wire)

    def test_a_refused_key_is_reported_with_the_status_and_never_shows_the_key(self):
        wire = Wire((403, "API key not valid"))
        with self.assertRaises(api_image.ApiImageError) as caught:
            edit("gemini", wire)
        self.assertEqual(caught.exception.status, 403)
        self.assertIn("Gemini HTTP 403", str(caught.exception))
        self.assertNotIn(KEY, str(caught.exception))


class GrokTests(unittest.TestCase):
    def test_the_frame_goes_to_the_edit_endpoint_as_a_data_uri(self):
        wire = Wire({"data": [{"b64_json": b64(png())}]})
        edit("grok", wire, model="grok-imagine-image-2.0")
        self.assertEqual(wire.requests[0].full_url, "https://api.x.ai/v1/images/edits")
        self.assertEqual(wire.header(0, "authorization"), f"Bearer {KEY}")
        body = wire.body()
        self.assertEqual(body["model"], "grok-imagine-image-2.0")
        self.assertEqual(body["image"]["type"], "image_url")
        self.assertTrue(body["image"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(body["response_format"], "b64_json")

    def test_a_picture_sent_as_a_link_is_downloaded(self):
        wire = Wire({"data": [{"url": "https://imgen.x.ai/one.png"}]}, png(color=(1, 2, 3)))
        result = edit("grok", wire)
        self.assertEqual(wire.requests[1].full_url, "https://imgen.x.ai/one.png")
        self.assertEqual(result.getpixel((0, 0)), (1, 2, 3))

    def test_an_option_it_does_not_know_is_retried_once_with_the_plain_request(self):
        wire = Wire((400, "unknown field response_format"), {"data": [{"b64_json": b64(png())}]})
        edit("grok", wire)
        self.assertIn("response_format", wire.body(0))
        self.assertNotIn("response_format", wire.body(1))

    def test_a_reply_without_a_picture_is_an_error(self):
        with self.assertRaisesRegex(api_image.ApiImageError, "Grok returned no picture. content policy"):
            edit("grok", Wire({"error": {"message": "content policy"}}))

    def test_a_blocked_account_gets_the_hint(self):
        with self.assertRaisesRegex(api_image.ApiImageError, "code 1010"):
            edit("grok", Wire((403, "error code: 1010")))


class OpenAiTests(unittest.TestCase):
    def test_the_frame_is_posted_as_a_form_to_the_edit_endpoint(self):
        wire = Wire({"data": [{"b64_json": b64(png(1536, 1024))}]})
        edit("openai", wire, model="gpt-image-2")
        request = wire.requests[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/images/edits")
        self.assertEqual(wire.header(0, "authorization"), f"Bearer {KEY}")
        self.assertIn("multipart/form-data", wire.header(0, "content-type"))
        body = request.data
        for text in (b'name="model"\r\n\r\ngpt-image-2', b'name="size"\r\n\r\n1536x1024', b'name="image"; filename="frame.png"', b"Add the title HEAT"):
            self.assertIn(text, body)

    def test_the_three_by_two_answer_is_cropped_to_sixteen_by_nine(self):
        result = edit("openai", Wire({"data": [{"b64_json": b64(png(1536, 1024))}]}))
        self.assertEqual(result.size, (1536, 864))


class SharedTests(unittest.TestCase):
    def test_sixteen_by_nine_is_left_alone_and_other_shapes_are_cropped_in_the_middle(self):
        wide = Image.new("RGB", (1920, 1080))
        self.assertIs(api_image.fit_wide(wide), wide)
        self.assertEqual(api_image.fit_wide(Image.new("RGB", (1000, 1000))).size, (1000, 562))
        self.assertEqual(api_image.fit_wide(Image.new("RGB", (2400, 1000))).size, (1778, 1000))

    def test_the_crop_keeps_the_middle(self):
        tall = Image.new("RGB", (160, 160), (0, 0, 0))
        for y in range(60, 100):
            for x in range(160):
                tall.putpixel((x, y), (255, 255, 255))
        cropped = api_image.fit_wide(tall)
        self.assertEqual(cropped.size, (160, 90))
        self.assertEqual(cropped.getpixel((80, 45)), (255, 255, 255))

    def test_a_big_frame_is_shrunk_before_it_is_sent(self):
        wire = Wire(gemini_reply(png()))
        edit("gemini", wire, source=Image.new("RGB", (4096, 2304)))
        sent = base64.b64decode(wire.body()["contents"][0]["parts"][1]["inlineData"]["data"])
        with Image.open(io.BytesIO(sent)) as image:
            self.assertEqual(image.size, (2048, 1152))

    def test_a_missing_key_or_prompt_is_refused_before_any_call(self):
        wire = Wire()
        with self.assertRaisesRegex(ValueError, "Add your Grok API key in the LLM Runner"):
            edit("grok", wire, key="  ")
        with patch.object(api_image, "urlopen_verified", wire), self.assertRaisesRegex(ValueError, "Write the prompt first"):
            api_image.edit_with_api("grok", KEY, "m", "  ", Image.new("RGB", (4, 4)))
        self.assertEqual(wire.requests, [])

    def test_a_network_failure_names_the_provider(self):
        def down(_request, timeout):
            raise urllib.error.URLError("no route")

        with patch.object(api_image, "urlopen_verified", down), self.assertRaisesRegex(api_image.ApiImageError, "OpenAI network error: no route"):
            api_image.edit_with_api("openai", KEY, "m", "p", Image.new("RGB", (4, 4)))


class MakeApiImageTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.project = os.path.join(self.folder.name, "project")
        os.makedirs(self.project)
        self.source = os.path.join(self.project, "source.png")
        Image.new("RGB", (640, 360), (20, 20, 20)).save(self.source)

    def payload(self, **extra):
        return {"project_folder": self.project, "source_path": self.source, "prompt": "Add HEAT.", "provider": "google", "api_key": KEY, **extra}

    def make(self, wire: Wire, **extra):
        with patch.object(api_image, "urlopen_verified", wire):
            return engine._make_thumbnail_api_image(self.payload(**extra))

    def test_the_picture_is_kept_as_the_next_numbered_thumbnail(self):
        first = self.make(Wire(gemini_reply(png())))
        second = self.make(Wire(gemini_reply(png())))
        self.assertEqual(os.path.basename(first["image_path"]), "thumbnail_0001.png")
        self.assertEqual(os.path.basename(second["image_path"]), "thumbnail_0002.png")
        self.assertEqual(os.path.dirname(first["image_path"]), os.path.join(self.project, "thumbnail"))
        self.assertEqual((first["width"], first["height"], first["provider"]), (640, 360, "gemini"))

    def test_it_shares_the_numbering_with_the_other_engines(self):
        thumbnail._save_thumbnail_image({"project_folder": self.project, "source_path": self.source})
        made = self.make(Wire(gemini_reply(png())))
        self.assertEqual(os.path.basename(made["image_path"]), "thumbnail_0002.png")

    def test_no_model_means_the_providers_first_one_and_a_chosen_one_is_used(self):
        wire = Wire(gemini_reply(png()), gemini_reply(png()))
        first = self.make(wire)
        second = self.make(wire, model="my-own-model.2")
        self.assertEqual(first["model"], api_image.default_api_image_model("gemini"))
        self.assertEqual(second["model"], "my-own-model.2")
        self.assertIn("/models/my-own-model.2:generateContent", wire.requests[1].full_url)

    def test_a_bad_request_is_refused_before_any_call(self):
        wire = Wire()
        for extra, message in (
            ({"project_folder": ""}, "Project folder is empty"),
            ({"source_path": "  "}, "Choose the frame first"),
            ({"provider": "anthropic"}, "cannot draw pictures"),
            ({"model": "bad model/../x"}, "not a valid image model name"),
            ({"api_key": ""}, "Add your Gemini API key"),
            ({"prompt": ""}, "Write the prompt first"),
        ):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self.make(wire, **extra)
        with self.assertRaisesRegex(FileNotFoundError, "The frame was not found"):
            self.make(wire, source_path=os.path.join(self.project, "nope.png"))
        self.assertEqual(wire.requests, [])

    def test_a_provider_error_is_raised_with_its_text_and_leaves_no_file_behind(self):
        with self.assertRaisesRegex(RuntimeError, "Gemini HTTP 429"):
            self.make(Wire((429, "quota")))
        folder = os.path.join(self.project, "thumbnail")
        self.assertFalse(os.path.isdir(folder) and os.listdir(folder))

    def test_the_build_says_it_can_use_an_image_api(self):
        capabilities = thumbnail._thumbnail_capabilities()
        self.assertEqual(set(capabilities["api_image"]["providers"]), {"gemini", "grok", "openai"})
        self.assertTrue(capabilities["overlay"])


if __name__ == "__main__":
    unittest.main()
