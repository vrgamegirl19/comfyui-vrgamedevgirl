"""Native video downloads advertise MP4 names without breaking range playback."""

import ast
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote

from aiohttp import web
from aiohttp.helpers import content_disposition_header
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]


class VideoPreviewDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        tree = ast.parse((ROOT / "builder/video_editor.py").read_text(encoding="utf-8"))
        handler = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.AsyncFunctionDef) and node.name == "vrgdg_video_editor_video")
        handler.decorator_list = []
        scope = {"asyncio": asyncio, "os": os, "web": web,
                 "content_disposition_header": content_disposition_header}
        module = ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[]))
        exec(compile(module, "builder/video_editor.py", "exec"), scope)
        app = web.Application()
        app.router.add_get("/video", scope[handler.name])
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    def clip(self, filename):
        path = Path(self.temporary.name) / filename
        data = b"test-video-bytes-for-range-playback"
        path.write_bytes(data)
        return path, data

    async def test_mp4_filename_mime_and_range_playback(self):
        path, data = self.clip("scene 001.mp4")
        response = await self.client.get("/video", params={"path": str(path)})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["Content-Type"], "video/mp4")
        self.assertTrue(response.headers["Content-Disposition"].startswith("inline;"))
        self.assertIn('filename="scene 001.mp4"', unquote(response.headers["Content-Disposition"]))
        self.assertIn("immutable", response.headers["Cache-Control"])
        self.assertEqual(await response.read(), data)
        ranged = await self.client.get("/video", params={"path": str(path)}, headers={"Range": "bytes=2-5"})
        self.assertEqual(ranged.status, 206)
        self.assertEqual(ranged.headers["Content-Type"], "video/mp4")
        self.assertEqual(await ranged.read(), data[2:6])

    async def test_m4v_container_downloads_as_mp4_with_safe_unicode_basename(self):
        path, _ = self.clip('scène one.m4v')
        response = await self.client.get("/video", params={"path": str(path)})
        self.assertEqual(response.headers["Content-Type"], "video/mp4")
        disposition = unquote(response.headers["Content-Disposition"])
        self.assertIn('scène one.mp4', disposition)
        self.assertNotIn(self.temporary.name, disposition)
        await response.read()

    async def test_other_container_keeps_its_own_extension(self):
        path, _ = self.clip("scene.webm")
        response = await self.client.get("/video", params={"path": str(path)})
        self.assertIn('filename="scene.webm"', response.headers["Content-Disposition"])
        await response.read()

    async def test_missing_video_keeps_existing_404_response(self):
        response = await self.client.get("/video", params={"path": str(Path(self.temporary.name) / "missing.mp4")})
        self.assertEqual(response.status, 404)
        self.assertFalse((await response.json())["ok"])


if __name__ == "__main__":
    unittest.main()
