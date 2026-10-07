"""Tests for the cache rule that keeps the browser from serving stale Video Builder modules."""

import asyncio
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

static_cache = importlib.import_module(f"{ROOT.name}.core.static_cache")


def _call(path):
    async def handler(_request):
        return web.Response(text="x")

    return asyncio.run(static_cache.revalidate_builder_modules(SimpleNamespace(path=path), handler))


class StaticCacheTests(unittest.TestCase):
    def test_this_packs_modules_must_revalidate(self):
        path = f"/extensions/{ROOT.name}/music_video_builder/builder.mjs"
        self.assertEqual(_call(path).headers["Cache-Control"], "no-cache")

    def test_other_files_and_other_packs_are_left_alone(self):
        for path in (
            f"/extensions/{ROOT.name}/music_video_builder/note.txt",
            f"/extensions/{ROOT.name}/VRGDG_MusicVideoBuilderUI.js",
            "/extensions/some-other-pack/module.mjs",
            "/",
            "/api/prompt",
        ):
            with self.subTest(path=path):
                self.assertNotIn("Cache-Control", _call(path).headers)


if __name__ == "__main__":
    unittest.main()
