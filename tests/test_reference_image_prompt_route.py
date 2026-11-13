"""The reference card prompt route has an engine-neutral name, and the old name still works."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_PATH = "/vrgdg/music_builder/flux_reference_zimage_prompt"
NEW_PATH = "/vrgdg/music_builder/reference_image_prompt"


class ReferenceImagePromptRouteTests(unittest.TestCase):
    def test_both_paths_are_served_by_one_handler(self) -> None:
        source = (ROOT / "builder" / "routes.py").read_text(encoding="utf-8")
        match = re.search(
            r'@server_instance\.routes\.post\("' + re.escape(NEW_PATH) + r'"\)\s*\n\s*'
            r'@server_instance\.routes\.post\("' + re.escape(OLD_PATH) + r'"\)\s*\n\s*async def (\w+)\(request\):',
            source,
        )
        self.assertIsNotNone(match, "both paths are stacked on one handler")
        self.assertIn("_generate_reference_image_prompt", source[match.end():match.end() + 400])

    def test_the_old_function_name_is_an_alias_of_the_new_one(self) -> None:
        source = (ROOT / "llm" / "image_prompt_generation.py").read_text(encoding="utf-8")
        self.assertIn("def _generate_reference_image_prompt(payload):", source)
        self.assertIn("_generate_flux_reference_zimage_prompt = _generate_reference_image_prompt", source)

    def test_the_builder_calls_the_new_path_only(self) -> None:
        callers = [path for path in (ROOT / "web").rglob("*.mjs") if OLD_PATH in path.read_text(encoding="utf-8")]
        self.assertEqual(callers, [])
        used = [path.name for path in (ROOT / "web" / "music_video_builder").glob("*.mjs") if NEW_PATH in path.read_text(encoding="utf-8")]
        self.assertEqual(sorted(used), ["gemma_runner.mjs", "reference_generation.mjs"])


if __name__ == "__main__":
    unittest.main()
