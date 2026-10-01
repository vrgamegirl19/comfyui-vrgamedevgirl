"""Api_Endpoints.md and agent_api/endpoints.json list every route and are up to date.

Regenerate both with scripts/export_api_endpoints.py after changing a route.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("export_api_endpoints", ROOT / "scripts" / "export_api_endpoints.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


class ApiEndpointsDocTests(unittest.TestCase):
    def test_every_route_is_described_and_the_files_are_current(self):
        document = module.build_document()  # exits with the missing routes if one has no description
        doc_path = ROOT / "Api_Endpoints.md"
        if doc_path.is_file():  # local-only (git-ignored), so a fresh clone does not have it
            self.assertEqual(_read(doc_path), document, "Api_Endpoints.md is stale. Run scripts/export_api_endpoints.py.")
        self.assertEqual(_read(ROOT / "agent_api" / "endpoints.json"), module.build_records(),
                         "agent_api/endpoints.json is stale. Run scripts/export_api_endpoints.py.")


if __name__ == "__main__":
    unittest.main()
