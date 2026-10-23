"""GET /references/scene-mapping?include=scenes (or ?scene_id=) lists each scene's mapped cards with their sources.

Each card shows its reference_type, its RefMod source (name, found on disk, kind, tokens, strength) when the card
uses a RefMod, and its picture (path, exists). The default response (the four id maps) is unchanged.
"""

import asyncio
import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

mutations = importlib.import_module(f"{ROOT.name}.agent_api.mutations")
errors = importlib.import_module(f"{ROOT.name}.agent_api.errors")
router = importlib.import_module(f"{ROOT.name}.agent_api.router")
library = importlib.import_module(f"{ROOT.name}.minimax.refmod_library")
from test_agent_api_references_llm import Base  # noqa: E402

API = "/vrgdg/api/v1"
LIBRARY = {
    "identity/Ayame_Realistic": {"name": "identity/Ayame_Realistic", "kind": "video", "tokens": 1458},
    "background/corridor": {"name": "background/corridor", "kind": "image", "tokens": 402},
}


class _Routes:
    def __init__(self):
        self.handlers = {}

    def __getattr__(self, method):
        def register(path):
            def decorator(handler):
                self.handlers[(method.upper(), path)] = handler
                return handler
            return decorator
        return register


class _Request:
    def __init__(self, query):
        self.match_info = {"pid": "Song"}
        self.query = query
        self.headers = {}
        self.can_read_body = False
        self.remote = "127.0.0.1"
        self.method = "GET"
        self.path = "/x"


def _handler():
    server = MagicMock()
    server.routes = _Routes()
    with patch.object(router, "_VRGDG_AGENT_API_ROUTES_REGISTERED", False):
        router.register_agent_api_routes(server)
    return server.routes.handlers[("GET", f"{API}/projects/{{pid}}/references/scene-mapping")]


def _call(query):
    import json
    response = asyncio.run(_handler()(_Request(query)))
    return response.status, json.loads(response.text)


class SceneMappingRefmodReadTests(Base):
    def setUp(self):
        super().setUp()
        session = self.read_session()
        refmod = lambda name, kind: {"name": name, "kind": kind, "tokens": 1, "strength": 1.0}  # noqa: E731
        session["flux_reference_builder"] = {
            "use_subject_reference": True, "use_location_references": True, "subject_count": 2,
            "subjects": [
                {"id": "ayame", "name": "Ayame", "reference_type": "character", "source": "refmod",
                 "image": {"path": self.image}, "refmod": refmod("identity/Ayame_Realistic", "video")},
                {"id": "glock", "name": "Glock 17", "reference_type": "prop", "source": "refmod",
                 "image": {"path": ""}, "refmod": refmod("prop/Glock17", "video")},
                {"id": "darrel", "name": "Darrel", "reference_type": "character", "image": {"path": self.image}},
            ],
            "locations": [{"id": "corridor", "name": "Corridor", "reference_type": "environment", "source": "refmod",
                           "image": {"path": "/nope/corridor.png"}, "refmod": refmod("background/corridor", "image")}],
            "subject_scene_map": {"seg_2": "ayame,glock", "seg_3": "darrel"},
            "scene_map": {"seg_2": "corridor"},
        }
        session["minimax_h3_settings"] = {"pipeline": "refmod"}
        self.write_session(session)
        p = patch.object(library, "find_refmod", lambda n: dict(LIBRARY[n]) if n in LIBRARY else None)
        p.start()
        self.addCleanup(p.stop)

    def test_default_response_is_unchanged(self):
        status, body = _call({})
        self.assertEqual(status, 200)
        self.assertEqual(set(body["data"]), {"subjects", "locations", "ingredients", "extras"})
        self.assertEqual(body["data"]["subjects"]["seg_2"], "ayame,glock")

    def test_include_scenes_lists_every_scene_with_its_cards_and_sources(self):
        status, body = _call({"include": "scenes"})
        self.assertEqual(status, 200)
        data = body["data"]
        self.assertEqual(data["pipeline"], "refmod")
        self.assertEqual(len(data["scenes"]), 10)
        s3 = data["scenes"][2]
        self.assertEqual((s3["scene_id"], s3["scene_number"]), ("seg_2", 3))
        cards = {c["card_id"]: c for c in s3["cards"]}
        self.assertEqual([c["card_id"] for c in s3["cards"]], ["ayame", "glock", "corridor"])
        self.assertEqual(cards["ayame"]["map"], "subjects")
        self.assertEqual(cards["glock"]["reference_type"], "prop")
        self.assertEqual(cards["ayame"]["source"], "refmod")
        self.assertEqual(cards["ayame"]["refmod"], {"name": "identity/Ayame_Realistic", "found": True, "kind": "video",
                                                    "tokens": 1458, "strength": 1.0})
        self.assertEqual(cards["glock"]["refmod"]["found"], False)
        self.assertEqual(cards["glock"]["refmod"]["kind"], "video")  # card's saved kind when the file is missing
        self.assertEqual(cards["corridor"]["refmod"]["kind"], "image")
        self.assertEqual(cards["ayame"]["picture"], {"path": self.image, "exists": True})
        self.assertIsNone(cards["glock"]["picture"])
        self.assertEqual(cards["corridor"]["picture"], {"path": "/nope/corridor.png", "exists": False})
        s4 = data["scenes"][3]
        self.assertEqual(s4["cards"][0]["source"], "image")
        self.assertIsNone(s4["cards"][0]["refmod"])
        self.assertEqual(data["scenes"][0]["cards"], [])

    def test_scene_id_filters_to_one_scene(self):
        for sid in ("seg_2", "3"):
            status, body = _call({"scene_id": sid})
            self.assertEqual(status, 200)
            self.assertEqual([s["scene_id"] for s in body["data"]["scenes"]], ["seg_2"])
        status, body = _call({"scene_id": "nope"})
        self.assertEqual(status, 404)

    def test_bad_include_is_400(self):
        status, _body = _call({"include": "bogus"})
        self.assertEqual(status, 400)

    def test_read_does_not_write(self):
        before = os.path.getmtime(self.session_file)
        _call({"include": "scenes"})
        self.assertEqual(os.path.getmtime(self.session_file), before)


if __name__ == "__main__":
    unittest.main()
