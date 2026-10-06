"""PUT /references/scene-mapping turns on the matching "use reference" switches, like the Builder."""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

pkg_name = ROOT.name
errors = importlib.import_module(f"{pkg_name}.agent_api.errors")
mutations = importlib.import_module(f"{pkg_name}.agent_api.mutations")
ref_orch = importlib.import_module(f"{pkg_name}.agent_api.orchestrator.reference_orchestrator")
from test_agent_api_references_llm import Base  # noqa: E402


class SceneMappingSwitchTests(Base):
    def setUp(self):
        super().setUp()
        self.add_locations(2)
        refs = self.read_session()["flux_reference_builder"]
        self.assertNotIn("use_subject_reference", refs)
        self.assertNotIn("use_location_references", refs)

    def refs(self):
        return self.read_session()["flux_reference_builder"]

    def test_saving_a_mapping_turns_both_switches_on(self):
        res = mutations.update_scene_reference_mapping("Song", {
            "subjects": {"seg_0": ["darrel"], "seg_1": ["darrel"]},
            "locations": {"seg_0": "loc1", "seg_1": "loc2"},
        }, if_match_revision=1)
        refs = self.refs()
        self.assertEqual(refs["subject_scene_map"], {"seg_0": ["darrel"], "seg_1": ["darrel"]})
        self.assertEqual(refs["scene_map"], {"seg_0": "loc1", "seg_1": "loc2"})
        self.assertIs(refs["use_subject_reference"], True)
        self.assertIs(refs["use_location_references"], True)
        self.assertEqual(res["scene_mapping"]["locations"]["seg_1"], "loc2")

    def test_only_the_switch_for_the_mapped_kind_changes(self):
        mutations.update_scene_reference_mapping("Song", {"locations": {"seg_0": "loc1"}})
        refs = self.refs()
        self.assertIs(refs["use_location_references"], True)
        self.assertNotIn("use_subject_reference", refs)

    def test_switches_match_assign_scenes_unchanged(self):
        mutations.update_scene_reference_mapping("Song", {
            "subjects": {"seg_0": ["darrel"]}, "locations": {"seg_0": "loc1"},
        })
        after_put = self.refs()
        ref_orch.assign_scenes("Song", {"character_pattern": "unchanged", "location_pattern": "unchanged", "scope": "all"})
        after_assign = self.refs()
        for key in ("use_subject_reference", "use_location_references", "subject_scene_map", "scene_map"):
            self.assertEqual(after_put[key], after_assign[key], key)

    def test_a_stale_revision_changes_nothing(self):
        with self.assertRaises(errors.RevisionConflictError):
            mutations.update_scene_reference_mapping("Song", {"locations": {"seg_0": "loc1"}}, if_match_revision=99)
        self.assertNotIn("use_location_references", self.refs())


if __name__ == "__main__":
    unittest.main()
