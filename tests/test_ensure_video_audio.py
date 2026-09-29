import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "Workflows" / "VideoUpscaleExperimental"
ORIGINAL_WORKFLOW = WORKFLOW_DIR / "VRGDG_MiniMaxH3_Upscaler.json"
V2_WORKFLOW = WORKFLOW_DIR / "VRGDG_MiniMaxH3_Upscaler_V2.json"


class EnsureVideoAudioWorkflowTests(unittest.TestCase):
    def test_v2_workflow_routes_all_audio_through_fallback_node(self):
        workflow = json.loads(V2_WORKFLOW.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        fallback = next(node for node in nodes.values() if node["type"] == "VRGDGEnsureVideoAudio")
        loader = next(node for node in nodes.values() if node["type"] == "VHS_LoadVideo")

        self.assertEqual(loader["outputs"][2]["links"], [55])
        self.assertEqual(set(fallback["outputs"][0]["links"]), {31, 39})
        self.assertEqual(workflow["last_node_id"], max(nodes))
        self.assertEqual(workflow["last_link_id"], max(link[0] for link in workflow["links"]))

        seen = set()
        for link_id, source_id, source_slot, target_id, target_slot, type_name in workflow["links"]:
            self.assertNotIn(link_id, seen)
            seen.add(link_id)
            source = nodes[source_id]["outputs"][source_slot]
            target = nodes[target_id]["inputs"][target_slot]
            self.assertEqual(source["type"], type_name)
            self.assertEqual(target["type"], type_name)
            self.assertIn(link_id, source["links"])
            self.assertEqual(target["link"], link_id)

    def test_original_workflow_remains_without_fallback_node(self):
        original = json.loads(ORIGINAL_WORKFLOW.read_text(encoding="utf-8"))
        self.assertNotIn("VRGDGEnsureVideoAudio", {node["type"] for node in original["nodes"]})


if __name__ == "__main__":
    unittest.main()
