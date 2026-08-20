import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(encoding="utf-8")


class BuilderSelectedCastCoverageTests(unittest.TestCase):
    def test_shared_contract_requires_every_selected_subject(self):
        self.assertIn("function selectedCastCoverageContract(segment, options = {})", BUILDER)
        self.assertIn("Every selected subject must appear visibly at least once.", BUILDER)
        self.assertIn("Performer/singer assignment controls only who sings or lip-syncs", BUILDER)
        self.assertIn("VISIBLE BAND PERFORMANCE — MANDATORY", BUILDER)
        self.assertIn("actively and believably playing their assigned instrument", BUILDER)
        self.assertIn("Do not invent an instrument for a subject whose reference does not assign one", BUILDER)

    def test_minimax_rotates_cast_only_without_an_assigned_performer(self):
        self.assertIn("labelMap: miniMaxH3SubjectLabelMapForSegment(segment, mode)", BUILDER)
        self.assertIn("every cut must change the featured band member", BUILDER)
        self.assertIn("an assigned performer/singer remains the featured primary subject on every lip-sync shot", BUILDER)
        self.assertIn("const featuredLabels = performerLabels.length ? performerLabels : labels;", BUILDER)
        self.assertIn("Use every selected subject label required by the selected-cast coverage", BUILDER)

    def test_ltx25_receives_contract_without_changing_legacy_ltx23(self):
        self.assertIn("function ltx25SelectedCastCoverageContract(segment)", BUILDER)
        self.assertIn('if (String(i2vVideoSettingsForSegment(segment)?.ltx_version || "2.5") === "2.3") return "";', BUILDER)
        self.assertIn("const selectedCastContract = ltx25SelectedCastCoverageContract(segment);", BUILDER)
        self.assertIn("if the prompt contains multiple shots or cuts, each new shot must feature a different selected member", BUILDER)
        self.assertIn("keep an assigned performer as the featured subject after every cut", BUILDER)


if __name__ == "__main__":
    unittest.main()
