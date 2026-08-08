import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = (ROOT / "web" / "VRGDG_MusicVideoBuilderUI.js").read_text(
    encoding="utf-8"
)


class BuilderMiniMaxSamplerDropdownTests(unittest.TestCase):
    def test_sampler_and_scheduler_stay_interactive_in_turbo_mode(self):
        self.assertIn("miniMaxSamplerName.disabled = false;", BUILDER_SOURCE)
        self.assertIn("miniMaxScheduler.disabled = false;", BUILDER_SOURCE)
        self.assertNotIn(
            "miniMaxSamplerName.disabled = settings.use_turbo_lora;",
            BUILDER_SOURCE,
        )
        self.assertNotIn(
            "miniMaxScheduler.disabled = settings.use_turbo_lora;",
            BUILDER_SOURCE,
        )

    def test_changing_standard_sampling_turns_off_turbo(self):
        self.assertIn(
            'miniMaxUseTurboLora.input.dispatchEvent(new Event("change"));',
            BUILDER_SOURCE,
        )
        self.assertIn(
            "Turbo was turned off so the selected standard ${label} will be used.",
            BUILDER_SOURCE,
        )

    def test_choices_are_loaded_from_current_comfyui_object_info(self):
        self.assertIn('getJson("/object_info/KSamplerSelect")', BUILDER_SOURCE)
        self.assertIn('getJson("/object_info/BasicScheduler")', BUILDER_SOURCE)
        self.assertIn("input[1]?.options", BUILDER_SOURCE)
        self.assertIn("Array.isArray(input[0])", BUILDER_SOURCE)

    def test_dynamic_choice_failure_keeps_builtin_fallbacks(self):
        self.assertIn(
            "Could not refresh sampler and scheduler choices; using built-in fallbacks:",
            BUILDER_SOURCE,
        )
        self.assertIn('sampler_name: "res_multistep"', BUILDER_SOURCE)
        self.assertIn('scheduler: "simple"', BUILDER_SOURCE)

    def test_selection_keeps_existing_state_persistence_and_render_payload(self):
        self.assertIn("sampler_name: miniMaxSamplerName.value", BUILDER_SOURCE)
        self.assertIn("scheduler: miniMaxScheduler.value", BUILDER_SOURCE)
        self.assertIn("sampler_name: miniMaxSettings.sampler_name", BUILDER_SOURCE)
        self.assertIn("scheduler: miniMaxSettings.scheduler", BUILDER_SOURCE)
        self.assertIn('autoSaveSessionQuiet("MiniMax H3 project settings")', BUILDER_SOURCE)


if __name__ == "__main__":
    unittest.main()
