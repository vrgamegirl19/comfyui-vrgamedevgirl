"""CPU checks of the chunked second pass node, with the sampler and upscaler stubbed (nothing touches the GPU).

Needs ComfyUI's own Python and modules, so ``test_minimax_chunked_pass2_runtime.py`` runs this file in a subprocess:

    ..\\python_embeded\\python.exe custom_nodes\\comfyui-vrgamedevgirl\\tests\\chunked_pass2_runtime_checks.py

The checks cover window coverage, joins, the locked predecessor head, the upscale modes and the report. They cannot
show visual quality or real VRAM use.
"""

import importlib
import inspect
import os
import sys
import types
import unittest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PACKAGE_ROOT))
sys.path.insert(0, COMFY_ROOT)
sys.argv = sys.argv[:1] + ["--cpu"]  # ComfyUI parses the command line when it is imported

try:
    import torch

    package = types.ModuleType("vrgdg_pkg")  # skip the pack's __init__, which wants a running server
    package.__path__ = [PACKAGE_ROOT]
    sys.modules["vrgdg_pkg"] = package
    core = importlib.import_module("vrgdg_pkg.minimax.chunked_pass2")
    plan_mod = importlib.import_module("vrgdg_pkg.minimax.chunk_plan")
except Exception as exc:  # not running under ComfyUI's Python
    print(f"chunked_pass2 runtime checks need ComfyUI's Python and modules: {type(exc).__name__}: {exc}")
    sys.exit(77)

NestedTensor = core.NestedTensor
TOTAL = 107  # 362 frames, the longest H3 clip


class StubUpscaler:
    """Doubles height and width by repeating, and records every call."""

    def __init__(self):
        self.calls = []

    def upscale(self, latent, upscale_model, scale):
        self.calls.append(tuple(latent["samples"].shape))
        return ({"samples": latent["samples"].repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)},)


class IdentityUpscaler(StubUpscaler):
    def upscale(self, latent, upscale_model, scale):
        self.calls.append(tuple(latent["samples"].shape))
        return ({"samples": latent["samples"].clone()},)


class StubSampler:
    """SamplerCustomAdvanced stand-in: adds the call number (1, 2, ...) to every unlocked video token."""

    windows = []

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, piece):
        video, audio = piece["samples"].unbind()
        mask = piece["noise_mask"].unbind()[0]
        cls.windows.append({"video": video.clone(), "mask": mask.clone(), "seed": noise.seed})
        return ({"samples": NestedTensor((video + mask * len(cls.windows), audio))},)


class StubGuider:
    def __init__(self, model):
        self.model = model

    def set_conds(self, conds):
        self.conds = conds


class StubNoise:
    def __init__(self, seed):
        self.seed = seed


def make_latent(tokens=TOTAL, height=4, width=6):
    video = torch.arange(tokens, dtype=torch.float32).view(1, 1, tokens, 1, 1).repeat(1, 24, 1, height, width)
    audio = torch.zeros(1, 32, 2, round(plan_mod.frame_at(tokens) * 40 / 24))
    return {"samples": NestedTensor((video, audio))}


class RunChunksTests(unittest.TestCase):
    def setUp(self):
        StubSampler.windows = []
        names = ("SamplerCustomAdvanced", "Guider_Basic", "Noise_RandomNoise")
        saved = {name: getattr(core, name) for name in names}
        core.SamplerCustomAdvanced, core.Guider_Basic, core.Noise_RandomNoise = StubSampler, StubGuider, StubNoise
        self.addCleanup(lambda: [setattr(core, name, value) for name, value in saved.items()])
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])

    def run_chunks(self, latent=None, upscaler=None, scale=2.0, **kwargs):
        upscaler = upscaler or StubUpscaler()
        latent = latent or make_latent()
        args = dict(chunk_seconds=3.0, overlap_seconds=1.6, overlap_mode="locked", seed=7, tail_trim_seconds=0.7)
        args.update(kwargs)
        result, report = core.run_chunks(
            latent, object(), [[torch.zeros(1, 4), {}]], {}, upscaler, object(), self.sigmas, scale,
            args.pop("chunk_seconds"), args.pop("overlap_seconds"), args.pop("overlap_mode"), args.pop("seed"), **args)
        return latent, result, report, upscaler

    @staticmethod
    def values(result):
        return result["samples"].unbind()[0][0, 0, :, 0, 0]

    @staticmethod
    def owners(plan):
        owner, covered = {}, 0
        for index, (_, _, keep_end) in enumerate(plan):
            for token in range(covered, keep_end):
                owner[token] = index
            covered = keep_end
        return owner

    # --- upscale modes, audio, seeds ---------------------------------------------------------------------------

    def test_per_window_is_the_default_and_never_upscales_the_whole_clip(self):
        self.assertEqual(inspect.signature(core.run_chunks).parameters["upscale_mode"].default, "per_window")
        _, _, _, upscaler = self.run_chunks()
        self.assertEqual(len(upscaler.calls), len(plan_mod.window_plan(TOTAL, 3.0, 1.6, 0.7)))
        self.assertTrue(all(call[2] < TOTAL for call in upscaler.calls))

    def test_whole_clip_upscales_once_and_cuts_windows(self):
        _, result, _, upscaler = self.run_chunks(upscale_mode="whole_clip")
        self.assertEqual(len(upscaler.calls), 1)
        self.assertEqual(tuple(result["samples"].unbind()[0].shape), (1, 24, TOTAL, 8, 12))

    def test_unknown_modes_are_refused(self):
        with self.assertRaises(ValueError):
            self.run_chunks(upscale_mode="sometimes")
        with self.assertRaises(ValueError):
            self.run_chunks(overlap_mode="blend")

    def test_audio_is_preserved_exactly(self):
        latent, result, _, _ = self.run_chunks()
        self.assertTrue(torch.equal(result["samples"].unbind()[1], latent["samples"].unbind()[1]))

    def test_seed_follows_the_window_start(self):
        self.run_chunks()
        plan = plan_mod.window_plan(TOTAL, 3.0, 1.6, 0.7)
        self.assertEqual([w["seed"] for w in StubSampler.windows], [7 + start for start, _, _ in plan])

    # --- token budget ------------------------------------------------------------------------------------------

    def test_small_budget_uses_the_minimum_window_and_reports_it(self):
        _, _, report, _ = self.run_chunks(max_window_tokens=100)
        self.assertEqual(StubSampler.windows[0]["video"].shape[2], plan_mod.MIN_WINDOW_TOKENS)
        self.assertIn("below the smallest supported window", report)

    def test_budget_counts_padded_odd_latents(self):
        self.run_chunks(latent=make_latent(height=69, width=121), upscaler=IdentityUpscaler(), scale=1.0, max_window_tokens=86000)
        self.assertEqual(StubSampler.windows[0]["video"].shape[2], 37)  # 2135 tokens/frame, not 2040

    # --- tail trim and joins -----------------------------------------------------------------------------------

    def test_locked_trim_places_each_token_from_its_owning_window(self):
        plan = plan_mod.window_plan(TOTAL, 3.0, 1.6, 0.7)
        latent, result, _, _ = self.run_chunks(overlap_mode="locked")
        original = latent["samples"].unbind()[0][0, 0, :, 0, 0]
        owner = self.owners(plan)
        for token in range(TOTAL):
            self.assertEqual(float(self.values(result)[token] - original[token]), owner[token] + 1, f"token {token}")

    def test_crossfade_blends_only_the_overlap_with_history(self):
        plan = plan_mod.window_plan(TOTAL, 3.0, 1.6, 0.7)
        latent, result, _, _ = self.run_chunks(overlap_mode="crossfade")
        original = latent["samples"].unbind()[0][0, 0, :, 0, 0]
        owner = self.owners(plan)
        blended = {}
        for j in range(len(plan) - 1):
            for token in range(plan[j + 1][0], plan[j][2]):
                blended[token] = j
        self.assertTrue(blended)
        for token in range(TOTAL):
            added = float(self.values(result)[token] - original[token])
            if token in blended:
                self.assertTrue(blended[token] + 1 < added < blended[token] + 2, f"token {token}")
            else:
                self.assertEqual(added, owner[token] + 1, f"token {token}")

    def test_both_modes_cover_the_clip_with_and_without_trim(self):
        for mode in ("locked", "crossfade"):
            for trim in (0.0, 0.7):
                _, result, _, _ = self.run_chunks(overlap_mode=mode, tail_trim_seconds=trim)
                self.assertEqual(result["samples"].unbind()[0].shape[2], TOTAL, f"{mode} trim {trim}")

    # --- locked predecessor head -------------------------------------------------------------------------------

    @staticmethod
    def head(tokens=7, value=-5.0, height=8, width=12):
        return lambda h, w: torch.full((1, 24, tokens, h, w), value)

    def test_the_head_is_locked_in_the_first_window_and_kept_exactly(self):
        for mode in ("locked", "crossfade"):
            with self.subTest(mode=mode):
                StubSampler.windows = []
                _, result, _, _ = self.run_chunks(overlap_mode=mode, head_for_size=self.head())
                self.assertTrue(torch.all(StubSampler.windows[0]["mask"][:, :, :7] == 0))
                self.assertTrue(torch.all(StubSampler.windows[0]["mask"][:, :, 7:] == 1))
                self.assertTrue(torch.all(StubSampler.windows[0]["video"][:, :, :7] == -5.0))
                self.assertTrue(torch.all(result["samples"].unbind()[0][:, :, :7] == -5.0), "the head must come out untouched")
                self.assertEqual(result["samples"].unbind()[0].shape[2], TOTAL)

    def test_the_head_is_asked_for_at_the_upscaled_size(self):
        sizes = []

        def head(height, width):
            sizes.append((height, width))
            return torch.zeros(1, 24, 7, height, width)

        self.run_chunks(head_for_size=head)
        self.assertEqual(sizes, [(8, 12)])

    def test_a_head_that_overlaps_the_second_window_is_not_blended_in_crossfade(self):
        # window 1 starts at token 10; a 12-token head reaches into it, and those tokens must stay exact
        _, result, _, _ = self.run_chunks(overlap_mode="crossfade", head_for_size=self.head(tokens=12, value=-3.0))
        self.assertTrue(torch.all(result["samples"].unbind()[0][:, :, :12] == -3.0))

    def test_a_head_too_long_for_the_first_window_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_chunks(head_for_size=self.head(tokens=20))

    def test_without_a_head_nothing_is_locked_in_the_first_window(self):
        self.run_chunks()
        self.assertTrue(torch.all(StubSampler.windows[0]["mask"] == 1))

    def test_windows_after_the_first_lock_their_history(self):
        self.run_chunks()
        plan = plan_mod.window_plan(TOTAL, 3.0, 1.6, 0.7)
        for window, (start, _, _), (_, _, previous_keep) in zip(StubSampler.windows[1:], plan[1:], plan):
            locked = previous_keep - start
            self.assertTrue(torch.all(window["mask"][:, :, :locked] == 0))
            self.assertTrue(torch.all(window["mask"][:, :, locked:] == 1))

    # --- report ------------------------------------------------------------------------------------------------

    def test_report_separates_upscale_and_sampling_peaks_per_window(self):
        _, _, report, _ = self.run_chunks()
        lines = report.splitlines()
        self.assertTrue(lines[0].startswith("Plan: "))
        window_lines = [line for line in lines if "peak VRAM" in line]
        self.assertEqual(len(window_lines), len(plan_mod.window_plan(TOTAL, 3.0, 1.6, 0.7)))
        for line in window_lines:
            self.assertIn("PyTorch allocated", line)
            self.assertIn("upscale ", line)
            self.assertIn("sampling ", line)


class JoinTests(unittest.TestCase):
    def test_crossfade_skips_the_protected_prefix(self):
        assembled = torch.zeros(1, 1, 10, 1, 1)
        assembled[:, :, :4] = 9.0
        chunk = torch.ones(1, 1, 8, 1, 1)
        covered = core.join_chunk(assembled, chunk, start=2, covered=6, mode="crossfade", protect=4)
        values = assembled[0, 0, :, 0, 0].tolist()
        self.assertEqual(values[:4], [9.0] * 4)  # protected
        self.assertAlmostEqual(values[4], 1 / 3, places=5)  # blended with rising weight
        self.assertAlmostEqual(values[5], 2 / 3, places=5)
        self.assertEqual(covered, 10)

    def test_locked_join_never_overwrites_history(self):
        assembled = torch.full((1, 1, 10, 1, 1), 5.0)
        chunk = torch.ones(1, 1, 8, 1, 1)
        core.join_chunk(assembled, chunk, start=2, covered=6, mode="locked")
        self.assertEqual(assembled[0, 0, :6, 0, 0].tolist(), [5.0] * 6)
        self.assertEqual(assembled[0, 0, 6:, 0, 0].tolist(), [1.0] * 4)


class NodeTests(unittest.TestCase):
    def test_node_defaults(self):
        node = core.NODE_CLASS_MAPPINGS["VRGDG_MiniMaxH3ChunkedPass2"]
        spec = node.INPUT_TYPES()
        self.assertEqual(spec["required"]["chunk_seconds"][1]["default"], 3.0)
        self.assertEqual(spec["required"]["overlap_seconds"][1]["default"], 1.6)
        self.assertEqual(spec["optional"]["upscale"][0][0], "per_window")
        self.assertEqual(spec["optional"]["tail_trim_seconds"][1]["default"], 0.7)
        self.assertIn("head_latent", spec["optional"])
        self.assertIn("head_vae", spec["optional"])
        self.assertEqual(inspect.signature(node.sample).parameters["upscale"].default, "per_window")
        self.assertEqual(node.RETURN_TYPES, ("LATENT", "STRING"))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
