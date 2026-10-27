"""Choosing the GPU of Gemma Local and Qwen Local: the number rules, the card list, the llama.cpp arguments and the settings that carry it."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
gpus = importlib.import_module(f"{pkg_name}.llm.gpus")
gguf = importlib.import_module(f"{pkg_name}.llm.gguf")
builder_runner = importlib.import_module(f"{pkg_name}.llm.builder_runner")
llm_runtime = importlib.import_module(f"{pkg_name}.agent_api.llm_runtime")
session_keys = importlib.import_module(f"{pkg_name}.agent_api.session_keys")
builder_project = importlib.import_module(f"{pkg_name}.builder.project")

GB = 1024 ** 3
RTX_5090 = {"slot": 0, "name": "NVIDIA GeForce RTX 5090", "uuid": "aaaa", "total_bytes": 32 * GB, "free_bytes": 25 * GB, "pci_bus": "00000000:01:00.0"}
RTX_4090 = {"slot": 1, "name": "NVIDIA GeForce RTX 4090", "uuid": "bbbb", "total_bytes": 24 * GB, "free_bytes": 15 * GB, "pci_bus": "00000000:03:00.0"}


def seen(number, card):
    return {"number": number, "name": card["name"], "uuid": card["uuid"], "total_bytes": card["total_bytes"]}


class GpuNumberTests(unittest.TestCase):
    def test_a_missing_or_odd_choice_is_automatic(self):
        for value in (None, "", "auto", "AUTO", "all", -1, "-3", True, False, "abc", [], {}):
            self.assertEqual(gpus.normalize_gpu_device(value), -1, repr(value))

    def test_a_number_is_one_gpu(self):
        self.assertEqual([gpus.normalize_gpu_device(v) for v in (0, 1, "1", " 2 ", 1.0, "1.0", 500)], [0, 1, 1, 2, 1, 1, gpus.MAX_GPU_NUMBER])

    def test_the_default_applies_only_to_what_is_not_a_choice(self):
        self.assertEqual(gpus.normalize_gpu_device(None, 1), 1)
        self.assertEqual(gpus.normalize_gpu_device("auto", 1), -1, "an explicit automatic stays automatic")

    def test_llama_cpp_is_told_to_keep_the_whole_model_on_one_gpu(self):
        self.assertEqual(gpus.gpu_kwargs(-1), {}, "automatic leaves llama.cpp as it always was")
        self.assertEqual(gpus.gpu_kwargs(None), {})
        self.assertEqual(gpus.gpu_kwargs(1), {"split_mode": 0, "main_gpu": 1}, "split mode none, main GPU 1")
        self.assertEqual(gpus.gpu_kwargs("0"), {"split_mode": 0, "main_gpu": 0})


class GpuListTests(unittest.TestCase):
    def test_a_launcher_that_hides_the_second_card_is_explained(self):
        # run_nvidia_gpu.bat has "set CUDA_VISIBLE_DEVICES=1": ComfyUI sees only the RTX 5090, as its GPU 0.
        listing = gpus.merge_gpus([RTX_5090, RTX_4090], [seen(0, RTX_5090)], comfyui_gpu=0, cuda_visible_devices="1")
        first, second = listing["gpus"]
        self.assertEqual((first["number"], first["visible"], first["comfyui_gpu"]), (0, True, True))
        self.assertEqual(first["label"], "GPU 0: NVIDIA GeForce RTX 5090 (32.0 GB, 25.0 GB free) - ComfyUI renders here")
        self.assertEqual((second["number"], second["visible"]), (None, False))
        self.assertEqual(second["label"], "NVIDIA GeForce RTX 4090 (24.0 GB, 15.0 GB free) - not visible to ComfyUI")
        self.assertEqual(listing["visible_count"], 1)
        self.assertIn('CUDA_VISIBLE_DEVICES is "1"', listing["note"])
        self.assertIn("--default-device 1,", listing["note"])
        self.assertIn("only see 1 of the 2 GPUs", listing["note"])

    def test_both_cards_visible_have_ComfyUIs_numbers_not_the_slot_numbers(self):
        # CUDA puts the RTX 4090 first although nvidia-smi lists the RTX 5090 as 0.
        listing = gpus.merge_gpus([RTX_5090, RTX_4090], [seen(0, RTX_4090), seen(1, RTX_5090)], comfyui_gpu=1)
        self.assertEqual([(g["number"], g["name"][-8:], g["slot"]) for g in listing["gpus"]], [(0, "RTX 4090", 1), (1, "RTX 5090", 0)])
        self.assertEqual([g["comfyui_gpu"] for g in listing["gpus"]], [False, True])
        self.assertEqual(listing["note"], "")

    def test_a_computer_without_the_card_list_still_lists_what_torch_sees(self):
        listing = gpus.merge_gpus([], [seen(0, RTX_4090)], comfyui_gpu=0)
        self.assertEqual([(g["number"], g["free_gb"], g["visible"]) for g in listing["gpus"]], [(0, None, True)])
        self.assertEqual(listing["gpus"][0]["label"], "GPU 0: NVIDIA GeForce RTX 4090 (24.0 GB) - ComfyUI renders here")

    def test_no_gpu_at_all_is_an_empty_list(self):
        listing = gpus.merge_gpus([], [])
        self.assertEqual((listing["gpus"], listing["visible_count"], listing["note"]), ([], 0, ""))

    def test_the_list_is_built_from_the_cards_and_the_process(self):
        with patch.object(gpus, "_torch_gpus", return_value=[seen(0, RTX_5090)]), patch.object(gpus, "_nvml_gpus", return_value=[RTX_5090, RTX_4090]), \
                patch.object(gpus, "_comfyui_gpu_number", return_value=0), patch.dict(gpus.os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            listing = gpus.list_gpus()
        self.assertEqual([g["visible"] for g in listing["gpus"]], [True, False])

    def test_a_number_that_is_not_visible_stops_with_the_list_of_those_that_are(self):
        with patch.object(gpus, "_torch_gpus", return_value=[seen(0, RTX_5090)]):
            gpus.check_gpu_device(-1)
            gpus.check_gpu_device(0)
            with self.assertRaises(ValueError) as caught:
                gpus.check_gpu_device(1)
        self.assertIn("GPU 1 is not one ComfyUI can see", str(caught.exception))
        self.assertIn("GPU 0: NVIDIA GeForce RTX 5090", str(caught.exception))

    def test_without_cuda_the_choice_does_not_matter(self):
        with patch.object(gpus, "_torch_gpus", return_value=[]):
            gpus.check_gpu_device(3)


class FakeLlama:
    calls = []

    def __init__(self, **kwargs):
        FakeLlama.calls.append(kwargs)

    def close(self):
        pass


class LoaderTests(unittest.TestCase):
    def fake_llama_cpp(self):
        module = types.ModuleType("llama_cpp")
        module.Llama = FakeLlama
        chat_format = types.ModuleType("llama_cpp.llama_chat_format")
        chat_format.Llava15ChatHandler = type("Llava15ChatHandler", (), {"__init__": lambda self, **kw: None})
        module.llama_chat_format = chat_format
        return patch.dict(sys.modules, {"llama_cpp": module, "llama_cpp.llama_chat_format": chat_format})

    def load(self, device, model_class=None):
        FakeLlama.calls.clear()
        llm = (model_class or gguf.VRGDG_SuperGemmaGGUFChat)()
        llm._gpu_device = device
        model_path = str(Path(__file__).resolve())  # any existing file: the fake loader never reads it
        with self.fake_llama_cpp(), patch.object(gpus, "_torch_gpus", return_value=[seen(0, RTX_4090), seen(1, RTX_5090)]), \
                patch.dict(gguf._GGUF_MODEL_CACHE, {}, clear=True):
            llm._load_gguf_model(model_path, 2048, 99, 8, "")
        return FakeLlama.calls[-1]

    def test_the_model_is_loaded_on_the_chosen_gpu(self):
        call = self.load(1)
        self.assertEqual((call["split_mode"], call["main_gpu"], call["n_gpu_layers"]), (0, 1, 99))

    def test_automatic_loads_it_the_way_it_always_was(self):
        call = self.load(-1)
        self.assertNotIn("main_gpu", call)
        self.assertNotIn("split_mode", call)

    def test_qwen_loads_on_the_chosen_gpu_too(self):
        call = self.load(0, gguf.VRGDG_QwenGGUF)
        self.assertEqual((call["split_mode"], call["main_gpu"]), (0, 0))

    def test_a_gpu_comfyui_cannot_see_is_refused_before_anything_is_loaded(self):
        with self.assertRaises(ValueError):
            self.load(5)
        self.assertEqual(FakeLlama.calls, [])

    def test_a_model_loaded_on_one_gpu_is_not_reused_for_the_other(self):
        llm = gguf.VRGDG_SuperGemmaGGUFChat()
        keys = []
        for device in (-1, 0, 1):
            llm._gpu_device = device
            keys.append(llm._gguf_model_key("m.gguf", 2048, 99, 8, "", ""))
        self.assertEqual(len(set(keys)), 3)

    def test_the_node_has_an_optional_gpu_input_that_does_not_move_the_other_widgets(self):
        for node in (gguf.VRGDG_SuperGemmaGGUFChat, gguf.VRGDG_QwenGGUF, gguf.VRGDG_GeneralGGUF):
            inputs = node.INPUT_TYPES()
            self.assertNotIn("gpu_device", inputs["required"], node.__name__)
            self.assertEqual(list(inputs["optional"])[-1], "gpu_device", node.__name__)
            self.assertEqual(inputs["optional"]["gpu_device"][1]["default"], -1)

    def test_the_builder_runner_hands_the_choice_to_the_model(self):
        for runner, kind in (("builtin", gguf.VRGDG_SuperGemmaGGUFChat), ("qwen_local", gguf.VRGDG_QwenGGUF)):
            llm = builder_runner._builder_local_llm({"text_runner": runner, "gpu_device": 1})
            self.assertIsInstance(llm, kind)
            self.assertEqual(llm._gpu_device, 1)
        self.assertEqual(builder_runner._builder_local_llm({"text_runner": "builtin"})._gpu_device, -1)

    def test_the_runner_dialog_gets_the_gpu_list_with_the_model_lists(self):
        listing = {"gpus": [{"number": 0, "label": "GPU 0: x"}], "visible_count": 1, "note": ""}
        with patch.object(builder_runner, "list_gpus", return_value=listing), patch.object(gguf.VRGDG_SuperGemmaGGUFChat, "_list_local_gemma_gguf_choices", return_value=["g"]), \
                patch.object(gguf.VRGDG_SuperGemmaGGUFChat, "_list_local_mmproj_choices", return_value=["p"]), \
                patch.object(gguf.VRGDG_QwenGGUF, "_list_local_qwen_gguf", return_value=["q"]), patch.object(gguf.VRGDG_QwenGGUF, "_list_local_qwen_mmproj", return_value=["qp"]):
            choices = builder_runner._gemma_choices()
        self.assertEqual(choices["gpus"], listing)
        self.assertEqual(choices["models"], ["g"])

    def test_a_failure_to_list_the_cards_does_not_break_the_dialog(self):
        with patch.object(builder_runner, "list_gpus", side_effect=RuntimeError("no nvml")):
            self.assertEqual(builder_runner._gpu_choices()["gpus"], [])


class SettingsTests(unittest.TestCase):
    def test_the_choice_is_saved_with_the_other_runner_settings(self):
        self.assertIn("gemma_gpu_device", builder_project.GLOBAL_LLM_RUNNER_FIELDS)
        self.assertIn("gemma_gpu_device", session_keys.BUILDER_SESSION_DATA_KEYS, "the Builder saves it, so the agent API knows the key")

    def test_an_agent_render_uses_the_saved_choice(self):
        with patch.object(llm_runtime, "global_llm_runner", return_value={}):
            self.assertEqual(llm_runtime.llm_payload_from_session({"text_gemma_runner": "builtin", "gemma_gpu_device": 1})["gpu_device"], 1)
            self.assertEqual(llm_runtime.llm_payload_from_session({"text_gemma_runner": "builtin"})["gpu_device"], -1)

    def test_the_agent_settings_show_the_choice(self):
        schemas = importlib.import_module(f"{pkg_name}.agent_api.schemas")
        self.assertEqual(schemas.LlmSettings().gemma_gpu_device, -1)


if __name__ == "__main__":
    unittest.main()
