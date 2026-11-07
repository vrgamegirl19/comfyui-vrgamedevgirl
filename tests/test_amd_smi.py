"""amd-smi: where the program is, how its readings are converted, and how the GPU list and the monitor get them."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

pkg_name = ROOT.name
amd_smi = importlib.import_module(f"{pkg_name}.core.amd_smi")
gpus = importlib.import_module(f"{pkg_name}.llm.gpus")

GiB = 1024 ** 3

STATIC = {
    "gpu_data": [
        {"gpu": 0, "asic": {"market_name": "AMD Radeon AI PRO R9700", "asic_serial": "0x7685A348C85BC8E5"},
         "bus": {"bdf": "0000:03:00.0"}, "vram": {"size": {"value": 32624, "unit": "MB"}},
         "limit": {"ppt0": {"socket_power_limit": {"value": 300, "unit": "W"}, "max_power_limit": {"value": 300, "unit": "W"}}}},
        {"gpu": 1, "asic": {"market_name": "AMD Radeon Graphics", "asic_serial": "N/A"}, "bus": {"bdf": "0000:7b:00.0"},
         "vram": {"size": {"value": 2048, "unit": "MB"}},
         "limit": {"ppt0": {"socket_power_limit": "N/A", "max_power_limit": "N/A"}}},
    ]
}

MONITOR = [
    {"gpu": 0, "power_usage": {"value": 52, "unit": "W"}, "hotspot_temperature": {"value": 56, "unit": "C"},
     "gfx_clk": {"value": 84, "unit": "MHz"}, "gfx": {"value": 5, "unit": "%"},
     "vram_used": {"value": 26.9, "unit": "GB"}, "vram_total": {"value": 31.9, "unit": "GB"}},
    {"gpu": 1, "power_usage": "N/A", "hotspot_temperature": "N/A", "gfx_clk": "N/A", "gfx": "N/A",
     "vram_used": {"value": 0.0, "unit": "GB"}, "vram_total": {"value": 30.2, "unit": "GB"}},
]

R9700 = {"slot": 0, "name": "AMD Radeon AI PRO R9700", "uuid": "37363835-6133-3438-6338-356263386535",
         "total_bytes": 32624 * 1000 * 1000, "free_bytes": 5724000000, "pci_bus": "0000:03:00.0"}
IGPU = {"slot": 1, "name": "AMD Radeon Graphics", "uuid": "", "total_bytes": 2048 * 1000 * 1000, "free_bytes": 2048 * 1000 * 1000,
        "pci_bus": "0000:7b:00.0"}


def fake_amd_smi(static, monitor):
    """``subprocess.check_output`` that answers the two commands with canned output."""
    answers = {"static": json.dumps(static), "monitor": json.dumps(monitor)}

    def check_output(command, **kwargs):
        return answers[command[1]]

    return patch.object(amd_smi.subprocess, "check_output", side_effect=check_output)


class ExecutableTests(unittest.TestCase):
    def test_an_activatable_venv_is_found_without_activating_it(self):
        venv = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin") / ("amd-smi.exe" if os.name == "nt" else "amd-smi")
        with patch.object(amd_smi.shutil, "which", return_value=None), patch.object(amd_smi.os.path, "isfile", return_value=True), \
                patch.object(amd_smi.os, "access", return_value=True):
            found = amd_smi.find_amd_smi()
        self.assertTrue(found and found.endswith(venv.name), found)
        self.assertIn(str(sys.prefix), found, "the environment that is running ComfyUI is searched even when its bin folder is not on PATH")

    def test_the_path_wins_when_the_tool_is_on_it(self):
        with patch.object(amd_smi.shutil, "which", return_value="/usr/bin/amd-smi"):
            self.assertEqual(amd_smi.find_amd_smi(), "/usr/bin/amd-smi")

    def test_a_told_location_is_used_and_a_wrong_one_is_not_replaced(self):
        with patch.dict(amd_smi.os.environ, {"VRGDG_AMD_SMI": "/opt/rocm/bin/amd-smi"}), \
                patch.object(amd_smi.os.path, "isfile", return_value=True), patch.object(amd_smi.os, "access", return_value=True):
            self.assertEqual(amd_smi.find_amd_smi(), "/opt/rocm/bin/amd-smi")
        with patch.dict(amd_smi.os.environ, {"VRGDG_AMD_SMI": "/opt/rocm/bin/amd-smi"}), \
                patch.object(amd_smi.shutil, "which", return_value="/usr/bin/amd-smi"):
            self.assertIsNone(amd_smi.find_amd_smi(), "an explicit setting that does not exist stays broken instead of hiding behind the PATH")

    def test_a_computer_without_the_tool_reports_none(self):
        with patch.object(amd_smi.shutil, "which", return_value=None), patch.object(amd_smi.os.path, "isfile", return_value=False):
            self.assertIsNone(amd_smi.find_amd_smi())

    def test_a_tool_installed_in_another_environment_is_not_used(self):
        """Only the environment that runs ComfyUI is searched. A tool sitting in some other venv on the computer stays invisible,
        which is what keeps the standalone-package and venv setups from reading each other's tools."""
        scripts = "Scripts" if os.name == "nt" else "bin"
        name = "amd-smi.exe" if os.name == "nt" else "amd-smi"
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / ".venv" / scripts
            other.mkdir(parents=True)
            outsider = other / name
            outsider.write_text("#!/bin/sh\n")
            outsider.chmod(0o755)
            running = (Path(tmp) / "comfy-env" / scripts)
            running.mkdir(parents=True)  # the environment we run in, without a tool of its own
            with patch.object(amd_smi.shutil, "which", return_value=None), patch.dict(amd_smi.os.environ, {"HOME": tmp}), \
                    patch.object(amd_smi.sys, "prefix", str(running.parent)), \
                    patch.object(amd_smi.sys, "executable", str(running / "python")):
                self.assertIsNone(amd_smi.find_amd_smi())


class ReadingTests(unittest.TestCase):
    def test_a_sensor_the_card_lacks_is_none_rather_than_a_crash(self):
        self.assertIsNone(amd_smi._clean("N/A"))
        self.assertIsNone(amd_smi._clean(None))
        self.assertIsNone(amd_smi._clean(float("inf")))
        self.assertIsNone(amd_smi._clean(True))
        self.assertEqual(amd_smi._clean(56), 56)

    def test_the_units_amd_smi_chose_become_bytes(self):
        self.assertEqual(amd_smi._bytes({"value": 32624, "unit": "MB"}), 32624 * 1000 ** 2)
        self.assertEqual(amd_smi._bytes({"value": 26.9, "unit": "GB"}), 26900000000)
        self.assertIsNone(amd_smi._bytes("N/A"))
        self.assertIsNone(amd_smi._bytes({"value": 1, "unit": "furlongs"}))

    def test_the_serial_is_written_the_way_torch_writes_the_uuid(self):
        self.assertEqual(amd_smi._torch_uuid("0x7685A348C85BC8E5"), "37363835-6133-3438-6338-356263386535")
        self.assertEqual(amd_smi._torch_uuid("N/A"), "")
        self.assertEqual(amd_smi._torch_uuid(None), "")

    def test_both_commands_are_put_together_by_gpu_number(self):
        with patch.object(amd_smi, "find_amd_smi", return_value="/usr/bin/amd-smi"), fake_amd_smi(STATIC, MONITOR):
            cards = amd_smi.gpus_amd()
        self.assertEqual([card["gpu"] for card in cards], [0, 1])
        r9700 = cards[0]
        self.assertEqual(r9700["name"], "AMD Radeon AI PRO R9700")
        self.assertEqual(r9700["pci_bus"], "0000:03:00.0")
        self.assertEqual(r9700["total"], 32624 * 1000 ** 2)
        self.assertEqual((r9700["used"], r9700["free"]), (26900000000, 32624 * 1000 ** 2 - 26900000000))
        self.assertEqual((r9700["load"], r9700["clock"], r9700["power"], r9700["power_limit"]), (5, 84, 52, 300))
        self.assertEqual(r9700["temperature"], 56)
        self.assertIsNone(r9700["fan"], "this card reports no fan")
        igpu = cards[1]
        self.assertEqual((igpu["load"], igpu["power"], igpu["power_limit"], igpu["temperature"]), (None, None, None, None))
        self.assertEqual(igpu["uuid"], "", "a card with no serial cannot be matched by UUID")

    def test_a_card_only_one_of_the_two_commands_knows_still_appears(self):
        with patch.object(amd_smi, "find_amd_smi", return_value="/usr/bin/amd-smi"), fake_amd_smi(STATIC, []):
            cards = amd_smi.gpus_amd()
        self.assertEqual([card["gpu"] for card in cards], [0, 1])
        self.assertEqual(cards[0]["used"], None)

    def test_a_command_that_fails_leaves_the_other_one_reading(self):
        with patch.object(amd_smi, "find_amd_smi", return_value="/usr/bin/amd-smi"), \
                patch.object(amd_smi.subprocess, "check_output", side_effect=[subprocess.TimeoutExpired("amd-smi", 2), json.dumps(MONITOR)]):
            cards = amd_smi.gpus_amd()
        self.assertEqual([(card["gpu"], card["name"], card["load"]) for card in cards], [(0, "", 5), (1, "", None)])

    def test_no_program_means_no_cards_and_no_attempt(self):
        with patch.object(amd_smi, "find_amd_smi", return_value=None), patch.object(amd_smi.subprocess, "check_output") as run:
            self.assertEqual(amd_smi.gpus_amd(), [])
        run.assert_not_called()


class ResourceMonitorTests(unittest.TestCase):
    def test_the_monitor_rows_keep_the_shape_the_browser_already_understands(self):
        with patch.object(amd_smi, "find_amd_smi", return_value="/usr/bin/amd-smi"), fake_amd_smi(STATIC, MONITOR):
            rows = amd_smi.parse_sensors(amd_smi.gpus_amd())
        self.assertEqual([(row["index"], row["name"]) for row in rows], [("0", "AMD Radeon AI PRO R9700"), ("1", "AMD Radeon Graphics")])
        self.assertEqual(set(rows[0]), {"index", "name", "load", "used", "total", "temperature", "fan", "clock", "power", "power_limit"})
        self.assertEqual(rows[0]["used"], 26900000000)
        self.assertIsNone(rows[1]["fan"])
        json.dumps({"gpus": rows}, allow_nan=False)


class GpuListTests(unittest.TestCase):
    def test_an_amd_computer_lists_its_cards_through_amd_smi(self):
        # No NVIDIA driver at all: nvidia-smi and NVML are gone, torch runs on ROCm and sees both cards.
        visible = [{"number": 0, "name": R9700["name"], "uuid": R9700["uuid"], "total_bytes": R9700["total_bytes"]},
                   {"number": 1, "name": IGPU["name"], "uuid": "", "total_bytes": 32414789632}]
        with patch.object(gpus, "_nvml_gpus", return_value=[]), patch.object(gpus, "_smi_gpus", return_value=[]), \
                patch.object(amd_smi, "find_amd_smi", return_value="/usr/bin/amd-smi"), fake_amd_smi(STATIC, MONITOR), \
                patch.object(gpus, "_torch_gpus", return_value=visible), patch.object(gpus, "_comfyui_gpu_number", return_value=0):
            listing = gpus.list_gpus()
        first, second = listing["gpus"]
        self.assertEqual((first["number"], first["visible"], first["comfyui_gpu"]), (0, True, True), "the UUID from the serial pairs the card")
        self.assertEqual(first["total_gb"], 30.4, "32624 MB of the card list is 30.4 GiB here")
        self.assertEqual(first["free_gb"], 5.3)
        self.assertEqual(second["number"], 1, "the iGPU with no serial is paired by its name")
        self.assertEqual(listing["note"], "")

    def test_two_visible_cards_that_share_a_name_are_not_paired_by_name_alone(self):
        card = {"slot": 0, "name": "AMD Radeon Graphics", "uuid": "", "total_bytes": 16 * GiB, "free_bytes": 8 * GiB, "pci_bus": "a"}
        visible = [{"number": number, "name": "AMD Radeon Graphics", "uuid": "", "total_bytes": 16 * GiB} for number in (0, 1)]
        listing = gpus.merge_gpus([card], visible)
        self.assertEqual([g["visible"] for g in listing["gpus"]], [True, True, False], "nobody is assigned a number on a guess when the name is shared")
        self.assertIsNone(listing["gpus"][-1]["number"])


if __name__ == "__main__":
    unittest.main()
