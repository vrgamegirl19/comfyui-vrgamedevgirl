"""The Custom Nodes dialog installs the nvidia-vfx pip library: allowlisted, from NVIDIA's index, without ComfyUI-Manager."""

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "core" / "system_routes.py"


def allowlist():
    """VIDEO_BUILDER_CUSTOM_NODES of core/system_routes.py as a syntax tree (the module needs a running ComfyUI server to import)."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(getattr(target, "id", "") == "VIDEO_BUILDER_CUSTOM_NODES" for target in node.targets))


def route_functions(installed, manager=False):
    """The status and install functions of core/system_routes.py with the outside world replaced by mocks."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    wanted = {"_node_status", "_install", "_run_pip_install", "_built_in"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    item = {"pip": "nvidia-vfx", "index_url": "https://pypi.nvidia.com", "is_installed": installed}
    namespace = {
        "sys": sys,
        "VIDEO_BUILDER_CUSTOM_NODES": {"nvidia_vfx": item, "kjnodes": {"install": "comfyui-kjnodes", "folders": ("ComfyUI-KJNodes",)}},
        "_manager_available": lambda: manager,
        "_installed_node_path": lambda _item: None,
        "_run_command": Mock(return_value="Successfully installed nvidia-vfx"),
        "_run_cm_cli": Mock(return_value=""),
        "has_built_in": lambda _since: False,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class PipLibraryEntryTests(unittest.TestCase):
    def test_the_entry_is_fixed_to_nvidia_vfx_and_nvidias_index(self) -> None:
        table = allowlist()
        entry = next(value for key, value in zip(table.keys, table.values) if key.value == "nvidia_vfx")
        fields = {key.value: value for key, value in zip(entry.keys, entry.values)}
        self.assertEqual(ast.literal_eval(fields["pip"]), "nvidia-vfx")
        self.assertEqual(ast.literal_eval(fields["index_url"]), "https://pypi.nvidia.com")
        self.assertEqual(fields["is_installed"].id, "nvidia_vsr_available")

    def test_only_the_nvidia_entry_is_a_pip_library(self) -> None:
        table = allowlist()
        pip = [key.value for key, entry in zip(table.keys, table.values)
               if any(getattr(name, "value", None) == "pip" for name in entry.keys)]
        self.assertEqual(pip, ["nvidia_vfx"])

    def test_the_status_row_says_pip_and_follows_whether_the_library_imports(self) -> None:
        for installed in (True, False):
            namespace = route_functions(lambda installed=installed: installed)
            row = namespace["_node_status"]("nvidia_vfx", namespace["VIDEO_BUILDER_CUSTOM_NODES"]["nvidia_vfx"])
            self.assertEqual((row["installed"], row["source"], row["install"]), (installed, "pip", "nvidia-vfx"))
            self.assertNotIn("built_in", row)

    def test_install_runs_pip_with_wheels_only_from_nvidias_index_and_needs_no_manager(self) -> None:
        state = {"installed": False}

        def installed():
            return state["installed"]

        namespace = route_functions(installed, manager=False)
        namespace["_run_command"].side_effect = lambda *_args, **_kwargs: state.update(installed=True) or "Successfully installed"
        results = namespace["_install"](["nvidia_vfx"])
        self.assertEqual([(item["id"], item["action"]) for item in results], [("nvidia_vfx", "installed")])
        command, tool = namespace["_run_command"].call_args.args[:2]
        self.assertEqual(tool, "pip")
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn("--only-binary", command)
        self.assertEqual(command[command.index("--only-binary") + 1], "nvidia-vfx")
        self.assertEqual(command[command.index("--extra-index-url") + 1], "https://pypi.nvidia.com")
        self.assertEqual(command.count("nvidia-vfx"), 2)
        namespace["_run_cm_cli"].assert_not_called()

    def test_an_installed_library_is_not_installed_again(self) -> None:
        namespace = route_functions(lambda: True)
        results = namespace["_install"](["nvidia_vfx"])
        self.assertEqual(results[0]["action"], "already installed")
        namespace["_run_command"].assert_not_called()

    def test_a_pip_run_that_leaves_the_library_unimportable_is_an_error(self) -> None:
        namespace = route_functions(lambda: False)
        with self.assertRaises(RuntimeError) as raised:
            namespace["_install"](["nvidia_vfx"])
        self.assertIn("still does not import", str(raised.exception))

    def test_a_custom_node_still_needs_the_manager(self) -> None:
        namespace = route_functions(lambda: False, manager=False)
        with self.assertRaises(RuntimeError) as raised:
            namespace["_install"](["kjnodes", "nvidia_vfx"])
        self.assertIn("ComfyUI-Manager is not installed", str(raised.exception))
        namespace["_run_command"].assert_not_called()

    def test_an_unknown_id_installs_nothing(self) -> None:
        namespace = route_functions(lambda: False)
        with self.assertRaises(RuntimeError):
            namespace["_install"](["../../evil"])
        namespace["_run_command"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
