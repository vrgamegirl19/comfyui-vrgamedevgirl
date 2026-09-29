import ast
import re
from pathlib import Path


WEB = Path(__file__).resolve().parents[1] / "web"
MODULES = WEB / "music_video_builder"


def as_script(source):
    # Tests treat the builder modules as plain script text, so drop module import/export syntax.
    source = re.sub(r"^import [^;]*;\r?\n", "", source, flags=re.M)
    return re.sub(r"^export ", "", source, flags=re.M)


def read_builder_source():
    files = [*sorted(MODULES.glob("*.mjs")), WEB / "VRGDG_MusicVideoBuilderUI.js"]
    return as_script("\n".join(path.read_text(encoding="utf-8") for path in files))


def read_storyboard_source():
    files = [*sorted((WEB / "storyboard_builder").glob("*.mjs")), WEB / "VRGDG_StoryboardBuilderUI.js"]
    return as_script("\n".join(path.read_text(encoding="utf-8") for path in files))


def read_builder_module(name):
    return as_script((MODULES / name).read_text(encoding="utf-8"))


REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")
REGEX_KEYWORD = re.compile(r"\b(?:return|typeof|case|void|throw)\s*$")


def function_source(source, name):
    """Full text of a function declaration, wherever its module placed it."""
    start = source.index(f"function {name}(")
    if source[start - 6:start] == "async ":
        start -= 6
    i = source.index("(", start)
    depth = 0
    templates = []  # brace depth at which each open ${ } started
    last = ""
    while i < len(source):
        ch = source[i]
        if ch in "\"'":
            i += 1
            while source[i] != ch:
                i += 2 if source[i] == "\\" else 1
        elif ch == "`" or (ch == "}" and templates and templates[-1] == depth):
            if ch == "}":
                templates.pop()
            i += 1
            while source[i] != "`":
                if source[i] == "\\":
                    i += 2
                    continue
                if source.startswith("${", i):
                    templates.append(depth)
                    i += 1
                    break
                i += 1
        elif source.startswith("//", i):
            i = source.index("\n", i)
            continue
        elif source.startswith("/*", i):
            i = source.index("*/", i) + 1
        elif ch == "/" and (last in REGEX_PRECEDERS or last == "" or REGEX_KEYWORD.search(source, max(0, i - 12), i)):
            i += 1
            in_class = False
            while in_class or source[i] != "/":
                if source[i] == "\\":
                    i += 1
                elif source[i] == "[":
                    in_class = True
                elif source[i] == "]":
                    in_class = False
                i += 1
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
            if depth == 0 and ch == "}":
                return source[start:i + 1]
        if not ch.isspace():
            last = ch
        i += 1
    raise ValueError(f"function {name} has no end")


BUILDER_BACKEND_FILES = (
    "builder/nodes.py",
    "builder/paths.py",
    "llm/output_checks.py",
    "builder/audio.py",
    "builder/media.py",
    "builder/project_copy.py",
    "builder/project.py",
    "llm/builder_instructions.py",
    "llm/builder_runner.py",
    "llm/video_prompt_generation.py",
    "llm/image_prompt_generation.py",
    "llm/builder_agent.py",
    "builder/routes.py",
)


def read_builder_backend_source():
    """The Video Builder backend: the node module and the builder/ and llm/ modules split out of it."""
    root = Path(__file__).resolve().parents[1]
    return "\n".join((root / name).read_text(encoding="utf-8") for name in BUILDER_BACKEND_FILES)


def python_function_source(source, *names):
    """Source of the named top-level Python functions, in the order given, wherever the split placed them."""
    functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    return "\n".join(ast.get_source_segment(source, functions[name]) for name in names)


RUNNER_FILES = (
    "runner/nodes.py",
    "runner/paths.py",
    "runner/models.py",
    "runner/api_graph.py",
    "runner/image_workflows.py",
    "runner/ltx_workflows.py",
    "runner/minimax_inputs.py",
    "runner/minimax_patches.py",
    "runner/minimax_workflows.py",
    "runner/utility_workflows.py",
    "runner/video_files.py",
    "runner/routes.py",
)


def read_runner_source():
    """The workflow runner: the node module and the runner/ modules split out of it."""
    root = Path(__file__).resolve().parents[1]
    return "\n".join((root / name).read_text(encoding="utf-8") for name in RUNNER_FILES)
