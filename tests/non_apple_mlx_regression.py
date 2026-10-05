# Run from the ComfyUI root with its venv python:
#   venv-3.13/bin/python custom_nodes/comfyui-vrgamedevgirl/tests/non_apple_mlx_regression.py
# Simulates Windows/Linux/Intel-Mac and checks the MLX additions never activate there.
import sys, types, importlib, importlib.util, os, tempfile
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import torch, comfy_aimdo
sys.modules.setdefault('comfy_aimdo.storage', types.ModuleType('comfy_aimdo.storage'))  # older comfy-aimdo lacks it
vrg = types.ModuleType('vrg'); vrg.__path__ = [REPO]; sys.modules['vrg'] = vrg
LLM = importlib.import_module('vrg.LLM')
import ast, platform
_src = open(os.path.join(REPO, 'VRGDG_WorkflowRunnerNodes.py')).read()
_ns = {'sys': sys, 'platform': platform}
for node in ast.parse(_src).body:
    if isinstance(node, ast.FunctionDef) and node.name.startswith('_require_') and node.name.endswith('_available'):
        exec(compile(ast.Module([node], []), 'runner', 'exec'), _ns)
runner = types.SimpleNamespace(**{k: v for k, v in _ns.items() if k.startswith('_require_')})
import platform
fails = []
def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond: fails.append(name)

check('no mlx modules imported at load', not [m for m in sys.modules if m.split('.')[0] in ('mlx','mlx_lm','mlx_vlm','mflux')])

# a fake mlx_lm/mlx_vlm "installed" must still be ignored off-Apple
real_find = importlib.util.find_spec
importlib.util.find_spec = lambda n, *a, **k: types.SimpleNamespace() if n in ('mlx_lm','mlx_vlm') else real_find(n, *a, **k)
orig_plat, orig_mach = sys.platform, platform.machine
for plat, mach in [('win32','AMD64'),('linux','x86_64'),('darwin','x86_64')]:
    sys.platform = plat; platform.machine = lambda m=mach: m
    check(f'{plat}/{mach}: gemma mlx text unavailable', LLM._gemma_mlx_available() is False)
    check(f'{plat}/{mach}: gemma mlx vision unavailable', LLM._gemma_mlx_available(vision=True) is False)
    for fn in (() if runner is None else ('_require_ltx2mlx_available','_require_flux2klein_mlx_available','_require_zimage_mlx_available','_require_krea2_zimage_mlx_available')):
        try: getattr(runner, fn)(); ok = False
        except RuntimeError as e: ok = 'Apple Silicon' in str(e)
        check(f'{plat}/{mach}: {fn} raises clear error', ok)
# darwin arm64 but package missing -> falls back
importlib.util.find_spec = lambda n, *a, **k: None if n in ('mlx_lm','mlx_vlm') else real_find(n, *a, **k)
sys.platform = 'darwin'; platform.machine = lambda: 'arm64'
check('arm64 without mlx_lm: unavailable', LLM._gemma_mlx_available() is False)
importlib.util.find_spec = real_find; sys.platform = 'win32'; platform.machine = lambda: 'AMD64'

# GGUF dispatch on non-Apple: _load_gguf_model must reach llama_cpp, never MLX
calls = []
fake = types.ModuleType('llama_cpp')
class Llama:
    def __init__(self, **kw): calls.append(kw)
    def create_chat_completion(self, **kw):
        calls.append(('chat', kw)); return {'choices':[{'message':{'content':'hello'}}]}
fake.Llama = Llama
fmt = types.ModuleType('llama_cpp.llama_chat_format'); fmt.Llava15ChatHandler = type('H',(),{})
sys.modules['llama_cpp'] = fake; sys.modules['llama_cpp.llama_chat_format'] = fmt
cls = LLM.VRGDG_GeneralGGUF
inst = cls.__new__(cls)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, 'gemma-3-27b-it.gguf'); open(p,'wb').write(b'x')
    h = inst._load_gguf_model(p, 2048, 0, 4, '', '')
    check('gguf handle is llama Llama (not MLX)', isinstance(h, Llama))
    out = inst._run_gguf_text_pipeline(h, 'hi', 0.7, 0.9, 16)
    check('gguf text pipeline returns text', out == 'hello')
    check('stop sequences still passed to llama', calls[-1][1].get('stop') == list(cls._GEMMA_STOP_SEQUENCES))
check('no mlx modules imported after dispatch', not [m for m in sys.modules if m.split('.')[0] in ('mlx','mlx_lm','mlx_vlm','mflux')])
r = LLM._clear_vrgdg_llm_caches(clear_cuda_cache=False)
check('cache clear works', isinstance(r, dict) and 'gguf_models_unloaded' in r)
print('FAILED' if fails else 'ALL PASS', fails)
