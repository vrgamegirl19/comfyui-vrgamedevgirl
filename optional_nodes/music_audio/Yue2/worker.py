"""Small process boundary around YuE2.

This file intentionally imports YuE2 only after startup so it can be executed by
an isolated Python environment whose Torch stack is independent from ComfyUI.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _configure_cuda_graph_attention(config: dict, torch) -> None:
    """Avoid YuE2's Flash Attention false-positive on some Torch builds."""

    if config.get("backend", "torch") != "torch" or not torch.cuda.is_available():
        return

    cuda_backend = getattr(torch.backends, "cuda", None)
    availability_check = getattr(cuda_backend, "is_flash_attention_available", None)
    if availability_check is None:
        return
    try:
        flash_available = bool(availability_check())
    except Exception:
        return
    if flash_available:
        return

    # Some Windows wheels publish the internal aten Flash Attention schema
    # even though its CUDA implementation was not compiled. YuE2 currently
    # treats the schema as proof that Flash Attention is usable and then fails
    # while capturing its CUDA graph.
    import yue2.cuda_graph as cuda_graph

    original = cuda_graph.GraphAR
    if getattr(original, "_vrgdg_attention_compat", False):
        return

    fallback = "cudnn" if torch.backends.cudnn.is_available() else "sdpa"

    class CompatibleGraphAR(original):
        _vrgdg_attention_compat = True

        def __init__(self, *args, **kwargs):
            # GraphAR's fourth positional argument is attention_backend. YuE2
            # currently omits it, but preserve any explicit upstream choice.
            if len(args) < 4 and kwargs.get("attention_backend", "auto") == "auto":
                kwargs["attention_backend"] = fallback
            super().__init__(*args, **kwargs)

    CompatibleGraphAR.__name__ = original.__name__
    CompatibleGraphAR.__qualname__ = original.__qualname__
    cuda_graph.GraphAR = CompatibleGraphAR

    # Support YuE2 releases which import GraphAR at sampling-module scope.
    try:
        import yue2.sampling as sampling

        if getattr(sampling, "GraphAR", None) is original:
            sampling.GraphAR = CompatibleGraphAR
    except Exception:
        pass

    print(
        "[VRGDG worker] Torch Flash Attention is not compiled in this build; "
        f"YuE2 CUDA graphs will use {fallback} attention.",
        file=sys.stderr,
        flush=True,
    )


def _load_pipeline(config: dict):
    try:
        from yue2 import YuE2Pipeline
        from yue2.protocol import GenerationConfig
        import torch
    except Exception as exc:
        raise RuntimeError(
            "YuE2 is not installed in the selected Python environment. Install "
            "the official yue2-infer package/repository in that environment."
        ) from exc

    _configure_cuda_graph_attention(config, torch)

    kwargs = {
        "vae": config["vae"],
        "device": config.get("device", "cuda"),
        "memory_budget_gib": float(config.get("memory_budget_gib", 24.0)),
        "backend": config.get("backend", "torch"),
        "quantization": config.get("quantization", "none"),
        "offload_ar": bool(config.get("offload_ar", False)),
        "local_files_only": bool(config.get("local_files_only", False)),
        "verify_hashes": bool(config.get("verify_hashes", False)),
        "progress": True,
        "generation_config": GenerationConfig(ode_steps=int(config.get("ode_steps", 32))),
    }
    cache_dir = str(config.get("cache_dir", "")).strip()
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    gpu_status = "CPU"
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu_status = (
            f"{torch.cuda.get_device_name(0)}; {free / 1024**3:.1f} GiB free / "
            f"{total / 1024**3:.1f} GiB total"
        )
    print(
        f"[VRGDG worker] Loading YuE2 model={config['model']} vae={config['vae']} "
        f"backend={config.get('backend', 'torch')} device={config.get('device', 'cuda')} "
        f"ode_steps={config.get('ode_steps', 32)} "
        f"gpu=({gpu_status})",
        file=sys.stderr, flush=True,
    )
    pipeline = YuE2Pipeline.from_pretrained(config["model"], **kwargs)
    print("[VRGDG worker] YuE2 pipeline is ready.", file=sys.stderr, flush=True)
    return pipeline


def _request_kwargs(request: dict) -> dict:
    values = {
        "style": request["style"],
        "lyrics": request["lyrics"],
        "cot": request.get("cot", "full"),
        "seed": int(request.get("seed", 831001)),
        "id": request.get("id", "song"),
    }
    abc = request.get("abc")
    if abc:
        values["abc"] = abc
    cfg_scale = request.get("cfg_scale")
    if cfg_scale is not None:
        values["cfg_scale"] = float(cfg_scale)
    return values


def run(payload: dict) -> dict:
    operation = payload["operation"]
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)

    pipe = _load_pipeline(payload["config"])
    try:
        request = _request_kwargs(payload["request"])
        print(
            f"[VRGDG worker] Starting {operation}: planning={request.get('cot')} "
            f"lyrics={len(request.get('lyrics', ''))} chars output={output_dir}",
            file=sys.stderr, flush=True,
        )
        if operation == "plan":
            plan = pipe.plan(**request)
            plan.save(output_dir)
            return {
                "status": "complete",
                "operation": "plan",
                "abc": plan.abc or "",
                "truncated": bool(plan.truncated),
                "timing": plan.timing,
                "output_dir": str(output_dir),
            }
        if operation != "generate":
            raise ValueError(f"Unknown YuE2 worker operation: {operation}")

        song = pipe(**request)
        print("[VRGDG worker] Generation finished; saving artifacts...", file=sys.stderr, flush=True)
        saved = song.save_artifacts(output_dir)
        print("[VRGDG worker] Artifacts saved.", file=sys.stderr, flush=True)
        return {
            "status": "complete",
            "operation": "generate",
            "abc": song.abc or "",
            "truncated": song.truncated,
            "sample_rate": int(song.sample_rate),
            "audio_path": str(output_dir / "audio.flac"),
            "output_dir": str(output_dir),
            "result": saved,
        }
    finally:
        pipe.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    response_path = Path(args.response)
    try:
        payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
        response = run(payload)
        code = 0
    except BaseException as exc:
        response = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        code = 1
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
