"""Isolated SheetSage2 transcription worker for the ComfyUI wrapper."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def run(payload: dict) -> dict:
    try:
        import torch
        from transformers import AutoModel
    except Exception as exc:
        raise RuntimeError(
            "SheetSage2 dependencies are not installed in the selected Python environment."
        ) from exc

    config = payload["config"]
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SheetSage2 was configured for CUDA, but CUDA is unavailable in its environment.")

    print(
        f"[VRGDG worker] Loading SheetSage2 model={config['model']} "
        f"parent={config.get('parent_model')} device={device}",
        file=sys.stderr, flush=True,
    )
    model = AutoModel.from_pretrained(
        config["model"],
        trust_remote_code=True,
        base_model_path=config.get("parent_model") or None,
        cache_dir=config.get("cache_dir") or None,
        local_files_only=bool(config.get("local_files_only", False)),
    ).eval().to(device)
    print("[VRGDG worker] SheetSage2 pipeline is ready.", file=sys.stderr, flush=True)
    try:
        def progress(event):
            print(f"[SheetSage2] {json.dumps(event, sort_keys=True)}", file=sys.stderr, flush=True)

        result = model.transcribe(
            payload["audio_path"],
            output_dir=str(output_dir),
            melody_only=bool(payload.get("melody_only", True)),
            progress=progress,
        )
        abc_path = output_dir / "score.abc"
        if abc_path.is_file():
            abc = abc_path.read_text(encoding="utf-8", errors="replace")
        elif isinstance(result, dict) and result.get("abc"):
            abc = str(result["abc"])
        else:
            raise RuntimeError("Transcription produced no score.abc file or ABC result.")
        artifacts = [str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()]
        return {
            "status": "complete",
            "operation": "transcribe",
            "melody_only": bool(payload.get("melody_only", True)),
            "abc": abc,
            "abc_path": str(abc_path),
            "output_dir": str(output_dir),
            "artifacts": artifacts,
            "result": _json_safe(result),
        }
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
