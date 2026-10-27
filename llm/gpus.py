"""The GPUs a local GGUF model (Gemma Local, Qwen Local) can be put on.

llama.cpp runs inside the ComfyUI process, so it only sees the GPUs that process sees, numbered the way CUDA numbers them there.
``CUDA_VISIBLE_DEVICES`` (for example ``set CUDA_VISIBLE_DEVICES=1`` in the launcher, or ComfyUI's ``--cuda-device``) hides the other cards and
renumbers the rest, and ``--default-device N`` moves card N to number 0. Windows Task Manager and ``nvidia-smi`` number the cards by PCI slot
instead, and CUDA puts the fastest card first, so the same card can have a different number there. Everything here uses the CUDA number of the
ComfyUI process, because that is the number ``main_gpu`` takes, and shows the name and memory beside it so the cards cannot be mixed up.
"""

import os
import subprocess
from typing import Any, Dict, List, Optional

AUTO_GPU = -1
MAX_GPU_NUMBER = 63
# llama.cpp's LLAMA_SPLIT_MODE_NONE: the whole model on ``main_gpu``. Its default spreads the layers over every visible card.
_SPLIT_MODE_NONE = 0


def normalize_gpu_device(value: Any, default: int = AUTO_GPU) -> int:
    """The GPU to use: ``-1`` for automatic (every visible GPU, split by llama.cpp), or the CUDA number of one GPU.

    Empty, ``auto``, a negative number or anything that is not a number is automatic when ``default`` is.
    """
    if value is None or isinstance(value, bool):
        return default
    text = str(value).strip().lower()
    if text in {"", "auto", "all", "default"}:
        return AUTO_GPU
    try:
        number = int(float(text))
    except (TypeError, ValueError):
        return default
    if number < 0:
        return AUTO_GPU
    return min(number, MAX_GPU_NUMBER)


def gpu_kwargs(device: Any) -> Dict[str, int]:
    """The llama-cpp-python arguments that keep the whole model on one GPU. Nothing for automatic, so llama.cpp spreads it as it always did."""
    number = normalize_gpu_device(device)
    if number < 0:
        return {}
    return {"split_mode": _SPLIT_MODE_NONE, "main_gpu": number}


def _clean_uuid(value: Any) -> str:
    return str(value or "").strip().lower().replace("gpu-", "")


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _torch_gpus() -> List[Dict[str, Any]]:
    """The GPUs this process sees, in CUDA's order. Reading their properties does not start a CUDA context."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        found = []
        for number in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(number)
            found.append({
                "number": number,
                "name": str(props.name),
                "total_bytes": int(props.total_memory),
                "uuid": _clean_uuid(getattr(props, "uuid", "")),
            })
        return found
    except Exception:
        return []


def _comfyui_gpu_number() -> Optional[int]:
    """The CUDA number of the GPU ComfyUI renders on."""
    try:
        import comfy.model_management as model_management
        device = model_management.get_torch_device()
        index = getattr(device, "index", None)
        if getattr(device, "type", "") == "cuda" and index is not None:
            return int(index)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except Exception:
        pass
    return None


def _nvml_gpus() -> List[Dict[str, Any]]:
    """Every GPU in the computer in PCI slot order, with memory in use, through NVML (it also lists the ones hidden from this process)."""
    try:
        import pynvml
    except Exception:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    try:
        found = []
        for slot in range(int(pynvml.nvmlDeviceGetCount())):
            handle = pynvml.nvmlDeviceGetHandleByIndex(slot)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            found.append({
                "slot": slot,
                "name": _text(pynvml.nvmlDeviceGetName(handle)),
                "uuid": _clean_uuid(_text(pynvml.nvmlDeviceGetUUID(handle))),
                "total_bytes": int(memory.total),
                "free_bytes": int(memory.free),
                "pci_bus": _text(pynvml.nvmlDeviceGetPciInfo(handle).busId),
            })
        return found
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _smi_gpus() -> List[Dict[str, Any]]:
    """The same list from ``nvidia-smi``, for a computer where NVML cannot be imported."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,pci.bus_id", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    found = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            found.append({
                "slot": int(parts[0]), "name": parts[1], "uuid": _clean_uuid(parts[2]),
                "total_bytes": int(float(parts[3]) * 1024 * 1024), "free_bytes": int(float(parts[4]) * 1024 * 1024), "pci_bus": parts[5],
            })
        except ValueError:
            continue
    return found


def _gb(size: Optional[int]) -> Optional[float]:
    return None if size is None else round(size / 1024 ** 3, 1)


def _label(entry: Dict[str, Any]) -> str:
    memory = f"{entry['total_gb']} GB" if entry.get("total_gb") is not None else ""
    if entry.get("free_gb") is not None:
        memory += f", {entry['free_gb']} GB free"
    detail = f" ({memory})" if memory else ""
    if entry["visible"]:
        mark = " - ComfyUI renders here" if entry["comfyui_gpu"] else ""
        return f"GPU {entry['number']}: {entry['name']}{detail}{mark}"
    return f"{entry['name']}{detail} - not visible to ComfyUI"


def merge_gpus(
    physical: List[Dict[str, Any]],
    visible: List[Dict[str, Any]],
    comfyui_gpu: Optional[int] = None,
    cuda_visible_devices: str = "",
) -> Dict[str, Any]:
    """Put the card list of the computer (``physical``) and the cards this process sees (``visible``) together, matched by UUID.

    A visible card keeps its CUDA ``number`` (the number to choose in the LLM runner). A card this process cannot see has ``number`` None.
    """
    by_uuid = {item["uuid"]: item for item in visible if item.get("uuid")}
    used = set()
    gpus: List[Dict[str, Any]] = []
    for card in physical:
        match = by_uuid.get(card["uuid"]) if card.get("uuid") else None
        if match is None and len(physical) == len(visible) == 1:
            match = visible[0]
        entry = {
            "number": match["number"] if match else None,
            "name": card.get("name") or (match or {}).get("name", "GPU"),
            "total_gb": _gb(card.get("total_bytes")),
            "free_gb": _gb(card.get("free_bytes")),
            "slot": card.get("slot"),
            "pci_bus": card.get("pci_bus", ""),
            "visible": match is not None,
            "comfyui_gpu": bool(match is not None and comfyui_gpu is not None and match["number"] == comfyui_gpu),
        }
        if match is not None:
            used.add(match["number"])
        gpus.append(entry)
    for card in visible:
        if card["number"] in used:
            continue
        gpus.append({
            "number": card["number"], "name": card["name"], "total_gb": _gb(card.get("total_bytes")), "free_gb": None, "slot": None, "pci_bus": "",
            "visible": True, "comfyui_gpu": comfyui_gpu is not None and card["number"] == comfyui_gpu,
        })
    gpus.sort(key=lambda item: (item["number"] is None, item["number"] if item["number"] is not None else 0, item.get("slot") or 0))
    for entry in gpus:
        entry["label"] = _label(entry)
    hidden = [entry for entry in gpus if not entry["visible"]]
    note = ""
    if hidden:
        first = str(cuda_visible_devices or "").split(",")[0].strip()
        render_on = first if first.isdigit() else "the number of the card it should render on"
        why = f"CUDA_VISIBLE_DEVICES is \"{cuda_visible_devices}\"" if cuda_visible_devices else "ComfyUI was started so that it cannot see them"
        note = (
            f"ComfyUI can only see {len(gpus) - len(hidden)} of the {len(gpus)} GPUs ({why}), so a local model can only be put on those. "
            "To let it use the others, remove any \"set CUDA_VISIBLE_DEVICES\" line from the ComfyUI launcher (.bat file) and start ComfyUI with "
            f"--default-device {render_on}, which keeps the card it renders on now as GPU 0 and leaves the rest visible. Then restart ComfyUI."
        )
    return {
        "gpus": gpus,
        "visible_count": len(visible),
        "comfyui_gpu": comfyui_gpu,
        "cuda_visible_devices": str(cuda_visible_devices or ""),
        "note": note,
    }


def list_gpus() -> Dict[str, Any]:
    """Every GPU of the computer with its CUDA number in this process (None when hidden from it), name, memory and a label to show."""
    visible = _torch_gpus()
    physical = _nvml_gpus() or _smi_gpus()
    if not physical:
        physical = [
            {"slot": None, "name": item["name"], "uuid": item["uuid"], "total_bytes": item["total_bytes"], "free_bytes": None, "pci_bus": ""}
            for item in visible
        ]
    return merge_gpus(physical, visible, _comfyui_gpu_number(), os.environ.get("CUDA_VISIBLE_DEVICES", ""))


def check_gpu_device(device: Any) -> None:
    """Stop with a readable message when the chosen GPU number is not one this process sees. Automatic always passes."""
    number = normalize_gpu_device(device)
    if number < 0:
        return
    visible = _torch_gpus()
    if not visible:
        return  # no CUDA at all: the model runs on the CPU and the choice does not matter
    if any(item["number"] == number for item in visible):
        return
    listing = "; ".join(f"GPU {item['number']}: {item['name']}" for item in visible)
    raise ValueError(
        f"GPU {number} is not one ComfyUI can see. It sees {len(visible)}: {listing}. "
        "Choose one of them in the LLM Runner, or choose Automatic."
    )
