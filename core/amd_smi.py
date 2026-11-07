"""Reading the AMD GPUs of the computer through ``amd-smi``, the AMD equivalent of ``nvidia-smi``.

The tool ships with ``rocm-smi-lib`` as a Python console script, so on a computer where ComfyUI runs on the ROCm PyTorch build it is usually only
inside the Python environment it was pip installed into: on the PATH after ``source ~/.venv/bin/activate`` (or ``.venv\\Scripts\\activate.bat``),
but not before. A program is not an import, so running ComfyUI inside that environment does not put the program on the PATH by itself;
``find_amd_smi`` therefore looks beside the interpreter that is running, and ``VRGDG_AMD_SMI`` can name the program outright.

``static`` answers what each card is: its name, its total VRAM, its power limit and its serial. ``monitor`` answers what it is doing now: how busy it
is, how much VRAM is in use, its temperature, its clock, its fan and its power draw. Each command takes a few tens of milliseconds, and
``gpus_amd()`` runs both and puts the two answers together by GPU number, so a caller gets one record per card either way.
Every reading is a ``{"value": 26.9, "unit": "GB"}`` object, and a sensor the card does not have is the string ``"N/A"`` rather than a number, so
every field can come back ``None``.

Memory numbers come back in decimal units (``32624 MB`` = 32.6 GB) while ComfyUI shows binary ones (30.4 GiB), so they are converted to bytes here
and nothing has to know which units ``amd-smi`` chose.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

# ``amd-smi`` writes plain decimal multiples: 1 kB is 1000 bytes, 1 GB is 1000^3.
_UNITS = {"B": 1, "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12}


def find_amd_smi() -> Optional[str]:
    """The ``amd-smi`` of the Python environment that is running ComfyUI, or ``None`` when that environment has none.

    The PATH is asked first: a system install puts it there, and so does activating an environment before starting ComfyUI. Failing that, the
    ``bin``/``Scripts`` folders of ``sys.prefix`` and of ``sys.executable`` are asked, which is the environment we are actually running in —
    ``rocm-smi-lib`` is a pip package, so it installs its program into it. No other environment on the computer is guessed at; a program installed
    somewhere else is pointed at with ``VRGDG_AMD_SMI`` instead.
    """
    override = str(os.environ.get("VRGDG_AMD_SMI") or "").strip()
    if override:
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return override
        return None
    found = shutil.which("amd-smi")
    if found:
        return found
    on_windows = os.name == "nt"
    scripts = "Scripts" if on_windows else "bin"
    name = "amd-smi.exe" if on_windows else "amd-smi"
    roots: List[str] = []
    for anchor in (sys.prefix, os.path.dirname(os.path.abspath(sys.executable))):
        if anchor and anchor not in roots:
            roots.append(anchor)
    for root in roots:
        for candidate in (os.path.join(root, scripts, name), os.path.join(root, name)):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _clean(value: Any) -> Optional[float]:
    """The number itself, or ``None``. A sensor the card does not have says ``"N/A"``, and a card the tool cannot reach says nothing at all,
    so both answers have to be tolerated."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _number(field: Any) -> Optional[float]:
    """The number of a reading like ``{"value": 52, "unit": "W"}``; older ``amd-smi`` builds print some readings as plain numbers, so those work too."""
    if isinstance(field, dict):
        return _clean(field.get("value"))
    return _clean(field)


def _bytes(value: Any) -> Optional[int]:
    """The number of bytes of a memory reading, whatever unit the tool happened to report it in."""
    if not isinstance(value, dict):
        return None
    amount = _clean(value.get("value"))
    multiplier = _UNITS.get(str(value.get("unit") or "").strip().upper())
    if amount is None or multiplier is None:
        return None
    return int(round(amount * multiplier))


def _rows(command: List[str], timeout: float = 8.0) -> List[Dict[str, Any]]:
    """The records ``amd-smi`` prints for its GPUs. ``static`` wraps them in ``{"gpu_data": [...]}``, ``monitor`` prints the list by itself."""
    try:
        payload = json.loads(subprocess.check_output(
            command, timeout=timeout, text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        ))
    except Exception:
        return []
    if isinstance(payload, dict):
        payload = payload.get("gpu_data") or []
    return [record for record in payload if isinstance(record, dict)]


def _nested(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def _torch_uuid(serial: Any) -> str:
    """The card's serial number in the form PyTorch reports the GPU UUID.

    The ROCm runtime builds the UUID out of the ASCII codes of the asic serial: the card whose serial is ``0x7685A348C85BC8E5`` shows up in
    ``torch.cuda.get_device_properties().uuid`` as ``37363835-6133-3438-6338-356263386535``, the serial digits ``7 6 8 5 a 3 4 8 ...`` written as
    their own bytes. Converting here lets ``gpus.merge_gpus`` pair an ``amd-smi`` card with the PyTorch card of the same piece of hardware by UUID,
    the same way it already pairs ``nvidia-smi`` and NVML cards.
    """
    text = str(serial or "").strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) != 16 or any(character not in "0123456789abcdef" for character in text):
        return ""
    raw = text.encode("ascii").hex()
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def _identification(cards: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """What the cards are, keyed by the GPU number ``amd-smi`` uses."""
    found: Dict[int, Dict[str, Any]] = {}
    for card in cards:
        gpu = card.get("gpu")
        if not isinstance(gpu, int):
            continue
        power_limit = _number(_nested(card, "limit", "ppt0", "socket_power_limit"))
        if power_limit is None:
            power_limit = _number(_nested(card, "limit", "ppt0", "max_power_limit"))
        found[gpu] = {
            "name": str(_nested(card, "asic", "market_name") or "").strip(),
            "total": _bytes(_nested(card, "vram", "size")),
            "power_limit": power_limit,
            "uuid": _torch_uuid(_nested(card, "asic", "asic_serial")),
            "pci_bus": str(_nested(card, "bus", "bdf") or "").strip().lower(),
        }
    return found


def _activity(cards: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """What the cards are doing right now, keyed by the GPU number ``amd-smi`` uses."""
    found: Dict[int, Dict[str, Any]] = {}
    for card in cards:
        gpu = card.get("gpu")
        if not isinstance(gpu, int):
            continue
        total = _bytes(card.get("vram_total"))
        fan = card.get("fan_speed") if card.get("fan_speed") is not None else card.get("fan")
        found[gpu] = {
            "load": _number(card.get("gfx")),
            "used": _bytes(card.get("vram_used")),
            # ``monitor`` reports the total as a rough figure next to the used amount; ``static`` reports the real card size and wins later.
            "total": total,
            "temperature": _number(card.get("hotspot_temperature")),
            "fan": _number(fan),
            "clock": _number(card.get("gfx_clk")),
            "power": _number(card.get("power_usage")),
        }
    return found


def gpus_amd(timeout: float = 8.0) -> List[Dict[str, Any]]:
    """Every AMD GPU ``amd-smi`` sees, one record per card: ``gpu``, ``name``, ``total``, ``used``, ``free``, ``temperature``, ``fan``, ``clock``,
    ``power``, ``power_limit``, ``uuid`` and ``pci_bus``. Memory is in bytes, every sensor can be ``None``, and the list is empty when this computer
    has no AMD GPU, no ``amd-smi``, or a driver that will not answer a non-administrator. ``timeout`` is per command."""
    executable = find_amd_smi()
    if not executable:
        return []
    known = _identification(_rows([executable, "static", "--json"], timeout))
    live = _activity(_rows([executable, "monitor", "--json"], timeout))
    gpus: List[Dict[str, Any]] = []
    for gpu in sorted(set(known) | set(live)):
        record: Dict[str, Any] = {"gpu": gpu, "name": "", "load": None, "total": None, "used": None, "free": None, "temperature": None, "fan": None,
                                  "clock": None, "power": None, "power_limit": None, "uuid": "", "pci_bus": ""}
        # ``live`` goes first so the solid card facts from ``static`` (real size, power limit, serial) replace the rough monitor figures.
        for part in (live.get(gpu), known.get(gpu)):
            for key, value in (part or {}).items():
                if value is not None:
                    record[key] = value
        if record["total"] is not None and record["used"] is not None:
            record["free"] = max(record["total"] - record["used"], 0)
        gpus.append(record)
    return gpus


def parse_sensors(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The live readings of ``gpus_amd()`` in the shape the resource monitor already uses for ``nvidia-smi``, so the browser keeps working unchanged.
    Memory becomes ``used`` and ``total`` in bytes, and ``index`` is the GPU number as text."""
    rows = []
    for record in records:
        rows.append({
            "index": str(record["gpu"]),
            "name": record["name"],
            "load": record["load"],
            "used": record["used"],
            "total": record["total"],
            "temperature": record["temperature"],
            "fan": record["fan"],
            "clock": record["clock"],
            "power": record["power"],
            "power_limit": record["power_limit"],
        })
    return rows
