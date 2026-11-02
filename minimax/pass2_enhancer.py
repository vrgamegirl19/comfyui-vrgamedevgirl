"""Resolve the two required pass-2 refinement LoRAs without loading weights."""


def resolve_enhancer_loras(available, dmad_name, refiner_name):
    """Keep registered subfolder paths; report every missing selection at once."""
    resolved = []
    missing = []
    for label, requested in (("DMAD", dmad_name), ("Open-H3 Refiner", refiner_name)):
        requested = str(requested or "").strip()
        normalized = requested.replace("\\", "/")
        matches = sorted(
            str(name) for name in available
            if str(name).replace("\\", "/").rsplit("/", 1)[-1] == normalized.rsplit("/", 1)[-1]
            and str(name) != "[none]"
        )
        exact = next((name for name in matches if name.replace("\\", "/") == normalized), None)
        if exact:
            resolved.append(exact)
        elif len(matches) == 1:
            resolved.append(matches[0])
        elif matches:
            missing.append(f"{label}: multiple files named '{requested}'; select its full subfolder path")
        else:
            missing.append(f"{label}: '{requested or '[none]'}' was not found")
    if missing:
        raise ValueError(
            "Pass 2 enhancer requires both LoRAs. " + "; ".join(missing)
            + ". Download them using the links below the enhancer checkbox, place them in "
            "ComfyUI/models/loras, refresh the model list and select the files."
        )
    return tuple(resolved)
