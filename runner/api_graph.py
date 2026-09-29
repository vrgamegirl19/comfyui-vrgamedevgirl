"""Workflow template loading and API graph helpers: node lookup, inputs, workflow-to-API conversion and subgraph expansion."""

import copy
import base64
import importlib
import inspect
import json
import os
import shutil
import sys
import time
import folder_paths

from .paths import _bool_payload, _workflow_template_path


_I2V_UNET_ALIASES = {
    "LTX-2.3-22B-distilled-11-Q6_K.gguf": "LTX-2.3-22B-distilled-1.1-Q6_K.gguf",
}


_PLACEHOLDER_I2I_IMAGE_NAME = "vrgdg_placeholder_i2i.png"


_PLACEHOLDER_I2I_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


def _clean_i2v_unet_name(value):
    text = str(value or "").strip()
    return _I2V_UNET_ALIASES.get(text, text)


def _replace_api_input_refs(prompt, old_ref, new_ref):
    old = [str(old_ref[0]), int(old_ref[1])]
    new = [str(new_ref[0]), int(new_ref[1])]
    replaced = 0
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in list(inputs.items()):
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == old[0] and int(value[1] or 0) == old[1]:
                inputs[key] = list(new)
                replaced += 1
    return replaced


def _collapse_ltx_video_model_switch(prompt, switch_id, selected_loader_id, unused_loader_id):
    switch_key = str(switch_id or "").strip()
    selected_key = str(selected_loader_id or "").strip()
    unused_key = str(unused_loader_id or "").strip()
    if not switch_key or not selected_key:
        return False
    if switch_key not in prompt or selected_key not in prompt:
        return False
    _replace_api_input_refs(prompt, (switch_key, 0), (selected_key, 0))
    prompt.pop(switch_key, None)
    if unused_key and unused_key != selected_key:
        prompt.pop(unused_key, None)
    return True


def _patch_ltx_video_model_loader(prompt, payload):
    use_gguf = _bool_payload(payload, "use_gguf_model", True)
    gguf_name = _clean_i2v_unet_name(payload.get("unet_name", ""))
    diffusion_name = str(payload.get("diffusion_model_name") or payload.get("model_name") or "").strip()
    if not diffusion_name:
        diffusion_name = gguf_name
    switch_id = _optional_api_node_id_by_class(prompt, "ComfySwitchNode", "Switch-use GGUF", fallback_ids=("955", "939", "959"))
    gguf_loader_id = _optional_api_node_id_by_class(prompt, "UnetLoaderGGUF", fallback_ids=("271:215", "969"))
    diffusion_loader_id = _optional_api_node_id_by_class(prompt, "DiffusionModelLoaderKJ", fallback_ids=("956", "938", "958"))
    if switch_id:
        _set_optional_api_input(prompt, switch_id, "switch", use_gguf)
    if gguf_loader_id:
        _set_optional_api_input(prompt, gguf_loader_id, "unet_name", gguf_name)
    if diffusion_loader_id:
        _set_optional_api_input(prompt, diffusion_loader_id, "model_name", diffusion_name)
    if switch_id and gguf_loader_id and diffusion_loader_id:
        if use_gguf:
            _collapse_ltx_video_model_switch(prompt, switch_id, gguf_loader_id, diffusion_loader_id)
        else:
            _collapse_ltx_video_model_switch(prompt, switch_id, diffusion_loader_id, gguf_loader_id)


def _load_workflow_template(path=None):
    raw_path = str(path or "").strip()
    if raw_path and not os.path.isabs(raw_path):
        raw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), raw_path)
    workflow_path = os.path.abspath(raw_path or _workflow_template_path())
    if not os.path.isfile(workflow_path):
        raise FileNotFoundError(f"Workflow template was not found: {workflow_path}")
    with open(workflow_path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        raise ValueError("Workflow template is not a valid ComfyUI workflow JSON.")
    return workflow_path, workflow


def _ltx25_diffusion_loader_node(payload):
    return {
        "inputs": {
            "model_name": str(payload.get("diffusion_model_name") or payload.get("model_name") or ""),
            "weight_dtype": "default",
            "compute_dtype": "default",
            "patch_cublaslinear": False,
            "sage_attention": "auto" if _bool_payload(payload, "use_sage_attention", False) else "disabled",
            "enable_fp16_accumulation": _bool_payload(payload, "enable_fp16_accumulation", False),
        },
        "class_type": "DiffusionModelLoaderKJ",
        "_meta": {"title": "LTX 2.5 Diffusion Model Loader"},
    }


def _load_api_template(path):
    api_path = os.path.abspath(path)
    if not os.path.isfile(api_path):
        raise FileNotFoundError(f"Workflow API template was not found: {api_path}")
    with open(api_path, "r", encoding="utf-8") as handle:
        prompt = json.load(handle)
    if not isinstance(prompt, dict) or not prompt:
        raise ValueError("Workflow API template is not a valid ComfyUI API prompt JSON.")
    return api_path, prompt


def _node_by_id(workflow, node_id):
    target = str(node_id)
    for node in workflow.get("nodes", []):
        if str(node.get("id")) == target:
            return node
    raise KeyError(f"Workflow node {node_id} was not found.")


def _set_widget(workflow, node_id, widget_index, value):
    node = _node_by_id(workflow, node_id)
    widgets = node.setdefault("widgets_values", [])
    if isinstance(widgets, dict):
        widgets[str(widget_index)] = value
        return
    while len(widgets) <= widget_index:
        widgets.append(None)
    widgets[widget_index] = value


def _set_widget_key(workflow, node_id, key, value):
    node = _node_by_id(workflow, node_id)
    widgets = node.setdefault("widgets_values", {})
    if not isinstance(widgets, dict):
        raise TypeError(f"Workflow node {node_id} does not use keyed widget values.")
    widgets[key] = value


def _workflow_node_id_by_class(workflow, class_type, fallback=None):
    for node in workflow.get("nodes", []):
        if node.get("type") == class_type or node.get("class_type") == class_type:
            return str(node.get("id"))
    if fallback is not None:
        _node_by_id(workflow, fallback)
        return str(fallback)
    raise KeyError(f"Workflow node class {class_type} was not found.")


def _api_node_id_by_class(prompt, class_type, fallback=None):
    for node_id, node in prompt.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return str(node_id)
    if fallback is not None and str(fallback) in prompt:
        return str(fallback)
    raise KeyError(f"API prompt node class {class_type} was not found.")


def _prepare_load_image_name(path="", data="", name="image.png"):
    raw_path = str(path or "").strip().strip('"')
    if raw_path:
        source_path = os.path.abspath(raw_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Image-to-image source was not found: {source_path}")
        ext = os.path.splitext(source_path)[1].lower() or ".png"
        input_dir = folder_paths.get_input_directory()
        target_name = f"vrgdg_i2i_{int(time.time() * 1000)}{ext}"
        shutil.copy2(source_path, os.path.join(input_dir, target_name))
        return target_name

    raw_data = str(data or "").strip()
    if raw_data:
        if "," in raw_data and raw_data.lower().startswith("data:"):
            header, encoded = raw_data.split(",", 1)
            ext = ".png"
            if "jpeg" in header.lower() or "jpg" in header.lower():
                ext = ".jpg"
            elif "webp" in header.lower():
                ext = ".webp"
        else:
            encoded = raw_data
            ext = os.path.splitext(str(name or ""))[1].lower() or ".png"
        input_dir = folder_paths.get_input_directory()
        target_name = f"vrgdg_i2i_{int(time.time() * 1000)}{ext}"
        with open(os.path.join(input_dir, target_name), "wb") as handle:
            handle.write(base64.b64decode(encoded))
        return target_name

    return ""


def _prepare_optional_input_image_name(image_info):
    if not isinstance(image_info, dict):
        return "(none)"

    raw_path = str(image_info.get("path") or image_info.get("filename") or "").strip().strip('"')
    if raw_path:
        if os.path.isabs(raw_path):
            return _prepare_load_image_name(raw_path, "", image_info.get("name") or "reference.png") or "(none)"
        clean_path = raw_path.replace("\\", "/")
        if "/" not in clean_path:
            return clean_path
        candidate_bases = [folder_paths.get_input_directory(), folder_paths.get_output_directory()]
        get_temp_directory = getattr(folder_paths, "get_temp_directory", None)
        if callable(get_temp_directory):
            candidate_bases.append(get_temp_directory())
        for base_dir in candidate_bases:
            candidate_path = os.path.abspath(os.path.join(base_dir, clean_path))
            try:
                if os.path.commonpath([os.path.abspath(base_dir), candidate_path]) != os.path.abspath(base_dir):
                    continue
            except ValueError:
                continue
            if os.path.isfile(candidate_path):
                return _prepare_load_image_name(candidate_path, "", image_info.get("name") or os.path.basename(clean_path)) or "(none)"

    image_name = str(image_info.get("name") or "reference.png")
    prepared = _prepare_load_image_name("", image_info.get("data") or "", image_name)
    return prepared or "(none)"


def _ensure_placeholder_load_image():
    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)
    target_path = os.path.join(input_dir, _PLACEHOLDER_I2I_IMAGE_NAME)
    if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
        try:
            from PIL import Image
            with Image.open(target_path) as image:
                image.verify()
            return _PLACEHOLDER_I2I_IMAGE_NAME
        except Exception:
            try:
                os.remove(target_path)
            except OSError:
                pass

    source_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "images",
        _PLACEHOLDER_I2I_IMAGE_NAME,
    )
    if os.path.isfile(source_path) and os.path.getsize(source_path) > 0:
        shutil.copy2(source_path, target_path)
    else:
        with open(target_path, "wb") as handle:
            handle.write(base64.b64decode(_PLACEHOLDER_I2I_IMAGE_BASE64))
    return _PLACEHOLDER_I2I_IMAGE_NAME


def _set_api_input(prompt, node_id, input_name, value):
    node = prompt.get(str(node_id))
    if not isinstance(node, dict):
        raise KeyError(f"API prompt node {node_id} was not found.")
    inputs = node.setdefault("inputs", {})
    inputs[input_name] = value


def _set_optional_api_input(prompt, node_id, input_name, value):
    node = prompt.get(str(node_id))
    if not isinstance(node, dict):
        return False
    inputs = node.setdefault("inputs", {})
    inputs[input_name] = value
    return True


def _api_node_title(node):
    meta = node.get("_meta") if isinstance(node, dict) else {}
    return str(meta.get("title", "") if isinstance(meta, dict) else "").strip()


def _optional_api_node_id_by_class(prompt, class_type, title="", fallback_ids=()):
    wanted_class = str(class_type or "").strip()
    wanted_title = str(title or "").strip()
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "") or "").strip() != wanted_class:
            continue
        if wanted_title and _api_node_title(node) != wanted_title:
            continue
        return str(node_id)
    for node_id in fallback_ids:
        node = prompt.get(str(node_id))
        if isinstance(node, dict) and str(node.get("class_type", "") or "").strip() == wanted_class:
            return str(node_id)
    return ""


def _get_comfy_node_mappings():
    comfy_nodes = sys.modules.get("nodes")
    if comfy_nodes is None or not hasattr(comfy_nodes, "NODE_CLASS_MAPPINGS"):
        comfy_nodes = importlib.import_module("nodes")
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, dict):
        raise RuntimeError("ComfyUI node mappings are not available yet.")
    return mappings


def _input_names_for_node(class_type, mappings):
    node_class = mappings.get(class_type)
    if node_class is None:
        raise KeyError(f"Node class is not loaded in ComfyUI: {class_type}")
    input_types = node_class.INPUT_TYPES()
    names = []
    for section in ("required", "optional"):
        values = input_types.get(section, {})
        if isinstance(values, dict):
            names.extend(values.keys())
    return names


def _node_input_names(class_type, mappings):
    try:
        return list(_input_names_for_node(class_type, mappings))
    except Exception:
        pass
    node_class = mappings.get(class_type) if isinstance(mappings, dict) else None
    if node_class is None:
        return []
    define_schema = getattr(node_class, "define_schema", None)
    if callable(define_schema):
        try:
            schema = define_schema()
            names = []
            for inp in getattr(schema, "inputs", None) or []:
                name = getattr(inp, "id", None) or getattr(inp, "name", None)
                if name:
                    names.append(str(name))
            if names:
                return names
        except Exception:
            pass
    execute = getattr(node_class, "execute", None)
    if callable(execute):
        try:
            names = []
            for name, param in inspect.signature(execute).parameters.items():
                if name in {"cls", "self"}:
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                names.append(name)
            return names
        except (TypeError, ValueError):
            pass
    return []


def _compat_node_inputs(class_type, mappings, required_inputs, extra_defaults=None):
    """Keep required inputs, and attach extras only when the installed node declares them."""
    inputs = dict(required_inputs)
    names = set(_node_input_names(class_type, mappings))
    for key, value in (extra_defaults or {}).items():
        if key in names:
            inputs[key] = value
    return inputs


def _api_widget_values(class_type, widget_values):
    values = list(widget_values or [])
    if class_type == "SamplerCustom" and len(values) >= 4:
        # ComfyUI stores seed control mode ("fixed", "randomize", etc.) in the
        # workflow widgets, but it is not an API input. The next real input is cfg.
        if str(values[2]).lower() in {"fixed", "randomize", "increment", "decrement"}:
            values.pop(2)
    return values


def _workflow_to_api_prompt(workflow):
    workflow = _expand_subgraphs(workflow)
    mappings = _get_comfy_node_mappings()
    links = {}
    for raw_link in workflow.get("links", []):
        if not isinstance(raw_link, list) or len(raw_link) < 6:
            continue
        link_id, origin_id, origin_slot = raw_link[0], raw_link[1], raw_link[2]
        links[int(link_id)] = [str(origin_id), int(origin_slot)]

    # Reroute is a canvas-only node. Resolve every link originating from one
    # directly to the reroute's incoming source before converting the graph.
    reroute_ids = {str(node.get("id")) for node in workflow.get("nodes", []) if node.get("type") == "Reroute"}
    reroute_inputs = {}
    for node in workflow.get("nodes", []):
        node_id = str(node.get("id"))
        if node_id not in reroute_ids:
            continue
        incoming = next((item.get("link") for item in node.get("inputs", []) or [] if item.get("link") is not None), None)
        if incoming is not None and int(incoming) in links:
            reroute_inputs[node_id] = list(links[int(incoming)])

    def resolve_source(source):
        seen = set()
        current = list(source)
        while str(current[0]) in reroute_inputs and str(current[0]) not in seen:
            seen.add(str(current[0]))
            current = list(reroute_inputs[str(current[0])])
        return current

    for link_id, source in list(links.items()):
        links[link_id] = resolve_source(source)

    set_values = {}
    get_nodes = {}
    for node in workflow.get("nodes", []):
        node_id = str(node.get("id"))
        class_type = node.get("type")
        widgets = node.get("widgets_values", [])
        if class_type == "SetNode" and isinstance(widgets, list) and widgets:
            input_link = None
            for input_info in node.get("inputs", []) or []:
                if input_info.get("link") is not None:
                    input_link = int(input_info.get("link"))
                    break
            if input_link is not None and input_link in links:
                set_values[str(widgets[0])] = links[input_link]
        elif class_type == "GetNode" and isinstance(widgets, list) and widgets:
            get_nodes[node_id] = str(widgets[0])

    prompt = {}
    for node in workflow.get("nodes", []):
        node_id = str(node.get("id"))
        class_type = node.get("type")
        if not node_id or not class_type:
            continue
        if class_type in {"SetNode", "GetNode", "MarkdownNote", "Reroute"}:
            continue

        linked_inputs = {}
        for input_info in node.get("inputs", []) or []:
            link_id = input_info.get("link")
            input_name = input_info.get("name")
            if link_id is not None and input_name and int(link_id) in links:
                source = links[int(link_id)]
                source_node_id = str(source[0])
                if source_node_id in get_nodes and get_nodes[source_node_id] in set_values:
                    source = set_values[get_nodes[source_node_id]]
                linked_inputs[input_name] = source

        inputs = dict(linked_inputs)
        raw_widget_values = node.get("widgets_values", [])
        keyed_widget_values = raw_widget_values if isinstance(raw_widget_values, dict) else None
        widget_values = [] if keyed_widget_values is not None else _api_widget_values(class_type, raw_widget_values)
        widget_index = 0
        for input_name in _input_names_for_node(class_type, mappings):
            if input_name in linked_inputs:
                continue
            if keyed_widget_values is not None:
                if input_name in keyed_widget_values and not isinstance(keyed_widget_values[input_name], dict):
                    inputs[input_name] = keyed_widget_values[input_name]
                continue
            if widget_index >= len(widget_values):
                break
            inputs[input_name] = widget_values[widget_index]
            widget_index += 1

        prompt[node_id] = {"class_type": class_type, "inputs": inputs}

    return prompt


def _expand_subgraphs(workflow, depth=0):
    definitions = {item.get("id"): item for item in workflow.get("definitions", {}).get("subgraphs", []) if isinstance(item, dict)}
    if not definitions or depth > 12:
        return workflow
    if not any(node.get("type") in definitions for node in workflow.get("nodes", [])):
        return workflow

    workflow = copy.deepcopy(workflow)
    outer_links = {}
    max_link_id = 0
    for raw_link in workflow.get("links", []):
        if isinstance(raw_link, list) and len(raw_link) >= 6:
            link_id = int(raw_link[0])
            max_link_id = max(max_link_id, link_id)
            outer_links[link_id] = [str(raw_link[1]), int(raw_link[2])]
        elif isinstance(raw_link, dict):
            link_id = int(raw_link.get("id", 0) or 0)
            max_link_id = max(max_link_id, link_id)
            outer_links[link_id] = [str(raw_link.get("origin_id")), int(raw_link.get("origin_slot", 0) or 0)]

    def new_link_id():
        nonlocal max_link_id
        max_link_id += 1
        return max_link_id

    def link_tuple(link_id, origin_id, origin_slot, target_id, target_slot, link_type):
        return [link_id, origin_id, origin_slot, target_id, target_slot, link_type]

    subgraph_node_ids = {str(node.get("id")) for node in workflow.get("nodes", []) if node.get("type") in definitions}
    expanded_nodes = []
    expanded_links = [
        link for link in workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 6 and str(link[1]) not in subgraph_node_ids and str(link[3]) not in subgraph_node_ids
    ]
    link_assignments = []
    subgraph_output_sources = {}

    for node in workflow.get("nodes", []):
        subgraph = definitions.get(node.get("type"))
        if not subgraph:
            expanded_nodes.append(node)
            continue

        node_id = str(node.get("id"))
        id_map = {str(inner.get("id")): f"{node_id}_{inner.get('id')}" for inner in subgraph.get("nodes", [])}
        external_inputs = node.get("inputs", []) or []
        external_widgets = list(node.get("widgets_values", []) or [])
        input_target_links = {}
        output_sources = {}

        for raw_link in subgraph.get("links", []) or []:
            if isinstance(raw_link, dict):
                link = {
                    "id": int(raw_link.get("id", 0) or 0),
                    "origin_id": raw_link.get("origin_id"),
                    "origin_slot": int(raw_link.get("origin_slot", 0) or 0),
                    "target_id": raw_link.get("target_id"),
                    "target_slot": int(raw_link.get("target_slot", 0) or 0),
                    "type": raw_link.get("type", "*"),
                }
            elif isinstance(raw_link, list) and len(raw_link) >= 6:
                link = {
                    "id": int(raw_link[0]),
                    "origin_id": raw_link[1],
                    "origin_slot": int(raw_link[2]),
                    "target_id": raw_link[3],
                    "target_slot": int(raw_link[4]),
                    "type": raw_link[5],
                }
            else:
                continue

            origin_id = str(link["origin_id"])
            target_id = str(link["target_id"])
            if origin_id == "-10":
                slot = int(link["origin_slot"])
                input_target_links.setdefault(slot, []).append(link)
                continue
            if target_id == "-20":
                output_sources[int(link["target_slot"])] = [id_map.get(origin_id, origin_id), int(link["origin_slot"])]
                continue

            if origin_id in id_map and target_id in id_map:
                new_id = new_link_id()
                expanded_links.append(link_tuple(new_id, id_map[origin_id], int(link["origin_slot"]), id_map[target_id], int(link["target_slot"]), link["type"]))
                link_assignments.append((id_map[target_id], int(link["target_slot"]), new_id))

        inner_nodes = []
        for inner in subgraph.get("nodes", []) or []:
            cloned = copy.deepcopy(inner)
            cloned["id"] = id_map[str(inner.get("id"))]
            for input_info in cloned.get("inputs", []) or []:
                if input_info.get("link") is not None:
                    input_info["link"] = None
            inner_nodes.append(cloned)

        inner_by_id = {str(inner.get("id")): inner for inner in inner_nodes}
        for slot, links_for_slot in input_target_links.items():
            outer_input = external_inputs[slot] if slot < len(external_inputs) else {}
            outer_link_id = outer_input.get("link")
            if outer_link_id is not None and int(outer_link_id) in outer_links:
                source = outer_links[int(outer_link_id)]
                for link in links_for_slot:
                    target = id_map.get(str(link["target_id"]))
                    if not target:
                        continue
                    new_id = new_link_id()
                    expanded_links.append(link_tuple(new_id, source[0], source[1], target, int(link["target_slot"]), link["type"]))
                    link_assignments.append((target, int(link["target_slot"]), new_id))
            else:
                value = external_widgets[slot] if slot < len(external_widgets) else None
                for link in links_for_slot:
                    target = id_map.get(str(link["target_id"]))
                    if not target or value is None:
                        continue
                    target_node = inner_by_id.get(str(target))
                    if not target_node:
                        continue
                    widgets = target_node.setdefault("widgets_values", [])
                    while len(widgets) <= int(link["target_slot"]):
                        widgets.append(None)
                    widgets[int(link["target_slot"])] = value

        subgraph_output_sources[node_id] = output_sources
        expanded_nodes.extend(inner_nodes)

    for raw_link in workflow.get("links", []) or []:
        if not isinstance(raw_link, list) or len(raw_link) < 6:
            continue
        link_id, origin_id, origin_slot, target_id, target_slot, link_type = raw_link[:6]
        output_sources = subgraph_output_sources.get(str(origin_id))
        if not output_sources:
            continue
        source = output_sources.get(int(origin_slot))
        if not source:
            continue
        new_id = new_link_id()
        expanded_links.append(link_tuple(new_id, source[0], source[1], target_id, target_slot, link_type))
        link_assignments.append((str(target_id), int(target_slot), new_id))

    workflow["nodes"] = expanded_nodes
    workflow["links"] = expanded_links
    nodes_by_id = {str(node.get("id")): node for node in workflow.get("nodes", [])}
    for target_id, target_slot, link_id in link_assignments:
        target_node = nodes_by_id.get(str(target_id))
        if not target_node:
            continue
        inputs = target_node.get("inputs", []) or []
        if 0 <= int(target_slot) < len(inputs):
            inputs[int(target_slot)]["link"] = link_id
    if any(node.get("type") in definitions for node in workflow.get("nodes", [])):
        return _expand_subgraphs(workflow, depth + 1)
    return workflow


def _find_final_vae_decode_id(prompt):
    video_combine_id = None
    for candidate in ("353", "142"):
        if candidate in prompt and prompt[candidate].get("class_type") == "VHS_VideoCombine":
            video_combine_id = candidate
            break
    if not video_combine_id:
        video_combine_id = _api_node_id_by_class(prompt, "VHS_VideoCombine", fallback="142")

    if video_combine_id and video_combine_id in prompt:
        images_ref = prompt[video_combine_id].get("inputs", {}).get("images")
        if isinstance(images_ref, list) and len(images_ref) >= 1:
            curr_id = str(images_ref[0])
            visited = set()
            while curr_id and curr_id in prompt and curr_id not in visited:
                visited.add(curr_id)
                curr_node = prompt.get(curr_id, {})
                if curr_node.get("class_type") in ("VAEDecode", "MiniMaxH3AVDecodeT8"):
                    return curr_id
                upstream = curr_node.get("inputs", {}).get("images") or curr_node.get("inputs", {}).get("image")
                if isinstance(upstream, list) and len(upstream) >= 1:
                    curr_id = str(upstream[0])
                else:
                    break
    return _api_node_id_by_class(prompt, "VAEDecode", fallback="122")


def _remap_api_prompt_references(prompt, prefix):
    """Copy an API prompt under collision-free IDs and rewrite socket links."""
    mapping = {str(node_id): f"{prefix}{str(node_id).replace(':', '_')}" for node_id in prompt}

    def remap(value):
        if isinstance(value, list):
            if len(value) == 2 and str(value[0]) in mapping and isinstance(value[1], int):
                return [mapping[str(value[0])], value[1]]
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value

    return {mapping[str(node_id)]: remap(copy.deepcopy(node)) for node_id, node in prompt.items()}, mapping


def _prune_api_prompt_to_roots(prompt, roots):
    """Keep only API nodes required to produce the requested output roots."""
    required = set()

    def visit(node_id):
        node_id = str(node_id)
        if node_id in required or node_id not in prompt:
            return
        required.add(node_id)
        node = prompt.get(node_id) or {}
        inputs = node.get("inputs") or {}
        for value in inputs.values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) and str(value[0]) in prompt:
                visit(value[0])

    for root in roots:
        visit(root)
    return {node_id: node for node_id, node in prompt.items() if node_id in required}
