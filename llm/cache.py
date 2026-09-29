"""Shared LLM model caches, the GGUF lock, and cache clearing used by the engines and the memory-cleanup route."""

import torch
import gc
import threading


_HF_PIPELINE_CACHE: dict[tuple, tuple] = {}


_GGUF_MODEL_CACHE: dict[tuple, object] = {}


_GGUF_CHAT_HANDLER_CACHE: dict[tuple, object] = {}


# Held while a cached GGUF model is in use and while entries are removed, so routes and nodes on
# other threads cannot close a llama.cpp model mid-generation.
_GGUF_LOCK = threading.RLock()


def _close_possible_resource(obj) -> None:
    if obj is None:
        return
    for name in ("close", "free"):
        close_fn = getattr(obj, name, None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass
            return


def _close_gguf_cached_resources(model, chat_handler=None) -> None:
    _close_possible_resource(model)
    _close_possible_resource(chat_handler)
    for attr in ("clip_model", "_clip_model", "clip_ctx", "_clip_ctx", "model"):
        try:
            _close_possible_resource(getattr(chat_handler, attr, None))
        except Exception:
            pass


def _clear_vrgdg_llm_caches(clear_cuda_cache: bool = True, clear_hf_pipeline_cache: bool = False) -> dict:
    hf_count = len(_HF_PIPELINE_CACHE) if clear_hf_pipeline_cache else 0

    with _GGUF_LOCK:
        gguf_count = len(_GGUF_MODEL_CACHE)
        for key, model in list(_GGUF_MODEL_CACHE.items()):
            chat_handler = _GGUF_CHAT_HANDLER_CACHE.pop(key, None)
            _close_gguf_cached_resources(model, chat_handler)
            del model, chat_handler
        _GGUF_MODEL_CACHE.clear()
        _GGUF_CHAT_HANDLER_CACHE.clear()

    if clear_hf_pipeline_cache:
        _HF_PIPELINE_CACHE.clear()

    gc.collect()
    cuda_cleared = False
    if clear_cuda_cache:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
                cuda_cleared = True
        except Exception:
            pass

    return {
        "gguf_models_unloaded": gguf_count,
        "hf_pipelines_unloaded": hf_count,
        "cuda_cache_cleared": cuda_cleared,
    }
