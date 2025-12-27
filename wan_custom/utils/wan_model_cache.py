# wan/runtime/wan_model_cache.py
from __future__ import annotations

import threading

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = {}


def get_or_load(key: str, loader_fn):
    """
    Singleton model cache.
    """
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        model = loader_fn()
        _MODEL_CACHE[key] = model
        return model
