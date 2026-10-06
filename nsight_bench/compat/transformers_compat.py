"""Bridge ``trust_remote_code`` models written for transformers 4.5x onto transformers 5.x.

Found while bringing up Kimi-Linear-48B-A3B (written against transformers 4.57) on 5.18.
Each shim below names the exact API it restores, so it can be deleted once upstream
checkpoints catch up. None of them changes what a model computes: they re-export moved
symbols, translate renamed keywords, and give an old cache class the methods 5.x calls.

Two further adaptations are not API shims but loading decisions, kept here because they
are equally specific to this class of checkpoint:

* **FP8 compressed-tensors** checkpoints must be loaded with ``use_optimized_inference``.
  Without it transformers keeps the weights compressed only until the first forward pass,
  then unpacks them to bf16 -- a 50 GB checkpoint becomes ~98 GB resident, and every
  measurement after that describes a bf16 model.
* **Hard-coded flash_attention_2.** Some remote models force it in ``__init__`` regardless
  of the requested implementation. Where flash-attn is not installed the attention layers
  are switched to SDPA, which was checked to match the model's eager reference path
  (0.33% logit difference, identical argmax, on a reduced Kimi-Linear).
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import Any

_APPLIED: list[str] = []


# --------------------------------------------------------------------------------------
# Import-time shims
# --------------------------------------------------------------------------------------


def apply_remote_code_shims() -> list[str]:
    """Restore transformers 4.5x symbols that remote modelling code imports.

    Must run before the remote module is imported -- i.e. before ``AutoTokenizer`` or
    ``AutoModel*.from_pretrained`` -- because the missing names fail at import time.
    Idempotent; returns the names of every shim that is in effect.
    """
    _shim_output_recorder()
    _shim_bytes_to_unicode()
    _shim_create_causal_mask()
    return list(_APPLIED)


def _mark(name: str) -> None:
    if name not in _APPLIED:
        _APPLIED.append(name)


def _shim_output_recorder() -> None:
    """``OutputRecorder`` moved from ``utils.generic`` to ``utils.output_capturing``."""
    try:
        import transformers.utils.generic as generic
    except Exception:                                            # noqa: BLE001
        return
    if hasattr(generic, "OutputRecorder"):
        return
    try:
        from transformers.utils.output_capturing import OutputRecorder
    except Exception:                                            # noqa: BLE001
        return
    generic.OutputRecorder = OutputRecorder
    _mark("transformers.utils.generic.OutputRecorder")


def _shim_bytes_to_unicode() -> None:
    """``bytes_to_unicode`` left ``models.gpt2.tokenization_gpt2``; tiktoken tokenizers import it."""
    try:
        import transformers.models.gpt2.tokenization_gpt2 as gpt2
    except Exception:                                            # noqa: BLE001
        return
    if hasattr(gpt2, "bytes_to_unicode"):
        return
    try:
        from transformers.convert_slow_tokenizer import bytes_to_unicode
    except Exception:                                            # noqa: BLE001
        return
    gpt2.bytes_to_unicode = bytes_to_unicode
    _mark("transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode")


def _shim_create_causal_mask() -> None:
    """``create_causal_mask(input_embeds=, cache_position=)`` became ``inputs_embeds=``.

    ``cache_position`` was dropped outright; 5.x derives the offset from the cache instead
    (see :func:`adapt_model_cache_api`). Remote code binds the function at import with
    ``from transformers.masking_utils import create_causal_mask``, so the module attribute
    is what has to be replaced.
    """
    try:
        import transformers.masking_utils as masking
    except Exception:                                            # noqa: BLE001
        return
    original = masking.create_causal_mask
    if getattr(original, "_nsbench_compat", False):
        return
    parameters = inspect.signature(original).parameters
    if "input_embeds" in parameters and "cache_position" in parameters:
        return

    def create_causal_mask(*args, input_embeds=None, cache_position=None, **kwargs):
        if input_embeds is not None and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = input_embeds
        if cache_position is not None and "cache_position" in parameters:
            kwargs["cache_position"] = cache_position
        return original(*args, **kwargs)

    create_causal_mask._nsbench_compat = True                   # type: ignore[attr-defined]
    create_causal_mask.__wrapped__ = original                   # type: ignore[attr-defined]
    masking.create_causal_mask = create_causal_mask
    _mark("transformers.masking_utils.create_causal_mask(input_embeds=, cache_position=)")


# --------------------------------------------------------------------------------------
# Post-load adaptation
# --------------------------------------------------------------------------------------


def _remote_cache_classes(model: Any) -> list[type]:
    """Cache classes defined in the model's own module, rather than imported from transformers."""
    module = sys.modules.get(type(model).__module__)
    if module is None:
        return []
    found = []
    for name, obj in vars(module).items():
        if (
            inspect.isclass(obj)
            and name.endswith("Cache")
            and obj.__module__ == module.__name__
            and callable(getattr(obj, "update", None))
        ):
            found.append(obj)
    return found


def model_manages_own_cache(model: Any) -> bool:
    """Whether the model must be allowed to create its own cache.

    Hybrid models carry state no generic KV cache has a slot for: Kimi-Linear's KDA layers
    keep a recurrent state and short-convolution state per layer, in a ``KimiDynamicCache``
    the model builds when handed ``past_key_values=None`` -- and asserts on when handed a
    transformers ``DynamicCache``. A model whose own module defines a cache class is
    treated as owning its cache.
    """
    return bool(_remote_cache_classes(model))


def adapt_model_cache_api(model: Any) -> list[str]:
    """Give a remote model's 4.5x-era cache class the methods 5.x masking calls.

    5.x calls ``cache.get_query_offset(layer_idx)`` and
    ``cache.get_mask_sizes(query_length: int, layer_idx)`` -- 4.5x passed the
    ``cache_position`` tensor as the first argument -- and reads ``cache.is_sliding``.
    Returns the names of the classes adapted.
    """
    import torch

    adapted = []
    for cls in _remote_cache_classes(model):
        if getattr(cls, "_nsbench_compat", False):
            adapted.append(cls.__name__)
            continue
        if not callable(getattr(cls, "get_seq_length", None)):
            continue

        if not hasattr(cls, "get_query_offset"):
            def get_query_offset(self, layer_idx=0):
                return self.get_seq_length(layer_idx)
            cls.get_query_offset = get_query_offset

        old_mask_sizes = getattr(cls, "get_mask_sizes", None)
        takes_tensor = old_mask_sizes is not None and "cache_position" in inspect.signature(
            old_mask_sizes
        ).parameters
        if old_mask_sizes is None or takes_tensor:
            def get_mask_sizes(self, query_length, layer_idx=0):
                if isinstance(query_length, torch.Tensor):
                    query_length = query_length.shape[0]
                return int(query_length) + self.get_seq_length(layer_idx), 0
            cls.get_mask_sizes = get_mask_sizes

        if not hasattr(cls, "is_sliding"):
            def is_sliding(self):
                return [False] * len(self)
            cls.is_sliding = property(is_sliding)

        cls._nsbench_compat = True
        adapted.append(cls.__name__)
    return adapted


def use_sdpa_if_flash_attn_missing(model: Any) -> str | None:
    """Switch a model that forced ``flash_attention_2`` to SDPA when flash-attn is absent.

    The attention layers read ``config._attn_implementation`` at call time, so setting it on
    every config object after construction takes effect. Returns the implementation that
    was replaced, or ``None`` when nothing changed.
    """
    config = getattr(model, "config", None)
    if getattr(config, "_attn_implementation", None) != "flash_attention_2":
        return None
    if importlib.util.find_spec("flash_attn") is not None:
        return None

    seen: set[int] = set()
    for module in model.modules():
        module_config = getattr(module, "config", None)
        if module_config is None or id(module_config) in seen:
            continue
        seen.add(id(module_config))
        if hasattr(module_config, "_attn_implementation"):
            module_config._attn_implementation = "sdpa"
    return "flash_attention_2"


# --------------------------------------------------------------------------------------
# Checkpoint inspection
# --------------------------------------------------------------------------------------


def is_fp8_compressed_tensors(config: dict) -> bool:
    """Whether a checkpoint's weights are FP8 under the compressed-tensors format.

    True only when every config group quantizes weights to 8-bit floats; a mixed scheme
    (FP8 attention with INT4 experts, say) is not, since the optimised loader would leave
    the INT4 part on the unpack-at-first-forward route.
    """
    quant = config.get("quantization_config") if isinstance(config, dict) else None
    if not isinstance(quant, dict) or str(quant.get("quant_method", "")).lower() != "compressed-tensors":
        return False
    groups = quant.get("config_groups") or {}
    if not groups:
        return False
    for group in groups.values():
        weights = (group or {}).get("weights") or {}
        if str(weights.get("type", "")).lower() != "float" or int(weights.get("num_bits") or 0) != 8:
            return False
    return True
