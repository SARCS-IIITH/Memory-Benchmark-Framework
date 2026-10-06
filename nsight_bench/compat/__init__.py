"""Compatibility adapters for checkpoints whose own code predates the installed libraries.

Checkpoints that ship ``trust_remote_code`` modelling files are pinned to whatever
transformers release they were written against, and the harness runs a newer one. Rather
than editing a downloaded snapshot -- which would change what is being benchmarked, and be
silently undone by the next fetch -- the gaps are bridged here, at runtime, and every
adapter that fires is recorded in the run manifest.
"""

from .transformers_compat import (
    adapt_model_cache_api,
    apply_remote_code_shims,
    is_fp8_compressed_tensors,
    model_manages_own_cache,
    use_sdpa_if_flash_attn_missing,
)

__all__ = [
    "adapt_model_cache_api",
    "apply_remote_code_shims",
    "is_fp8_compressed_tensors",
    "model_manages_own_cache",
    "use_sdpa_if_flash_attn_missing",
]
