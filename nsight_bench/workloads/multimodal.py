"""Multimodal workload -- interface wired, input generation deliberately minimal.

Vision and audio models are in scope for this harness but were not the first target, so this
module implements the contract and the plumbing without pretending to cover every processor
API. It is written to be filled in, not worked around.

What already works: the phase structure, NVTX placement, memory tracking and the ncu scoping
convention are all inherited from :class:`TextGenerationWorkload`, and none of them are
text-specific. A multimodal run splits into prefill and decode exactly the same way -- with
the wrinkle that its prefill also contains the vision or audio encoder, which is worth
annotating separately because its memory behaviour is nothing like the language model's.

What needs doing: :meth:`_build_multimodal_inputs` has to construct whatever the specific
checkpoint's ``AutoProcessor`` expects. There is no single shape for that across Qwen-VL,
Nemotron-Omni and the rest, which is why it is not guessed at here.
"""

from __future__ import annotations

from typing import Any

from ..backends.base import Backend
from ..instrumentation.memory import MemorySampler
from ..instrumentation.nvtx import nvtx_range
from .base import ProfileMode, WorkloadResult, register
from .text_generation import TextGenerationWorkload

#: NVTX range for the vision/audio encoder. Separate from the language prefill because the
#: encoder is a different memory regime -- convolutional or patch-embedding traffic with very
#: different L2 reuse than transformer GEMMs.
ENCODER_RANGE = "nsbench.encoder"


@register
class MultimodalWorkload(TextGenerationWorkload):
    """Image/audio + text prefill followed by text decode."""

    kind = "multimodal"

    def run(
        self,
        backend: Backend,
        mode: ProfileMode = ProfileMode.BASELINE,
        sampler: MemorySampler | None = None,
    ) -> WorkloadResult:
        cfg = self.config
        if cfg.image_count == 0 and cfg.audio_seconds == 0:
            # No media requested: this is a text run wearing a different name. Fall through
            # rather than failing, so a mixed sweep config does not need special-casing.
            result = super().run(backend, mode, sampler)
            result.notes.append(
                "multimodal workload ran text-only (image_count=0, audio_seconds=0)"
            )
            return result

        raise NotImplementedError(
            "Multimodal input generation is not implemented yet.\n"
            "  The phase structure, NVTX scoping and memory tracking are all in place and\n"
            "  reusable -- what is missing is building the processor inputs for a specific\n"
            "  checkpoint, which differs across Qwen-VL, Nemotron-Omni and others.\n"
            "  Implement _build_multimodal_inputs() in workloads/multimodal.py, and add a\n"
            "  matching prepare_inputs() path to the backend.\n"
            "  For now, set workload.kind='text-generation' to benchmark the language model."
        )

    def _build_multimodal_inputs(self, backend: Backend) -> Any:
        """Construct processor inputs for the checkpoint under test.

        Implementation sketch, kept here so the next person does not start from nothing::

            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                backend.model_config.path,
                trust_remote_code=backend.model_config.trust_remote_code,
            )
            images = [synthetic_image(*self.config.image_size)
                      for _ in range(self.config.image_count)]
            return processor(images=images, text=prompt, return_tensors="pt").to("cuda")

        Use a deterministic synthetic image rather than a real one: content does not affect
        the kernel mix, and a generated tensor keeps runs reproducible without shipping
        binary fixtures.
        """
        raise NotImplementedError

    def _encode_media(self, backend: Backend, inputs: Any) -> Any:
        """Run the vision/audio encoder inside its own NVTX range.

        Kept separate so the report can attribute encoder traffic independently of the
        language model's -- on a VLM the encoder can dominate prefill entirely.
        """
        with nvtx_range(ENCODER_RANGE):
            raise NotImplementedError
