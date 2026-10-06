"""Force or observe mixture-of-experts routing, to bound expert-weight traffic.

How many bytes a MoE layer reads depends on how many *distinct* experts the tokens of one
forward pass route to, not on how many tokens there are. With top-k of E experts and T tokens
per pass, that count sits anywhere between k (every token picks the same experts) and
min(E, k*T) (no two tokens share one). The router decides where in that range a real run
lands, so a measurement of natural routing alone cannot say how much of its traffic is the
routing and how much is the hardware. Forcing the two extremes gives the bounds:

* ``fixed`` -- every token routes to experts ``0..k-1`` in every layer. The lower bound.
* ``disjoint`` -- the tokens of a pass take consecutive, non-overlapping blocks of k experts,
  and the blocks advance on every pass, so no expert is reused while any is unused. The
  upper bound.
* ``natural`` -- the model's own router, untouched.

Forcing is done with a forward hook on each gate module, replacing its ``(topk_idx,
topk_weight)`` output after the gate has run. The checkpoint and its modelling code are not
edited. The gate's own scoring still runs, so its kernels stay in the timeline; the experts
then process tokens the model never chose, which makes the generated text meaningless but
leaves the weight traffic real. Forced weights are uniform at the gate's own per-token total
(for Kimi, ``routed_scaling_factor``), so activations stay at a sane magnitude.

The observer records, per layer and phase, how many distinct experts each pass used and how
often each expert was chosen. It costs a few small kernels per MoE layer, so it is installed
only for baseline runs, where it does not disturb a profiler's kernel accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROUTING_MODES = ("natural", "fixed", "disjoint")


def is_routing_gate(module: Any) -> bool:
    """Whether ``module`` is a top-k router returning ``(topk_idx, topk_weight)``.

    Matched by shape rather than by class, so any DeepSeek-V3-style gate qualifies -- Kimi's
    ``KimiMoEGate`` among them. Routers that return logits for the block to top-k itself
    (Qwen3-MoE, Mixtral) do not, and are left alone.
    """
    return (
        type(module).__name__.endswith("Gate")
        and isinstance(getattr(module, "top_k", None), int)
        and isinstance(getattr(module, "num_experts", getattr(module, "n_routed_experts", None)), int)
    )


def _num_experts(gate: Any) -> int:
    return int(getattr(gate, "num_experts", None) or getattr(gate, "n_routed_experts"))


def expected_distinct_experts(mode: str, tokens_per_pass: int, num_experts: int, top_k: int) -> float:
    """Distinct experts one layer reads in a pass of ``tokens_per_pass`` tokens.

    ``natural`` assumes each token draws its k experts uniformly and independently, which is
    the right baseline but not a guarantee -- real routers favour some experts, which pulls
    the true count below this. The observer measures it.
    """
    tokens = max(1, tokens_per_pass)
    if mode == "fixed":
        return float(top_k)
    if mode == "disjoint":
        return float(min(num_experts, top_k * tokens))
    return num_experts * (1.0 - (1.0 - top_k / num_experts) ** tokens)


@dataclass
class _PhaseStats:
    """Device-side accumulators for one phase. Read back only in :meth:`summary`."""

    histogram: Any = None          # [layers, experts] int64 -- times each expert was chosen
    distinct: Any = None           # [layers] int64 -- sum over passes of distinct experts
    passes: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)


class RoutingController:
    """Installed forward hooks on every gate, plus what they observed."""

    def __init__(self, gates: list[Any], mode: str, observe: bool) -> None:
        self.gates = gates
        self.mode = mode
        self.observe = observe
        #: Set by the backend before each forward pass, so observations split by phase.
        self.phase = "prefill"
        self.num_experts = _num_experts(gates[0]) if gates else 0
        self.top_k = int(gates[0].top_k) if gates else 0
        self._calls = [0] * len(gates)
        self._stats: dict[str, _PhaseStats] = {}
        self._handles = [
            gate.register_forward_hook(self._make_hook(i)) for i, gate in enumerate(gates)
        ]

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ---- the hook ---------------------------------------------------------------------

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            idx, weight = output
            if self.mode != "natural":
                idx, weight = self._forced(layer, idx, weight)
            if self.observe:
                self._record(layer, idx)
            self._calls[layer] += 1
            return (idx, weight) if self.mode != "natural" else None
        return hook

    def _forced(self, layer: int, idx: Any, weight: Any) -> tuple[Any, Any]:
        """Replacement ``(topk_idx, topk_weight)`` of the same shape, dtype and device.

        Built from host-side integers only (shapes and a call counter), so forcing adds no
        device-to-host synchronisation.
        """
        import torch

        tokens, k = idx.shape
        e = self.num_experts
        if self.mode == "fixed":
            forced = torch.arange(k, device=idx.device, dtype=idx.dtype).expand(tokens, k)
        else:
            # Token t takes experts offset + t*k .. offset + t*k + k-1, wrapping at E. The
            # offset advances by the whole pass each call, so successive steps rotate through
            # the expert set rather than re-reading the same block.
            offset = (self._calls[layer] * tokens * k) % e
            forced = (torch.arange(tokens * k, device=idx.device, dtype=idx.dtype) + offset) % e
            forced = forced.view(tokens, k)
        uniform = (weight.sum(dim=-1, keepdim=True) / k).expand(tokens, k)
        return forced.contiguous(), uniform.contiguous()

    def _record(self, layer: int, idx: Any) -> None:
        import torch

        stats = self._stats.get(self.phase)
        if stats is None:
            n = len(self.gates)
            stats = _PhaseStats(
                histogram=torch.zeros(n, self.num_experts, dtype=torch.int64, device=idx.device),
                distinct=torch.zeros(n, dtype=torch.int64, device=idx.device),
                passes=[0] * n, tokens=[0] * n,
            )
            self._stats[self.phase] = stats
        counts = torch.bincount(idx.reshape(-1), minlength=self.num_experts)
        stats.histogram[layer] += counts
        stats.distinct[layer] += (counts > 0).sum()
        stats.passes[layer] += 1
        stats.tokens[layer] += idx.shape[0]

    # ---- reporting --------------------------------------------------------------------

    def summary(self) -> dict:
        """What the run's manifest records. Synchronises, so call it after measurement."""
        out: dict[str, Any] = {
            "mode": self.mode,
            "gates_hooked": len(self.gates),
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "observed": self.observe,
        }
        phases: dict[str, Any] = {}
        for phase, stats in self._stats.items():
            passes = max(stats.passes) or 1
            tokens_per_pass = stats.tokens[0] / max(1, stats.passes[0])
            distinct = [
                d / max(1, p) for d, p in zip(stats.distinct.tolist(), stats.passes)
            ]
            histogram = stats.histogram.float()
            # Share of all assignments that went to each layer's k most-chosen experts: k/E
            # for perfectly even routing, 1.0 when routing never leaves one set.
            top_share = (
                histogram.topk(self.top_k, dim=1).values.sum(1)
                / histogram.sum(1).clamp(min=1)
            ).tolist()
            phases[phase] = {
                "passes": passes,
                "tokens_per_pass": tokens_per_pass,
                "distinct_experts_per_pass": {
                    "mean": sum(distinct) / len(distinct),
                    "min": min(distinct),
                    "max": max(distinct),
                    "per_layer": [round(d, 2) for d in distinct],
                },
                "expected_distinct_experts_per_pass": round(expected_distinct_experts(
                    self.mode, round(tokens_per_pass), self.num_experts, self.top_k), 2),
                "experts_ever_used_per_layer": (stats.histogram > 0).sum(1).tolist(),
                "top_k_share_per_layer": [round(s, 4) for s in top_share],
            }
        out["phases"] = phases
        return out


def install_routing(model: Any, mode: str, observe: bool = True) -> RoutingController | None:
    """Hook every routing gate in ``model``; ``None`` when there is nothing to do.

    Raises when a forced mode is requested but no gate is found: silently running natural
    routing under a ``fixed`` or ``disjoint`` label would produce bounds that are not bounds.
    """
    if mode not in ROUTING_MODES:
        raise ValueError(f"routing must be one of {ROUTING_MODES}, got {mode!r}")
    if mode == "natural" and not observe:
        return None
    gates = [m for m in model.modules() if is_routing_gate(m)]
    if not gates:
        if mode != "natural":
            raise RuntimeError(
                f"routing={mode!r} requested, but no top-k gate returning (topk_idx, "
                "topk_weight) was found in this model. Forced routing supports "
                "DeepSeek-V3-style gates (e.g. Kimi-Linear's KimiMoEGate) only."
            )
        return None
    return RoutingController(gates, mode, observe)
