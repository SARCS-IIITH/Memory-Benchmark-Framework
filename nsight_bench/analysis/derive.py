"""Derive physical quantities from raw Nsight Compute counters.

Counters are sectors, wavefronts and instructions. What anyone actually wants is bytes,
hit rates, bandwidth and arithmetic intensity. This module does that conversion, and it is
where the GB10-specific reasoning lives.

Two rules govern everything here.

**Rates are re-derived, never averaged.** ``lts__t_sector_hit_rate.pct`` is a per-kernel
percentage. Taking the mean across kernels weights a 200-sector elementwise kernel the same
as a GEMM moving hundreds of megabytes, which is why aggregate hit rates are computed as
``sum(hits) / sum(hits + misses)`` from the raw counts instead.

**A missing counter is not a zero.** On GB10 there are no ``dram__*`` counters at all. If an
absent counter were read as 0.0 the report would confidently state that a decode step moved
no data through memory. Missing values propagate as ``None`` and surface as "not measured".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..metrics import SECTOR_BYTES, WAVEFRONT_BYTES, Level
from ..parsers.ncu_parse import KernelRecord, NcuReport


def _sum(kernels: list[KernelRecord], metric: str) -> float | None:
    """Sum a metric across kernels, returning None if no kernel reported it.

    The distinction matters: 0.0 means "measured, and it was zero"; None means "this GPU
    does not expose that counter".
    """
    values = [k.metrics[metric] for k in kernels if metric in k.metrics]
    return sum(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _pct(part: float | None, whole: float | None) -> float | None:
    ratio = _ratio(part, whole)
    return ratio * 100.0 if ratio is not None else None


@dataclass
class LevelSummary:
    """Traffic and behaviour at one rung of the memory hierarchy."""

    level: Level
    bytes_total: float | None = None
    bytes_read: float | None = None
    bytes_write: float | None = None
    sectors: float | None = None
    requests: float | None = None
    hit_rate_pct: float | None = None
    detail: dict[str, float | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.level.label

    @property
    def measured(self) -> bool:
        return self.bytes_total is not None

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "label": self.label,
            "bytes_total": self.bytes_total,
            "bytes_read": self.bytes_read,
            "bytes_write": self.bytes_write,
            "sectors": self.sectors,
            "requests": self.requests,
            "hit_rate_pct": self.hit_rate_pct,
            **{f"detail_{k}": v for k, v in self.detail.items()},
        }


@dataclass
class HierarchySummary:
    """The full memory picture for one phase."""

    scope: str = ""
    kernel_count: int = 0
    gpu_time_ns: float = 0.0
    levels: dict[Level, LevelSummary] = field(default_factory=dict)

    # ---- work ----
    tensor_ops: float | None = None
    scalar_flops: float | None = None

    # ---- integrity ----
    sentinels_ok: bool = True
    sentinel_detail: dict[str, float | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ---- helpers -----------------------------------------------------------------------

    def level(self, which: Level) -> LevelSummary:
        return self.levels.get(which, LevelSummary(level=which))

    @property
    def dram_bytes(self) -> float | None:
        return self.level(Level.DRAM).bytes_total

    @property
    def gpu_time_s(self) -> float:
        return self.gpu_time_ns / 1e9

    @property
    def achieved_dram_bandwidth_gbps(self) -> float | None:
        """Bytes past L2 divided by GPU-busy time, in GB/s.

        Uses summed kernel duration rather than wall clock: wall clock for an ncu collection
        includes replay and profiler overhead, which would understate bandwidth by an order
        of magnitude.
        """
        if self.dram_bytes is None or self.gpu_time_s <= 0:
            return None
        return self.dram_bytes / self.gpu_time_s / 1e9

    @property
    def total_flops(self) -> float | None:
        """Best available FLOP count.

        Tensor-path ops dominate transformer GEMMs; the scalar FMA counters cover the rest
        (normalisation, activation, attention softmax) at 2 FLOPs per instruction.
        """
        parts = [p for p in (self.tensor_ops, self.scalar_flops) if p is not None]
        return sum(parts) if parts else None

    @property
    def arithmetic_intensity(self) -> float | None:
        """FLOPs per byte fetched from memory -- the roofline x-coordinate."""
        return _ratio(self.total_flops, self.dram_bytes)

    def amplification(self) -> dict[str, float | None]:
        """How traffic grows or shrinks between rungs.

        ``l2_to_dram`` is the one that matters most for LLM decode: it is the fraction of
        what L1 asked of L2 that L2 could not satisfy and had to fetch from LPDDR5X. A value
        near 1.0 means L2 is providing no reuse at all.
        """
        l1 = self.level(Level.L1TEX).bytes_total
        l2 = self.level(Level.L2).bytes_total
        dram = self.level(Level.DRAM).bytes_total
        return {
            "sm_to_l2": _ratio(l2, l1),
            "l2_to_dram": _ratio(dram, l2),
            "sm_to_dram": _ratio(dram, l1),
        }

    def bytes_per_token(self, tokens: int) -> dict[str, float | None]:
        """Bytes moved at each level per generated token.

        This is the headline comparison number across models and quantization schemes: it
        normalises away batch size and sequence length, so a 4-bit and a 16-bit checkpoint
        can be put side by side.
        """
        if tokens <= 0:
            return {}
        out: dict[str, float | None] = {}
        for which in (Level.LOCAL, Level.SHARED, Level.L1TEX, Level.L2, Level.DRAM):
            total = self.level(which).bytes_total
            out[which.value] = total / tokens if total is not None else None
        return out

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "kernel_count": self.kernel_count,
            "gpu_time_ns": self.gpu_time_ns,
            "levels": {lvl.value: summary.to_dict() for lvl, summary in self.levels.items()},
            "tensor_ops": self.tensor_ops,
            "scalar_flops": self.scalar_flops,
            "total_flops": self.total_flops,
            "arithmetic_intensity_flops_per_byte": self.arithmetic_intensity,
            "achieved_dram_bandwidth_gbps": self.achieved_dram_bandwidth_gbps,
            "amplification": self.amplification(),
            "sentinels_ok": self.sentinels_ok,
            "sentinel_detail": self.sentinel_detail,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------------------
# The roll-up
# --------------------------------------------------------------------------------------


def summarize(report: NcuReport, scope: str | None = None) -> HierarchySummary:
    """Roll one ncu collection up into a :class:`HierarchySummary`."""
    kernels = report.kernels
    summary = HierarchySummary(
        scope=scope or report.scope,
        kernel_count=len(kernels),
        gpu_time_ns=_sum(kernels, "gpu__time_duration.sum") or 0.0,
    )
    summary.warnings.extend(report.warnings)

    if not kernels:
        summary.warnings.append("no kernels in this collection; nothing to summarise")
        return summary

    summary.levels[Level.REGISTER] = _registers(kernels)
    summary.levels[Level.LOCAL] = _local(kernels)
    summary.levels[Level.SHARED] = _shared(kernels)
    summary.levels[Level.L1TEX] = _l1(kernels)
    summary.levels[Level.L2] = _l2(kernels)
    summary.levels[Level.DRAM] = _dram(kernels, summary)

    summary.tensor_ops = _sum(kernels, "sm__ops_path_tensor_src_bf16_dst_fp32.sum")
    ffma = _sum(kernels, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
    hfma = _sum(kernels, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum")
    fma_parts = [p for p in (ffma, hfma) if p is not None]
    # An FMA instruction is two floating-point operations.
    summary.scalar_flops = 2.0 * sum(fma_parts) if fma_parts else None

    return summary


def _registers(kernels: list[KernelRecord]) -> LevelSummary:
    """Register file usage.

    Registers hold no traffic counter -- they are a capacity resource, not a cache -- so this
    reports occupancy pressure instead. Spill traffic, which is the part that costs memory
    bandwidth, is reported at the LOCAL level below.
    """
    per_thread = [
        k.metrics["launch__registers_per_thread"]
        for k in kernels if "launch__registers_per_thread" in k.metrics
    ]
    summary = LevelSummary(level=Level.REGISTER)
    if per_thread:
        summary.detail = {
            "max_registers_per_thread": max(per_thread),
            "mean_registers_per_thread": sum(per_thread) / len(per_thread),
            "kernels_at_register_limit": float(sum(1 for v in per_thread if v >= 255)),
        }
        summary.notes.append(
            "registers are a capacity resource; spill traffic is reported under Local"
        )
        if summary.detail["kernels_at_register_limit"]:
            summary.notes.append(
                f"{int(summary.detail['kernels_at_register_limit'])} kernel(s) at the 255 "
                "register/thread ceiling -- these are spilling to local memory"
            )
    return summary


def _local(kernels: list[KernelRecord]) -> LevelSummary:
    """Local memory: register spills.

    Local memory is physically the same LPDDR5X as global memory, so spills are not a
    register-file problem, they are a bandwidth problem. Traffic here is real DRAM traffic
    that produces no useful work.
    """
    ld = _sum(kernels, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum")
    st = _sum(kernels, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")
    parts = [p for p in (ld, st) if p is not None]
    sectors = sum(parts) if parts else None

    summary = LevelSummary(
        level=Level.LOCAL,
        sectors=sectors,
        bytes_total=sectors * SECTOR_BYTES if sectors is not None else None,
        bytes_read=ld * SECTOR_BYTES if ld is not None else None,
        bytes_write=st * SECTOR_BYTES if st is not None else None,
    )
    if sectors:
        summary.notes.append(
            "non-zero spill traffic: this consumes memory bandwidth without doing work"
        )
    return summary


def _shared(kernels: list[KernelRecord]) -> LevelSummary:
    """Shared memory.

    Bytes are approximated from wavefronts, because shared memory has no sector counter. The
    approximation is honest in one direction: bank conflicts inflate the wavefront count, so
    a conflict-heavy kernel reports more shared bytes than its data actually needs -- which
    is exactly the effect worth seeing.
    """
    wavefronts = _sum(kernels, "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum")
    ld_wf = _sum(kernels, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum")
    st_wf = _sum(kernels, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum")
    conflicts = _sum(kernels, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")
    ld_inst = _sum(kernels, "smsp__inst_executed_op_shared_ld.sum")
    st_inst = _sum(kernels, "smsp__inst_executed_op_shared_st.sum")

    static = [
        k.metrics["launch__shared_mem_per_block_static"]
        for k in kernels if "launch__shared_mem_per_block_static" in k.metrics
    ]
    dynamic = [
        k.metrics["launch__shared_mem_per_block_dynamic"]
        for k in kernels if "launch__shared_mem_per_block_dynamic" in k.metrics
    ]

    summary = LevelSummary(
        level=Level.SHARED,
        bytes_total=wavefronts * WAVEFRONT_BYTES if wavefronts is not None else None,
        bytes_read=ld_wf * WAVEFRONT_BYTES if ld_wf is not None else None,
        bytes_write=st_wf * WAVEFRONT_BYTES if st_wf is not None else None,
        detail={
            "wavefronts": wavefronts,
            "bank_conflicts": conflicts,
            "load_instructions": ld_inst,
            "store_instructions": st_inst,
            "max_static_bytes_per_block": max(static) if static else None,
            "max_dynamic_bytes_per_block": max(dynamic) if dynamic else None,
        },
    )
    summary.notes.append(
        f"bytes estimated as wavefronts x {WAVEFRONT_BYTES} B; there is no shared-memory "
        "sector counter"
    )

    total_inst = sum(p for p in (ld_inst, st_inst) if p is not None)
    if conflicts and total_inst:
        summary.detail["conflicts_per_instruction"] = conflicts / total_inst
        if conflicts / total_inst > 0.1:
            summary.notes.append(
                f"{conflicts / total_inst:.2f} bank conflicts per shared instruction -- "
                "accesses are serialising across banks"
            )
    return summary


def _l1(kernels: list[KernelRecord]) -> LevelSummary:
    sectors = _sum(kernels, "l1tex__t_sectors.sum")
    hits = _sum(kernels, "l1tex__t_sectors_lookup_hit.sum")
    misses = _sum(kernels, "l1tex__t_sectors_lookup_miss.sum")
    lookups = sum(p for p in (hits, misses) if p is not None) or None

    ld = _sum(kernels, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
    st = _sum(kernels, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")

    # Prefer the hardware byte counter when present; fall back to sectors x 32.
    bytes_total = _sum(kernels, "l1tex__t_bytes.sum")
    if bytes_total is None and sectors is not None:
        bytes_total = sectors * SECTOR_BYTES

    summary = LevelSummary(
        level=Level.L1TEX,
        bytes_total=bytes_total,
        bytes_read=ld * SECTOR_BYTES if ld is not None else None,
        bytes_write=st * SECTOR_BYTES if st is not None else None,
        sectors=sectors,
        requests=_sum(kernels, "l1tex__t_requests.sum"),
        # Recomputed from counts rather than averaging the per-kernel .pct values.
        hit_rate_pct=_pct(hits, lookups),
    )

    coalescing = [
        k.metrics["l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"]
        for k in kernels
        if k.metrics.get(
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 0
        ) > 0
    ]
    if coalescing:
        worst = max(coalescing)
        summary.detail["worst_sectors_per_global_load_request"] = worst
        summary.detail["mean_sectors_per_global_load_request"] = (
            sum(coalescing) / len(coalescing)
        )
        # A fully coalesced 32-thread warp load of 4-byte values touches 4 sectors.
        if worst > 8:
            summary.notes.append(
                f"worst-case {worst:.1f} sectors per global load request (4 is perfectly "
                "coalesced) -- some kernel is scattering its accesses"
            )
    return summary


def _l2(kernels: list[KernelRecord]) -> LevelSummary:
    sectors = _sum(kernels, "lts__t_sectors.sum")
    hits = _sum(kernels, "lts__t_sectors_lookup_hit.sum")
    misses = _sum(kernels, "lts__t_sectors_lookup_miss.sum")
    lookups = sum(p for p in (hits, misses) if p is not None) or None
    read = _sum(kernels, "lts__t_sectors_op_read.sum")
    write = _sum(kernels, "lts__t_sectors_op_write.sum")

    summary = LevelSummary(
        level=Level.L2,
        bytes_total=sectors * SECTOR_BYTES if sectors is not None else None,
        bytes_read=read * SECTOR_BYTES if read is not None else None,
        bytes_write=write * SECTOR_BYTES if write is not None else None,
        sectors=sectors,
        requests=_sum(kernels, "lts__t_requests.sum"),
        hit_rate_pct=_pct(hits, lookups),
        detail={"hit_sectors": hits, "miss_sectors": misses},
    )
    return summary


def _dram(kernels: list[KernelRecord], summary: HierarchySummary) -> LevelSummary:
    """Traffic reaching the unified LPDDR5X pool.

    There is no ``dram__*`` counter on GB10, so this is derived from the L2 sysmem-aperture
    miss counters. The derivation was validated against a streaming kernel of known size --
    512 MiB of traffic measured as 537.0 MB -- and the device/peer aperture counters are
    carried as sentinels that must stay zero for it to remain valid.
    """
    miss = _sum(kernels, "lts__t_sectors_aperture_sysmem_lookup_miss.sum")
    read = _sum(kernels, "lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum")
    write = _sum(kernels, "lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum")
    saved = _sum(kernels, "lts__t_sectors_aperture_sysmem_lookup_hit.sum")
    fill = _sum(kernels, "lts__d_sectors_fill_sysmem.sum")

    level = LevelSummary(
        level=Level.DRAM,
        bytes_total=miss * SECTOR_BYTES if miss is not None else None,
        bytes_read=read * SECTOR_BYTES if read is not None else None,
        bytes_write=write * SECTOR_BYTES if write is not None else None,
        sectors=miss,
        detail={
            "sectors_served_by_l2": saved,
            "fill_sectors_from_sysmem": fill,
            "bytes_saved_by_l2": saved * SECTOR_BYTES if saved is not None else None,
        },
    )
    level.notes.append(
        "derived from lts__t_sectors_aperture_sysmem_lookup_miss x 32 B; GB10 exposes no "
        "dram__* counters"
    )

    device_miss = _sum(kernels, "lts__t_sectors_aperture_device_lookup_miss.sum")
    peer_miss = _sum(kernels, "lts__t_sectors_aperture_peer_lookup_miss.sum")
    summary.sentinel_detail = {
        "device_aperture_miss_sectors": device_miss,
        "peer_aperture_miss_sectors": peer_miss,
    }
    if device_miss or peer_miss:
        summary.sentinels_ok = False
        summary.warnings.append(
            "SENTINEL TRIPPED: the device or peer L2 aperture reported non-zero traffic "
            f"(device={device_miss}, peer={peer_miss}). On this unified-memory part both "
            "must be zero. The sysmem-only DRAM derivation is therefore incomplete and the "
            "DRAM figures in this report understate real traffic."
        )
    return level


# --------------------------------------------------------------------------------------
# Expectation checks
# --------------------------------------------------------------------------------------


@dataclass
class DecodeExpectation:
    """Predicted versus measured DRAM traffic for one decode step.

    A decode step must read every weight and the whole KV cache to emit one token, so its
    DRAM traffic is predictable from first principles. Comparing prediction to measurement
    is the strongest available check that the NVTX scoping and the sector-to-byte derivation
    are both correct -- if they are not, this ratio goes badly wrong in an obvious way.
    """

    weight_bytes: int = 0
    kv_bytes: int = 0
    measured_bytes: float | None = None
    context_len: int | None = None
    #: "resident" (measured from the loaded model) or "on-disk" (the checkpoint's size).
    #: The two differ whenever a checkpoint stores tied embeddings twice, so the ratio below
    #: cannot be read without knowing which one it was computed against.
    weight_bytes_source: str = "resident"

    #: For a mixture of experts, the whole checkpoint -- against which ``weight_bytes`` is the
    #: routed-active subset. Zero for a dense model, where the two are the same thing.
    total_weight_bytes: int = 0
    expert_activation_ratio: float | None = None

    #: Recurrent state a linear-attention layer writes back in full every step (the read is
    #: already in ``kv_bytes``). Zero for models without linear-attention layers.
    state_write_bytes: int = 0

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_activation_ratio is not None and self.total_weight_bytes)

    @property
    def expected_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.state_write_bytes

    @property
    def ratio(self) -> float | None:
        if self.measured_bytes is None or not self.expected_bytes:
            return None
        return self.measured_bytes / self.expected_bytes

    def verdict(self) -> str:
        ratio = self.ratio
        if ratio is None:
            return "not evaluated (no measured DRAM traffic)"
        basis = (
            "the routed-active weights" if self.is_moe else "the model once per token"
        )
        if 0.75 <= ratio <= 1.6:
            return (
                f"consistent ({ratio:.2f}x expected) -- a decode step is reading roughly "
                f"{basis}, as it must"
            )
        if ratio < 0.75:
            if self.is_moe:
                return (
                    f"LOWER than expected ({ratio:.2f}x against the routed-active weights). "
                    "Either fewer experts are genuinely being touched than top-k implies -- "
                    "tokens in a batch can route to overlapping experts -- or L2 is holding "
                    "part of the hot expert set across steps."
                )
            return (
                f"LOWER than expected ({ratio:.2f}x). Either L2 is holding a real part of "
                "the weights across steps -- plausible here given 25 MB of L2 -- or the "
                "profiled NVTX range captured only part of the step."
            )
        return (
            f"HIGHER than expected ({ratio:.2f}x). The usual causes, most common first: a "
            "growing KV cache implementation that reallocates and copies the whole cache "
            "each step (transformers' DynamicCache concatenates K and V on every update, so "
            "the cache is read and rewritten rather than appended to); activation and "
            "workspace traffic; an unfused dequantization pass; or a KV cache physically "
            "larger than the analytic estimate. The backend's recorded kv_cache_class in the "
            "run manifest distinguishes the first from the rest."
        )

    def to_dict(self) -> dict:
        return {
            "weight_bytes": self.weight_bytes,
            "weight_bytes_source": self.weight_bytes_source,
            "is_moe": self.is_moe,
            "total_weight_bytes": self.total_weight_bytes or None,
            "expert_activation_ratio": self.expert_activation_ratio,
            "kv_bytes": self.kv_bytes,
            "state_write_bytes": self.state_write_bytes or None,
            "expected_bytes": self.expected_bytes,
            "measured_bytes": self.measured_bytes,
            "ratio": self.ratio,
            "ratio_vs_total_weights": (
                self.measured_bytes / self.total_weight_bytes
                if self.measured_bytes and self.total_weight_bytes else None
            ),
            "context_len": self.context_len,
            "verdict": self.verdict(),
        }


def check_decode_expectation(
    summary: HierarchySummary,
    weight_bytes: int,
    kv_bytes: int,
    context_len: int | None = None,
    weight_bytes_source: str = "resident",
    total_weight_bytes: int = 0,
    expert_activation_ratio: float | None = None,
    state_write_bytes: int = 0,
) -> DecodeExpectation:
    return DecodeExpectation(
        weight_bytes=weight_bytes,
        kv_bytes=kv_bytes,
        measured_bytes=summary.dram_bytes,
        context_len=context_len,
        weight_bytes_source=weight_bytes_source,
        total_weight_bytes=total_weight_bytes,
        expert_activation_ratio=expert_activation_ratio,
        state_write_bytes=state_write_bytes,
    )


def kernel_traffic_rows(report: NcuReport, top_n: int | None = None) -> list[dict]:
    """Per-kernel traffic table, ranked by bytes reaching LPDDR5X.

    Ranked by DRAM bytes rather than duration on purpose: this harness is about memory, and
    the kernel that moves the most data is not always the one that takes the most time.
    """
    rows: list[dict] = []
    for kernel in report.kernels:
        dram_sectors = kernel.metrics.get("lts__t_sectors_aperture_sysmem_lookup_miss.sum")
        l2_sectors = kernel.metrics.get("lts__t_sectors.sum")
        l1_sectors = kernel.metrics.get("l1tex__t_sectors.sum")
        duration = kernel.duration_ns

        dram_bytes = dram_sectors * SECTOR_BYTES if dram_sectors is not None else None
        rows.append({
            "kernel_short": kernel.short_name,
            "kernel_name": kernel.name,
            "grid": kernel.grid,
            "block": kernel.block,
            "duration_ns": duration,
            "l1_bytes": l1_sectors * SECTOR_BYTES if l1_sectors is not None else None,
            "l2_bytes": l2_sectors * SECTOR_BYTES if l2_sectors is not None else None,
            "dram_bytes": dram_bytes,
            "l1_hit_rate_pct": kernel.metrics.get("l1tex__t_sector_hit_rate.pct"),
            "l2_hit_rate_pct": kernel.metrics.get("lts__t_sector_hit_rate.pct"),
            "dram_bandwidth_gbps": (
                dram_bytes / (duration / 1e9) / 1e9
                if dram_bytes is not None and duration > 0 else None
            ),
            "registers_per_thread": kernel.metrics.get("launch__registers_per_thread"),
            "shared_bytes_per_block": (
                (kernel.metrics.get("launch__shared_mem_per_block_static") or 0)
                + (kernel.metrics.get("launch__shared_mem_per_block_dynamic") or 0)
            ),
            "local_spill_bytes": (
                ((kernel.metrics.get("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum") or 0)
                 + (kernel.metrics.get("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum") or 0))
                * SECTOR_BYTES
            ),
            "sm_throughput_pct": kernel.metrics.get(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            ),
            "mem_pipe_throughput_pct": kernel.metrics.get(
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
            ),
        })

    rows.sort(key=lambda r: r["dram_bytes"] or 0, reverse=True)
    return rows[:top_n] if top_n else rows
