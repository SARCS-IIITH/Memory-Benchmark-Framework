"""The memory-hierarchy metric registry.

This module is the single source of truth for *which* Nsight Compute counters we collect and
*what they mean* on this hardware. Everything downstream -- the ncu command line, the parsers,
the hierarchy roll-up, the reports -- is generated from the tables here.

Why this file is opinionated about GB10
---------------------------------------
Every NVIDIA profiling guide tells you to read DRAM traffic from ``dram__bytes.sum``. On the
DGX Spark's GB10 (chip GB20B, sm_121) that counter **does not exist**. Neither do the
``ctc__*`` NVLink-C2C counters -- Nsight Compute's C2CLink section is gated to CC_90/CC_100,
so it never applies to this part.

What does exist is the L2 (LTS) aperture breakdown. On this integrated part every access that
misses L2 is tagged with the ``sysmem`` aperture, because the GPU has no private VRAM: it
shares one coherent LPDDR5X pool with the Grace CPU. That was verified empirically -- a
streaming kernel writing 256 MiB reported::

    lts__t_sectors_aperture_sysmem_lookup_miss.sum = 8,395,760 sectors
    8,395,760 x 32 B                               = 268.7 MB   (256 MiB = 268.4 MB)
    lts__t_sectors_aperture_device_lookup_miss.sum = 0
    lts__t_sectors_aperture_peer_lookup_miss.sum   = 0

So ``sysmem_lookup_miss x 32 B`` is our DRAM-bytes substitute, and the ``device``/``peer``
aperture counters are kept as *sentinels*: they must remain zero. If they ever go non-zero the
derivation is no longer sound and the report says so loudly rather than quietly reporting a
wrong number.

Every metric marked ``TIER1`` below was confirmed to actually collect on this GPU, by running
a real one-kernel ncu collection -- not by trusting ``ncu --query-metrics``. That distinction
matters: ``launch__*`` metrics collect fine but are absent from the query listing, while
``derived__local_spilling_requests`` appears usable but is section-internal and fails as a
standalone ``--metrics`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# --------------------------------------------------------------------------------------
# Fundamental constants
# --------------------------------------------------------------------------------------

#: L1/L2 cache-line sector size in bytes. Constant across all current NVIDIA architectures;
#: this is the multiplier that converts every ``*_sectors`` counter into bytes.
SECTOR_BYTES = 32


class Level(str, Enum):
    """A rung of the memory hierarchy, ordered from closest-to-ALU to furthest."""

    REGISTER = "register"
    LOCAL = "local"          # register spills -- physically off-chip, logically per-thread
    SHARED = "shared"
    L1TEX = "l1tex"
    L2 = "l2"
    DRAM = "dram"            # on GB10: the unified LPDDR5X pool, reached via the sysmem aperture
    HOST = "host"            # allocator / process-level footprint, not an ncu counter

    @property
    def order(self) -> int:
        return _LEVEL_ORDER[self]

    @property
    def label(self) -> str:
        return _LEVEL_LABELS[self]


_LEVEL_ORDER = {
    Level.REGISTER: 0,
    Level.LOCAL: 1,
    Level.SHARED: 2,
    Level.L1TEX: 3,
    Level.L2: 4,
    Level.DRAM: 5,
    Level.HOST: 6,
}

_LEVEL_LABELS = {
    Level.REGISTER: "Registers",
    Level.LOCAL: "Local (spill)",
    Level.SHARED: "Shared memory",
    Level.L1TEX: "L1 / TEX cache",
    Level.L2: "L2 cache",
    Level.DRAM: "Unified LPDDR5X",
    Level.HOST: "Host / allocator",
}

#: Hierarchy rungs in traversal order, excluding HOST (which is not an ncu counter).
DEVICE_LEVELS: tuple[Level, ...] = (
    Level.REGISTER,
    Level.LOCAL,
    Level.SHARED,
    Level.L1TEX,
    Level.L2,
    Level.DRAM,
)


class Role(str, Enum):
    """What a metric is used for, which decides how the analysis layer treats it."""

    TRAFFIC = "traffic"        # a sector/byte count -- summable across kernels
    RATE = "rate"              # a percentage or ratio -- must be re-derived, never summed
    CONFIG = "config"          # a static launch property -- per-kernel, not summable
    TIME = "time"              # a duration
    THROUGHPUT = "throughput"  # % of peak sustained
    SENTINEL = "sentinel"      # must stay zero on this platform; non-zero invalidates results


@dataclass(frozen=True)
class MetricSpec:
    """One Nsight Compute counter, with everything needed to collect and interpret it."""

    name: str
    """Fully-qualified ncu metric name, including the ``.sum``/``.pct``/``.ratio`` suffix."""

    level: Level
    role: Role
    unit: str
    description: str

    #: Short column name used in the tidy CSV output and report tables.
    short: str = ""

    #: If True, absence of this metric is a hard error rather than a degraded run.
    required: bool = False

    def __post_init__(self) -> None:
        if not self.short:
            object.__setattr__(self, "short", self.name.split(".")[0])

    @property
    def is_sector_count(self) -> bool:
        """Whether multiplying by :data:`SECTOR_BYTES` yields bytes."""
        return "_sectors" in self.name and self.role in (Role.TRAFFIC, Role.SENTINEL)


# --------------------------------------------------------------------------------------
# Tier 1 -- the triage set. Cheap enough to run over every kernel in a decode step.
# Every entry here was confirmed to collect on GB10 / sm_121.
# --------------------------------------------------------------------------------------

_M = MetricSpec

TIER1: tuple[MetricSpec, ...] = (
    # ---- Registers & occupancy limits -------------------------------------------------
    _M("launch__registers_per_thread", Level.REGISTER, Role.CONFIG, "reg/thread",
       "Registers allocated per thread. High values throttle occupancy and, past 255, force spills.",
       short="regs_per_thread", required=True),
    _M("launch__occupancy_limit_registers", Level.REGISTER, Role.CONFIG, "block",
       "Blocks per SM permitted by register pressure alone.",
       short="occ_limit_regs"),
    _M("launch__occupancy_limit_shared_mem", Level.SHARED, Role.CONFIG, "block",
       "Blocks per SM permitted by shared-memory usage alone.",
       short="occ_limit_smem"),

    # ---- Local memory (register spills) ----------------------------------------------
    # Local memory is backed by the same physical DRAM as global memory, so spills are
    # expensive in exactly the way a naive register-pressure reading does not suggest.
    _M("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", Level.LOCAL, Role.TRAFFIC, "sector",
       "Sectors read from local memory -- i.e. register spill reloads.",
       short="local_ld_sectors"),
    _M("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", Level.LOCAL, Role.TRAFFIC, "sector",
       "Sectors written to local memory -- i.e. register spill stores.",
       short="local_st_sectors"),

    # ---- Shared memory ----------------------------------------------------------------
    _M("launch__shared_mem_per_block_static", Level.SHARED, Role.CONFIG, "byte/block",
       "Statically declared shared memory per block.",
       short="smem_static"),
    _M("launch__shared_mem_per_block_dynamic", Level.SHARED, Role.CONFIG, "byte/block",
       "Dynamically allocated shared memory per block (the extern __shared__ argument).",
       short="smem_dynamic"),
    _M("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", Level.SHARED, Role.TRAFFIC, "conflict",
       "Shared-memory bank conflicts. Each one serialises an otherwise parallel access.",
       short="smem_bank_conflicts"),
    _M("smsp__inst_executed_op_shared_ld.sum", Level.SHARED, Role.TRAFFIC, "inst",
       "Shared-memory load instructions executed.",
       short="smem_ld_inst"),
    _M("smsp__inst_executed_op_shared_st.sum", Level.SHARED, Role.TRAFFIC, "inst",
       "Shared-memory store instructions executed.",
       short="smem_st_inst"),
    # Wavefronts are the only path to shared-memory *bytes*: there is no sector counter for
    # shared memory, so without these the shared rung of the hierarchy has no traffic figure
    # at all. One wavefront moves up to WAVEFRONT_BYTES across the 32 banks.
    _M("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", Level.SHARED, Role.TRAFFIC,
       "wavefront",
       "Shared-memory wavefronts through the LSU data pipe. x128 B approximates bytes moved; "
       "bank conflicts inflate this above the ideal count.",
       short="smem_wavefronts"),
    _M("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", Level.SHARED, Role.TRAFFIC,
       "wavefront", "Shared-memory load wavefronts.", short="smem_ld_wavefronts"),
    _M("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", Level.SHARED, Role.TRAFFIC,
       "wavefront", "Shared-memory store wavefronts.", short="smem_st_wavefronts"),

    # ---- L1 / TEX ---------------------------------------------------------------------
    _M("l1tex__t_requests.sum", Level.L1TEX, Role.TRAFFIC, "request",
       "Requests arriving at L1TEX from the SM.",
       short="l1_requests"),
    _M("l1tex__t_sectors.sum", Level.L1TEX, Role.TRAFFIC, "sector",
       "Sectors moved through L1TEX. x32 B gives bytes the SM demanded of the cache hierarchy.",
       short="l1_sectors", required=True),
    _M("l1tex__t_bytes.sum", Level.L1TEX, Role.TRAFFIC, "byte",
       "Bytes moved through L1TEX, reported directly by hardware.",
       short="l1_bytes"),
    _M("l1tex__t_sector_hit_rate.pct", Level.L1TEX, Role.RATE, "%",
       "Fraction of L1TEX sector lookups that hit. Must be re-derived from counts, never averaged.",
       short="l1_hit_rate"),
    # Raw hit/miss counts, so an aggregate hit rate over many kernels can be computed as
    # sum(hits)/sum(lookups). Averaging the per-kernel .pct values instead would weight a
    # trivial 200-sector kernel the same as a GEMM moving hundreds of megabytes.
    _M("l1tex__t_sectors_lookup_hit.sum", Level.L1TEX, Role.TRAFFIC, "sector",
       "L1TEX sector lookups that hit.", short="l1_hit_sectors"),
    _M("l1tex__t_sectors_lookup_miss.sum", Level.L1TEX, Role.TRAFFIC, "sector",
       "L1TEX sector lookups that missed and went to L2.", short="l1_miss_sectors"),
    _M("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
       Level.L1TEX, Role.RATE, "sector/request",
       "Sectors per global load request. 4.0 is perfectly coalesced for a 128 B warp access; "
       "higher means the warp is scattering across lines.",
       short="l1_sectors_per_req_ld"),
    _M("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", Level.L1TEX, Role.TRAFFIC, "sector",
       "Global-load sectors through L1TEX.",
       short="l1_global_ld_sectors"),
    _M("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", Level.L1TEX, Role.TRAFFIC, "sector",
       "Global-store sectors through L1TEX.",
       short="l1_global_st_sectors"),

    # ---- L2 (LTS) ---------------------------------------------------------------------
    _M("lts__t_requests.sum", Level.L2, Role.TRAFFIC, "request",
       "Requests arriving at L2.",
       short="l2_requests"),
    _M("lts__t_sectors.sum", Level.L2, Role.TRAFFIC, "sector",
       "Sectors moved through L2. x32 B gives bytes L1 demanded of L2.",
       short="l2_sectors", required=True),
    _M("lts__t_sector_hit_rate.pct", Level.L2, Role.RATE, "%",
       "Fraction of L2 sector lookups that hit. With 25 MB of L2 on this part, this is the "
       "single most informative number for whether a model's working set is being captured.",
       short="l2_hit_rate"),
    _M("lts__t_sectors_op_read.sum", Level.L2, Role.TRAFFIC, "sector",
       "Read sectors at L2.",
       short="l2_read_sectors"),
    _M("lts__t_sectors_op_write.sum", Level.L2, Role.TRAFFIC, "sector",
       "Write sectors at L2.",
       short="l2_write_sectors"),
    _M("lts__t_sectors_lookup_hit.sum", Level.L2, Role.TRAFFIC, "sector",
       "L2 sector lookups that hit.", short="l2_hit_sectors"),
    _M("lts__t_sectors_lookup_miss.sum", Level.L2, Role.TRAFFIC, "sector",
       "L2 sector lookups that missed.", short="l2_miss_sectors"),

    # ---- Past L2 -> unified LPDDR5X ---------------------------------------------------
    # This block is the dram__* replacement. See the module docstring for the calibration.
    _M("lts__t_sectors_aperture_sysmem_lookup_miss.sum", Level.DRAM, Role.TRAFFIC, "sector",
       "L2 misses to the sysmem aperture. On GB10 this IS the traffic that reaches the unified "
       "LPDDR5X pool -- the stand-in for dram__bytes.sum, which does not exist on this chip.",
       short="dram_sectors", required=True),
    _M("lts__t_sectors_aperture_sysmem_lookup_hit.sum", Level.DRAM, Role.TRAFFIC, "sector",
       "L2 hits on sysmem-aperture lines: traffic that was *saved* from going to LPDDR5X.",
       short="dram_saved_sectors"),
    _M("lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum", Level.DRAM, Role.TRAFFIC, "sector",
       "Read sectors that missed L2 and went to LPDDR5X.",
       short="dram_read_sectors"),
    _M("lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum", Level.DRAM, Role.TRAFFIC, "sector",
       "Write sectors that missed L2 and went to LPDDR5X.",
       short="dram_write_sectors"),
    _M("lts__d_sectors_fill_sysmem.sum", Level.DRAM, Role.TRAFFIC, "sector",
       "L2 data-stage fill sectors sourced from sysmem: an independent view of inbound traffic, "
       "used to cross-check the lookup_miss derivation.",
       short="dram_fill_sectors"),
    _M("lts__d_sectors.sum", Level.DRAM, Role.TRAFFIC, "sector",
       "Total L2 data-stage sectors.",
       short="l2_data_sectors"),

    # ---- Sentinels: must be zero on GB10 ----------------------------------------------
    _M("lts__t_sectors_aperture_device_lookup_miss.sum", Level.DRAM, Role.SENTINEL, "sector",
       "L2 misses to the device (vidmem) aperture. GB10 has no private VRAM, so this must read 0. "
       "Non-zero means the sysmem-aperture derivation is incomplete and results are suspect.",
       short="sentinel_device_miss"),
    _M("lts__t_sectors_aperture_peer_lookup_miss.sum", Level.DRAM, Role.SENTINEL, "sector",
       "L2 misses to a peer GPU aperture. Single-GPU system, so this must read 0.",
       short="sentinel_peer_miss"),

    # ---- Time & throughput ------------------------------------------------------------
    _M("gpu__time_duration.sum", Level.DRAM, Role.TIME, "ns",
       "Kernel duration as measured by the profiler.",
       short="duration_ns", required=True),
    _M("sm__throughput.avg.pct_of_peak_sustained_elapsed", Level.L1TEX, Role.THROUGHPUT, "%",
       "SM pipeline throughput as a fraction of peak.",
       short="sm_throughput_pct"),
    _M("sm__memory_throughput.avg.pct_of_peak_sustained_elapsed", Level.L1TEX, Role.THROUGHPUT, "%",
       "SM-side memory pipeline throughput as a fraction of peak.",
       short="sm_mem_throughput_pct"),
    _M("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
       Level.L2, Role.THROUGHPUT, "%",
       "Compute-memory pipeline throughput: the closest single number to 'how saturated is the "
       "memory system', given no DRAM counters are available.",
       short="mem_pipe_throughput_pct"),
    _M("sm__pipe_tensor_cycles_active.sum", Level.REGISTER, Role.TRAFFIC, "cycle",
       "Tensor-core active cycles, used to separate tensor-bound from memory-bound kernels.",
       short="tensor_cycles"),

    # ---- Work counters, for arithmetic intensity ---------------------------------------
    # Arithmetic intensity (FLOPs per byte from memory) is what places a kernel on the
    # roofline, and it needs a real FLOP count. Tensor ops dominate transformer GEMMs, so
    # measuring them directly beats inferring FLOPs from parameter counts.
    _M("sm__ops_path_tensor_src_bf16_dst_fp32.sum", Level.REGISTER, Role.TRAFFIC, "op",
       "Tensor-core operations with bf16 inputs and fp32 accumulate -- the dominant work "
       "path for bf16 transformer GEMMs.",
       short="tensor_ops_bf16"),
    _M("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", Level.REGISTER, Role.TRAFFIC,
       "inst",
       "Non-tensor fp32 fused multiply-add thread-instructions (2 FLOPs each).",
       short="ffma_inst"),
    _M("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", Level.REGISTER, Role.TRAFFIC,
       "inst",
       "Non-tensor fp16 fused multiply-add thread-instructions (2 FLOPs each).",
       short="hfma_inst"),
)

#: Bytes moved by one shared-memory wavefront across the 32 banks (32 banks x 4 B).
WAVEFRONT_BYTES = 128


# --------------------------------------------------------------------------------------
# Tier 2 / Tier 3 -- section-based deep dives
# --------------------------------------------------------------------------------------

#: Tier 2 collects whole Nsight Compute sections rather than a metric list, because the
#: interesting derived quantities (spill requests, roofline points, memory chart edges) are
#: section-internal and cannot be requested individually.
#:
#: Caveat on this chip: MemoryWorkloadAnalysis references ``dram__bytes.sum.per_second``, which
#: GB10 does not have. Those rows come back as n/a. The parser tolerates that rather than failing.
TIER2_SECTIONS: tuple[str, ...] = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Tables",
    "MemoryWorkloadAnalysis_Chart",
    "ComputeWorkloadAnalysis",
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "InstructionStats",
    "WorkloadDistribution",
)

#: Tier 3 adds source-level attribution. Expensive and only meaningful when the binary carries
#: line info, which stock PyTorch/cuBLAS kernels generally do not -- so this is opt-in and
#: documented as best-effort.
TIER3_SECTIONS: tuple[str, ...] = TIER2_SECTIONS + ("SourceCounters",)

#: Sections that are known NOT to apply to GB10, recorded so preflight can explain the absence
#: rather than leaving a confusing gap in the report.
UNAVAILABLE_SECTIONS: dict[str, str] = {
    "C2CLink": (
        "Gated to CC_90/CC_100 in C2CLink.section; GB10 is CC_121. The ctc__* NVLink-C2C "
        "counters are not exposed on this part."
    ),
    "Nvlink": "No NVLink peer topology on a single-GPU DGX Spark.",
    "Nvlink_Tables": "No NVLink peer topology on a single-GPU DGX Spark.",
    "Nvlink_Topology": "No NVLink peer topology on a single-GPU DGX Spark.",
    "NumaAffinity": "Single NUMA node on this host; the section carries no signal.",
}

#: Metrics that look collectable but are not, with the reason. Preflight uses this to avoid
#: re-probing known dead ends, and the docs cite it so the absence is explained rather than
#: rediscovered.
KNOWN_MISSING: dict[str, str] = {
    "dram__bytes.sum": (
        "No dram__* counters exist on GB20B. Use lts__t_sectors_aperture_sysmem_lookup_miss "
        "x 32 B instead."
    ),
    "dram__bytes_read.sum": "No dram__* counters on GB20B.",
    "dram__bytes_write.sum": "No dram__* counters on GB20B.",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "No dram__* counters on GB20B.",
    "ctc__rx_bytes_data_user.sum": "No ctc__* (NVLink-C2C) counters on GB20B.",
    "ctc__tx_bytes_data_user.sum": "No ctc__* (NVLink-C2C) counters on GB20B.",
    "derived__local_spilling_requests": (
        "Section-internal derived metric; fails as a standalone --metrics entry. Collected "
        "indirectly via the MemoryWorkloadAnalysis section in tier 2."
    ),
    "sass__inst_executed_register_spilling_mem_local": (
        "Section-internal; requires SourceCounters and SASS line info."
    ),
}


# --------------------------------------------------------------------------------------
# Lookup helpers
# --------------------------------------------------------------------------------------

BY_NAME: dict[str, MetricSpec] = {m.name: m for m in TIER1}
BY_SHORT: dict[str, MetricSpec] = {m.short: m for m in TIER1}


def tier1_names() -> list[str]:
    """Metric names for the tier-1 ``ncu --metrics`` argument, in registry order."""
    return [m.name for m in TIER1]


def tier1_metrics_arg(available: set[str] | None = None) -> str:
    """Build the comma-separated ``--metrics`` value for a tier-1 collection.

    Args:
        available: If given, restrict to metrics known to collect on the attached GPU
            (as recorded by preflight). Passing ``None`` requests the full registry, which
            is correct for the very first probe.
    """
    names = tier1_names()
    if available is not None:
        names = [n for n in names if n in available]
    return ",".join(names)


def required_names() -> list[str]:
    """Metrics without which a run cannot be meaningfully analysed."""
    return [m.name for m in TIER1 if m.required]


def sentinels() -> list[MetricSpec]:
    """Metrics that must read zero on this platform for the derivation to hold."""
    return [m for m in TIER1 if m.role is Role.SENTINEL]


def by_level(level: Level) -> list[MetricSpec]:
    """All tier-1 metrics attributed to one rung of the hierarchy."""
    return [m for m in TIER1 if m.level is level]


def traffic_metrics() -> list[MetricSpec]:
    """Summable traffic counters, excluding sentinels."""
    return [m for m in TIER1 if m.role is Role.TRAFFIC]


def sections_for_tier(tier: int) -> tuple[str, ...]:
    """Nsight Compute sections to request for a given deep-dive tier."""
    if tier <= 1:
        return ()
    if tier == 2:
        return TIER2_SECTIONS
    return TIER3_SECTIONS


@dataclass
class MetricAvailability:
    """Outcome of probing which registry metrics actually collect on the attached GPU."""

    available: set[str] = field(default_factory=set)
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every metric flagged ``required`` is collectable."""
        return all(name in self.available for name in required_names())

    def missing_required(self) -> list[str]:
        return [n for n in required_names() if n not in self.available]

    def summary(self) -> str:
        total = len(TIER1)
        return (
            f"{len(self.available)}/{total} tier-1 metrics available"
            + (f", {len(self.missing)} missing" if self.missing else "")
        )
