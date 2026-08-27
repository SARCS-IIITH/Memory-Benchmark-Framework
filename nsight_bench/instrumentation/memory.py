"""Memory footprint tracking on a unified-memory part.

On a discrete GPU you would ask NVML how much VRAM is in use and be done. On GB10 that path
is closed: ``nvidia-smi --query-gpu=memory.used`` returns ``N/A``, because there is no
separate VRAM to report -- the GPU shares one coherent LPDDR5X pool with the Grace CPU.

So footprint is reconstructed from three independent sources, each with different blind
spots, and the harness cross-checks them against each other:

1. **The PyTorch caching allocator** (``torch.cuda.memory_stats``). Exact for tensors torch
   allocated, but blind to cuBLAS workspaces, the CUDA context, and anything a fused kernel
   allocates itself. It also reports *cached* memory, which is held but not in use.
2. **The CUDA driver** (``cudaMemGetInfo`` via ``torch.cuda.mem_get_info``). Sees everything
   CUDA has taken, including the context and library workspaces. On this part the "total" is
   the whole 128 GB unified pool, so "free" moves when *host* processes allocate too.
3. **The host kernel** (``/proc/meminfo``, ``/proc/self/status``). The only view that shows
   GPU and CPU allocations competing for one budget, which is the thing that actually bites
   on a unified-memory box.

Disagreement between them is signal, not noise, and the report surfaces it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a dict of bytes."""
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                # Values are in kB except for a few unitless counters.
                scale = 1024 if len(parts) > 1 and parts[1] == "kB" else 1
                result[key.strip()] = int(parts[0]) * scale
    except OSError:
        pass
    return result


def _proc_rss() -> int:
    """Resident set size of this process, in bytes."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


@dataclass
class MemorySnapshot:
    """One instant across all three views of memory."""

    timestamp: float = 0.0
    label: str = ""

    # --- torch caching allocator ---
    torch_allocated: int = 0
    torch_reserved: int = 0
    torch_max_allocated: int = 0
    torch_max_reserved: int = 0

    # --- CUDA driver (cudaMemGetInfo) ---
    cuda_free: int = 0
    cuda_total: int = 0

    # --- host kernel ---
    host_mem_available: int = 0
    host_mem_free: int = 0
    process_rss: int = 0

    @property
    def cuda_used(self) -> int:
        """Bytes taken from the unified pool as the CUDA driver sees it."""
        return self.cuda_total - self.cuda_free

    @property
    def torch_cache_overhead(self) -> int:
        """Reserved-but-unused bytes held by the caching allocator."""
        return max(0, self.torch_reserved - self.torch_allocated)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cuda_used"] = self.cuda_used
        d["torch_cache_overhead"] = self.torch_cache_overhead
        return d


def snapshot(label: str = "") -> MemorySnapshot:
    """Capture all three views at once."""
    snap = MemorySnapshot(timestamp=time.time(), label=label)

    try:
        import torch

        if torch.cuda.is_available():
            snap.torch_allocated = torch.cuda.memory_allocated()
            snap.torch_reserved = torch.cuda.memory_reserved()
            snap.torch_max_allocated = torch.cuda.max_memory_allocated()
            snap.torch_max_reserved = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info()
            snap.cuda_free, snap.cuda_total = free, total
    except Exception:                                            # noqa: BLE001
        pass

    info = _meminfo()
    snap.host_mem_available = info.get("MemAvailable", 0)
    snap.host_mem_free = info.get("MemFree", 0)
    snap.process_rss = _proc_rss()
    return snap


#: Run-level high-water marks, carried across the per-phase counter resets below.
#:
#: ``torch.cuda.reset_peak_memory_stats`` is what makes a per-phase peak possible, but it
#: also destroys the run's own peak -- and the peak that matters is usually prefill's, several
#: resets before the process ends. Reading ``allocated_bytes.all.peak`` at exit therefore
#: reports the peak since the *last* reset, which is a smaller and quietly wrong number.
#: These two carry the real maximum forward.
_RUN_PEAK_ALLOCATED = 0
_RUN_PEAK_RESERVED = 0


def _absorb_peaks_into_run_maximum() -> None:
    """Fold torch's current peak counters into the run-level maxima before they are cleared."""
    global _RUN_PEAK_ALLOCATED, _RUN_PEAK_RESERVED
    try:
        import torch

        if not torch.cuda.is_available():
            return
        _RUN_PEAK_ALLOCATED = max(_RUN_PEAK_ALLOCATED, torch.cuda.max_memory_allocated())
        _RUN_PEAK_RESERVED = max(_RUN_PEAK_RESERVED, torch.cuda.max_memory_reserved())
    except Exception:                                            # noqa: BLE001
        pass


def run_peaks() -> dict[str, int]:
    """The run's true allocator high-water marks, across every phase reset."""
    _absorb_peaks_into_run_maximum()
    return {
        "run_peak_allocated_bytes": _RUN_PEAK_ALLOCATED,
        "run_peak_reserved_bytes": _RUN_PEAK_RESERVED,
    }


def reset_peaks() -> None:
    """Clear torch's peak counters so the next phase measures its own high-water mark.

    The outgoing peaks are folded into the run-level maxima first, so per-phase resolution
    does not cost the run-level figure.
    """
    _absorb_peaks_into_run_maximum()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:                                            # noqa: BLE001
        pass


@dataclass
class MemoryTimeline:
    """A time series of snapshots, plus the labelled marks taken around each phase."""

    samples: list[MemorySnapshot] = field(default_factory=list)
    marks: list[MemorySnapshot] = field(default_factory=list)

    def to_rows(self) -> list[dict]:
        rows = [{**s.to_dict(), "kind": "sample"} for s in self.samples]
        rows += [{**s.to_dict(), "kind": "mark"} for s in self.marks]
        rows.sort(key=lambda r: r["timestamp"])
        if rows:
            t0 = rows[0]["timestamp"]
            for row in rows:
                row["t_rel_s"] = row["timestamp"] - t0
        return rows

    def peak(self, field_name: str = "cuda_used") -> int:
        values = [
            getattr(s, field_name, None) or s.to_dict().get(field_name, 0)
            for s in self.samples + self.marks
        ]
        return max((v for v in values if isinstance(v, int)), default=0)


class MemorySampler:
    """Background thread sampling memory at a fixed rate for the whole run.

    A sampled timeline is what makes transient peaks visible -- weight loading, the first
    prefill's workspace allocation, KV-cache growth -- none of which show up in a
    before/after pair. The default 20 Hz is fast enough to catch those without meaningfully
    perturbing the workload.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.timeline = MemoryTimeline()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.timeline.samples.append(snapshot())
            self._stop.wait(self.interval_s)

    def start(self) -> MemorySampler:
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="nsbench-memory-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> MemoryTimeline:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.timeline

    def mark(self, label: str) -> MemorySnapshot:
        """Take a labelled snapshot, e.g. at a phase boundary."""
        snap = snapshot(label)
        self.timeline.marks.append(snap)
        return snap

    def __enter__(self) -> MemorySampler:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


@dataclass
class PhaseMemoryDelta:
    """What one phase did to memory, measured across all three views."""

    phase: str
    before: MemorySnapshot
    after: MemorySnapshot
    peak_torch_allocated: int = 0
    peak_torch_reserved: int = 0

    @property
    def torch_delta(self) -> int:
        return self.after.torch_allocated - self.before.torch_allocated

    @property
    def cuda_delta(self) -> int:
        return self.after.cuda_used - self.before.cuda_used

    @property
    def host_delta(self) -> int:
        """Host-visible consumption. Positive means the pool lost that many bytes."""
        return self.before.host_mem_available - self.after.host_mem_available

    def agreement_ratio(self) -> float | None:
        """How closely the driver and allocator views agree on this phase.

        Values near 1.0 mean torch accounts for essentially all of the allocation. Values
        well above 1.0 mean something outside the allocator -- a cuBLAS workspace, the CUDA
        context, a library-internal pool -- took memory that torch cannot see.
        """
        if self.torch_delta <= 0:
            return None
        return self.cuda_delta / self.torch_delta

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "torch_delta_bytes": self.torch_delta,
            "cuda_delta_bytes": self.cuda_delta,
            "host_delta_bytes": self.host_delta,
            "peak_torch_allocated_bytes": self.peak_torch_allocated,
            "peak_torch_reserved_bytes": self.peak_torch_reserved,
            "driver_vs_allocator_ratio": self.agreement_ratio(),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


class PhaseMemoryTracker:
    """Context manager measuring one phase's memory delta and peak."""

    def __init__(self, phase_name: str, sampler: MemorySampler | None = None) -> None:
        self.phase_name = phase_name
        self.sampler = sampler
        self.delta: PhaseMemoryDelta | None = None
        self._before: MemorySnapshot | None = None

    def __enter__(self) -> PhaseMemoryTracker:
        reset_peaks()
        self._before = snapshot(f"{self.phase_name}:before")
        if self.sampler is not None:
            self.sampler.timeline.marks.append(self._before)
        return self

    def __exit__(self, *_exc) -> None:
        after = snapshot(f"{self.phase_name}:after")
        if self.sampler is not None:
            self.sampler.timeline.marks.append(after)
        assert self._before is not None
        self.delta = PhaseMemoryDelta(
            phase=self.phase_name,
            before=self._before,
            after=after,
            peak_torch_allocated=after.torch_max_allocated,
            peak_torch_reserved=after.torch_max_reserved,
        )


# --------------------------------------------------------------------------------------
# Allocator history -- the fine-grained view
# --------------------------------------------------------------------------------------


def start_allocator_history(max_entries: int = 100_000) -> bool:
    """Begin recording every allocator event, for a post-hoc allocation trace.

    This is torch's own memory profiler. It captures each alloc/free with a Python stack, so
    a fragmented or leaking run can be traced back to the line responsible -- detail that no
    counter-based view provides.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
        return True
    except Exception:                                            # noqa: BLE001
        return False


def dump_allocator_history(path: str | Path) -> Path | None:
    """Write the recorded allocator history and stop recording.

    The resulting pickle opens at https://pytorch.org/memory_viz for a visual timeline.
    """
    try:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))
        torch.cuda.memory._record_memory_history(enabled=None)
        return path
    except Exception:                                            # noqa: BLE001
        return None


def allocator_stats() -> dict:
    """Flatten ``torch.cuda.memory_stats`` into a plain dict of the fields worth keeping."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        stats = torch.cuda.memory_stats()
        keep = (
            "allocated_bytes.all.current", "allocated_bytes.all.peak",
            "reserved_bytes.all.current", "reserved_bytes.all.peak",
            "active_bytes.all.current", "active_bytes.all.peak",
            "inactive_split_bytes.all.current",
            "num_alloc_retries", "num_ooms",
            "allocation.all.allocated", "allocation.all.freed",
            "segment.all.allocated", "segment.all.freed",
        )
        out = {k: stats[k] for k in keep if k in stats}
        # The ``.peak`` fields above are peaks since the last per-phase reset. The run's own
        # high-water mark is tracked separately, because it is usually set during prefill --
        # several resets earlier -- and is the figure the footprint report should quote.
        out.update(run_peaks())
        return out
    except Exception:                                            # noqa: BLE001
        return {}
