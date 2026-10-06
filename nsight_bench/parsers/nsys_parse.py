"""Query an exported Nsight Systems SQLite database.

Everything here goes through SQL against the exported database rather than shelling out to
``nsys stats``. The built-in reports are a fixed menu of summaries; the joins we actually
need -- kernels attributed to the NVTX phase that launched them, allocation events correlated
with those phases -- are not on that menu.

The projection of GPU work onto NVTX ranges is the important piece. NVTX ranges are recorded
on the CPU when the host thread pushes them; kernels are recorded on the GPU when they
execute, potentially long after. Linking the two requires going through the CUDA runtime
call that launched the kernel::

    kernel.correlationId -> runtime.correlationId -> runtime.start
    -> the innermost NVTX range on the same thread containing that timestamp

Matching on kernel *execution* time instead would misattribute every kernel that outlived
the range which launched it -- which, for an asynchronous decode step, is most of them.
"""

from __future__ import annotations

import bisect
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

#: The sampled all-L2-traffic series added by configs/nsys/gb20b_l2.config.
L2_SECTORS_METRIC = "L2 Sectors [Sectors]"
L2_SECTOR_BYTES = 32
#: Kernels of one phase closer together than this count as one GPU-busy window.
L2_WINDOW_MERGE_NS = 1_000_000

#: NVTX event types in the nsys schema.
NVTX_PUSH_POP_RANGE = 59
NVTX_START_END_RANGE = 60
NVTX_MARK = 34

#: Memory operations recorded in CUDA_GPU_MEMORY_USAGE_EVENTS.
MEM_OP_ALLOCATION = 0
MEM_OP_DEALLOCATION = 1


@dataclass
class KernelEvent:
    """One kernel execution from the timeline."""

    start_ns: int = 0
    end_ns: int = 0
    name: str = ""
    short_name: str = ""
    stream_id: int = 0
    grid: tuple[int, int, int] = (0, 0, 0)
    block: tuple[int, int, int] = (0, 0, 0)
    registers_per_thread: int = 0
    static_shared_bytes: int = 0
    dynamic_shared_bytes: int = 0
    local_memory_total: int = 0
    correlation_id: int = 0
    phase: str = ""

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def to_row(self) -> dict:
        return {
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "kernel_name": self.name,
            "kernel_short": self.short_name,
            "phase": self.phase,
            "stream": self.stream_id,
            "grid_x": self.grid[0], "grid_y": self.grid[1], "grid_z": self.grid[2],
            "block_x": self.block[0], "block_y": self.block[1], "block_z": self.block[2],
            "registers_per_thread": self.registers_per_thread,
            "static_shared_bytes": self.static_shared_bytes,
            "dynamic_shared_bytes": self.dynamic_shared_bytes,
            "local_memory_total_bytes": self.local_memory_total,
        }


@dataclass
class NvtxRange:
    """One NVTX push/pop range."""

    start_ns: int = 0
    end_ns: int = 0
    text: str = ""
    tid: int = 0

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def phase(self) -> str:
        """The range name with any ``:detail`` suffix stripped."""
        return self.text.split(":", 1)[0]

    def to_row(self) -> dict:
        return {
            "phase": self.phase,
            "text": self.text,
            "start_ns": self.start_ns,
            "duration_ns": self.duration_ns,
            "tid": self.tid,
        }


@dataclass
class MemoryEvent:
    """One GPU allocation or free."""

    timestamp_ns: int = 0
    bytes: int = 0
    kind: str = ""
    operation: str = ""
    name: str = ""

    def to_row(self) -> dict:
        return {
            "timestamp_ns": self.timestamp_ns,
            "bytes": self.bytes,
            "memory_kind": self.kind,
            "operation": self.operation,
            "name": self.name,
        }


@dataclass
class NsysReport:
    """Everything extracted from one nsys database."""

    source: str = ""
    kernels: list[KernelEvent] = field(default_factory=list)
    nvtx_ranges: list[NvtxRange] = field(default_factory=list)
    memory_events: list[MemoryEvent] = field(default_factory=list)
    memcpy_bytes: dict[str, int] = field(default_factory=dict)
    gpu_metrics: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    um_page_faults: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ---- derived views ----------------------------------------------------------------

    def kernels_in_phase(self, phase: str) -> list[KernelEvent]:
        return [k for k in self.kernels if k.phase == phase]

    def phase_durations_ns(self) -> dict[str, int]:
        """Total wall time per NVTX phase, summed over its instances."""
        totals: dict[str, int] = {}
        for rng in self.nvtx_ranges:
            totals[rng.phase] = totals.get(rng.phase, 0) + rng.duration_ns
        return totals

    def phase_instance_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rng in self.nvtx_ranges:
            counts[rng.phase] = counts.get(rng.phase, 0) + 1
        return counts

    def phase_occupancy(self, phase: str) -> dict | None:
        """How much of a phase's wall time the GPU was actually executing something.

        This is the number that separates "memory-bound" from "launch-bound", and nothing
        else in the harness can answer it. A per-kernel profile says how fast each kernel
        ran; summed kernel duration says how much work there was. Neither notices that the
        GPU sat idle between them.

        It matters most for decode. A step is a long chain of small kernels, and if the host
        cannot queue them faster than the device drains them -- or worse, blocks on a
        synchronisation every step -- the phase is bounded by launch latency, not by
        bandwidth. Quoting a bandwidth utilisation figure for such a phase invites the wrong
        optimisation entirely: fewer bytes will not help a GPU that is idle a third of the
        time.

        Busy time is the **union** of kernel intervals, not their sum: kernels on different
        streams can overlap, and summing would double-count that into a busy fraction above
        100%. Only kernels are counted, so a phase whose time goes to memcpy shows as idle,
        which is itself the finding.
        """
        windows = [r for r in self.nvtx_ranges if r.phase == phase and r.end_ns > r.start_ns]
        if not windows or not self.kernels:
            return None

        ordered = sorted(self.kernels, key=lambda k: k.start_ns)
        starts = [k.start_ns for k in ordered]

        wall_ns = 0
        busy_ns = 0
        for window in windows:
            wall_ns += window.duration_ns
            busy_ns += _union_length(ordered, starts, window.start_ns, window.end_ns)

        if wall_ns <= 0:
            return None
        return {
            "instances": len(windows),
            "wall_ns": wall_ns,
            "gpu_busy_ns": busy_ns,
            "gpu_idle_ns": max(0, wall_ns - busy_ns),
            "busy_pct": 100.0 * busy_ns / wall_ns,
            "wall_ns_per_instance": wall_ns / len(windows),
            "gpu_busy_ns_per_instance": busy_ns / len(windows),
        }

    def kernel_time_by_name(
        self, phase: str | None = None, key: Callable[[str], str] | None = None,
    ) -> dict[str, tuple[int, int]]:
        """``kernel name -> (total ns, launch count)``, optionally scoped to a phase.

        ``key`` maps each demangled name to the name it is grouped under. Pass
        :func:`~nsight_bench.parsers.ncu_parse.base_identifier` to group the way ncu's
        ``--kernel-name`` filter matches, so every template instantiation of one function
        counts toward a single entry.
        """
        totals: dict[str, tuple[int, int]] = {}
        for kernel in self.kernels:
            if phase is not None and kernel.phase != phase:
                continue
            name = key(kernel.name) if key else kernel.name
            if not name:
                continue
            total, count = totals.get(name, (0, 0))
            totals[name] = (total + kernel.duration_ns, count + 1)
        return totals

    def sampled_l2_bytes_per_instance(self, phase: str) -> float | None:
        """L2 traffic per instance of ``phase``, from the sampled ``lts__t_sectors`` series.

        Only present when the extended metric set ran (``nsys.sample_l2_traffic``). The phase's
        kernels are merged into GPU-busy windows (gaps under :data:`L2_WINDOW_MERGE_NS`
        joined), and every sample whose interval ends inside a window, or within one sampling
        period after it, is counted. A sample is the counter delta over the interval ending at
        its timestamp. Validated against ncu on Qwen3-0.6B: within 1% for prefill and 4% for
        decode.
        """
        samples = self.gpu_metrics.get(L2_SECTORS_METRIC)
        kernels = self.kernels_in_phase(phase)
        instances = self.phase_instance_counts().get(phase, 0)
        if not samples or not kernels or not instances:
            return None
        times = [t for t, _ in samples]
        prefix = [0.0]
        for _, value in samples:
            prefix.append(prefix[-1] + value)
        period = (times[-1] - times[0]) / max(1, len(times) - 1)

        windows: list[list[int]] = []
        for start, end in sorted((k.start_ns, k.end_ns) for k in kernels):
            if windows and start - windows[-1][1] <= L2_WINDOW_MERGE_NS:
                windows[-1][1] = max(windows[-1][1], end)
            else:
                windows.append([start, end])

        sectors = 0.0
        for start, end in windows:
            lo = bisect.bisect_right(times, start)
            hi = bisect.bisect_right(times, end + period)
            sectors += prefix[hi] - prefix[lo]
        return sectors * L2_SECTOR_BYTES / instances

    def top_kernels(
        self, phase: str | None = None, n: int = 10,
        key: Callable[[str], str] | None = None,
    ) -> list[tuple[str, int, int]]:
        """Kernels ranked by total GPU time: ``(name, total_ns, launches)``."""
        ranked = sorted(
            ((name, total, count)
             for name, (total, count) in self.kernel_time_by_name(phase, key).items()),
            key=lambda item: item[1], reverse=True,
        )
        return ranked[:n]

    @property
    def allocation_timestamps_available(self) -> bool:
        """Whether allocation events carry usable timestamps.

        On this nsys build every ``CUDA_GPU_MEMORY_USAGE_EVENTS.start`` is 0 -- the events
        are recorded in order, with correct sizes, but with no time. Plotting them against a
        collapsed time axis would produce a chart where every point sits at t=0, which reads
        as a bug rather than as missing data. The allocation *sequence* is still meaningful,
        so the caller falls back to event index and says so.
        """
        return any(e.timestamp_ns for e in self.memory_events)

    def allocation_timeline(self) -> list[dict]:
        """Cumulative GPU bytes outstanding, in allocation order.

        This is the substitute for a VRAM usage graph on GB10, where NVML reports nothing.
        Deallocations are subtracted so the series tracks live bytes, not gross allocation.

        Ordering falls back to insertion order when timestamps are unavailable (see
        :attr:`allocation_timestamps_available`); the database returns the events in
        recording order regardless, so the sequence stays correct either way.
        """
        timed = self.allocation_timestamps_available
        events = (
            sorted(self.memory_events, key=lambda e: e.timestamp_ns)
            if timed else self.memory_events
        )
        rows: list[dict] = []
        outstanding = 0
        for index, event in enumerate(events):
            delta = event.bytes if event.operation == "Allocation" else -event.bytes
            outstanding += delta
            rows.append({
                "event_index": index,
                "timestamp_ns": event.timestamp_ns if timed else None,
                "delta_bytes": delta,
                "outstanding_bytes": outstanding,
                "memory_kind": event.kind,
                "operation": event.operation,
                "name": event.name,
            })
        return rows

    def peak_allocated_bytes(self) -> int:
        rows = self.allocation_timeline()
        return max((r["outstanding_bytes"] for r in rows), default=0)

    def clock_summary(self) -> dict:
        """Min/mean/max of the sampled GPC clock, in MHz.

        Clock stability is what tells you whether two runs are comparable. A run whose clock
        sagged partway through produced real numbers under conditions the other run did not
        share.

        The stored values need rescaling: nsys names the metric ``GPC Clock Frequency
        [MHz]`` but writes raw cycles-per-second into GPU_METRICS, applying the 1e-6
        multiplier only at display time. Taken at face value a 1.43 GHz clock reads as
        1.4 billion MHz. Rescaling is decided by magnitude rather than by parsing the label,
        because the label is exactly what is misleading here.
        """
        series = None
        for name, samples in self.gpu_metrics.items():
            if "GPC Clock" in name:
                series = samples
                break
        if not series:
            return {}
        values = [v for _, v in series if v > 0]
        if not values:
            return {}
        if max(values) > 1e6:                # plainly Hz: no GPU runs at a million MHz
            values = [v / 1e6 for v in values]
        return {
            "gpc_clock_mhz_min": min(values),
            "gpc_clock_mhz_mean": sum(values) / len(values),
            "gpc_clock_mhz_max": max(values),
            "gpc_clock_samples": len(values),
            "gpc_clock_spread_pct": (
                100.0 * (max(values) - min(values)) / max(values) if max(values) else 0.0
            ),
        }


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _union_length(
    ordered: list[KernelEvent], starts: list[int], window_start: int, window_end: int
) -> int:
    """Total length of the union of kernel intervals clipped to ``[window_start, window_end]``.

    ``ordered`` must be sorted by start time and ``starts`` must be its start timestamps, so
    the scan can bisect to the first candidate instead of walking every kernel in the trace
    for every range -- with tens of thousands of kernels and a range per generated token, the
    naive form is quadratic.

    A kernel that begins before the window is still counted for the part that falls inside
    it, which is what makes the fraction correct for the first kernel of a phase.
    """
    index = bisect.bisect_left(starts, window_start)
    # Step back over kernels that started earlier but are still running at the window's edge.
    while index > 0 and ordered[index - 1].end_ns > window_start:
        index -= 1

    total = 0
    covered_to = window_start
    while index < len(ordered) and ordered[index].start_ns < window_end:
        start = max(ordered[index].start_ns, window_start, covered_to)
        end = min(ordered[index].end_ns, window_end)
        if end > start:
            total += end - start
            covered_to = end
        index += 1
    return total


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?", (name,)
    ).fetchone()
    return row is not None


def parse_sqlite(path: str | Path) -> NsysReport:
    """Extract kernels, NVTX phases, allocations and GPU metrics from an nsys database."""
    path = Path(path)
    report = NsysReport(source=str(path))
    if not path.exists():
        report.warnings.append(f"nsys database not found: {path}")
        return report

    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        _load_nvtx(conn, report)
        _load_kernels(conn, report)
        _project_phases(conn, report)
        _load_memory_events(conn, report)
        _load_memcpy(conn, report)
        _load_gpu_metrics(conn, report)
        _load_um_faults(conn, report)

    return report


def _load_nvtx(conn: sqlite3.Connection, report: NsysReport) -> None:
    if not _table_exists(conn, "NVTX_EVENTS"):
        report.warnings.append("no NVTX_EVENTS table -- was nvtx included in --trace?")
        return

    # Range text is stored inline for short strings and in StringIds for longer ones.
    rows = conn.execute(
        """
        SELECT e.start, e.end, e.globalTid,
               COALESCE(e.text, s.value) AS label
        FROM NVTX_EVENTS e
        LEFT JOIN StringIds s ON s.id = e.textId
        WHERE e.eventType IN (?, ?) AND e.end IS NOT NULL
        ORDER BY e.start
        """,
        (NVTX_PUSH_POP_RANGE, NVTX_START_END_RANGE),
    ).fetchall()

    for row in rows:
        label = row["label"] or ""
        if not label:
            continue
        report.nvtx_ranges.append(
            NvtxRange(
                start_ns=row["start"], end_ns=row["end"],
                text=label, tid=row["globalTid"] or 0,
            )
        )

    if not report.nvtx_ranges:
        report.warnings.append(
            "no NVTX ranges captured -- phase attribution will be unavailable. If the "
            "capture range was set to cudaProfilerApi, check the workload reached "
            "cudaProfilerStart."
        )


def _load_kernels(conn: sqlite3.Connection, report: NsysReport) -> None:
    if not _table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        report.warnings.append("no kernel activity table -- was cuda included in --trace?")
        return

    rows = conn.execute(
        """
        SELECT k.start, k.end, k.streamId, k.correlationId,
               k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
               k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory,
               k.localMemoryTotal,
               COALESCE(d.value, s.value) AS name,
               s.value AS short_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds d ON d.id = k.demangledName
        LEFT JOIN StringIds s ON s.id = k.shortName
        ORDER BY k.start
        """
    ).fetchall()

    for row in rows:
        report.kernels.append(
            KernelEvent(
                start_ns=row["start"], end_ns=row["end"],
                name=row["name"] or "", short_name=row["short_name"] or "",
                stream_id=row["streamId"] or 0,
                grid=(row["gridX"] or 0, row["gridY"] or 0, row["gridZ"] or 0),
                block=(row["blockX"] or 0, row["blockY"] or 0, row["blockZ"] or 0),
                registers_per_thread=row["registersPerThread"] or 0,
                static_shared_bytes=row["staticSharedMemory"] or 0,
                dynamic_shared_bytes=row["dynamicSharedMemory"] or 0,
                local_memory_total=row["localMemoryTotal"] or 0,
                correlation_id=row["correlationId"] or 0,
            )
        )


def _project_phases(conn: sqlite3.Connection, report: NsysReport) -> None:
    """Attribute each kernel to the innermost NVTX range that launched it.

    Uses the launching runtime call's timestamp, not the kernel's execution timestamp:
    kernel execution is asynchronous and routinely outlives the range that queued it, so
    matching on execution time would file decode kernels under whatever range happened to be
    open when the GPU got round to them.
    """
    if not report.kernels or not report.nvtx_ranges:
        return
    if not _table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        report.warnings.append(
            "no CUDA runtime activity -- kernels cannot be attributed to NVTX phases. "
            "Add 'cuda' to --trace."
        )
        return

    launch_times: dict[int, tuple[int, int]] = {}
    for row in conn.execute(
        "SELECT correlationId, start, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        launch_times[row["correlationId"]] = (row["start"], row["globalTid"] or 0)

    # Innermost wins. The loop below assigns on every match without breaking, so the *last*
    # containing range in iteration order is the one that sticks -- which means the list must
    # be ordered longest-first, so the shortest (tightest) match is visited last. That is why
    # this is reverse=True. Sorting the other way would file every decode_step kernel under
    # the enclosing nsbench.decode range instead, silently emptying the per-step attribution
    # the whole report is built on.
    ranges = sorted(report.nvtx_ranges, key=lambda r: r.duration_ns, reverse=True)

    unmatched = 0
    for kernel in report.kernels:
        launch = launch_times.get(kernel.correlation_id)
        if launch is None:
            unmatched += 1
            continue
        launch_ns, tid = launch
        for rng in ranges:
            if rng.start_ns <= launch_ns <= rng.end_ns and (not rng.tid or rng.tid == tid):
                kernel.phase = rng.phase

    if unmatched:
        report.warnings.append(
            f"{unmatched} kernels had no matching runtime launch record and are unattributed"
        )


def _load_memory_events(conn: sqlite3.Connection, report: NsysReport) -> None:
    if not _table_exists(conn, "CUDA_GPU_MEMORY_USAGE_EVENTS"):
        report.warnings.append(
            "no GPU memory usage events -- pass --cuda-memory-usage=true. On this part that "
            "is the only footprint timeline available, since NVML reports nothing."
        )
        return

    kinds = {
        row["id"]: row["label"]
        for row in conn.execute("SELECT id, label FROM ENUM_CUDA_MEM_KIND")
    } if _table_exists(conn, "ENUM_CUDA_MEM_KIND") else {}
    opers = {
        row["id"]: row["label"]
        for row in conn.execute("SELECT id, label FROM ENUM_CUDA_DEV_MEM_EVENT_OPER")
    } if _table_exists(conn, "ENUM_CUDA_DEV_MEM_EVENT_OPER") else {}

    for row in conn.execute(
        """
        SELECT start, bytes, memKind, memoryOperationType, name
        FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start
        """
    ):
        report.memory_events.append(
            MemoryEvent(
                timestamp_ns=row["start"],
                bytes=row["bytes"] or 0,
                kind=kinds.get(row["memKind"], str(row["memKind"])),
                operation=opers.get(row["memoryOperationType"],
                                    str(row["memoryOperationType"])),
                name=row["name"] or "",
            )
        )


def _load_memcpy(conn: sqlite3.Connection, report: NsysReport) -> None:
    """Sum explicit host/device copies by direction.

    On a unified-memory part these should be small or absent during steady state; a decode
    loop that is still doing H2D copies every step is a finding in itself.
    """
    if not _table_exists(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return
    labels = {
        row["id"]: row["label"]
        for row in conn.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")
    } if _table_exists(conn, "ENUM_CUDA_MEMCPY_OPER") else {}

    for row in conn.execute(
        "SELECT copyKind, SUM(bytes) AS total FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
    ):
        label = labels.get(row["copyKind"], f"kind_{row['copyKind']}")
        report.memcpy_bytes[label] = int(row["total"] or 0)


def _load_gpu_metrics(conn: sqlite3.Connection, report: NsysReport) -> None:
    """Load the sampled hardware counter series.

    On GB10 the ``gb20b`` set carries clocks, engine activity and warp occupancy -- but no
    memory bandwidth rows. Bandwidth comes from the ncu pass; this series is for clock
    stability and occupancy context.
    """
    if not (_table_exists(conn, "GPU_METRICS")
            and _table_exists(conn, "TARGET_INFO_GPU_METRICS")):
        return

    names = {
        (row["typeId"], row["metricId"]): row["metricName"]
        for row in conn.execute(
            "SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS"
        )
    }
    for row in conn.execute(
        "SELECT timestamp, typeId, metricId, value FROM GPU_METRICS ORDER BY timestamp"
    ):
        name = names.get((row["typeId"], row["metricId"]))
        if name is None:
            continue
        report.gpu_metrics.setdefault(name, []).append((row["timestamp"], row["value"]))


def _load_um_faults(conn: sqlite3.Connection, report: NsysReport) -> None:
    """Count unified-memory page faults.

    These matter far more here than on a discrete GPU: CPU and GPU share one coherent pool,
    so a fault means a genuine migration or first-touch, not a PCIe transfer. A steady-state
    decode loop should show approximately none.
    """
    for table, label in (
        ("CUDA_UM_CPU_PAGE_FAULT_EVENTS", "cpu_page_faults"),
        ("CUDA_UM_GPU_PAGE_FAULT_EVENTS", "gpu_page_faults"),
    ):
        if _table_exists(conn, table):
            count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            report.um_page_faults[label] = int(count or 0)
