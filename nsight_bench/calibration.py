"""Calibration: prove the derivation before trusting any model measurement.

Everything this harness reports about memory rests on one substitution -- that on GB10,
``lts__t_sectors_aperture_sysmem_lookup_miss x 32 B`` is the traffic that reaches the unified
LPDDR5X pool, standing in for the ``dram__*`` counters the chip does not have. That claim is
load-bearing, so it is not taken on faith: it is re-verified against a known quantity every
time, and the result is a pass/fail gate recorded in the run manifest.

Two microbenchmarks run here:

**Byte accounting.** A streaming kernel touches a precisely known number of bytes. The
derived figure must match. When this was developed, a 256 MiB elementwise multiply -- 256 MiB
read plus 256 MiB written -- measured 537.0 MB against 536.9 MB expected, and the device and
peer aperture counters both read zero. If a future driver or architecture starts routing
traffic through the device aperture, this gate fails loudly instead of silently halving every
reported DRAM figure.

**L2 knee and bandwidth ceiling.** A working-set sweep from well inside L2 to well beyond it.
This locates the 25.17 MB L2 empirically and establishes the achievable streaming bandwidth,
which is the memory-bound ceiling on the roofline. Without a measured ceiling the roofline
would need a datasheet number, and datasheet bandwidth is not a bound any real kernel reaches.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .metrics import SECTOR_BYTES

#: Working-set sizes for the sweep, in MiB. Chosen to bracket the 25.17 MB L2 closely --
#: 16 and 24 MiB sit inside it, 32 and 48 just outside -- so the knee lands between sample
#: points rather than being inferred from a wide gap.
DEFAULT_SWEEP_MIB = (1, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024)

#: Tolerance for the byte-accounting gate. Real traffic includes a small amount of
#: instruction fetch and page-table walk that the analytic figure does not model, so an exact
#: match is not expected -- but anything beyond this indicates a broken derivation, not noise.
BYTE_ACCOUNTING_TOLERANCE = 0.05


@dataclass
class ByteAccountingResult:
    """Outcome of the derivation gate."""

    expected_bytes: int = 0
    measured_bytes: float = 0.0
    device_aperture_sectors: float = 0.0
    peer_aperture_sectors: float = 0.0
    kernel_name: str = ""
    duration_ns: float = 0.0
    error: str = ""

    @property
    def relative_error(self) -> float | None:
        if not self.expected_bytes:
            return None
        return (self.measured_bytes - self.expected_bytes) / self.expected_bytes

    @property
    def sentinels_clean(self) -> bool:
        return not (self.device_aperture_sectors or self.peer_aperture_sectors)

    @property
    def passed(self) -> bool:
        error = self.relative_error
        return (
            not self.error
            and error is not None
            and abs(error) <= BYTE_ACCOUNTING_TOLERANCE
            and self.sentinels_clean
        )

    @property
    def replay_bandwidth_gbps(self) -> float | None:
        """Measured bytes over the kernel's **profiled** duration.

        This is not a performance figure and must never be quoted as one. The duration comes
        from a kernel that ncu replayed with L2 flushed before each pass, so it is neither
        the time the kernel takes normally nor a rate the machine can sustain. It is kept
        only as a sanity check that the gate's kernel did roughly what a streaming kernel
        should -- the achievable ceiling comes from the unprofiled sweep instead.
        """
        if self.duration_ns <= 0:
            return None
        return self.measured_bytes / (self.duration_ns / 1e9) / 1e9

    def summary(self) -> str:
        if self.error:
            return f"FAIL -- {self.error}"
        error = self.relative_error
        if error is None:
            return "FAIL -- nothing measured"
        status = "PASS" if self.passed else "FAIL"
        line = (
            f"{status}: expected {self.expected_bytes / 1e6:,.1f} MB, "
            f"measured {self.measured_bytes / 1e6:,.1f} MB ({error:+.2%})"
        )
        if not self.sentinels_clean:
            line += (
                f"; SENTINEL TRIPPED device={self.device_aperture_sectors:,.0f} "
                f"peer={self.peer_aperture_sectors:,.0f} sectors"
            )
        # The replay rate is deliberately absent here. This gate is about byte accounting,
        # and appending a GB/s figure derived from a profiled, cache-flushed replay invites
        # exactly the misreading the rest of the harness works to prevent.
        return line

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update({
            "relative_error": self.relative_error,
            "sentinels_clean": self.sentinels_clean,
            "passed": self.passed,
            "replay_bandwidth_gbps_not_performance": self.replay_bandwidth_gbps,
            "summary": self.summary(),
        })
        return data


@dataclass
class SweepPoint:
    """One working-set size in the L2 sweep."""

    #: Total live footprint the kernel touches -- BOTH arrays, source and destination.
    #: Reporting only the source array size would place the knee at half its true working
    #: set and make it look as though L2 stopped helping well before capacity.
    working_set_mib: float = 0.0
    #: Size of one of the two arrays, which is what the sweep iterates over.
    array_mib: float = 0.0
    bytes_touched: int = 0
    seconds: float = 0.0
    bandwidth_gbps: float = 0.0
    l2_hit_rate_pct: float | None = None
    dram_bytes: float | None = None

    @property
    def fits_in_l2(self) -> bool | None:
        return None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationResult:
    """Everything the calibration produced."""

    byte_accounting: ByteAccountingResult = field(default_factory=ByteAccountingResult)
    sweep: list[SweepPoint] = field(default_factory=list)
    l2_cache_bytes: int = 0

    #: Best bandwidth seen with the working set inside L2. This is an on-chip cache figure --
    #: several times the DRAM rate -- and must never be used as the roofline's memory bound.
    peak_l2_bandwidth_gbps: float = 0.0

    #: Best bandwidth seen with the working set well beyond L2, so every access reaches
    #: LPDDR5X. This is the memory-bound ceiling for the roofline.
    peak_dram_bandwidth_gbps: float = 0.0

    #: Measured dense bf16 GEMM throughput -- the roofline's compute ceiling.
    peak_compute_gflops: float = 0.0
    compute_peak_detail: dict = field(default_factory=dict)

    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.byte_accounting.passed

    def classify_sweep(self) -> None:
        """Split the sweep into cache-resident and memory-resident regimes.

        Conflating the two is the easy mistake here: an L2-resident streaming kernel on this
        part runs at roughly five times the LPDDR5X rate, so taking the sweep maximum as
        "peak bandwidth" would put the roofline's memory ceiling far above anything a real
        model could reach, and every kernel would look memory-efficient by comparison.
        """
        if not self.sweep or not self.l2_cache_bytes:
            if self.sweep:
                self.peak_dram_bandwidth_gbps = max(p.bandwidth_gbps for p in self.sweep)
            return

        l2_mib = self.l2_cache_bytes / (1024 * 1024)
        resident = [p for p in self.sweep if p.working_set_mib <= l2_mib * 0.75]
        # 4x L2 leaves no doubt that the working set cannot be held, so the figure is a
        # genuine memory-side rate rather than a partially-cached blend.
        streaming = [p for p in self.sweep if p.working_set_mib >= l2_mib * 4]

        if resident:
            self.peak_l2_bandwidth_gbps = max(p.bandwidth_gbps for p in resident)
        if streaming:
            self.peak_dram_bandwidth_gbps = max(p.bandwidth_gbps for p in streaming)
        elif self.sweep:
            self.peak_dram_bandwidth_gbps = min(p.bandwidth_gbps for p in self.sweep)

    def knee_mib(self) -> float | None:
        """Working-set size at which bandwidth falls off, i.e. where L2 stops helping.

        Found as the largest drop between adjacent points rather than by comparing against
        the L2 size, so the sweep reports what the hardware does rather than confirming what
        the datasheet says.
        """
        if len(self.sweep) < 3:
            return None
        biggest_drop, knee = 0.0, None
        for previous, current in zip(self.sweep, self.sweep[1:]):
            if previous.bandwidth_gbps <= 0:
                continue
            drop = (previous.bandwidth_gbps - current.bandwidth_gbps) / previous.bandwidth_gbps
            if drop > biggest_drop:
                biggest_drop, knee = drop, current.working_set_mib
        return knee if biggest_drop > 0.15 else None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "byte_accounting": self.byte_accounting.to_dict(),
            "sweep": [p.to_dict() for p in self.sweep],
            "l2_cache_bytes": self.l2_cache_bytes,
            "knee_mib": self.knee_mib(),
            "peak_l2_bandwidth_gbps": self.peak_l2_bandwidth_gbps,
            "peak_dram_bandwidth_gbps": self.peak_dram_bandwidth_gbps,
            "peak_compute_gflops": self.peak_compute_gflops,
            "compute_peak_detail": self.compute_peak_detail,
            "notes": self.notes,
        }

    @property
    def ridge_point(self) -> float | None:
        """Arithmetic intensity where the roofline turns from memory- to compute-bound.

        Below this, a kernel cannot be anything but memory-bound no matter how well written.
        LLM decode sits far below it, which is the structural reason decode is slow.
        """
        if not (self.peak_compute_gflops and self.peak_dram_bandwidth_gbps):
            return None
        return self.peak_compute_gflops / self.peak_dram_bandwidth_gbps


# --------------------------------------------------------------------------------------
# The GPU-side microbenchmarks (run in the worker process, possibly under ncu)
# --------------------------------------------------------------------------------------


def _stream_kernel(elements: int, iterations: int = 1):
    """Elementwise multiply over ``elements`` fp32 values.

    Reads one array and writes another, so bytes touched is exactly ``2 x 4 x elements``.
    ``torch.mul(..., out=)`` is used rather than ``copy_`` because a same-device copy is
    serviced by the copy engine, which produces no kernel for ncu to profile.
    """
    import torch

    a = torch.randn(elements, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    torch.cuda.synchronize()
    for _ in range(iterations):
        torch.mul(a, 2.0, out=b)
    torch.cuda.synchronize()
    return 2 * 4 * elements * iterations


def run_stream_microbenchmark(megabytes: int = 256) -> int:
    """Run the byte-accounting kernel once. Returns bytes touched.

    Wrapped in an NVTX range so ncu can scope to it and ignore the setup allocations.
    """
    from .instrumentation.nvtx import Phase, nvtx_range

    elements = (megabytes * 1024 * 1024) // 4
    import torch

    a = torch.randn(elements, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    torch.cuda.synchronize()

    with nvtx_range(Phase.CALIBRATE.range_name):
        torch.mul(a, 2.0, out=b)
        torch.cuda.synchronize()

    return 2 * 4 * elements


def run_compute_peak(size: int = 8192, iterations: int = 20, dtype: str = "bfloat16") -> dict:
    """Measure achievable dense bf16 GEMM throughput -- the roofline's compute ceiling.

    Measured rather than taken from a spec sheet for the same reason the bandwidth ceiling
    is: a datasheet number includes sparsity and clock assumptions no real kernel meets, so
    plotting against it would make every kernel look far from the roof regardless of how
    well it is actually doing.

    A square GEMM at this size is comfortably compute-bound, so the result is a genuine
    ceiling rather than a bandwidth measurement in disguise.
    """
    import time

    import torch

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        dtype, torch.bfloat16
    )
    a = torch.randn(size, size, device="cuda", dtype=torch_dtype)
    b = torch.randn(size, size, device="cuda", dtype=torch_dtype)

    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        _ = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # A square matmul is 2*N^3 floating point operations (one multiply, one add per term).
    flops = 2.0 * size ** 3 * iterations
    return {
        "dtype": dtype,
        "matrix_size": size,
        "seconds": elapsed,
        "gflops": flops / elapsed / 1e9 if elapsed > 0 else 0.0,
    }


def run_bandwidth_sweep(sizes_mib=DEFAULT_SWEEP_MIB, iterations: int = 20) -> list[dict]:
    """Time the streaming kernel across working-set sizes.

    Run without a profiler attached: this measures achievable bandwidth, and ncu's replay
    would report the timing of a serialised re-execution instead.
    """
    import time

    import torch

    results = []
    for mib in sizes_mib:
        elements = (int(mib) * 1024 * 1024) // 4
        if elements < 1024:
            continue
        try:
            a = torch.randn(elements, device="cuda", dtype=torch.float32)
            b = torch.empty_like(a)
        except torch.cuda.OutOfMemoryError:
            break

        # Warm up so the measurement excludes first-touch page faults, which on a unified
        # memory part are paid on the host side and are substantial.
        for _ in range(3):
            torch.mul(a, 2.0, out=b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            torch.mul(a, 2.0, out=b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        bytes_touched = 2 * 4 * elements * iterations
        results.append({
            # Two arrays are live, so the footprint competing for L2 is twice the array size.
            "working_set_mib": 2.0 * float(mib),
            "array_mib": float(mib),
            "bytes_touched": bytes_touched,
            "seconds": elapsed,
            "bandwidth_gbps": bytes_touched / elapsed / 1e9 if elapsed > 0 else 0.0,
        })
        del a, b
        torch.cuda.empty_cache()

    return results


# --------------------------------------------------------------------------------------
# Parsing the profiled calibration
# --------------------------------------------------------------------------------------


def parse_byte_accounting(csv_path: str | Path, expected_bytes: int) -> ByteAccountingResult:
    """Read the ncu export from the calibration run and evaluate the gate."""
    from .parsers.ncu_parse import parse_csv

    report = parse_csv(csv_path, scope="calibration", tier=1)
    result = ByteAccountingResult(expected_bytes=expected_bytes)

    if not report.kernels:
        result.error = (
            "the calibration kernel was not profiled. "
            + ("; ".join(report.warnings) if report.warnings else "no kernels in export")
        )
        return result

    # The streaming multiply is the kernel that moved the most; the setup randn also runs
    # inside the process and would otherwise be picked up.
    kernel = max(
        report.kernels,
        key=lambda k: k.metrics.get("lts__t_sectors_aperture_sysmem_lookup_miss.sum", 0),
    )
    result.kernel_name = kernel.short_name
    result.duration_ns = kernel.duration_ns
    result.measured_bytes = (
        kernel.metrics.get("lts__t_sectors_aperture_sysmem_lookup_miss.sum", 0.0)
        * SECTOR_BYTES
    )
    result.device_aperture_sectors = kernel.metrics.get(
        "lts__t_sectors_aperture_device_lookup_miss.sum", 0.0
    )
    result.peer_aperture_sectors = kernel.metrics.get(
        "lts__t_sectors_aperture_peer_lookup_miss.sum", 0.0
    )
    return result


# --------------------------------------------------------------------------------------
# Worker entry point
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run one calibration task in this process.

    Launched directly for the bandwidth sweep, and under ncu for the byte-accounting gate.
    """
    parser = argparse.ArgumentParser(
        prog="nsight_bench.calibration",
        description="Run a memory calibration microbenchmark.",
    )
    parser.add_argument("--task", choices=["stream", "sweep", "compute"], default="stream")
    parser.add_argument("--megabytes", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    payload: dict = {"task": args.task}
    try:
        if args.task == "stream":
            payload["bytes_touched"] = run_stream_microbenchmark(args.megabytes)
            payload["megabytes"] = args.megabytes
        elif args.task == "compute":
            payload["compute"] = run_compute_peak(iterations=args.iterations)
        else:
            payload["sweep"] = run_bandwidth_sweep(iterations=args.iterations)
            payload["compute"] = run_compute_peak(iterations=args.iterations)
        payload["ok"] = True
    except Exception as exc:                                     # noqa: BLE001
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"

    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    else:
        print(text)

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
