"""Parse Nsight Compute CSV exports into tidy per-kernel records.

``ncu --csv --page raw`` emits a *wide* table -- one row per profiled kernel launch, with
every metric as its own column -- preceded by a units row. It looks like this::

    "ID","Process ID",...,"Kernel Name",...,"gpu__time_duration.sum","lts__t_sectors.sum",...
    "","",...,"",...,"ms","sector",...
    "0","25586",...,"void vectorized_elementwise_kernel<...>",...,"1.516736","8,400,907",...

Three things about that format bite anyone parsing it naively, and all three are handled here:

**The units row is not optional information.** ``gpu__time_duration.sum`` comes back in
*milliseconds*, not the nanoseconds the metric's own description implies, and byte-valued
metrics are scaled to Kbyte/Mbyte/Gbyte depending on magnitude. Reading the numbers without
the units row produces values wrong by factors of a million, in a direction that still looks
plausible. Every value is therefore normalised to a base unit (ns, byte, sector) on the way
in, and the raw unit is retained.

**Values carry thousands separators.** ``"8,400,907"`` is one number, not a CSV field
boundary -- it survives because ncu quotes every field.

**Unavailable metrics appear as empty strings, not zeros.** On GB10 every ``dram__*`` column
that the MemoryWorkloadAnalysis section requests comes back blank, because those counters do
not exist on this chip. Storing them as 0.0 would report the GPU as doing no DRAM traffic at
all, so they are recorded as unavailable instead.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Non-metric columns that ncu always emits before the metric columns.
_META_COLUMNS = {
    "ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Function Name",
    "Demangled Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC",
    "Section Name", "Metric Name", "Metric Unit", "Metric Value", "Estimated Speedup",
    "Estimated Speedup Type", "Rule Name", "Rule Type", "Rule Description",
}

#: Values meaning "not collected". Distinct from 0.0 throughout the analysis, because a
#: counter that does not exist on this chip and a counter that measured zero traffic are
#: entirely different findings.
_NA_VALUES = {"", "n/a", "N/A", "nan", "NaN", "-", "inf", "Inf", "<null>"}

#: Multipliers to a base unit. ncu autoscales magnitudes, so the same metric can arrive in
#: different units in different reports.
_UNIT_SCALE: dict[str, tuple[float, str]] = {
    # time -> nanoseconds
    "ns": (1.0, "ns"), "nsecond": (1.0, "ns"),
    "us": (1e3, "ns"), "usecond": (1e3, "ns"),
    "ms": (1e6, "ns"), "msecond": (1e6, "ns"),
    "s": (1e9, "ns"), "second": (1e9, "ns"),
    # bytes -> bytes
    "byte": (1.0, "byte"), "bytes": (1.0, "byte"),
    "Kbyte": (1e3, "byte"), "Mbyte": (1e6, "byte"), "Gbyte": (1e9, "byte"),
    "Tbyte": (1e12, "byte"),
    "byte/block": (1.0, "byte/block"),
    "Kbyte/block": (1e3, "byte/block"), "Mbyte/block": (1e6, "byte/block"),
}


def normalize_value(value: float, unit: str) -> tuple[float, str]:
    """Scale a value to its base unit. Returns ``(value, base_unit)``."""
    scale, base = _UNIT_SCALE.get(unit.strip(), (1.0, unit.strip()))
    return value * scale, base


@dataclass
class KernelRecord:
    """One profiled kernel launch with all its metrics, in base units."""

    launch_id: str = ""
    name: str = ""
    grid: str = ""
    block: str = ""
    stream: str = ""
    context: str = ""
    device: str = ""
    compute_capability: str = ""

    #: metric name -> value, normalised to ns / byte / sector / %.
    metrics: dict[str, float] = field(default_factory=dict)
    #: metric name -> base unit after normalisation.
    units: dict[str, str] = field(default_factory=dict)
    #: metric name -> unit exactly as ncu reported it, kept for auditability.
    raw_units: dict[str, str] = field(default_factory=dict)
    #: Metrics ncu emitted as blank -- requested but not collectable on this GPU.
    unavailable: list[str] = field(default_factory=list)
    #: Non-numeric columns (device attributes, config strings).
    attributes: dict[str, str] = field(default_factory=dict)

    def get(self, metric: str, default: float = 0.0) -> float:
        return self.metrics.get(metric, default)

    def has(self, metric: str) -> bool:
        return metric in self.metrics

    @property
    def duration_ns(self) -> float:
        return self.get("gpu__time_duration.sum")

    @property
    def short_name(self) -> str:
        """A readable kernel name for tables.

        Full names run to several hundred characters of template arguments; the leading
        identifier is what a person actually reads.
        """
        return shorten_kernel_name(self.name)

    def to_row(self) -> dict:
        row: dict = {
            "launch_id": self.launch_id,
            "kernel_name": self.name,
            "kernel_short": self.short_name,
            "grid": self.grid,
            "block": self.block,
            "stream": self.stream,
        }
        row.update(self.metrics)
        return row


#: Base identifiers that carry no information on their own -- cuBLAS names most of its
#: kernels ``kernel`` and cutlass names most of its ``Kernel2``. For these the informative
#: part lives in the template arguments, so a distinctive token is pulled from there instead.
_GENERIC_BASES = {"kernel", "Kernel", "Kernel2", "kernel_", "device_kernel"}

#: Recognisable families worth surfacing when the base name is generic, most specific first.
_FAMILY_PATTERNS = (
    re.compile(r"(nvjet_[a-z0-9_]+)", re.IGNORECASE),
    re.compile(r"(cutlass[a-z0-9_]*gemm[a-z0-9_]*)", re.IGNORECASE),
    re.compile(r"(flash_[a-z0-9_]+)", re.IGNORECASE),
    re.compile(r"(cublas[A-Za-z]*Gemv[A-Za-z]*)"),
    re.compile(r"(cublas[A-Za-z]*Gemm[A-Za-z]*)"),
    re.compile(r"([a-z0-9_]*gemm[a-z0-9_]*)", re.IGNORECASE),
)


def base_identifier(name: str) -> str:
    """The bare function identifier, matching what ncu's kernel-name filter uses.

    Bracket depth is tracked rather than splitting on the first ``<``, because template
    brackets routinely appear in the *return type*: ``std::enable_if<!T7, void>::type
    internal::kernel<...>(T13)`` would otherwise yield ``std::enable_if``.
    """
    if not name:
        return ""

    depth = 0
    buffer: list[str] = []
    candidates: list[str] = []
    for char in name:
        if char == "<":
            if depth == 0:
                candidates.append("".join(buffer))
                buffer = []
            depth += 1
        elif char == ">":
            depth = max(0, depth - 1)
            if depth == 0:
                buffer = []
        elif char == "(" and depth == 0:
            candidates.append("".join(buffer))
            break
        elif depth == 0:
            buffer.append(char)
    else:
        candidates.append("".join(buffer))

    for candidate in reversed(candidates):
        text = candidate.strip()
        if not text:
            continue
        token = (text.split()[-1] if text.split() else text).split("::")[-1].strip()
        if token:
            return token
    return name.strip()


def shorten_kernel_name(name: str, max_len: int = 44) -> str:
    """A readable kernel label for tables.

    Full signatures run to several hundred characters of template arguments. The base
    identifier is usually what a person reads -- except when a library names every kernel the
    same thing, in which case the identity lives in the template arguments and a distinctive
    family token is pulled from there instead.
    """
    if not name:
        return ""

    base = base_identifier(name)
    if base in _GENERIC_BASES or not base:
        for pattern in _FAMILY_PATTERNS:
            match = pattern.search(name)
            if match:
                candidate = match.group(1)
                if base and base not in _GENERIC_BASES:
                    candidate = f"{candidate}"
                return _clip(candidate, max_len)
    return _clip(base or name, max_len)


def _clip(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "\u2026"


@dataclass
class NcuReport:
    """All kernels from one ncu collection, plus what could not be collected."""

    source: str = ""
    scope: str = ""
    tier: int = 1
    kernels: list[KernelRecord] = field(default_factory=list)
    #: metric name -> how many kernels reported it as unavailable.
    unavailable_metrics: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.kernels)

    def metric_names(self) -> list[str]:
        names: dict[str, None] = {}
        for kernel in self.kernels:
            for name in kernel.metrics:
                names.setdefault(name, None)
        return list(names)

    def total(self, metric: str) -> float:
        """Sum a metric across every profiled launch."""
        return sum(k.get(metric) for k in self.kernels)

    def to_rows(self) -> list[dict]:
        return [k.to_row() for k in self.kernels]

    def by_name(self) -> dict[str, list[KernelRecord]]:
        grouped: dict[str, list[KernelRecord]] = {}
        for kernel in self.kernels:
            grouped.setdefault(kernel.name, []).append(kernel)
        return grouped

    def top_by(self, metric: str, n: int = 10) -> list[KernelRecord]:
        return sorted(self.kernels, key=lambda k: k.get(metric), reverse=True)[:n]


def _parse_number(raw: str) -> float | None:
    """Convert one ncu cell to a float, or None when it is not a number."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text in _NA_VALUES:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _is_metric_column(column: str) -> bool:
    """Metric columns are the qualified ``unit__name`` ones; everything else is metadata."""
    return "__" in column and column not in _META_COLUMNS


def parse_csv(path: str | Path, scope: str = "", tier: int = 1) -> NcuReport:
    """Parse one ``ncu --csv --page raw`` export into an :class:`NcuReport`."""
    path = Path(path)
    report = NcuReport(source=str(path), scope=scope, tier=tier)

    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        report.warnings.append(f"could not read {path}: {exc}")
        return report

    # ncu prefixes the CSV with progress banners ("==PROF== ..."). Find the real header
    # rather than assuming it is the first line.
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith('"ID"') or ('"Kernel Name"' in line and line.startswith('"')):
            start = index
            break
    if start is None:
        report.warnings.append(
            "no CSV header found -- was this exported with 'ncu --import ... --csv --page raw'?"
        )
        return report

    reader = csv.reader(io.StringIO("\n".join(lines[start:])))
    try:
        header = next(reader)
    except StopIteration:
        report.warnings.append("CSV contained only a header")
        return report

    # The row after the header holds units. It is identifiable because its metadata columns
    # are blank while its metric columns carry unit strings. A report with zero profiled
    # kernels has no units row at all, so this is detected rather than assumed.
    units_row: list[str] = []
    try:
        candidate = next(reader)
    except StopIteration:
        report.warnings.append("no kernels were profiled (header only)")
        return report

    id_index = header.index("ID") if "ID" in header else 0
    if not candidate[id_index].strip():
        units_row = candidate
        data_rows = list(reader)
    else:
        data_rows = [candidate, *reader]

    units = {
        column: (units_row[i].strip() if i < len(units_row) else "")
        for i, column in enumerate(header)
    }

    metric_columns = [c for c in header if _is_metric_column(c)]
    unavailable_counts: dict[str, int] = {}

    for values in data_rows:
        if not values or len(values) < len(header) // 2:
            continue
        row = dict(zip(header, values))

        record = KernelRecord(
            launch_id=(row.get("ID") or "").strip(),
            name=(row.get("Kernel Name") or row.get("Function Name") or "").strip(),
            grid=(row.get("Grid Size") or "").strip(),
            block=(row.get("Block Size") or "").strip(),
            stream=(row.get("Stream") or "").strip(),
            context=(row.get("Context") or "").strip(),
            device=(row.get("Device") or "").strip(),
            compute_capability=(row.get("CC") or "").strip(),
        )

        for column in metric_columns:
            raw = row.get(column)
            number = _parse_number(raw)
            if number is None:
                if raw is not None and str(raw).strip() in _NA_VALUES:
                    record.unavailable.append(column)
                    unavailable_counts[column] = unavailable_counts.get(column, 0) + 1
                elif raw is not None and str(raw).strip():
                    record.attributes[column] = str(raw).strip()
                continue

            raw_unit = units.get(column, "")
            value, base_unit = normalize_value(number, raw_unit)
            record.metrics[column] = value
            record.units[column] = base_unit
            if raw_unit:
                record.raw_units[column] = raw_unit

        if record.name or record.metrics:
            report.kernels.append(record)

    report.unavailable_metrics = unavailable_counts

    dram_missing = [m for m in unavailable_counts if m.startswith("dram__")]
    if dram_missing:
        report.warnings.append(
            f"{len(dram_missing)} dram__* metrics were requested by a section but do not "
            "exist on this GPU (expected on GB10). DRAM traffic is derived from the L2 "
            "sysmem aperture instead -- see nsight_bench/metrics.py."
        )
    if not report.kernels:
        report.warnings.append("no kernel rows found in the export")

    return report


def load_collections(collection_records: list[dict]) -> dict[tuple[str, int], NcuReport]:
    """Parse every successful collection from an ncu runner record.

    Returns ``(scope, tier) -> NcuReport``, which is how prefill and decode stay separated
    all the way through to the report.
    """
    reports: dict[tuple[str, int], NcuReport] = {}
    for record in collection_records:
        csv_path = record.get("csv_path")
        if not csv_path or not record.get("ok"):
            continue
        scope = record.get("scope", "")
        tier = int(record.get("tier", 1))
        reports[(scope, tier)] = parse_csv(csv_path, scope=scope, tier=tier)
    return reports
