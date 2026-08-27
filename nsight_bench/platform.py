"""Platform detection and capability probing.

The harness refuses to guess about hardware. Before any benchmark runs, this module records
what the machine actually is and what the profilers can actually do on it, into a
``platform_profile.json`` that every run manifest references.

That matters more than usual here. GB10 is new enough that the standard Nsight metric names
from NVIDIA's own documentation do not all exist, GPU memory reporting through NVML returns
nothing at all, and Nsight Systems' CPU sampling is unavailable under the kernel's default
paranoid level. Discovering each of those at analysis time -- after a long profiling run --
would waste hours, so we discover them up front and fail loudly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import metrics as M

# Chip -> Nsight Systems GPU metric set alias. nsys will not auto-select the right set, and
# passing a mismatched one is a hard error, so the mapping is explicit.
NSYS_METRIC_SET_BY_CHIP = {
    "GB20B": "gb20b",   # DGX Spark / GB10
    "GB10B": "gb10b",
    "GB20": "gb20x",
    "GB10": "gb10x",
    "GH100": "gh100",
    "GA100": "ga100",
    "AD10": "ad10x",
}


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run a command, never raising. Returns (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s: {' '.join(cmd)}"


def _first_group(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


# --------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------


@dataclass
class GpuInfo:
    name: str = ""
    chip: str = ""
    compute_capability: str = ""
    sm_count: int = 0
    total_memory_bytes: int = 0
    l2_cache_bytes: int = 0
    shared_mem_per_block_bytes: int = 0
    shared_mem_per_sm_bytes: int = 0
    regs_per_sm: int = 0
    warp_size: int = 32
    is_integrated: bool = False
    driver_version: str = ""
    uuid: str = ""

    @property
    def unified_memory(self) -> bool:
        """True when GPU and CPU share one physical pool, as on GB10.

        This flips several analysis decisions: NVML memory queries are meaningless, host RSS
        and GPU allocations draw on the same budget, and past-L2 traffic is tagged ``sysmem``
        rather than ``device``.
        """
        return self.is_integrated


@dataclass
class ToolInfo:
    nsys_path: str = ""
    nsys_version: str = ""
    ncu_path: str = ""
    ncu_version: str = ""
    cuda_version: str = ""
    nvcc_path: str = ""


@dataclass
class PermissionInfo:
    ncu_counters_allowed: bool = False
    ncu_permission_detail: str = ""
    restrict_profiling_to_admin: bool | None = None
    perf_event_paranoid: int | None = None
    nsys_cpu_sampling_available: bool = False
    has_passwordless_sudo: bool = False

    @property
    def nsys_sampling_flags(self) -> list[str]:
        """Flags forcing nsys off CPU sampling when the kernel will not permit it.

        Without these nsys still profiles the GPU correctly but emits a wall of warnings and
        wastes setup time attempting perf_event_open.
        """
        if self.nsys_cpu_sampling_available:
            return []
        return ["--sample=none", "--cpuctxsw=none"]


@dataclass
class PlatformProfile:
    """Everything the harness knows about this machine. Serialised to platform_profile.json."""

    detected_at: str = ""
    hostname: str = ""
    kernel: str = ""
    arch: str = ""
    cpu_model: str = ""
    cpu_count: int = 0
    numa_nodes: int = 1
    host_memory_bytes: int = 0

    gpu: GpuInfo = field(default_factory=GpuInfo)
    tools: ToolInfo = field(default_factory=ToolInfo)
    permissions: PermissionInfo = field(default_factory=PermissionInfo)

    python_executable: str = ""
    torch_version: str = ""
    torch_cuda_version: str = ""

    nsys_gpu_metric_set: str = ""
    metric_available: list[str] = field(default_factory=list)
    metric_missing: dict[str, str] = field(default_factory=dict)
    unavailable_sections: dict[str, str] = field(default_factory=dict)

    notes: list[str] = field(default_factory=list)

    # ---- serialisation ----------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> PlatformProfile:
        raw = json.loads(Path(path).read_text())
        profile = cls(**{k: v for k, v in raw.items()
                         if k not in ("gpu", "tools", "permissions")})
        profile.gpu = GpuInfo(**raw.get("gpu", {}))
        profile.tools = ToolInfo(**raw.get("tools", {}))
        profile.permissions = PermissionInfo(**raw.get("permissions", {}))
        return profile

    # ---- convenience ------------------------------------------------------------------

    @property
    def availability(self) -> M.MetricAvailability:
        return M.MetricAvailability(
            available=set(self.metric_available), missing=dict(self.metric_missing)
        )

    def unprobed_metrics(self) -> list[str]:
        """Registry metrics this profile has no verdict on.

        A profile saved before a metric was added to the registry lists it as neither
        available nor missing. The runner filters its ``--metrics`` argument against the
        available set, so an unprobed metric is silently dropped from every collection --
        and its absence then shows up much later as an unexplained blank column in the
        report. Detecting the gap turns that into an actionable "re-run preflight".
        """
        known = set(self.metric_available) | set(self.metric_missing)
        return [name for name in M.tier1_names() if name not in known]

    @property
    def stale(self) -> bool:
        return bool(self.unprobed_metrics())

    def describe(self) -> str:
        g = self.gpu
        lines = [
            f"host      {self.hostname}  ({self.arch}, {self.cpu_count} cores, "
            f"{self.host_memory_bytes / 1e9:.0f} GB RAM)",
            f"gpu       {g.name} [{g.chip}] sm_{g.compute_capability.replace('.', '')} "
            f"{g.sm_count} SMs"
            + ("  UNIFIED MEMORY" if g.unified_memory else ""),
            f"memory    {g.total_memory_bytes / 1e9:.1f} GB   L2 {g.l2_cache_bytes / 1e6:.2f} MB   "
            f"shared {g.shared_mem_per_sm_bytes // 1024} KB/SM",
            f"tools     nsys {self.tools.nsys_version}   ncu {self.tools.ncu_version}   "
            f"cuda {self.tools.cuda_version}",
            f"torch     {self.torch_version} (cuda {self.torch_cuda_version})",
            f"ncu       counters {'allowed' if self.permissions.ncu_counters_allowed else 'BLOCKED'}",
            f"nsys      cpu sampling "
            f"{'available' if self.permissions.nsys_cpu_sampling_available else 'unavailable'}"
            f"   gpu metric set '{self.nsys_gpu_metric_set or 'none'}'",
            f"metrics   {self.availability.summary()}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Individual probes
# --------------------------------------------------------------------------------------


def probe_host() -> dict:
    uname = os.uname()
    info = {
        "hostname": uname.nodename,
        "kernel": f"{uname.sysname} {uname.release}",
        "arch": uname.machine,
        "cpu_count": os.cpu_count() or 0,
        "cpu_model": "",
        "numa_nodes": 1,
        "host_memory_bytes": 0,
    }

    try:
        lscpu = _run(["lscpu"])[1]
        models = re.findall(r"^Model name:\s+(.+)$", lscpu, re.MULTILINE)
        # Heterogeneous ARM SoCs report one line per core cluster (e.g. Cortex-X925 + A725).
        info["cpu_model"] = " + ".join(dict.fromkeys(m.strip() for m in models))
        nodes = _first_group(r"^NUMA node\(s\):\s+(\d+)", lscpu, "1")
        info["numa_nodes"] = int(nodes)
    except Exception:                                            # noqa: BLE001
        pass

    try:
        meminfo = Path("/proc/meminfo").read_text()
        kb = int(_first_group(r"MemTotal:\s+(\d+)", meminfo, "0"))
        info["host_memory_bytes"] = kb * 1024
    except Exception:                                            # noqa: BLE001
        pass

    return info


def probe_gpu() -> GpuInfo:
    """Read GPU properties, preferring torch (richer) and falling back to nvidia-smi."""
    gpu = GpuInfo()

    rc, out, _ = _run([
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,uuid",
        "--format=csv,noheader",
    ])
    if rc == 0 and out.strip():
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        if len(parts) >= 4:
            gpu.name, gpu.compute_capability, gpu.driver_version, gpu.uuid = parts[:4]

    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gpu.name = p.name
            gpu.compute_capability = f"{p.major}.{p.minor}"
            gpu.sm_count = p.multi_processor_count
            gpu.total_memory_bytes = p.total_memory
            gpu.l2_cache_bytes = getattr(p, "L2_cache_size", 0)
            gpu.shared_mem_per_block_bytes = getattr(p, "shared_memory_per_block", 0)
            gpu.shared_mem_per_sm_bytes = getattr(p, "shared_memory_per_multiprocessor", 0)
            gpu.regs_per_sm = getattr(p, "regs_per_multiprocessor", 0)
            gpu.warp_size = getattr(p, "warp_size", 32)
            gpu.is_integrated = bool(getattr(p, "is_integrated", 0))
    except Exception:                                            # noqa: BLE001
        pass

    gpu.chip = probe_chip_name() or gpu.chip
    return gpu


def probe_chip_name() -> str:
    """Resolve the internal chip name (e.g. GB20B), which the nsys metric set is keyed on.

    The marketing name ('NVIDIA GB10') and the chip name ('GB20B') differ on this part, and
    only the chip name selects the right ``--gpu-metrics-set``.
    """
    rc, out, _ = _run(["ncu", "--devices", "0", "--query-metrics"], timeout=120)
    if rc == 0:
        # ncu prints e.g. "Device NVIDIA GB10 (GB20B)"
        chip = _first_group(r"^Device .*\(([A-Z0-9]+)\)", out, "")
        if chip:
            return chip

    rc, out, _ = _run(["nsys", "profile", "--gpu-metrics-devices=help"], timeout=60)
    if rc == 0:
        # e.g. "0: Blackwell GB20B | NVIDIA GB10 PCI[...]"
        return _first_group(r"^\s*0:\s+\S+\s+([A-Z0-9]+)\s*\|", out, "")
    return ""


def probe_tools() -> ToolInfo:
    tools = ToolInfo()

    tools.nsys_path = shutil.which("nsys") or ""
    if tools.nsys_path:
        _, out, _ = _run([tools.nsys_path, "--version"])
        tools.nsys_version = _first_group(r"version\s+(\S+)", out, out.strip()[:40])

    tools.ncu_path = shutil.which("ncu") or ""
    if tools.ncu_path:
        _, out, _ = _run([tools.ncu_path, "--version"])
        tools.ncu_version = _first_group(r"Version\s+(\S+)", out, "")

    tools.nvcc_path = shutil.which("nvcc") or ""
    if tools.nvcc_path:
        _, out, _ = _run([tools.nvcc_path, "--version"])
        tools.cuda_version = _first_group(r"release ([\d.]+)", out, "")

    return tools


def probe_permissions() -> PermissionInfo:
    """Determine what the profilers are permitted to do without escalation."""
    perms = PermissionInfo()

    # GPU performance counters. Reading the driver's modprobe config tells us the intent;
    # the authoritative answer comes from actually collecting a metric (probe_metrics).
    for conf in Path("/etc/modprobe.d").glob("*.conf"):
        try:
            text = conf.read_text()
        except OSError:
            continue
        value = _first_group(r"NVreg_RestrictProfilingToAdminUsers=(\d)", text, "")
        if value:
            perms.restrict_profiling_to_admin = value != "0"
            break

    try:
        perms.perf_event_paranoid = int(
            Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
        )
    except Exception:                                            # noqa: BLE001
        pass

    # nsys needs perf_event_open for CPU sampling. Paranoid > 2 forbids it for non-root.
    rc, out, _ = _run(["nsys", "status", "--environment"], timeout=60)
    if rc == 0:
        perms.nsys_cpu_sampling_available = "CPU Profiling Environment (process-tree): OK" in out
    else:
        paranoid = perms.perf_event_paranoid
        perms.nsys_cpu_sampling_available = paranoid is not None and paranoid <= 2

    rc, _, _ = _run(["sudo", "-n", "true"], timeout=15)
    perms.has_passwordless_sudo = rc == 0

    return perms


PROBE_KERNEL_SOURCE = """\
import torch

# Two allocations that comfortably exceed L2 so the kernel actually touches memory, kept
# small enough that the probe stays fast under ncu's multi-pass replay.
n = 4 * 1024 * 1024
a = torch.randn(n, device="cuda", dtype=torch.float32)
b = torch.empty_like(a)
torch.cuda.synchronize()
torch.mul(a, 2.0, out=b)
torch.cuda.synchronize()
print("probe-ok", float(b[0]))
"""


def probe_metrics(
    python_executable: str | None = None,
    workdir: str | Path | None = None,
    timeout: int = 900,
) -> M.MetricAvailability:
    """Discover which registry metrics genuinely collect on the attached GPU.

    This runs a real one-kernel ncu collection rather than consulting
    ``ncu --query-metrics``, because the query listing is not a reliable oracle in either
    direction: ``launch__*`` metrics collect fine but never appear in it, while
    ``derived__*`` metrics appear collectable but fail as standalone ``--metrics`` entries.

    On failure ncu names the offending metric in its error output, so we drop that one and
    retry. A handful of iterations converges on the collectable subset.
    """
    python_executable = python_executable or sys.executable
    workdir = Path(workdir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "_metric_probe.py"
    script.write_text(PROBE_KERNEL_SOURCE)

    candidates = M.tier1_names()
    missing: dict[str, str] = dict(M.KNOWN_MISSING)
    detail = ""

    # Each failed pass eliminates at least one metric, so this terminates. The cap is a
    # backstop against an ncu error format we do not recognise.
    for _ in range(len(candidates) + 2):
        if not candidates:
            break
        rc, out, err = _run(
            [
                "ncu",
                "--target-processes", "all",
                "--launch-count", "1",
                "--kernel-name", "regex:.*",
                "--metrics", ",".join(candidates),
                "--clock-control", "none",
                python_executable, "-B", str(script),
            ],
            timeout=timeout,
        )
        combined = out + err
        if rc == 0 and "Failed to find metric" not in combined:
            detail = "collected"
            break

        bad = re.findall(r"Failed to find metric (?:regex:\^)?([A-Za-z0-9_.]+?)(?:\\\.|\s|$)",
                         combined)
        if not bad:
            detail = (combined.strip().splitlines() or ["ncu collection failed"])[-1]
            break
        for name in bad:
            for candidate in list(candidates):
                if candidate.startswith(name):
                    candidates.remove(candidate)
                    missing[candidate] = "ncu reported: metric not found on this device"

    script.unlink(missing_ok=True)
    availability = M.MetricAvailability(available=set(candidates), missing=missing)
    if detail and detail != "collected":
        availability.missing.setdefault("_probe_error", detail)
    return availability


def probe_ncu_counter_access(
    python_executable: str | None = None, workdir: str | Path | None = None
) -> tuple[bool, str]:
    """Check whether ncu can read GPU performance counters as this user.

    On a locked-down driver this is the failure that stops the whole project, and its error
    message is distinctive ("ERR_NVGPUCTRPERM"), so it is worth testing on its own.
    """
    python_executable = python_executable or sys.executable
    workdir = Path(workdir or Path.cwd())
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "_perm_probe.py"
    script.write_text(PROBE_KERNEL_SOURCE)

    rc, out, err = _run(
        [
            "ncu", "--target-processes", "all", "--launch-count", "1",
            "--kernel-name", "regex:.*",
            "--metrics", "gpu__time_duration.sum",
            "--clock-control", "none",
            python_executable, "-B", str(script),
        ],
        timeout=600,
    )
    script.unlink(missing_ok=True)
    combined = out + err

    if "ERR_NVGPUCTRPERM" in combined:
        return False, (
            "GPU performance counters are restricted to admin users. Fix with: "
            "echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | "
            "sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo update-initramfs -u && reboot"
        )
    if rc != 0:
        return False, (combined.strip().splitlines() or ["ncu failed"])[-1]
    return True, "counters readable without elevation"


# --------------------------------------------------------------------------------------
# Top-level detection
# --------------------------------------------------------------------------------------


def detect(
    probe_metric_availability: bool = True,
    python_executable: str | None = None,
    workdir: str | Path | None = None,
) -> PlatformProfile:
    """Build a full :class:`PlatformProfile` for this machine."""
    profile = PlatformProfile(
        detected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        python_executable=python_executable or sys.executable,
    )
    profile.__dict__.update(probe_host())
    profile.gpu = probe_gpu()
    profile.tools = probe_tools()
    profile.permissions = probe_permissions()
    profile.unavailable_sections = dict(M.UNAVAILABLE_SECTIONS)

    try:
        import torch

        profile.torch_version = torch.__version__
        profile.torch_cuda_version = torch.version.cuda or ""
    except Exception:                                            # noqa: BLE001
        profile.notes.append("torch not importable in this interpreter")

    profile.nsys_gpu_metric_set = NSYS_METRIC_SET_BY_CHIP.get(profile.gpu.chip, "")
    if not profile.nsys_gpu_metric_set and profile.gpu.chip:
        profile.notes.append(
            f"No nsys GPU metric set mapped for chip {profile.gpu.chip}; "
            "GPU metric sampling will be disabled. Check 'nsys profile --gpu-metrics-set=help'."
        )

    workdir = Path(workdir or Path.cwd())
    allowed, detail = probe_ncu_counter_access(profile.python_executable, workdir)
    profile.permissions.ncu_counters_allowed = allowed
    profile.permissions.ncu_permission_detail = detail

    if probe_metric_availability and allowed:
        availability = probe_metrics(profile.python_executable, workdir)
        profile.metric_available = sorted(availability.available)
        profile.metric_missing = availability.missing
        if not availability.ok:
            profile.notes.append(
                "Required metrics unavailable: " + ", ".join(availability.missing_required())
            )
    elif not allowed:
        profile.notes.append(f"Skipped metric probe -- {detail}")

    _add_platform_notes(profile)
    return profile


def _add_platform_notes(profile: PlatformProfile) -> None:
    """Record the consequences of this hardware, so reports can explain themselves."""
    gpu = profile.gpu

    if gpu.unified_memory:
        profile.notes.append(
            "Unified memory part: GPU and CPU share one physical pool. Past-L2 traffic is "
            "tagged with the sysmem aperture, and NVML GPU memory queries return nothing -- "
            "footprint is tracked via cudaMemGetInfo, the torch allocator and /proc/meminfo."
        )

    if "dram__bytes.sum" not in profile.metric_available:
        profile.notes.append(
            "No dram__* counters on this chip. DRAM bytes are derived from "
            "lts__t_sectors_aperture_sysmem_lookup_miss x 32 B, calibrated against a "
            "known-size streaming kernel before every run."
        )

    if not profile.permissions.nsys_cpu_sampling_available:
        paranoid = profile.permissions.perf_event_paranoid
        profile.notes.append(
            f"nsys CPU sampling unavailable (perf_event_paranoid={paranoid}). GPU tracing is "
            "unaffected; --sample=none --cpuctxsw=none is passed automatically."
        )

    rc, out, _ = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"])
    if rc == 0 and "N/A" in out:
        profile.notes.append(
            "nvidia-smi reports GPU memory as N/A on this part; do not use NVML for footprint."
        )


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "configs" / "platform_profile.json"


def load_or_detect(path: str | Path | None = None, **kwargs) -> PlatformProfile:
    """Load a cached profile, detecting and caching one if absent."""
    path = Path(path or DEFAULT_PROFILE_PATH)
    if path.exists():
        return PlatformProfile.load(path)
    profile = detect(**kwargs)
    profile.save(path)
    return profile
