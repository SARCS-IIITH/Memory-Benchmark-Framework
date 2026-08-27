"""Shared runner machinery: the run directory, subprocess execution, provenance.

Every collector writes into the same directory layout and records the same provenance, so
two runs from different days on different models can be compared without archaeology. The
verbatim command lines are kept in the manifest deliberately -- a profiler result is only
reproducible if you know exactly which flags produced it, and those flags vary with the
platform probe.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp() -> str:
    """Timestamp used in run directory names: sortable, filesystem-safe."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunPaths:
    """Directory layout for a single run."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def raw(self) -> Path:
        """Profiler artefacts exactly as the tools wrote them (.nsys-rep, .ncu-rep, .sqlite)."""
        return self.root / "raw"

    @property
    def metrics(self) -> Path:
        """Tidy, machine-readable extracts (CSV/JSON)."""
        return self.root / "metrics"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def create(self) -> RunPaths:
        for directory in (self.root, self.raw, self.metrics, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class CommandResult:
    """Outcome of one external command, kept for the manifest."""

    argv: list[str] = field(default_factory=list)
    returncode: int = 0
    seconds: float = 0.0
    stdout_path: str = ""
    stderr_path: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def command_line(self) -> str:
        """The command as you would retype it in a shell."""
        return " ".join(shlex.quote(a) for a in self.argv)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["command_line"] = self.command_line
        data["ok"] = self.ok
        return data


def run_command(
    argv: list[str],
    log_dir: Path,
    log_name: str,
    timeout: int = 3600,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> CommandResult:
    """Run a command, streaming both streams to files and keeping a tail for the manifest.

    Profiler output is far too large to hold in memory or paste into a report -- ncu alone
    can emit tens of megabytes of per-kernel tables -- so it goes to disk and only the last
    few lines, which is where the errors are, come back inline.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{log_name}.stdout.log"
    stderr_path = log_dir / f"{log_name}.stderr.log"

    full_env = {**os.environ, **(env or {})}
    started = time.perf_counter()

    with stdout_path.open("w") as out, stderr_path.open("w") as err:
        try:
            proc = subprocess.run(
                argv, stdout=out, stderr=err, timeout=timeout,
                env=full_env, cwd=str(cwd) if cwd else None, check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            err.write(f"\n[nsbench] timed out after {timeout}s\n")
        except FileNotFoundError as exc:
            returncode = 127
            err.write(f"\n[nsbench] {exc}\n")

    return CommandResult(
        argv=argv,
        returncode=returncode,
        seconds=time.perf_counter() - started,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_tail=_tail(stdout_path),
        stderr_tail=_tail(stderr_path),
    )


def _tail(path: Path, lines: int = 20) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError:
        return ""


def file_digest(path: Path, chunk_bytes: int = 8 << 20, full: bool = False) -> str:
    """Content digest for a weight file.

    Multi-gigabyte safetensors shards make a full hash expensive -- tens of seconds each,
    repeated for every run -- so by default this samples the head and tail plus the file
    size. That is enough to detect a checkpoint being swapped or truncated, which is what
    the digest is for. Pass ``full=True`` when an exact hash is genuinely needed.
    """
    import hashlib

    hasher = hashlib.sha256()
    try:
        size = path.stat().st_size
        hasher.update(str(size).encode())
        with path.open("rb") as handle:
            if full or size <= 2 * chunk_bytes:
                for block in iter(lambda: handle.read(chunk_bytes), b""):
                    hasher.update(block)
            else:
                hasher.update(handle.read(chunk_bytes))
                handle.seek(-chunk_bytes, os.SEEK_END)
                hasher.update(handle.read(chunk_bytes))
        prefix = "sha256" if (full or size <= 2 * chunk_bytes) else "sha256-sampled"
        return f"{prefix}:{hasher.hexdigest()[:32]}"
    except OSError:
        return ""


def model_provenance(model_path: str | Path, files: list[str], full_hash: bool = False) -> dict:
    """Record enough about the weights to detect them changing between runs."""
    directory = Path(model_path)
    entries = []
    total = 0
    for name in files:
        path = directory / name
        if not path.exists():
            continue
        stat = path.stat()
        total += stat.st_size
        entries.append({
            "file": name,
            "bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "digest": file_digest(path, full=full_hash),
        })
    return {"path": str(directory), "total_bytes": total, "files": entries}


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    """Write rows to CSV without requiring pandas.

    The parsers produce plain dicts, and keeping the CSV writer dependency-free means the
    metrics directory is populated even if the analysis stack is missing.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path

    if columns is None:
        seen: dict[str, None] = {}
        for row in rows:
            for key in row:
                seen.setdefault(key, None)
        columns = list(seen)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
