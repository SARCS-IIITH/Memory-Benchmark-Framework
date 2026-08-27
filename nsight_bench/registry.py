"""The model catalogue: selection, fetching, and the shared weight store.

A benchmark suite has a bootstrapping problem the single-run path does not. ``nsbench run``
is handed a checkpoint directory that already exists; a suite is handed a list of *names* and
has to turn them into directories, which means resolving a registry entry, finding out whether
the weights are already on disk, and downloading a few hundred gigabytes if they are not.

Two decisions shape this module.

**Weights live in one shared store, not per-user caches.** These checkpoints run to 70 GB
each. A second user benchmarking the same model should not re-download it, and a machine
should not carry three copies of Qwen3-30B because three people tried it. The store is a
standard HuggingFace cache layout under one directory, so it is reused simply by pointing
``HF_HOME`` at it -- no bespoke path convention that only this harness understands.

**Fetching is separate from benchmarking, and both are resumable.** A 284 GB download and a
15-hour profiling run fail for entirely different reasons, and recovering from one should not
mean redoing the other. ``nsbench fetch`` can run days ahead of ``nsbench suite``, and the
suite skips any model already complete in the store.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "model-registry.yaml"

#: Where weights go unless overridden. Chosen to be outside any one user's home so a second
#: user can read the same 70 GB checkpoint instead of fetching their own copy.
DEFAULT_STORE = Path("/opt/ai-models")


@dataclass
class RegistryEntry:
    """One selectable model."""

    key: str
    repo: str
    label: str = ""
    family: str = ""
    model_class: str = "dense"
    params: float = 0.0
    active_params: float = 0.0
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    note: str = ""

    @property
    def is_moe(self) -> bool:
        return self.model_class == "moe"

    @property
    def display(self) -> str:
        return self.label or self.key

    def to_dict(self) -> dict:
        return {
            "key": self.key, "repo": self.repo, "label": self.display,
            "family": self.family, "class": self.model_class,
            "params": self.params, "active_params": self.active_params,
            "dtype": self.dtype, "attn_implementation": self.attn_implementation,
            "note": self.note,
        }


@dataclass
class Registry:
    """The parsed catalogue."""

    entries: dict[str, RegistryEntry] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    store: Path = DEFAULT_STORE

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str) -> RegistryEntry:
        if key not in self.entries:
            available = ", ".join(sorted(self.entries))
            raise KeyError(f"Unknown model '{key}'. Available: {available}")
        return self.entries[key]

    def resolve(self, selection: str | list[str] | None) -> list[RegistryEntry]:
        """Turn a selection expression into entries, preserving order and de-duplicating.

        Accepts a comma-separated string, a list, ``@group`` references, and ``all``. Group
        references expand in place so ``@small,qwen3-30b`` means what it looks like.
        """
        if selection is None:
            selection = "@all"
        if isinstance(selection, str):
            tokens = [t.strip() for t in selection.split(",") if t.strip()]
        else:
            tokens = [str(t).strip() for t in selection if str(t).strip()]

        keys: list[str] = []
        for token in tokens:
            if token.startswith("@") or token in self.groups:
                name = token.lstrip("@")
                if name not in self.groups:
                    raise KeyError(
                        f"Unknown group '@{name}'. Available: "
                        + ", ".join(f"@{g}" for g in sorted(self.groups))
                    )
                keys.extend(self.groups[name])
            else:
                keys.append(token)

        seen: set[str] = set()
        ordered: list[RegistryEntry] = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            ordered.append(self.get(key))
        return ordered


def load_registry(path: str | Path | None = None, store: str | Path | None = None) -> Registry:
    """Read the catalogue."""
    import yaml

    path = Path(path or DEFAULT_REGISTRY)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults") or {}

    registry = Registry(
        store=Path(store or data.get("store") or DEFAULT_STORE),
        groups={k: list(v) for k, v in (data.get("groups") or {}).items()},
    )
    for key, raw in (data.get("models") or {}).items():
        registry.entries[key] = RegistryEntry(
            key=key,
            repo=raw["repo"],
            label=raw.get("label", ""),
            family=raw.get("family", ""),
            model_class=raw.get("class", "dense"),
            params=float(raw.get("params") or 0),
            active_params=float(raw.get("active_params") or 0),
            dtype=raw.get("dtype", defaults.get("dtype", "bfloat16")),
            attn_implementation=raw.get(
                "attn_implementation", defaults.get("attn_implementation", "sdpa")
            ),
            note=(raw.get("note") or "").strip(),
        )

    # A group naming a model that is not in the catalogue is a typo that would otherwise
    # only surface hours into a suite run.
    for name, members in registry.groups.items():
        missing = [m for m in members if m not in registry.entries]
        if missing:
            raise ValueError(
                f"Group '@{name}' references unknown models: {', '.join(missing)}"
            )
    return registry


# --------------------------------------------------------------------------------------
# The shared store
# --------------------------------------------------------------------------------------


def store_env(store: str | Path) -> dict[str, str]:
    """Environment pointing the HuggingFace libraries at the shared store.

    ``HF_HOME`` is set rather than a bespoke variable so anything else on the machine --
    a plain ``transformers`` script, another user's notebook -- picks up the same cache by
    exporting one standard variable.
    """
    store = str(Path(store).expanduser())
    return {"HF_HOME": store, "HUGGINGFACE_HUB_CACHE": str(Path(store) / "hub")}


def imported_path(entry: RegistryEntry, store: str | Path) -> Path:
    """Where :func:`install_imported` puts a model transferred in from another machine."""
    return Path(store).expanduser() / "imported" / entry.key


def snapshot_path(entry: RegistryEntry, store: str | Path) -> Path | None:
    """Local directory holding this model's weights, or None if not fetched.

    A snapshot counts as present only when it has a config *and* at least one weight shard.
    An interrupted download leaves the directory and the small files behind, so testing for
    the directory alone would report a half-fetched 70 GB model as ready and fail hours later
    inside the benchmark.

    Imported models win over the hub cache. They are plain directories of real files rather
    than the symlink farm the cache uses, because a checkpoint that travelled on an external
    disk cannot arrive in cache layout: exFAT has no symlinks at all and NTFS refuses them
    without Developer Mode, so the cache degrades into keeping two copies of every shard.
    Both layouts hold identical files and ``transformers`` cannot tell them apart.
    """
    imported = imported_path(entry, store)
    if (imported / "config.json").exists() and any(imported.glob("*.safetensors")):
        return imported

    hub = Path(store).expanduser() / "hub"
    repo_dir = hub / ("models--" + entry.repo.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    for candidate in sorted(snapshots.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        if (candidate / "config.json").exists() and any(candidate.glob("*.safetensors")):
            return candidate
    return None


#: Files a benchmark actually opens. A repository often carries GGUF conversions, ONNX
#: exports or original-format duplicates alongside the safetensors, and fetching those can
#: double the transfer for files the harness will never read.
DEFAULT_PATTERNS = [
    "*.safetensors", "*.safetensors.index.json", "config.json",
    "generation_config.json", "tokenizer*", "*.model", "preprocessor_config.json",
    "chat_template.*", "*.py",
]

#: Seconds of zero progress before a transfer is treated as dead.
#:
#: Sized by what a false positive costs, which is far more than it looks. huggingface_hub 1.x
#: streams every file into a ``uuid4``-suffixed temporary name and deletes it on the way out
#: (``file_download.py``, PR #4228: a shared ``<etag>.incomplete`` corrupts the cache when
#: ``flock`` silently lies, as it does on Lustre, GPFS and some NFS mounts). Their own comment
#: is explicit that the partial "could not be reused anyway since the temporary name is unique
#: to this download". So resume is whole-file granular: killing a transfer 95% through a 10 GB
#: shard throws away all 10 GB.
#:
#: At the ~8 MB/s this link sustains, that is ~20 minutes to redo. Killing early to save four
#: costs more than it saves, so the threshold sits above the re-download cost of the largest
#: shard we fetch. The genuine hang this exists for ran for tens of minutes, and is still
#: caught well inside one retry.
STALL_TIMEOUT_S = 900

#: How often the watchdog samples on-disk bytes.
POLL_INTERVAL_S = 10

#: How long a ``.incomplete`` file must sit untouched before it counts as abandoned.
#:
#: Comfortably longer than any gap a live transfer leaves between writes, and shorter than
#: anything an operator would wait on. Its real job is to keep the sweep from deleting a file
#: some *other* process is actively writing, which no size comparison can detect.
PARTIAL_STALE_S = 300


def _download_worker_source() -> str:
    """The child process body. Kept as source because it must run in its own interpreter."""
    return (
        "import inspect, json, os, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "args = json.loads(sys.argv[1])\n"
        "kwargs = {\n"
        "    'repo_id': args['repo'],\n"
        "    'cache_dir': args['cache_dir'],\n"
        "    'allow_patterns': args['patterns'],\n"
        "    'max_workers': args['workers'],\n"
        "    'resume_download': True,\n"
        "}\n"
        # `resume_download` was removed in huggingface_hub 1.x, where resuming is automatic,
        # and passing it raises. Filtering against the installed signature keeps this working
        # on both major versions instead of whichever one happens to be installed.
        "accepted = inspect.signature(snapshot_download).parameters\n"
        "kwargs = {k: v for k, v in kwargs.items() if k in accepted}\n"
        "print(snapshot_download(**kwargs))\n"
    )


def _blob_dir(entry: RegistryEntry, store: str | Path) -> Path | None:
    """The cache blob directory for one repo, or ``None`` if nothing has been fetched yet."""
    hub = Path(store).expanduser() / "hub"
    blobs = hub / ("models--" + entry.repo.replace("/", "--")) / "blobs"
    return blobs if blobs.is_dir() else None


def sweep_orphan_partials(
    entry: RegistryEntry, store: str | Path, stale_after: float = PARTIAL_STALE_S,
) -> tuple[int, int]:
    """Delete abandoned ``.incomplete`` files, returning ``(files, bytes)`` reclaimed.

    A partial is normally removed by the downloader's own ``finally``. A killed process never
    reaches it, so every stall leaves a full-size carcass behind -- 9.5 GB, in the case that
    prompted this. *Every* leftover partial is garbage, not just the duplicates: the temporary
    name carries a fresh uuid per download and is opened with ``"wb"``, so no later attempt
    can ever adopt one. Keeping the biggest would only keep the biggest piece of litter.

    Staleness is judged by mtime rather than size, because size gets it backwards exactly when
    it matters. Immediately after a kill the abandoned file is the *larger* one and the live
    retry has just started from zero. mtime separates them correctly, and it is also what
    makes this safe to call while an unrelated transfer into the same repo is running -- a
    file being written now is never ``stale_after`` seconds old.
    """
    blobs = _blob_dir(entry, store)
    if blobs is None:
        return 0, 0

    cutoff = time.time() - stale_after
    files = reclaimed = 0
    for blob in blobs.iterdir():
        if not blob.name.endswith(".incomplete"):
            continue
        try:
            stat = blob.stat()
            if stat.st_mtime > cutoff:
                continue  # still being written
            blob.unlink()
        except OSError:
            continue
        files += 1
        reclaimed += stat.st_size
    return files, reclaimed


def fetch(
    entry: RegistryEntry,
    store: str | Path,
    allow_patterns: list[str] | None = None,
    attempts: int = 6,
    log=print,
) -> Path:
    """Download one model into the shared store, resuming and retrying through stalls.

    The download runs in a **subprocess under a watchdog**, which is more machinery than a
    library call should need and is here for a measured reason. ``snapshot_download`` hangs
    indefinitely against an unauthenticated, rate-limited hub: the remote closes the
    connection, the sockets sit in CLOSE-WAIT, the worker thread parks in a futex, and the
    parent blocks joining it. ``HF_HUB_DOWNLOAD_TIMEOUT`` does not rescue this -- it governs a
    single request, not a pool of threads one of which will never return -- so no in-process
    timeout can end it. Killing the process is the only thing that reliably can.

    A kill is not free, and the cost decides the tuning. Resume in huggingface_hub 1.x is
    whole-file granular -- see :data:`STALL_TIMEOUT_S` -- so a stall forfeits the shard in
    flight while every file already linked into ``blobs/`` is kept. Re-running ``fetch`` after
    any failure therefore does resume, just at file boundaries rather than byte offsets. The
    threshold is set so that waiting out a suspected stall is cheaper than mistakenly
    restarting one, and abandoned partials are swept between attempts so they neither
    accumulate on disk nor inflate the reported progress.
    """
    import subprocess
    import threading
    import time

    store = Path(store).expanduser()
    for key, value in store_env(store).items():
        os.environ[key] = value

    payload = json.dumps({
        "repo": entry.repo,
        "cache_dir": str(store / "hub"),
        "patterns": allow_patterns or DEFAULT_PATTERNS,
        # Fewer workers than the default: against a rate-limited endpoint, more parallel
        # requests earned no extra throughput here and made a stall more likely, not less.
        "workers": 4,
    })

    last_error = ""
    for attempt in range(1, attempts + 1):
        # Safe here and only here: no child of ours is touching this repo between attempts.
        swept, reclaimed = sweep_orphan_partials(entry, store)
        if swept:
            log(f"    reclaimed {reclaimed / 1e9:,.1f} GB from {swept} abandoned partial(s)")

        complete, in_flight = repo_bytes_on_disk(entry, store)
        if attempt > 1:
            log(f"    attempt {attempt}/{attempts} "
                f"(resuming from {complete / 1e9:,.1f} GB of completed files)")
        else:
            log(f"    fetching {entry.repo}")

        process = subprocess.Popen(
            [sys.executable, "-c", _download_worker_source(), payload],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "HF_HUB_DOWNLOAD_TIMEOUT": "30"},
        )

        stalled = threading.Event()

        def watchdog() -> None:
            previous = sum(repo_bytes_on_disk(entry, store))
            idle = 0.0
            while process.poll() is None:
                time.sleep(POLL_INTERVAL_S)
                current = sum(repo_bytes_on_disk(entry, store))
                idle = 0.0 if current > previous else idle + POLL_INTERVAL_S
                previous = current
                if idle >= STALL_TIMEOUT_S:
                    stalled.set()
                    process.kill()
                    return

        watcher = threading.Thread(target=watchdog, daemon=True)
        watcher.start()
        _, stderr = process.communicate()
        watcher.join(timeout=POLL_INTERVAL_S * 2)

        if process.returncode == 0 and not stalled.is_set():
            path = snapshot_path(entry, store)
            if path is not None:
                return path
            last_error = "download reported success but no usable snapshot appeared"
        elif stalled.is_set():
            last_error = f"no progress for {STALL_TIMEOUT_S}s; killed and retrying"
            log(f"    stalled -- {last_error}")
        else:
            last_error = (stderr or "").strip().splitlines()[-1][:300] if stderr else \
                f"exit {process.returncode}"
            log(f"    failed: {last_error}")

        if attempt < attempts:
            # Backoff, capped: a rate limit clears on its own, and hammering it does not help.
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))

    raise RuntimeError(
        f"{entry.repo} could not be fetched after {attempts} attempts. Last error: "
        f"{last_error}. Partial data is kept, so re-running resumes rather than restarting."
    )


def repo_bytes_on_disk(entry: RegistryEntry, store: str | Path) -> tuple[int, int]:
    """``(complete_bytes, in_flight_bytes)`` for one repo's blobs.

    Reads the blob directory rather than the snapshot, because that is where a transfer in
    progress actually lives: ``huggingface_hub`` streams into ``<hash>.<id>.incomplete`` and
    only links the file into the snapshot once it is whole. Counting only finished snapshots
    -- which is the right test for "can I benchmark this" -- makes a model that is 80% fetched
    indistinguishable from one that has not started, and over a multi-hour download that is
    the single thing an operator wants to know.
    """
    blobs = _blob_dir(entry, store)
    if blobs is None:
        return 0, 0
    # A partial only counts as progress while something is still writing it. A killed transfer
    # leaves its bytes behind, and those bytes are unrecoverable -- the retry opens a new file
    # rather than continuing that one -- so counting them would report a model as more fetched
    # than it is. That is the specific bug that showed gemma4-e4b at 63% when 9.5 GB of the
    # figure was a carcass. Same mtime test the sweep uses, so the two always agree.
    cutoff = time.time() - PARTIAL_STALE_S
    complete = in_flight = 0
    for blob in blobs.iterdir():
        try:
            stat = blob.stat()
        except OSError:
            continue
        if blob.name.endswith(".incomplete"):
            if stat.st_mtime > cutoff:
                in_flight += stat.st_size
        else:
            complete += stat.st_size
    return complete, in_flight


#: Filename the export script writes beside each downloaded model.
MANIFEST_NAME = "nsbench-manifest.json"


def validate_model_dir(directory: Path) -> tuple[bool, list[str]]:
    """``(ok, problems)`` for a directory that should hold one complete checkpoint.

    Worth doing properly, because the failure this catches is expensive and silent: a shard
    truncated by a full disk or a yanked USB cable leaves a file that exists, opens, and is
    the wrong length. Nothing notices until ``transformers`` faults hours into a suite run.

    Three checks, cheapest first. The shard list in ``model.safetensors.index.json`` is the
    checkpoint's own statement of what it consists of, so a missing shard is detectable
    without contacting the hub at all. Sizes come from the manifest the export script wrote,
    which is what makes a *truncated* file distinguishable from a complete one.
    """
    problems: list[str] = []
    if not directory.is_dir():
        return False, [f"{directory} is not a directory"]
    if not (directory / "config.json").exists():
        problems.append("no config.json")

    shards = sorted(p.name for p in directory.glob("*.safetensors"))
    if not shards:
        problems.append("no *.safetensors weight files")

    index = directory / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            expected = sorted(set(weight_map.values()))
            missing = [s for s in expected if s not in shards]
            if missing:
                problems.append(
                    f"index lists {len(expected)} shard(s), {len(missing)} missing: "
                    + ", ".join(missing[:4]) + ("..." if len(missing) > 4 else "")
                )
        except (ValueError, OSError) as exc:
            problems.append(f"unreadable {index.name}: {exc}")

    manifest = directory / MANIFEST_NAME
    if manifest.exists():
        try:
            recorded = json.loads(manifest.read_text()).get("files", {})
        except (ValueError, OSError) as exc:
            recorded = {}
            problems.append(f"unreadable manifest: {exc}")
        for name, size in recorded.items():
            actual = directory / name
            if not actual.exists():
                problems.append(f"missing {name}")
            elif actual.stat().st_size != size:
                problems.append(
                    f"{name} is {actual.stat().st_size:,} bytes, expected {size:,} "
                    "(truncated or still copying)"
                )
    else:
        problems.append(
            f"no {MANIFEST_NAME} -- sizes cannot be verified, only presence"
        )

    hard = [p for p in problems if not p.startswith("no " + MANIFEST_NAME)]
    return not hard, problems


def install_imported(
    entry: RegistryEntry, source: Path, store: str | Path, move: bool = False,
) -> Path:
    """Put a validated checkpoint directory where :func:`snapshot_path` will find it.

    Defaults to copying. ``move`` is offered because these are 70 GB directories and a rename
    within one filesystem is instant, but it is not the default: it consumes the only other
    copy, and the source here is usually the transfer disk someone may still need.
    """
    destination = imported_path(entry, store)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    if move:
        try:
            source.rename(destination)
            return destination
        except OSError:
            pass  # across filesystems; fall through to a copy
    shutil.copytree(source, destination)
    return destination


def store_status(registry: Registry, entries: list[RegistryEntry]) -> list[dict]:
    """What the store holds, including transfers still in progress."""
    rows = []
    for entry in entries:
        path = snapshot_path(entry, registry.store)
        complete, in_flight = repo_bytes_on_disk(entry, registry.store)
        on_disk = complete + in_flight
        # The registry's parameter count gives an expected size at 2 bytes per weight. It is
        # an estimate, so a percentage derived from it is capped rather than allowed to read
        # above 100 and look like a bug.
        expected = int(entry.params * 2) if entry.params else 0
        pct = min(100.0, 100.0 * on_disk / expected) if expected and on_disk else None

        if path is not None:
            state = "ready"
        elif on_disk > 0:
            state = "fetching"
        else:
            state = "absent"

        rows.append({
            "key": entry.key,
            "label": entry.display,
            "repo": entry.repo,
            "present": path is not None,
            "state": state,
            "path": str(path) if path else None,
            "bytes": on_disk,
            "in_flight_bytes": in_flight,
            "expected_bytes": expected,
            "pct": pct,
        })
    return rows


def store_free_bytes(store: str | Path) -> int:
    """Free space on the filesystem holding the store, walking up to a path that exists."""
    path = Path(store).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free
