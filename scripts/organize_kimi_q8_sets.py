"""Put every Kimi Q8 llama.cpp set under runs/kimi-q8-llamacpp/ and fix the paths that point at it.

Layout (user, 2026-10-13): runs/kimi-q8-llamacpp/results.md (all 22 workloads,
scripts/summarize_kimi_q8_all.py) and one folder per set -- natural-b1/, forced-b1/, later
batch/ -- each with its run folders and queue/, plus validation-forced-routing/ and
validation-batch/.

    ~/envs/nsbench/bin/python scripts/organize_kimi_q8_sets.py [--wait]

--wait first waits (polling each minute) until the forced-routing chain has finished: its
queue logged QUEUE END and scripts/run_forced_routing_after_queue.sh logged its last line, so
nothing is still writing into the folders being moved. Folders already moved are skipped.

Rewrites the moved paths in progress.md, docs/*.md and the text files (.md/.log/.txt) under
runs/kimi-q8-llamacpp/, then refreshes the set tables and the all-22 table, and points the
queue scripts' defaults at the new layout (replacing each file atomically, so a bash process
still reading the old file is not disturbed).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = "runs/kimi-q8-llamacpp"
MOVES = {
    "runs/kimi-q8-llamacpp-sweep": f"{ROOT}/natural-b1",
    "runs/kimi-q8-llamacpp-batch-validation": f"{ROOT}/validation-batch",
    "runs/kimi-q8-llamacpp-forced-routing": f"{ROOT}/forced-b1",
    "runs/kimi-q8-llamacpp-forced-validation": f"{ROOT}/validation-forced-routing",
}
FORCED_QLOG = REPO / "runs/kimi-q8-llamacpp-forced-routing/queue/queue.log"
CHAIN_LOG = REPO / "runs/kimi-q8-llamacpp-forced-validation/validation.log"
#: (script, old text, new text): defaults that should point at the new layout
DEFAULTS = [
    ("scripts/run_kimi_q8_llamacpp_queue.sh",
     'SET_DIR="${SET_DIR:-runs/kimi-q8-llamacpp-sweep}"',
     'SET_DIR="${SET_DIR:-runs/kimi-q8-llamacpp/natural-b1}"'),
    ("scripts/run_kimi_q8_llamacpp_queue.sh",
     "# runs/kimi-q8-llamacpp-sweep): the run directories themselves",
     "# runs/kimi-q8-llamacpp/natural-b1): the run directories themselves"),
]


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def wait_for_chain() -> None:
    log("waiting for the forced-routing queue and its chain to finish")
    while True:
        qdone = FORCED_QLOG.exists() and "QUEUE END" in (FORCED_QLOG.read_text().splitlines() or [""])[-1]
        chain = CHAIN_LOG.exists() and "forced-routing queue ended" in CHAIN_LOG.read_text()
        if qdone and chain:
            return
        time.sleep(60)


def replace_atomically(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".new")
    tmp.write_text(text)
    os.chmod(tmp, os.stat(path).st_mode)
    os.replace(tmp, path)


def main() -> int:
    if "--wait" in sys.argv:
        wait_for_chain()
    moves = {o: n for o, n in MOVES.items() if (REPO / o).is_dir()}
    for old, new in moves.items():
        if (REPO / new).exists():
            sys.exit(f"{new} already exists; not moving {old}")
        (REPO / new).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(REPO / old), str(REPO / new))
        log(f"{old} -> {new}")
    if moves:
        pat = re.compile("|".join(re.escape(k) + r"(?=[/\s`'\")]|$)" for k in moves))
        files = [REPO / "progress.md", *(REPO / "docs").glob("*.md"),
                 *(p for p in (REPO / ROOT).rglob("*") if p.is_file()
                   and p.suffix in (".md", ".log", ".txt") and "raw" not in p.parts)]
        for f in files:
            s = f.read_text(errors="surrogateescape")
            n = len(pat.findall(s))
            if n:
                f.write_text(pat.sub(lambda m: moves[m.group(0)], s), errors="surrogateescape")
                log(f"  rewrote {n} path(s) in {f.relative_to(REPO)}")
    for script, old, new in DEFAULTS:
        p = REPO / script
        s = p.read_text()
        if old in s:
            replace_atomically(p, s.replace(old, new))
            log(f"  default updated in {script}")
    py = sys.executable
    for qdir in sorted((REPO / ROOT).glob("*/queue")):
        if (qdir / "queue.log").exists():
            subprocess.run([py, str(REPO / "scripts/summarize_kimi_q8_queue.py"),
                            str(qdir.relative_to(REPO))], capture_output=True)
            log(f"  refreshed {qdir.relative_to(REPO)}/results.md")
    subprocess.run([py, str(REPO / "scripts/summarize_kimi_q8_all.py")], capture_output=True)
    log(f"refreshed {ROOT}/results.md; done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
