"""Move one finished set of runs (a queue) into its own folder under runs/, and fix the paths.

Every set of runs gets one folder: the run directories themselves plus queue/ (status log,
summary, per-run logs). Queues started with scripts/run_kimi_q8_llamacpp_queue.sh already
write there; this is for sets that were written straight into runs/ (as the first Kimi Q8
llama.cpp queue was).

    ~/envs/nsbench/bin/python scripts/move_run_set.py runs/kimi-q8-queue runs/kimi-q8-llamacpp-sweep
    ~/envs/nsbench/bin/python scripts/move_run_set.py ... --dry-run    # print what would move

What moves: every run directory named by a ``run=`` field in <queue dir>/queue.log (plus any
--extra paths), into <set dir>/, and the queue dir itself to <set dir>/queue. What is rewritten:
those paths, wherever they appear in the moved queue folder's text files, progress.md and
docs/*.md (not scripts/: they hold queue locations as settings, fix those by hand). Files
inside the run directories keep their old absolute paths (historical).

Refuses while the queue is still running (no QUEUE END in queue.log): a live run keeps
writing to its old path. All moves are renames on the same disk.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("queue_dir", help="the queue's folder, e.g. runs/kimi-q8-queue")
    ap.add_argument("set_dir", help="the new set folder, e.g. runs/kimi-q8-llamacpp-sweep")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="more run dirs or files (relative to the repo) that belong to the set")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    qdir, set_dir = Path(args.queue_dir), Path(args.set_dir)
    qlog = REPO / qdir / "queue.log"
    if not qlog.exists():
        sys.exit(f"no {qlog}")
    text = qlog.read_text()
    if "QUEUE END" not in text.splitlines()[-1]:
        sys.exit(f"{qlog} does not end with QUEUE END: the queue is still running (or was cut "
                 "off). Move it once it has finished.")
    if (REPO / set_dir).exists() and any((REPO / set_dir).iterdir()):
        sys.exit(f"{set_dir} already exists and is not empty")

    runs = sorted(set(re.findall(r"run=(runs/\S+)", text)) - {"runs/unknown"})
    mapping = {r: f"{set_dir}/{Path(r).name}" for r in runs}
    mapping.update({e.rstrip("/"): f"{set_dir}/{Path(e).name}" for e in args.extra})
    mapping[str(qdir).rstrip("/")] = f"{set_dir}/queue"
    missing = [old for old in mapping if not (REPO / old).exists()]
    if missing:
        sys.exit("missing: " + ", ".join(missing))

    for old, new in mapping.items():
        print(f"{old}  ->  {new}")
    if args.dry_run:
        return 0

    (REPO / set_dir).mkdir(parents=True, exist_ok=True)
    for old, new in mapping.items():
        shutil.move(str(REPO / old), str(REPO / new))

    pat = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    files = [REPO / "progress.md", *(REPO / "docs").glob("*.md"),
             *(p for p in (REPO / set_dir / "queue").iterdir()
               if p.is_file() and p.suffix in (".log", ".md", ".txt"))]
    for f in files:
        s = f.read_text(errors="surrogateescape")
        n = len(pat.findall(s))
        if n:
            f.write_text(pat.sub(lambda m: mapping[m.group(0)], s), errors="surrogateescape")
            print(f"  rewrote {n:3d} path(s) in {f.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
