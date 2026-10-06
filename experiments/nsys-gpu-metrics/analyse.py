import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
ours = {i: n for i, n in names.items() if n.startswith("L2 ")}
if not ours:
    print("  no custom metric recorded; metrics present:", len(names)); sys.exit()
tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
src = "CUPTI_ACTIVITY_KIND_KERNEL" if "CUPTI_ACTIVITY_KIND_KERNEL" in tables else "CUPTI_ACTIVITY_KIND_MEMCPY"
a, b, busy, n, nbytes = db.execute(f"SELECT MIN(start), MAX(end), SUM(end-start), COUNT(*), " +
    ("SUM(bytes)" if src.endswith("MEMCPY") else "0") + f" FROM {src}").fetchone()
print(f"  {n} ops in {src[20:]}, {busy/1e6:.1f} ms busy, window {(b-a)/1e6:.1f} ms, memcpy bytes {nbytes/1e9:.2f} GB")
for mid, name in ours.items():
    rows = db.execute("SELECT timestamp, value FROM GPU_METRICS WHERE metricId=? ORDER BY timestamp", (mid,)).fetchall()
    inside = [(t, v) for t, v in rows if a - 1_000_000 <= t <= b + 1_000_000]
    if "per second" in name:
        # integrate rate over sample spacing
        tot = sum(v * (inside[i+1][0]-inside[i][0]) / 1e9 for i, (t, v) in enumerate(inside[:-1]))
    else:
        tot = sum(v for _, v in inside)
    print(f"  {name}: {len(rows)} samples ({len(inside)} in window) -> {tot:.4g} sectors = {tot*32/1e9:.2f} GB")
