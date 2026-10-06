# Sampling L2 traffic with nsys on GB10

The stock `gb20b` nsys GPU-metric set has no memory counters. A custom metric file can add
`lts__t_sectors` (all L2 traffic). Tested 2026-10-06 against 10 copies of a 2 GiB tensor:
**42.98 GB measured against 42.94 GB expected (0.1%).**

Every other L2 counter tried is silently dropped. nsys exits 0 and records only the stock 19
metrics. The dropped counters are:
- `lts__t_sectors_aperture_sysmem_lookup_miss`
- `lts__d_sectors_fill_sysmem`
- `lts__t_sectors_lookup_miss` and `lts__t_sectors_lookup_hit`
- `lts__t_sectors_op_read` and `lts__t_sectors_op_write`
- `lts__d_sectors`

So nsys gives per-phase **L2** traffic for free, but not DRAM bytes. L2 traffic is an upper bound on DRAM.

| File | Use |
|---|---|
| `l2_sectors.config` | `gb20b.config` plus `lts__t_sectors`. The `alias:` must equal the file name (`l2_sectors`), or nsys refuses it. |
| `copy_test.py` | The known-size workload |
| `run_test.sh` | Profile, export, analyse. GPU must be idle. |
| `analyse.py` | Sums the sampled sectors over the copy window, times 32 B |

Use it with `nsys profile --gpu-metrics-devices=0 --gpu-metrics-set=file:$PWD/l2_sectors.config ...`
