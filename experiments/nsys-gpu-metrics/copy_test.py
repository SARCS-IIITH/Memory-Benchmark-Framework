# Known-size traffic: N copies of a 2 GiB bf16 tensor. Each copy reads 2 GiB and writes 2 GiB.
import torch, torch.cuda.nvtx as nvtx
N = 10
src = torch.empty(1 << 30, dtype=torch.bfloat16, device="cuda").normal_()
dst = torch.empty_like(src)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
nvtx.range_push("copies")
for _ in range(N):
    dst.copy_(src)
torch.cuda.synchronize()
nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()
print(f"expected read {N*src.numel()*2/1e9:.2f} GB, write {N*src.numel()*2/1e9:.2f} GB")
