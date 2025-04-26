import torch

torch.ops.load_library("./_C.so")

def make_inputs(nD, bD, pD):
    A = torch.randn(nD, nD, pD, pD, device='cuda', dtype=torch.float64)
    B = torch.randn(nD, bD, pD, device='cuda', dtype=torch.float64)
    A = A.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)
    B = B.permute(2, 1, 0).contiguous().permute(2, 1, 0)
    return A, B

def measure_cutensor_runtime(build_fn, A, B, nwarmup=5, niter=1000):
    for _ in range(nwarmup):
        build_fn(A, B)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(niter):
        C = build_fn(A, B)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / niter, C

def measure_torch_runtime(einsum_eq, A, B, nwarmup=5, niter=1000):
    for _ in range(nwarmup):
        torch.einsum(einsum_eq, A, B)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(niter):
        C_true = torch.einsum(einsum_eq, A, B)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / niter, C_true

def benchmark(build_fn, einsum_eq, nD, bD, pD):
    A, B = make_inputs(nD, bD, pD)
    cutensor_time, C = measure_cutensor_runtime(build_fn, A, B)
    torch_time, C_true = measure_torch_runtime(einsum_eq, A, B)

    C_true = C_true.permute(2, 1, 0).reshape(nD, bD, pD)
    rel_error = (C_true - C).abs().max() / C_true.abs().max()

    print(f"[test] error = {rel_error.item():.3e}")
    print(f"[timing] cuTENSOR: {cutensor_time:.4f} ms")
    print(f"[timing] torch.einsum: {torch_time:.4f} ms")

def benchmark_build_s1():
    benchmark(torch.ops.cutensor_ext.build_s1, 'YXpQ,XUQ->YUp', 16, 8, 8)

def benchmark_build_s2():
    benchmark(torch.ops.cutensor_ext.build_s2, 'YXPq,YVP->XVq', 16, 8, 8)

def main():
    benchmark_build_s1()
    benchmark_build_s2()

if __name__ == "__main__":
    main()
