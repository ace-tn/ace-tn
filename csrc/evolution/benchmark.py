import torch

torch.ops.load_library("./_C.so")

def make_inputs(nD, bD, pD):
    n12g = torch.randn(nD, nD, pD, pD, device='cuda', dtype=torch.float64)
    n12g = n12g.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)

    n12 = torch.randn(nD, nD, nD, nD, device='cuda', dtype=torch.float64)
    n12 = n12.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)

    a2r = torch.randn(nD, bD, pD, device='cuda', dtype=torch.float64)
    a2r = a2r.permute(2, 1, 0).contiguous().permute(2, 1, 0)

    return n12, n12g, a2r

def measure_time(build_fn, A, B, nwarmup=50, niter=10):
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

def benchmark_build_s1(nD, bD, pD):
    _, n12g, a2r = make_inputs(nD, bD, pD)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.build_s1, n12g, a2r)
    torch_time, C_true = measure_time(s1_reference, n12g, a2r)
    C_true = C_true.permute(2,1,0).reshape(nD,bD,pD)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[build_s1] error = {rel_error.item():.3e}")
    print(f"[build_s1] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

def benchmark_build_s2(nD, bD, pD):
    _, n12g, a2r = make_inputs(nD, bD, pD)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.build_s2, n12g, a2r)
    torch_time, C_true = measure_time(s2_reference, n12g, a2r)
    C_true = C_true.permute(2,1,0).reshape(nD,bD,pD)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[build_s2] error = {rel_error.item():.3e}")
    print(f"[build_s2] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

def benchmark_build_r1(nD, bD, pD):
    n12, _, a2r = make_inputs(nD, bD, pD)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.build_r1, n12, a2r)
    torch_time, C_true = measure_time(r1_reference, n12, a2r)
    C_true = C_true.permute(3,2,1,0).reshape(nD,bD,nD,bD)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[build_r1] error = {rel_error.item():.3e}")
    print(f"[build_r1] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

def benchmark_build_r2(nD, bD, pD):
    n12, _, a1r = make_inputs(nD, bD, pD)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.build_r2, n12, a1r)
    torch_time, C_true = measure_time(r2_reference, n12, a1r)
    C_true = C_true.permute(3,2,1,0).reshape(nD,bD,nD,bD)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[build_r2] error = {rel_error.item():.3e}")
    print(f"[build_r2] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

def s1_reference(A, B):
    return torch.einsum('YXpQ,XUQ->YUp', A, B)

def s2_reference(A, B):
    return torch.einsum('YXPq,YVP->XVq', A, B)

def r1_reference(A, B):
    R = torch.einsum('yxYX,xuq->yYXuq', A, B)
    return torch.einsum('yYXuQ,XUQ->YUyu', R, B)

def r2_reference(A, B):
    R = torch.einsum('yxYX,yvp->xYXvp', A, B)
    return torch.einsum('xYXvP,YVP->XVxv', R, B)

def main():
    nD, bD, pD = 16, 8, 2
    benchmark_build_s1(nD, bD, pD)
    benchmark_build_s2(nD, bD, pD)
    benchmark_build_r1(nD, bD, pD)
    benchmark_build_r2(nD, bD, pD)

if __name__ == "__main__":
    main()
