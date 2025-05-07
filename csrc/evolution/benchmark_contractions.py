import torch

torch.ops.load_library("./als_contractions.so")

def make_inputs(nD, bD, pD):
    n12g = torch.randn(nD, nD, pD, pD, device='cuda', dtype=torch.float64)
    n12g = n12g.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)

    n12 = torch.randn(nD, nD, nD, nD, device='cuda', dtype=torch.float64)
    n12 = n12.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)

    ar = torch.randn(nD, bD, pD, device='cuda', dtype=torch.float64)
    ar = ar.permute(2, 1, 0).contiguous().permute(2, 1, 0)

    return n12, n12g, ar

def measure_time(build_fn, *args, nwarmup=10, niter=20):
    for _ in range(nwarmup):
        build_fn(*args)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(niter):
        C = build_fn(*args)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / niter, C

def benchmark_build_s1(nD, bD, pD):
    _, n12g, ar = make_inputs(nD, bD, pD)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.build_s1, n12g, ar)
    torch_time, C_true = measure_time(s1_reference, n12g, ar)
    C = C.reshape(pD,bD,nD).permute(2,1,0)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[build_s1] error = {rel_error.item():.3e}")
    print(f"[build_s1] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

def benchmark_build_s2(nD, bD, pD):
    _, n12g, ar = make_inputs(nD, bD, pD)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.build_s2, n12g, ar)
    torch_time, C_true = measure_time(s2_reference, n12g, ar)
    C = C.reshape(pD,bD,nD).permute(2,1,0)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[build_s2] error = {rel_error.item():.3e}")
    print(f"[build_s2] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

def benchmark_build_r1(nD, bD, pD):
    n12, _, ar = make_inputs(nD, bD, pD)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.build_r1, n12, ar)
    torch_time, C_true = measure_time(r1_reference, n12, ar)
    C = C.reshape(bD,nD,bD,nD).permute(3,2,1,0)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[build_r1] error = {rel_error.item():.3e}")
    print(f"[build_r1] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

def benchmark_build_r2(nD, bD, pD):
    n12, _, ar = make_inputs(nD, bD, pD)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.build_r2, n12, ar)
    torch_time, C_true = measure_time(r2_reference, n12, ar)
    C = C.reshape(bD,nD,bD,nD).permute(3,2,1,0)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[build_r2] error = {rel_error.item():.3e}")
    print(f"[build_r2] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

def benchmark_calculate_cost(nD, bD, pD):
    n12, _, ar = make_inputs(nD, bD, pD)
    a12g = torch.einsum("yup,xuq->yxpq", ar, ar)
    a12g = a12g.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)

    cutensor_time, C = measure_time(torch.ops.cutensor_ext.calculate_cost, ar, ar, a12g, n12)
    torch_time, C_true = measure_time(calculate_cost_reference, ar, ar, a12g, n12)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print(f"[calculate_cost] error = {rel_error.item():.3e}")
    print(f"[calculate_cost] cuTENSOR: {cutensor_time:.4f} ms, torch.einsum: {torch_time:.4f} ms")

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

def calculate_cost_reference(a1r, a2r, a12g, n12):
    a12n = torch.einsum("yup,xuq->yxpq", a1r, a2r)

    d2 = torch.einsum("yxYX,yxpq->YXpq", n12, a12n)
    d2 = torch.einsum("YXpq,YXpq->", d2, a12n)

    d3 = torch.einsum("yxYX,yxpq->YXpq", n12, a12g)
    d3 = torch.einsum("YXpq,YXpq->", d3, a12n)

    return d2.real - 2*d3.real

def main():
    nD, bD, pD = 64, 8, 8
    benchmark_build_s1(nD, bD, pD)
    benchmark_build_s2(nD, bD, pD)
    benchmark_build_r1(nD, bD, pD)
    benchmark_build_r2(nD, bD, pD)
    benchmark_calculate_cost(nD, bD, pD)

if __name__ == "__main__":
    main()
