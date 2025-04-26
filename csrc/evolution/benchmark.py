import torch
torch.ops.load_library("./_C.so")

def test_build_s1():
    nD, bD, pD = 16, 8, 2
    A = torch.randn(nD, nD, pD, pD, device='cuda', dtype=torch.float64)
    B = torch.randn(nD, bD, pD, device='cuda', dtype=torch.float64)

    A = A.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)
    B = B.permute(2, 1, 0).contiguous().permute(2, 1, 0)

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    nwarmup = 5
    niter = 1000

    # warmup
    for _ in range(nwarmup):
        torch.ops.cutensor_ext.build_s1(A, B)

    start_evt.record()
    for _ in range(niter):
        C = torch.ops.cutensor_ext.build_s1(A, B)
    end_evt.record()
    torch.cuda.synchronize()
    cutensor_time = start_evt.elapsed_time(end_evt) / niter
    # warmup
    for _ in range(nwarmup):
        torch.einsum('YXpQ,XUQ->YUp', A, B)

    start_evt.record()
    for _ in range(niter):
        C_true = torch.einsum('YXpQ,XUQ->YUp', A, B)
    end_evt.record()
    torch.cuda.synchronize()
    torch_time = start_evt.elapsed_time(end_evt) / niter
    C_true = C_true.permute(2,1,0).reshape(nD, bD, pD)

    rel_error = (C_true - C).abs().max() / C_true.abs().max()
    print("[test] error =", rel_error.item())
    print(f"[timing] cuTENSOR: {cutensor_time:.4f} ms")
    print(f"[timing] torch.einsum: {torch_time:.4f} ms")

def main():
    test_build_s1()

if __name__ == "__main__":
    main()
