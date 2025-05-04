import torch
import time

torch.ops.load_library("./cholesky_solve.so")

def make_symmetric_posdef(n):
    A = torch.randn(n, n, device='cuda', dtype=torch.float64)
    B = A @ A.T + 1e0 * torch.eye(n, device='cuda', dtype=torch.float64)
    return B

def measure_time(build_fn, A, B, nwarmup=50, niter=500):
    for _ in range(nwarmup):
        build_fn(A.clone(), B.clone())
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(niter):
        C = build_fn(A.clone(), B.clone())
    torch.cuda.synchronize()
    return (time.time() - start) * 1000 / niter, C

def benchmark_cholesky_solve(n, nrhs):
    A = make_symmetric_posdef(n)
    B = torch.randn(n,nrhs, device='cuda', dtype=torch.float64).t().contiguous().t()

    cusolver_time, X1 = measure_time(torch.ops.cusolver_ext.cholesky_solve, A, B)
    torch_time, X2 = measure_time(torch.linalg.solve, A, B)

    rel_error = (X1 - X2).abs().max() / X2.abs().max()
    print(f"[cholesky_solve] error = {rel_error.item():.3e}")
    print(f"[cholesky_solve] cuSOLVER: {cusolver_time:.4f} ms, torch: {torch_time:.4f} ms")

def main():
    n, nrhs = 512, 8
    benchmark_cholesky_solve(n, nrhs)

if __name__ == "__main__":
    main()
