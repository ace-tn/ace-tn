import torch
import time
from torch.linalg import cholesky, solve_triangular

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

def benchmark_cholesky_solve(nD,bD,pD):
    n, nrhs = nD*bD, pD
    A = make_symmetric_posdef(n)
    B = torch.randn(n,nrhs, device='cuda', dtype=torch.float64).t().contiguous().t()

    cusolver_time, X1 = measure_time(torch.ops.cusolver_ext.cholesky_solve, A, B)
    X1 = X1.reshape(nD,bD,pD)
    torch_time, X2 = measure_time(torch_cholesky_solve, A, B)
    X2 = X2.reshape(nD,bD,pD)

    rel_error = (X1 - X2).abs().max() / X2.abs().max()
    print(f"[cholesky_solve] error = {rel_error.item():.3e}")
    print(f"[cholesky_solve] cuSOLVER: {cusolver_time:.4f} ms, torch: {torch_time:.4f} ms")

def torch_cholesky_solve(A, B):
    L = cholesky(A)
    Y = solve_triangular(L, B, upper=False)
    return solve_triangular(L.mH, Y, upper=True)

def main():
    bD = 8
    pD = 2
    nD = bD*pD
    benchmark_cholesky_solve(nD,bD,pD)

if __name__ == "__main__":
    main()
