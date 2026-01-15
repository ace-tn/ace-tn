import torch


def svd_lowrank(A, q=6, niter=2):
    """
    Compute the randomized SVD of a matrix.

    This function computes an approximate SVD using random projections,
    equivalent to torch.svd_lowrank.

    Args:
        A (Tensor): Input matrix of shape (m, n).
        q (int): Target rank for the approximation. Default: 6.
        niter (int): Number of power iterations for accuracy improvement. Default: 2.
        M (Tensor, optional): Not used, included for API compatibility with torch.svd_lowrank.

    Returns:
        tuple[Tensor, Tensor, Tensor]: 
            - U: Left singular vectors of shape (m, q).
            - S: Singular values of shape (q,).
            - V: Right singular vectors of shape (n, q).

    Note:
        The returned V is not transposed (same as torch.svd_lowrank convention),
        so A ≈ U @ diag(S) @ V.T
    """
    m, n = A.shape
    dtype = A.dtype
    device = A.device

    q = min(q, m, n)
    Omega = torch.randn(n, q, dtype=dtype, device=device)
    Y = A @ Omega

    for _ in range(niter):
        Y, _ = torch.linalg.qr(Y)
        Y = A @ (A.mH @ Y)

    Q, _ = torch.linalg.qr(Y)
    B = Q.mH @ A
    U_B, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_B
    V = Vh.mH
    return U, S, V
