import torch


def fused_matmul_svd_lowrank(A, B, q=6, niter=2):
    """
    Compute the randomized SVD of the matrix product A @ B without forming it explicitly.

    This is more memory efficient than computing A @ B followed by svd_lowrank,
    as it avoids materializing the full product matrix.

    Args:
        A (Tensor): Left input matrix of shape (m, k).
        B (Tensor): Right input matrix of shape (k, n).
        q (int): Target rank for the approximation. Default: 6.
        niter (int): Number of power iterations for accuracy improvement. Default: 2.

    Returns:
        tuple[Tensor, Tensor, Tensor]: 
            - U: Left singular vectors of shape (m, q).
            - S: Singular values of shape (q,).
            - V: Right singular vectors of shape (n, q).

    Note:
        The returned V is not transposed, so A @ B ≈ U @ diag(S) @ V.T
    """
    m = A.shape[0]
    n = B.shape[1]
    dtype = A.dtype
    device = A.device

    q = min(q, m, n)
    Omega = torch.randn(n, q, dtype=dtype, device=device)

    # Y = (A @ B) @ Omega = A @ (B @ Omega)
    Y = A @ (B @ Omega)

    for _ in range(niter):
        Y, _ = torch.linalg.qr(Y)
        # (A @ B)^H @ Y = B^H @ (A^H @ Y)
        # (A @ B) @ ((A @ B)^H @ Y) = A @ (B @ (B^H @ (A^H @ Y)))
        Y = A @ (B @ (B.mH @ (A.mH @ Y)))

    Q, _ = torch.linalg.qr(Y)

    # B_proj = Q^H @ (A @ B) = (Q^H @ A) @ B
    B = (Q.mH @ A) @ B

    U_B, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_B
    V = Vh.mH

    return U, S, V
