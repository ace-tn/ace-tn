import torch


def fused_3matmul_svd_lowrank(A, B, C, D, q=6, niter=2):
    """
    Compute the randomized SVD of A @ B @ C @ D without forming the product explicitly.

    This avoids materializing any intermediate or final product matrices,
    which is more memory efficient for large tensors.

    Args:
        A (Tensor): First matrix of shape (m, k1).
        B (Tensor): Second matrix of shape (k1, k2).
        C (Tensor): Third matrix of shape (k2, k3).
        D (Tensor): Fourth matrix of shape (k3, n).
        q (int): Target rank for the approximation. Default: 6.
        niter (int): Number of power iterations for accuracy improvement. Default: 2.

    Returns:
        tuple[Tensor, Tensor, Tensor]: 
            - U: Left singular vectors of shape (m, q).
            - S: Singular values of shape (q,).
            - V: Right singular vectors of shape (n, q).

    Note:
        The returned V is not transposed, so A @ B @ C @ D ≈ U @ diag(S) @ V.T
    """
    m = A.shape[0]
    n = D.shape[1]
    dtype = A.dtype
    device = A.device

    q = min(q, m, n)
    Omega = torch.randn(n, q, dtype=dtype, device=device)

    # Y = A @ B @ C @ D @ Omega = A @ (B @ (C @ (D @ Omega)))
    Y = A @ (B @ (C @ (D @ Omega)))

    for _ in range(niter):
        Y, _ = torch.linalg.qr(Y)
        # (A @ B @ C @ D)^H @ Y = D^H @ (C^H @ (B^H @ (A^H @ Y)))
        Y = D.mH @ (C.mH @ (B.mH @ (A.mH @ Y)))
        Y, _ = torch.linalg.qr(Y)
        # A @ B @ C @ D @ Y
        Y = A @ (B @ (C @ (D @ Y)))

    Q, _ = torch.linalg.qr(Y)

    # B_proj = Q^H @ A @ B @ C @ D = ((Q^H @ A) @ B @ C) @ D
    B_proj = (((Q.mH @ A) @ B) @ C) @ D

    U_B, S, Vh = torch.linalg.svd(B_proj, full_matrices=False)
    U = Q @ U_B
    V = Vh.mH

    return U, S, V
