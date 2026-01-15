import pytest
import torch
from acetn.linalg import svd_lowrank, fused_matmul_svd_lowrank, fused_3matmul_svd_lowrank


class TestSvdLowrank:
    """Tests for svd_lowrank function."""

    @pytest.fixture
    def random_matrix(self):
        torch.manual_seed(42)
        return torch.randn(100, 80, dtype=torch.float64)

    @pytest.fixture
    def low_rank_matrix(self):
        """Create a matrix with known low rank structure."""
        torch.manual_seed(42)
        m, n, r = 100, 80, 10
        U = torch.randn(m, r, dtype=torch.float64)
        V = torch.randn(n, r, dtype=torch.float64)
        return U @ V.T

    def test_output_shapes(self, random_matrix):
        """Test that output shapes are correct."""
        A = random_matrix
        q = 20
        U, S, V = svd_lowrank(A, q=q, niter=2)

        assert U.shape == (A.shape[0], q)
        assert S.shape == (q,)
        assert V.shape == (A.shape[1], q)

    def test_reconstruction_error(self, random_matrix):
        """Test that reconstruction approximates the original matrix."""
        A = random_matrix
        q = 40
        U, S, V = svd_lowrank(A, q=q, niter=2)

        A_approx = U @ torch.diag(S) @ V.T
        rel_error = torch.norm(A - A_approx) / torch.norm(A)

        # Reconstruction should capture most of the matrix
        assert rel_error < 0.5

    def test_low_rank_exact_recovery(self, low_rank_matrix):
        """Test that a low-rank matrix is recovered accurately."""
        A = low_rank_matrix
        q = 15  # Slightly more than true rank of 10
        U, S, V = svd_lowrank(A, q=q, niter=3)

        A_approx = U @ torch.diag(S) @ V.T
        rel_error = torch.norm(A - A_approx) / torch.norm(A)

        # Should recover low-rank matrix very accurately
        assert rel_error < 1e-10

    def test_singular_values_ordering(self, random_matrix):
        """Test that singular values are in descending order."""
        A = random_matrix
        U, S, V = svd_lowrank(A, q=20, niter=2)

        # Singular values should be non-negative and descending
        assert torch.all(S >= 0)
        assert torch.all(S[:-1] >= S[1:])

    def test_orthogonality(self, random_matrix):
        """Test that U and V have orthonormal columns."""
        A = random_matrix
        U, S, V = svd_lowrank(A, q=20, niter=2)

        # U^T U should be close to identity
        U_orth = U.T @ U
        assert torch.allclose(U_orth, torch.eye(U.shape[1], dtype=A.dtype), atol=1e-10)

        # V^T V should be close to identity
        V_orth = V.T @ V
        assert torch.allclose(V_orth, torch.eye(V.shape[1], dtype=A.dtype), atol=1e-10)

    def test_comparison_with_torch(self, random_matrix):
        """Test that results are similar to torch.svd_lowrank."""
        torch.manual_seed(123)
        A = random_matrix
        q = 20

        torch.manual_seed(456)
        U1, S1, V1 = svd_lowrank(A, q=q, niter=2)
        A_approx1 = U1 @ torch.diag(S1) @ V1.T

        torch.manual_seed(456)
        U2, S2, V2 = torch.svd_lowrank(A, q=q, niter=2)
        A_approx2 = U2 @ torch.diag(S2) @ V2.T

        # Both should give similar reconstruction errors
        err1 = torch.norm(A - A_approx1) / torch.norm(A)
        err2 = torch.norm(A - A_approx2) / torch.norm(A)

        assert abs(err1 - err2) < 0.1


class TestFusedMatmulSvdLowrank:
    """Tests for fused_matmul_svd_lowrank function."""

    @pytest.fixture
    def matrix_pair(self):
        torch.manual_seed(42)
        A = torch.randn(60, 50, dtype=torch.float64)
        B = torch.randn(50, 70, dtype=torch.float64)
        return A, B

    @pytest.fixture
    def low_rank_pair(self):
        """Create matrices whose product has low rank."""
        torch.manual_seed(42)
        r = 8
        A = torch.randn(60, r, dtype=torch.float64)
        B = torch.randn(r, 70, dtype=torch.float64)
        return A, B

    def test_output_shapes(self, matrix_pair):
        """Test that output shapes are correct."""
        A, B = matrix_pair
        q = 20
        U, S, V = fused_matmul_svd_lowrank(A, B, q=q, niter=2)

        assert U.shape == (A.shape[0], q)
        assert S.shape == (q,)
        assert V.shape == (B.shape[1], q)

    def test_equivalence_to_explicit(self, matrix_pair):
        """Test that fused version gives same result as explicit product."""
        A, B = matrix_pair
        q = 25
        
        # Explicit approach
        C = A @ B
        torch.manual_seed(789)
        U1, S1, V1 = svd_lowrank(C, q=q, niter=2)
        C_approx1 = U1 @ torch.diag(S1) @ V1.T

        # Fused approach
        torch.manual_seed(789)
        U2, S2, V2 = fused_matmul_svd_lowrank(A, B, q=q, niter=2)
        C_approx2 = U2 @ torch.diag(S2) @ V2.T

        # Both should give similar reconstruction
        err1 = torch.norm(C - C_approx1) / torch.norm(C)
        err2 = torch.norm(C - C_approx2) / torch.norm(C)

        assert abs(err1 - err2) < 0.05

    def test_low_rank_exact_recovery(self, low_rank_pair):
        """Test that a low-rank product is recovered accurately."""
        A, B = low_rank_pair
        C = A @ B
        q = 12

        U, S, V = fused_matmul_svd_lowrank(A, B, q=q, niter=3)
        C_approx = U @ torch.diag(S) @ V.T

        rel_error = torch.norm(C - C_approx) / torch.norm(C)
        assert rel_error < 1e-10

    def test_reconstruction_quality(self, matrix_pair):
        """Test reconstruction quality of the fused SVD."""
        A, B = matrix_pair
        C = A @ B
        q = 30

        U, S, V = fused_matmul_svd_lowrank(A, B, q=q, niter=2)
        C_approx = U @ torch.diag(S) @ V.T

        rel_error = torch.norm(C - C_approx) / torch.norm(C)
        assert rel_error < 0.5


class TestFused3MatmulSvdLowrank:
    """Tests for fused_3matmul_svd_lowrank function."""

    @pytest.fixture
    def matrix_quad(self):
        torch.manual_seed(42)
        A = torch.randn(40, 35, dtype=torch.float64)
        B = torch.randn(35, 30, dtype=torch.float64)
        C = torch.randn(30, 35, dtype=torch.float64)
        D = torch.randn(35, 45, dtype=torch.float64)
        return A, B, C, D

    @pytest.fixture
    def low_rank_quad(self):
        """Create matrices whose product has low rank."""
        torch.manual_seed(42)
        r = 6
        A = torch.randn(40, r, dtype=torch.float64)
        B = torch.eye(r, dtype=torch.float64)
        C = torch.eye(r, dtype=torch.float64)
        D = torch.randn(r, 45, dtype=torch.float64)
        return A, B, C, D

    def test_output_shapes(self, matrix_quad):
        """Test that output shapes are correct."""
        A, B, C, D = matrix_quad
        q = 20
        U, S, V = fused_3matmul_svd_lowrank(A, B, C, D, q=q, niter=2)

        assert U.shape == (A.shape[0], q)
        assert S.shape == (q,)
        assert V.shape == (D.shape[1], q)

    def test_equivalence_to_explicit(self, matrix_quad):
        """Test that fused version gives same result as explicit product."""
        A, B, C, D = matrix_quad
        q = 20

        # Explicit approach
        F = A @ B @ C @ D
        torch.manual_seed(101)
        U1, S1, V1 = svd_lowrank(F, q=q, niter=2)
        F_approx1 = U1 @ torch.diag(S1) @ V1.T

        # Fused approach
        torch.manual_seed(101)
        U2, S2, V2 = fused_3matmul_svd_lowrank(A, B, C, D, q=q, niter=2)
        F_approx2 = U2 @ torch.diag(S2) @ V2.T

        # Both should give similar reconstruction
        err1 = torch.norm(F - F_approx1) / torch.norm(F)
        err2 = torch.norm(F - F_approx2) / torch.norm(F)

        assert abs(err1 - err2) < 0.05

    def test_low_rank_exact_recovery(self, low_rank_quad):
        """Test that a low-rank product is recovered accurately."""
        A, B, C, D = low_rank_quad
        F = A @ B @ C @ D
        q = 10

        U, S, V = fused_3matmul_svd_lowrank(A, B, C, D, q=q, niter=3)
        F_approx = U @ torch.diag(S) @ V.T

        rel_error = torch.norm(F - F_approx) / torch.norm(F)
        assert rel_error < 1e-10

    def test_reconstruction_quality(self, matrix_quad):
        """Test reconstruction quality of the fused SVD."""
        A, B, C, D = matrix_quad
        F = A @ B @ C @ D
        q = 25

        U, S, V = fused_3matmul_svd_lowrank(A, B, C, D, q=q, niter=2)
        F_approx = U @ torch.diag(S) @ V.T

        rel_error = torch.norm(F - F_approx) / torch.norm(F)
        assert rel_error < 0.5

    def test_singular_values_ordering(self, matrix_quad):
        """Test that singular values are in descending order."""
        A, B, C, D = matrix_quad
        U, S, V = fused_3matmul_svd_lowrank(A, B, C, D, q=20, niter=2)

        assert torch.all(S >= 0)
        assert torch.all(S[:-1] >= S[1:])


class TestComplexMatrices:
    """Tests for complex-valued matrices."""

    def test_svd_lowrank_complex(self):
        """Test svd_lowrank with complex matrices."""
        torch.manual_seed(42)
        A = torch.randn(50, 40, dtype=torch.complex128)

        U, S, V = svd_lowrank(A, q=20, niter=2)
        # For complex matrices: A ≈ U @ diag(S) @ V^H
        A_approx = U @ torch.diag(S.to(A.dtype)) @ V.mH

        rel_error = torch.norm(A - A_approx) / torch.norm(A)
        assert rel_error < 0.5

    def test_fused_matmul_complex(self):
        """Test fused_matmul_svd_lowrank with complex matrices."""
        torch.manual_seed(42)
        A = torch.randn(40, 35, dtype=torch.complex128)
        B = torch.randn(35, 45, dtype=torch.complex128)

        C = A @ B
        U, S, V = fused_matmul_svd_lowrank(A, B, q=15, niter=2)
        # For complex matrices: C ≈ U @ diag(S) @ V^H
        C_approx = U @ torch.diag(S.to(C.dtype)) @ V.mH

        rel_error = torch.norm(C - C_approx) / torch.norm(C)
        assert rel_error < 0.5

    def test_fused_3matmul_complex(self):
        """Test fused_3matmul_svd_lowrank with complex matrices."""
        torch.manual_seed(42)
        A = torch.randn(30, 25, dtype=torch.complex128)
        B = torch.randn(25, 20, dtype=torch.complex128)
        C = torch.randn(20, 25, dtype=torch.complex128)
        D = torch.randn(25, 35, dtype=torch.complex128)

        F = A @ B @ C @ D
        U, S, V = fused_3matmul_svd_lowrank(A, B, C, D, q=15, niter=2)
        # For complex matrices: F ≈ U @ diag(S) @ V^H
        F_approx = U @ torch.diag(S.to(F.dtype)) @ V.mH

        rel_error = torch.norm(F - F_approx) / torch.norm(F)
        assert rel_error < 0.5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_q_larger_than_dimensions(self):
        """Test when q is larger than matrix dimensions."""
        torch.manual_seed(42)
        A = torch.randn(20, 15, dtype=torch.float64)

        # q=50 is larger than both dimensions
        U, S, V = svd_lowrank(A, q=50, niter=2)

        # Should clamp to min(m, n)
        assert U.shape[1] <= min(A.shape)
        assert S.shape[0] <= min(A.shape)
        assert V.shape[1] <= min(A.shape)

    def test_square_matrix(self):
        """Test with square matrix."""
        torch.manual_seed(42)
        A = torch.randn(30, 30, dtype=torch.float64)

        U, S, V = svd_lowrank(A, q=15, niter=2)
        A_approx = U @ torch.diag(S) @ V.T

        rel_error = torch.norm(A - A_approx) / torch.norm(A)
        assert rel_error < 0.6

    def test_tall_matrix(self):
        """Test with tall (m > n) matrix."""
        torch.manual_seed(42)
        A = torch.randn(100, 20, dtype=torch.float64)

        U, S, V = svd_lowrank(A, q=15, niter=2)

        assert U.shape == (100, 15)
        assert V.shape == (20, 15)

    def test_wide_matrix(self):
        """Test with wide (m < n) matrix."""
        torch.manual_seed(42)
        A = torch.randn(20, 100, dtype=torch.float64)

        U, S, V = svd_lowrank(A, q=15, niter=2)

        assert U.shape == (20, 15)
        assert V.shape == (100, 15)

    def test_niter_zero(self):
        """Test with zero power iterations."""
        torch.manual_seed(42)
        A = torch.randn(50, 40, dtype=torch.float64)

        # Should still work, just less accurate
        U, S, V = svd_lowrank(A, q=20, niter=0)
        A_approx = U @ torch.diag(S) @ V.T

        rel_error = torch.norm(A - A_approx) / torch.norm(A)
        # Less accurate but should still be reasonable
        assert rel_error < 1.0
