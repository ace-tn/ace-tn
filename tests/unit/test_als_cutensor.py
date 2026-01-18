"""
Tests for cuTENSOR C++ extension.

These tests compare the cuTENSOR backend against the PyTorch backend
to ensure numerical equivalence. Tests are skipped if the cuTENSOR
extension is not available.
"""

import pytest
import torch
from torch import einsum

from acetn.evolution._extensions import CUTENSOR_AVAILABLE
from acetn.evolution.als_solver import ALSSolver
from acetn.evolution.full_update import positive_approx, gauge_fix
from acetn.ipeps.ipeps_config import EvolutionConfig


# Skip all tests in this module if cuTENSOR is not available
pytestmark = pytest.mark.skipif(
    not CUTENSOR_AVAILABLE,
    reason="cuTENSOR extension not available"
)


@pytest.fixture
def setup_tensors_float64():
    """Setup float64 test tensors for ALS solver comparison."""
    dtype = torch.float64
    device = torch.device("cuda")
    nD, bD, pD = 8, 5, 3

    a1r = torch.rand(nD, bD, pD, dtype=dtype, device=device)
    a2r = torch.rand(nD, bD, pD, dtype=dtype, device=device)

    a12g = einsum("yup,xuq->yxpq", a1r, a2r)
    a12g += 1e-4 * torch.rand_like(a12g)

    n12 = torch.rand(nD, nD, nD, nD, dtype=dtype, device=device)

    nz = positive_approx(n12, nD)
    n12, a12g, *_ = gauge_fix(nz, a12g, nD)

    return n12, a12g, (nD, bD, pD)


@pytest.fixture
def setup_tensors_float32():
    """Setup float32 test tensors for ALS solver comparison."""
    dtype = torch.float32
    device = torch.device("cuda")
    nD, bD, pD = 8, 5, 3

    a1r = torch.rand(nD, bD, pD, dtype=dtype, device=device)
    a2r = torch.rand(nD, bD, pD, dtype=dtype, device=device)

    a12g = einsum("yup,xuq->yxpq", a1r, a2r)
    a12g += 1e-4 * torch.rand_like(a12g)

    n12 = torch.rand(nD, nD, nD, nD, dtype=dtype, device=device)

    nz = positive_approx(n12, nD)
    n12, a12g, *_ = gauge_fix(nz, a12g, nD)

    return n12, a12g, (nD, bD, pD)


@pytest.fixture
def setup_tensors_complex128():
    """Setup complex128 test tensors for ALS solver comparison."""
    dtype = torch.complex128
    device = torch.device("cuda")
    nD, bD, pD = 6, 4, 2

    a1r = torch.rand(nD, bD, pD, dtype=dtype, device=device)
    a2r = torch.rand(nD, bD, pD, dtype=dtype, device=device)

    a12g = einsum("yup,xuq->yxpq", a1r, a2r)
    a12g += 1e-4 * torch.rand_like(a12g)

    n12 = torch.rand(nD, nD, nD, nD, dtype=dtype, device=device)

    nz = positive_approx(n12, nD)
    n12, a12g, *_ = gauge_fix(nz, a12g, nD)

    return n12, a12g, (nD, bD, pD)


@pytest.fixture
def setup_tensors_complex64():
    """Setup complex64 test tensors for ALS solver comparison."""
    dtype = torch.complex64
    device = torch.device("cuda")
    nD, bD, pD = 6, 4, 2

    a1r = torch.rand(nD, bD, pD, dtype=dtype, device=device)
    a2r = torch.rand(nD, bD, pD, dtype=dtype, device=device)

    a12g = einsum("yup,xuq->yxpq", a1r, a2r)
    a12g += 1e-4 * torch.rand_like(a12g)

    n12 = torch.rand(nD, nD, nD, nD, dtype=dtype, device=device)

    nz = positive_approx(n12, nD)
    n12, a12g, *_ = gauge_fix(nz, a12g, nD)

    return n12, a12g, (nD, bD, pD)


class TestCutensorBackendEquivalence:
    """Test that cuTENSOR backend produces same results as PyTorch backend."""

    def _run_als_comparison(self, n12, a12g, ar_shape, rtol=0.01):
        """Helper to compare torch and cutensor backends."""
        config_torch = EvolutionConfig(
            backend="torch", als_niter=50, als_tol=1e-15
        )
        config_cutensor = EvolutionConfig(
            backend="cutensor", als_niter=50, als_tol=1e-15
        )

        solver_torch = ALSSolver(n12.clone(), a12g.clone(), ar_shape, config_torch)
        solver_cutensor = ALSSolver(n12.clone(), a12g.clone(), ar_shape, config_cutensor)

        a1r_torch, a2r_torch = solver_torch.solve()
        a1r_cutensor, a2r_cutensor = solver_cutensor.solve()

        cost_torch = ALSSolver.calculate_cost(a1r_torch, a2r_torch, a12g, n12)
        cost_cutensor = ALSSolver.calculate_cost(a1r_cutensor, a2r_cutensor, a12g, n12)

        rel_diff = abs(cost_torch - cost_cutensor) / (abs(cost_torch) + 1e-10)
        assert rel_diff < rtol, f"Cost difference too large: torch={cost_torch}, cutensor={cost_cutensor}"

    def test_als_solve_float64(self, setup_tensors_float64):
        """Test ALS solve with float64 - cuTENSOR vs PyTorch."""
        n12, a12g, ar_shape = setup_tensors_float64
        self._run_als_comparison(n12, a12g, ar_shape)

    def test_als_solve_float32(self, setup_tensors_float32):
        """Test ALS solve with float32 - cuTENSOR vs PyTorch."""
        n12, a12g, ar_shape = setup_tensors_float32
        self._run_als_comparison(n12, a12g, ar_shape, rtol=0.05)

    def test_als_solve_complex128(self, setup_tensors_complex128):
        """Test ALS solve with complex128 - cuTENSOR vs PyTorch."""
        n12, a12g, ar_shape = setup_tensors_complex128
        self._run_als_comparison(n12, a12g, ar_shape)

    def test_als_solve_complex64(self, setup_tensors_complex64):
        """Test ALS solve with complex64 - cuTENSOR vs PyTorch."""
        n12, a12g, ar_shape = setup_tensors_complex64
        self._run_als_comparison(n12, a12g, ar_shape, rtol=0.05)

    def test_output_shapes(self, setup_tensors_float64):
        """Test that cuTENSOR backend returns correct shapes."""
        n12, a12g, ar_shape = setup_tensors_float64
        nD, bD, pD = ar_shape

        config = EvolutionConfig(backend="cutensor", als_niter=10)
        solver = ALSSolver(n12, a12g, ar_shape, config)

        a1r, a2r = solver.solve()

        assert a1r.shape == (nD, bD, pD), f"Wrong a1r shape: {a1r.shape}"
        assert a2r.shape == (nD, bD, pD), f"Wrong a2r shape: {a2r.shape}"

    def test_no_nan_in_output(self, setup_tensors_float64):
        """Test that cuTENSOR backend doesn't produce NaN values."""
        n12, a12g, ar_shape = setup_tensors_float64

        config = EvolutionConfig(backend="cutensor", als_niter=50)
        solver = ALSSolver(n12, a12g, ar_shape, config)

        a1r, a2r = solver.solve()

        assert not torch.isnan(a1r).any(), "a1r contains NaN"
        assert not torch.isnan(a2r).any(), "a2r contains NaN"
        assert not torch.isinf(a1r).any(), "a1r contains Inf"
        assert not torch.isinf(a2r).any(), "a2r contains Inf"

    def test_cost_decreases(self, setup_tensors_float64):
        """Test that ALS iterations decrease the cost function."""
        n12, a12g, ar_shape = setup_tensors_float64

        config = EvolutionConfig(backend="cutensor", als_niter=50)
        solver = ALSSolver(n12, a12g, ar_shape, config)

        a1r_init, a2r_init, _ = solver.initialize_tensors()
        cost_initial = ALSSolver.calculate_cost(a1r_init, a2r_init, a12g, n12)

        a1r_final, a2r_final = solver.solve()
        cost_final = ALSSolver.calculate_cost(a1r_final, a2r_final, a12g, n12)

        assert cost_final <= cost_initial * 1.01, \
            f"Cost increased: {cost_initial} -> {cost_final}"


class TestCutensorContractions:
    """Test individual cuTENSOR contractions against PyTorch."""

    def test_single_iteration_complex128(self, setup_tensors_complex128):
        """Test a single ALS iteration with complex128."""
        n12, a12g, ar_shape = setup_tensors_complex128

        # Run with just 1 iteration to compare
        config_torch = EvolutionConfig(backend="torch", als_niter=1, als_tol=1e-15)
        config_cutensor = EvolutionConfig(backend="cutensor", als_niter=1, als_tol=1e-15)

        solver_torch = ALSSolver(n12.clone(), a12g.clone(), ar_shape, config_torch)
        solver_cutensor = ALSSolver(n12.clone(), a12g.clone(), ar_shape, config_cutensor)

        a1r_torch, a2r_torch = solver_torch.solve()
        a1r_cutensor, a2r_cutensor = solver_cutensor.solve()

        # Compare the solutions directly
        a1r_diff = (a1r_torch - a1r_cutensor).abs().max()
        a2r_diff = (a2r_torch - a2r_cutensor).abs().max()

        print(f"a1r max diff: {a1r_diff:.2e}")
        print(f"a2r max diff: {a2r_diff:.2e}")
        print(f"a1r_torch max: {a1r_torch.abs().max():.2e}")
        print(f"a1r_cutensor max: {a1r_cutensor.abs().max():.2e}")

        # Check costs
        cost_torch = ALSSolver.calculate_cost(a1r_torch, a2r_torch, a12g, n12)
        cost_cutensor = ALSSolver.calculate_cost(a1r_cutensor, a2r_cutensor, a12g, n12)
        print(f"Cost torch: {cost_torch:.2e}")
        print(f"Cost cutensor: {cost_cutensor:.2e}")

        # The solutions should be similar after 1 iteration
        assert a1r_diff < 1.0, f"a1r differs too much: {a1r_diff}"
        assert a2r_diff < 1.0, f"a2r differs too much: {a2r_diff}"


class TestCutensorAvailability:
    """Test extension availability and loading."""

    def test_extension_loaded(self):
        """Test that the extension is properly loaded."""
        assert CUTENSOR_AVAILABLE, "cuTENSOR extension should be available"

    def test_als_solve_available(self):
        """Test that als_solve is available in the extension module."""
        from acetn.evolution._extensions import _C_cutensor
        assert hasattr(_C_cutensor, "als_solve"), "als_solve not available in _C_cutensor"
