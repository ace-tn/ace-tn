import pytest
import torch
from torch import einsum
from acetn.evolution.als_solver import ALSSolver
from acetn.evolution.full_update import *
from acetn.ipeps.ipeps_config import EvolutionConfig

@pytest.fixture
def setup_als_solver():
    # Setup for ALSSolver
    dtype = torch.float64
    device = torch.device("cpu")
    nD, bD, pD = 8, 5, 3  # Example dimensions

    # Generate random reduced tensors a1r and a2r
    a1r = torch.rand(nD, bD, pD, dtype=dtype, device=device)
    a2r = torch.rand(nD, bD, pD, dtype=dtype, device=device)

    # Correct a12g initialization using einsum
    a12g = einsum("yup,xuq->yxpq", a1r, a2r)
    a12g += 1e-4 * torch.rand_like(a12g)  # Add small perturbation

    # Generate random norm tensor n12
    n12 = torch.rand(nD, nD, nD, nD, dtype=dtype, device=device)

    # Apply positive approximation and gauge fixing to the tensors
    nz = positive_approx(n12, nD)
    n12, a12g, *_ = gauge_fix(nz, a12g, nD)

    # Create ALSSolver instance with EvolutionConfig
    als_solver = ALSSolver(n12, a12g, bD, pD, nD, config=EvolutionConfig(als_niter=20, als_tol=1e-15))
    
    return als_solver

def test_als_solver_initialization(setup_als_solver):
    als_solver = setup_als_solver
    # Test shapes of n12 and a12g
    assert als_solver.n12.shape == (8, 8, 8, 8)  # Norm tensor shape
    assert als_solver.a12g.shape == (8, 8, 3, 3)  # Reduced tensor shape

def test_als_solver_solution(setup_als_solver):
    als_solver = setup_als_solver
    # Solve and check the shapes of the returned tensors
    a1r, a2r = als_solver.solve()
    
    # Test shapes of solution tensors a1r and a2r
    assert a1r.shape == (8, 5, 3)  # Reduced tensor a1r
    assert a2r.shape == (8, 5, 3)  # Reduced tensor a2r

def test_als_solver_convergence(setup_als_solver):
    als_solver = setup_als_solver

    als_solver.niter=99
    a1r, a2r = als_solver.solve()
    prev_distance = als_solver.calculate_distance(a1r, a2r)

    als_solver.niter=100
    a1r, a2r = als_solver.solve()
    next_distance = als_solver.calculate_distance(a1r, a2r)

    assert abs(next_distance - prev_distance) < als_solver.tol, f"Convergence failed"
