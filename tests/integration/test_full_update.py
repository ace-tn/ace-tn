import pytest
import torch
from acetn.ipeps import Ipeps
from acetn.evolution.gate import Gate
from acetn.evolution.full_update import FullUpdater
from acetn.model.model_factory import model_factory

@pytest.fixture
def setup_ipeps_and_gate():
    """
    Fixture to set up a mock Ipeps object and a gate for testing.
    """
    # Create a mock Ipeps object with IpepsConfig
    config = {'TN':{'nx':2, 'ny':2, 'dims':{'phys':2, 'bond':5, 'chi':8}}, 
              'model':{'name':'heisenberg', 'params':{'J':1.0}}}
    ipeps = Ipeps(config)

    # Set up mock gate tensors for testing
    model = model_factory.create(ipeps.config.model)
    gate = Gate(model, 0.01, ipeps.bond_list, ipeps.site_list)

    # Mock config object with required parameters
    class Config:
        als_niter = 100
        als_tol = 1e-15
        als_method = "cholesky"
        als_epsilon = 1e-12
        use_gauge_fix = False
        gauge_fix_atol = 1e-12
        positive_approx_cutoff = 1e-12

    config = Config()

    return ipeps, gate, config

def test_tensor_update(setup_ipeps_and_gate):
    """
    Test the tensor update process in the FullUpdater class.
    """
    ipeps, gate, config = setup_ipeps_and_gate

    # Create FullUpdater instance
    full_updater = FullUpdater(ipeps, gate, config)

    # Create some dummy tensors a1 and a2 with new dimensions (5, 5, 5, 5, 3)
    a1 = torch.rand(5, 5, 5, 5, 2, dtype=ipeps.dtype)
    a2 = torch.rand(5, 5, 5, 5, 2, dtype=ipeps.dtype)
    
    bond = [(0, 0), (0, 1), 2]  # Sample bond (site (0, 1) and site (0, 1) with bond index 0)

    # Perform tensor update
    updated_a1, updated_a2 = full_updater.tensor_update(a1, a2, bond)

    # Check if the shapes of the updated tensors are consistent with expectations
    assert updated_a1.shape == a1.shape, "Updated a1 tensor has unexpected shape"
    assert updated_a2.shape == a2.shape, "Updated a2 tensor has unexpected shape"

    # Verify if tensor norms are correct (e.g., they should be normalized)
    assert torch.isclose(updated_a1.norm(), torch.tensor(1.0, dtype=ipeps.dtype), atol=1e-6), "Updated a1 tensor is not normalized"
    assert torch.isclose(updated_a2.norm(), torch.tensor(1.0, dtype=ipeps.dtype), atol=1e-6), "Updated a2 tensor is not normalized"

    # Check consistency of tensor update
    updated_a12, updated_a22 = full_updater.tensor_update(a1, a2, bond)
    tensor_diff_a1 = torch.norm(updated_a1 - updated_a12)
    tensor_diff_a2 = torch.norm(updated_a2 - updated_a22)

    assert tensor_diff_a1 < 1e-6, "a1 tensor update is not sufficiently small"
    assert tensor_diff_a2 < 1e-6, "a2 tensor update is not sufficiently small"
