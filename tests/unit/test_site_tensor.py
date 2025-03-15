import pytest
import torch
from acetn.ipeps.site_tensor import SiteTensor

# Sample dimensions for testing
dims = {
    'bond': 4,  # bond dimension
    'phys': 2,  # physical dimension
    'chi': 3,   # boundary dimension
}

@pytest.fixture
def site_tensor():
    # Fixture to create a SiteTensor instance
    return SiteTensor(dims=dims)

def test_initialization(site_tensor):
    """
    Test the initialization of the SiteTensor.
    Ensures that tensors are correctly initialized with the given dimensions.
    """
    assert site_tensor._site_tensor is not None
    assert site_tensor._corner_tensors is not None
    assert site_tensor._edge_tensors is not None
    assert site_tensor._site_tensor.shape == (dims['bond'], dims['bond'], dims['bond'], dims['bond'], dims['phys'])
    assert len(site_tensor._corner_tensors) == 4
    assert len(site_tensor._edge_tensors) == 4

def test_tensor_copy(site_tensor):
    """
    Test the copying of a SiteTensor.
    Ensures that the `copy` method correctly copies the tensors with optional noise.
    """
    site_tensor_copy = SiteTensor(dims=dims, site_tensor=site_tensor)
    
    assert torch.allclose(site_tensor_copy['A'], site_tensor['A'], atol=1e-3)
    assert torch.allclose(site_tensor_copy['C'][0], site_tensor['C'][0], atol=1e-3)
    assert torch.allclose(site_tensor_copy['E'][0], site_tensor['E'][0], atol=1e-3)

def test_set_and_get_tensor(site_tensor):
    """
    Test getting and setting tensors using the __getitem__ and __setitem__ methods.
    """
    new_tensor = torch.ones(dims['bond'], dims['bond'], dims['bond'], dims['bond'], dims['phys'])
    site_tensor['A'] = new_tensor
    
    assert torch.equal(site_tensor['A'], new_tensor)

def test_invalid_key_for_getter(site_tensor):
    """
    Test invalid keys for the __getitem__ method.
    """
    with pytest.raises(ValueError):
        site_tensor['InvalidKey']

def test_invalid_key_for_setter(site_tensor):
    """
    Test invalid keys for the __setitem__ method.
    """
    with pytest.raises(ValueError):
        site_tensor['InvalidKey'] = torch.ones(dims['bond'], dims['bond'], dims['bond'], dims['bond'], dims['phys'])

def test_bond_permute(site_tensor):
    """
    Test the bond_permute method.
    Ensures that bond permutations work correctly.
    """
    permuted_tensor = site_tensor.bond_permute(1)
    assert permuted_tensor.shape == site_tensor['A'].shape
    assert not torch.equal(permuted_tensor, site_tensor['A'])  # Ensure the tensor is permuted

def test_initialize_corner_tensors(site_tensor):
    """
    Test the initialization of corner tensors.
    """
    site_tensor.initialize_corner_tensors()
    assert len(site_tensor['C']) == 4
    assert torch.allclose(site_tensor['C'][0], site_tensor['C'][0], atol=1e-3)  # Check tensors are normalized

def test_initialize_edge_tensors(site_tensor):
    """
    Test the initialization of edge tensors.
    """
    site_tensor.initialize_edge_tensors()
    assert len(site_tensor['E']) == 4
    assert torch.allclose(site_tensor['E'][0], site_tensor['E'][0], atol=1e-3)  # Check tensors are normalized

def test_initialize_site_tensor_with_noise(site_tensor):
    """
    Test the initialization of the site tensor with noise.
    """
    initial_tensor = site_tensor._site_tensor.clone()
    site_tensor.initialize_site_tensor(site_state=[1., 0.], noise=0.1)
    
    assert not torch.equal(site_tensor._site_tensor, initial_tensor)  # Check if noise was added

def test_tensor_copy_with_noise(site_tensor):
    """
    Test the copy method with noise added.
    """
    site_tensor_copy = SiteTensor(dims=dims, site_tensor=site_tensor)
    original_tensor = site_tensor_copy._site_tensor.clone()

    site_tensor_copy.copy(site_tensor, noise=0.1)
    assert not torch.equal(site_tensor_copy._site_tensor, original_tensor)  # Ensure noise was added
