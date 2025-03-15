import pytest
import torch
from acetn.measurement.rdm import RDM
from acetn.ipeps.site_tensor import SiteTensor

# Mock iPEPS class for testing purposes
class MockIPEPS:
    def __init__(self):
        self.site_tensors = {}
    
    def __getitem__(self, key):
        return self.site_tensors.get(key, {})
    
    def __setitem__(self, key, value):
        self.site_tensors[key] = value

@pytest.fixture
def rdm_object():
    ipeps = MockIPEPS()
    
    # Set some mock tensors in the iPEPS object
    ipeps[(0, 0)] = SiteTensor({'phys':3, "bond":4, "chi":5})
    ipeps[(1, 0)] = SiteTensor({'phys':3, "bond":4, "chi":5})
    
    # Return an instance of the RDM object
    return RDM(ipeps)

# Test for the `build_site_rdm` method
def test_build_site_rdm(rdm_object):
    # Using a mock site (0, 0) to build the site RDM
    rdm = rdm_object[(0, 0)]  # Call the `__getitem__` method to access site RDM
    expected_shape = (rdm_object.ipeps[(0, 0)]['A'].shape[4],)*2  # Shape should match the physical dimension
    
    # Test if the RDM has the expected shape
    assert rdm.shape == expected_shape

    # Check if the RDM is a tensor (we can also check specific values if needed)
    assert isinstance(rdm, torch.Tensor)

# Test for the `build_bond_rdm` method
def test_build_bond_rdm(rdm_object):
    # Using mock sites (0, 0) and (1, 0) with bond index 0
    rdm = rdm_object[[(0, 0), (1, 0), 0]]  # Call the `__getitem__` method to access bond RDM
    
    # We expect the RDM to have shape based on the bond and physical dimensions
    expected_shape = (rdm_object.ipeps[(0, 0)]['A'].shape[4],)*4
    
    # Test if the RDM has the expected shape
    assert rdm.shape == expected_shape

    # Check if the RDM is a tensor (we can also check specific values if needed)
    assert isinstance(rdm, torch.Tensor)

# Test for an invalid index (KeyError case)
def test_invalid_index_access(rdm_object):
    with pytest.raises(KeyError):
        # Try accessing a non-existent site (5, 5)
        rdm_object[(5, 5)]
