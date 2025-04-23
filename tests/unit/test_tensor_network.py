import pytest
import torch
from acetn.ipeps.site_tensor import SiteTensor
from acetn.ipeps.tensor_network import TensorNetwork
from acetn.ipeps.ipeps_config import TNConfig

@pytest.fixture
def setup_tensor_network():
    # Create a TNConfig instance with default values
    config = TNConfig(nx=2, ny=2, dims={'phys': 2, 'bond': 3, 'chi': 10})
    dtype = torch.float32
    device = torch.device('cpu')  # Use CPU for simplicity
    
    # Create TensorNetwork instance with the provided config
    tensor_network = TensorNetwork(None, config, dtype, device)
    return tensor_network, config


def test_initialization(setup_tensor_network):
    # Test the initialization of TensorNetwork
    tensor_network, config = setup_tensor_network
    
    assert tensor_network.nx == config.nx  # Should match the config values
    assert tensor_network.ny == config.ny
    assert tensor_network.dims == config.dims  # Should match the config values
    assert tensor_network.dtype == torch.float32
    assert tensor_network.device == torch.device('cpu')
    assert len(tensor_network.site_list) == config.nx * config.ny  # 2x2 grid
    assert len(tensor_network.bond_list) == 8  # 2x2 grid with horizontal and vertical bonds


def test_getitem_setitem(setup_tensor_network):
    # Test the __getitem__ and __setitem__ methods
    tensor_network, _ = setup_tensor_network
    site = (0, 0)
    
    # Check if the tensor for site (0, 0) is a SiteTensor
    tensor = tensor_network[site]
    assert isinstance(tensor, SiteTensor)
    
    # Set a new tensor at site (0, 0)
    new_site_tensor = SiteTensor(tensor_network.dims, site_state=[1.0, 0.0], dtype=tensor_network.dtype, device=tensor_network.device)
    tensor_network[site] = new_site_tensor
    
    # Ensure that the tensor is updated correctly
    assert tensor_network[site] == new_site_tensor


def test_copy_from(setup_tensor_network):
    # Test the copy_from method
    tensor_network, config = setup_tensor_network
    new_tensor_network = TensorNetwork(None, config, tensor_network.dtype, tensor_network.device)
    
    # Copy the tensor network from the original one
    new_tensor_network.copy_from(tensor_network)
    
    # Check if the site tensors have been copied correctly
    for site in tensor_network.site_list:
        assert torch.equal(tensor_network[site]['A'], new_tensor_network[site]['A'])


def test_save_load(setup_tensor_network, tmp_path):
    tensor_network, config = setup_tensor_network
    save_path = tmp_path / "test_tensor_network"

    # Ensure the temporary file does not already exist
    if save_path.exists():
        save_path.unlink()  # Remove the file if exists
    
    # Save the tensor network to the temporary path
    tensor_network.save(str(save_path))
    
    # Create a new TensorNetwork instance and load from the saved file
    loaded_tensor_network = TensorNetwork(None, config, tensor_network.dtype, tensor_network.device)
    loaded_tensor_network.load(str(save_path) + ".pt")
    
    # Check if the loaded tensor network has the same tensors
    for site in tensor_network.site_list:
        assert torch.equal(tensor_network[site]['A'], loaded_tensor_network[site]['A'])


def test_initialize_tensors(setup_tensor_network):
    # Test the tensor initialization
    tensor_network, _ = setup_tensor_network
    tensor_network.initialize_site_tensors(tensor_network.initial_site_state_map)

    # Ensure that the site list has been populated correctly
    assert len(tensor_network.site_list) == 4  # 2x2 grid of sites

    # Check that tensors have been initialized
    for site in tensor_network.site_list:
        assert isinstance(tensor_network[site], SiteTensor)
        assert tensor_network[site]['A'].shape == (3, 3, 3, 3, 2)


def test_build_bond_list(setup_tensor_network):
    # Test building the bond list
    tensor_network, _ = setup_tensor_network
    bond_list = tensor_network.build_bond_list()
    
    # There should be 8 bonds (4 horizontal + 4 vertical for a 2x2 grid)
    assert len(bond_list) == 8
    
    # Check if the bonds are correctly defined
    for bond in bond_list:
        assert len(bond) == 3  # Each bond should contain 2 sites and the bond direction
        assert isinstance(bond[0], tuple)  # The first site
        assert isinstance(bond[1], tuple)  # The second site
        assert bond[2] in [1, 2]  # Bond direction should be either 1 (vertical) or 2 (horizontal)
