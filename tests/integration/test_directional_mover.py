import torch
import torch.distributed as dist
import pytest
from acetn.ipeps import Ipeps
from acetn.renormalization.directional_mover import DirectionalMover

@pytest.fixture
def ipeps_and_directional_mover():
    dims = {'phys': 2, 'bond': 6, 'chi': 36}
    dtype = torch.float64
    device = torch.device('cpu')

    ipeps_config = {
        'dtype': dtype,
        'device': device,
        'TN': {
            'nx': 2,
            'ny': 2,
            'dims': dims,
        },
        'ctmrg': {
            'projectors': 'half-system',
        },
    }
    ipeps = Ipeps(ipeps_config)
    directional_mover = DirectionalMover(ipeps.config.ctmrg)

    return ipeps, directional_mover

@pytest.fixture
def test_sites():
    return [(0, 0), (1, 0), (1, 1), (0, 1)]

def test_directional_mover_initialization(ipeps_and_directional_mover):
    ipeps, directional_mover = ipeps_and_directional_mover
    assert isinstance(directional_mover, DirectionalMover)
    assert callable(directional_mover.calculate_projectors)

@pytest.mark.parametrize('move_func', ['left_move', 'up_move', 'right_move', 'down_move'])
def test_move_methods(ipeps_and_directional_mover, move_func):
    ipeps, directional_mover = ipeps_and_directional_mover
    xi, yi = 0, 0  # Arbitrary test coordinates for moves
    move_method = getattr(directional_mover, move_func)
    move_method(ipeps, xi if 'left' in move_func or 'right' in move_func else yi)

@pytest.mark.skipif(condition=not dist.is_initialized(), 
                    reason="Distributed environment is not initialized, skipping distributed test.")
@pytest.mark.parametrize('move_func', ['left_right_move_dist', 'up_down_move_dist'])
def test_distributed_moves(ipeps_and_directional_mover, move_func):
    ipeps, directional_mover = ipeps_and_directional_mover
    coord1, coord2 = 0, 1  # Arbitrary coordinates for distributed moves
    move_method = getattr(directional_mover, move_func)
    move_method(ipeps, coord1, coord2)

@pytest.mark.parametrize('dir_func, site_idx', [
    ('calculate_left_projectors', 0),
    ('calculate_right_projectors', 1),
    ('calculate_up_projectors', 2),
    ('calculate_down_projectors', 3)
])
def test_projectors(ipeps_and_directional_mover, test_sites, dir_func, site_idx):
    ipeps, directional_mover = ipeps_and_directional_mover
    xi, yi = test_sites[site_idx]
    proj1, proj2 = getattr(directional_mover, dir_func)(ipeps, xi, yi)
    assert isinstance(proj1, torch.Tensor)
    assert isinstance(proj2, torch.Tensor)

def test_renormalize_boundary(ipeps_and_directional_mover, test_sites):
    ipeps, directional_mover = ipeps_and_directional_mover
    proj1, proj2 = {}, {}
    for yi in range(2):
        proj1[yi], proj2[yi] = directional_mover.calculate_left_projectors(ipeps, test_sites[yi][0], test_sites[yi][1])
    
    s1, s2 = (0, 0), (1, 0)  # Example boundary sites.
    directional_mover.renormalize_boundary(ipeps, proj1, proj2, s1, s2, 0, 1, 0)
    assert isinstance(ipeps[s2]['C'][0], torch.Tensor)  # Ensure that the C tensor was updated.
