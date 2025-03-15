import torch
import pytest
from acetn.ipeps import Ipeps
from acetn.renormalization.projectors import ProjectorCalculator

@pytest.fixture
def ipeps_and_projector_calculator():
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
    projector_calculator = ProjectorCalculator(ipeps.config.ctmrg)

    return ipeps, projector_calculator

@pytest.fixture
def test_sites():
    return [(0, 0), (1, 0), (1, 1), (0, 1)]

def test_projector_calculator_initialization(ipeps_and_projector_calculator):
    ipeps, projector_calculator = ipeps_and_projector_calculator
    assert isinstance(projector_calculator, ProjectorCalculator)

def test_projector_calculator_calculate(ipeps_and_projector_calculator, test_sites):
    ipeps, projector_calculator = ipeps_and_projector_calculator
    for k in range(4):
        projector_calculator.calculate(ipeps, test_sites, k)

def test_projector_calculator_different_projectors(ipeps_and_projector_calculator, test_sites):
    projector_types = ['half-system', 'full-system']
    for proj_type in projector_types:
        ipeps, _ = ipeps_and_projector_calculator
        config = ipeps.config.ctmrg
        config.projectors = proj_type
        projector_calculator = ProjectorCalculator(config)
        for k in range(4):
            projector_calculator.calculate(ipeps, test_sites, k)

def test_projector_calculator_output_structure(ipeps_and_projector_calculator, test_sites):
    ipeps, projector_calculator = ipeps_and_projector_calculator
    projectors = projector_calculator.calculate(ipeps, test_sites, k=0)
    bD, cD = ipeps.dims['bond'], ipeps.dims['chi']
    expected_shape = (cD, bD, bD, cD)

    assert len(projectors) == 2
    assert all(isinstance(p, torch.Tensor) for p in projectors)
    assert all(p.shape == expected_shape for p in projectors)

def test_projector_calculator_invalid_sites(ipeps_and_projector_calculator):
    ipeps, projector_calculator = ipeps_and_projector_calculator
    invalid_sites = [(100, 100), (200, 200), (300, 300), (400, 400)]

    with pytest.raises(ValueError):
        projector_calculator.calculate(ipeps, invalid_sites, k=0)
