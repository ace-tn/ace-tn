import torch
import pytest
from acetn.ipeps import Ipeps
from acetn.renormalization.ctmrg import ctmrg

@pytest.fixture
def ipeps():
    dims = {'phys': 2, 'bond': 3, 'chi': 9}
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
            'steps': 3,
            'disable_progressbar': True,
        },
    }
    ipeps = Ipeps(ipeps_config)

    return ipeps


def test_ctmrg(ipeps):
    # Check that ipeps tensors are updated by ctmrg
    config = ipeps.config.ctmrg
    for projectors in ['half-system', 'full-system']:
        config.projectors = projectors
        C0_before = {}
        for site in [(0,0),(1,0),(1,1),(0,1)]:
            for k in range(4):
                C0_before[site + (k,)] = ipeps[site]['C'][k]

        ctmrg(ipeps, config)

        for site in [(0,0),(1,0),(1,1),(0,1)]:
            for k in range(4):
                assert isinstance(ipeps, Ipeps)
                assert not (ipeps[site]['C'][k] == C0_before[site + (k,)]).all()
