from ipeps import Ipeps
from model import Model
from model.pauli_matrix import pauli_matrices
import numpy as np

if __name__=='__main__':
    dims = {}
    dims['phys'] = 2
    dims['bond'] = 2
    dims['chi'] = 10

    ctmrg_steps = 40

    dtype = "float64"
    device = "cpu"

    ipeps_config = {
        'dtype': dtype,
        'device': device,
        'TN':{
            'dims': dims,
            'nx': 2,
            'ny': 2,
        },
        'ctmrg':{
            'steps': ctmrg_steps,
            'projectors': 'full-system',
        },
    }

    class CompassModel(Model):
        def __init__(self, config):
            super().__init__(config)

        def one_site_observables(self, site):
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            observables = {"sx": X, "sz": Z}
            return observables

        def two_site_observables(self, bond):
            observables = {}
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            if self.bond_direction(bond) in ["+x","-x"]:
                observables["phi"] = X*X
            elif self.bond_direction(bond) in ["+y","-y"]:
                observables["phi"] = -Z*Z
            observables["chi"] = X*Z - Z*X
            return observables

        def one_site_hamiltonian(self, site):
            hx = self.params.get('hx')
            hz = self.params.get('hz')
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            return -hx*X - hz*Z

        def two_site_hamiltonian(self, bond):
            jx = self.params.get('jx')
            jz = self.params.get('jz')
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            if self.bond_direction(bond) in ['+x','-x']:
                return -jx*X*X
            elif self.bond_direction(bond) in ['+y','-y']:
                return -jz*Z*Z

    ipeps = Ipeps(ipeps_config)
    ipeps.set_model(CompassModel, {'jz':-1.0/4.,'jx':-1.0/4.,'hz':1.0/2.,'hx':1.0/2.})

    dtau = 0.1
    steps = 50
    ipeps.evolve(dtau, steps=steps)

    dtau = 0.01
    steps = 100
    for _ in range(5):
        ipeps.evolve(dtau, steps=steps)

    dtau = 0.005
    steps = 400
    ipeps.evolve(dtau, steps=steps)

    dtau = 0.001
    steps = 400
    ipeps.evolve(dtau, steps=steps)

    dtau = 0.0001
    steps = 400
    ipeps.evolve(dtau, steps=steps)
