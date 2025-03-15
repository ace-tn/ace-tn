from acetn.ipeps import Ipeps
from acetn.model import Model
from acetn.model.pauli_matrix import pauli_matrices
import numpy as np

class HoneycombIsingModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def one_site_observables(self, site):
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        observables = {}
        mx_A = X*I
        mx_B = I*X
        mz_A = Z*I
        mz_B = I*Z
        mx = 0.5*(mx_A + mx_B)
        mz = 0.5*(mz_A + mz_B)
        observables["mag_x(A)"] = mx_A*0.5
        observables["mag_x(B)"] = mx_B*0.5
        observables["mag_z(A)"] = mz_A*0.5
        observables["mag_z(B)"] = mz_B*0.5
        observables["mag_x"] = mx*0.5
        observables["mag_z"] = mz*0.5
        return observables

    def one_site_hamiltonian(self, site):
        jz = self.params.get('jz')
        hx = self.params.get('hx')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return -jz*(Z*Z) - hx*((X*I)+(I*X))

    def two_site_hamiltonian(self, bond):
        jz = self.params.get('jz')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        match self.bond_direction(bond):
            case '+x':
                return -jz*(I*Z)*(Z*I)
            case '-x':
                return -jz*(Z*I)*(I*Z)
            case '+y':
                return -jz*(Z*I)*(I*Z)
            case '-y':
                return -jz*(I*Z)*(Z*I)


if __name__=='__main__':
    dims = {}
    dims['phys'] = 4
    dims['bond'] = 2
    dims['chi'] = 10

    ctmrg_steps = 20

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
            'projectors': 'half-system',
        },
    }

    ipeps = Ipeps(ipeps_config)
    ipeps.set_model(HoneycombIsingModel, {'jz':1.0/4.,'hx':0.0/2.})

    dtau = 0.1
    steps = 10
    ipeps.evolve(dtau, steps=steps)

    dtau = 0.01
    steps = 50
    hx_list = np.arange(0.1, 2.0, 0.1)
    for hx in hx_list:
        ipeps.set_model_params(hx=hx/2.)
        for _ in range(10):
            ipeps.evolve(dtau, steps=steps)
