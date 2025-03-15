from acetn.ipeps import Ipeps
from acetn.model import Model
from acetn.model.pauli_matrix import pauli_matrices
import numpy as np

class KagomeIsingModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def one_site_observables(self, site):
        X,Y,Z,I = self.pauli_matrices
        mx_A = X*I*I
        mx_B = I*X*I
        mx_C = I*I*X
        mz_A = Z*I*I
        mz_B = I*Z*I
        mz_C = I*I*Z
        observables = {
            "mag_x(A)": 0.5*mx_A,
            "mag_x(B)": 0.5*mx_B,
            "mag_x(C)": 0.5*mx_C,
            "mag_z(A)": 0.5*mz_A,
            "mag_z(B)": 0.5*mz_B,
            "mag_z(C)": 0.5*mz_C,
        }
        return observables

    def one_site_hamiltonian(self, site):
        jz = self.params.get('jz')
        hx = self.params.get('hx')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return -jz*((I*Z*Z)+(Z*Z*I)) \
                -hx*((X*I*I)+(I*X*I)+(I*I*X))

    def two_site_hamiltonian(self, bond):
        jz = self.params.get('jz')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        match self.bond_direction(bond):
            case '-x':
                return -jz*(Z*I*I)*(I*I*Z) -jz*(I*Z*I)*(I*I*Z)
            case '+x':
                return -jz*(I*I*Z)*(Z*I*I) -jz*(I*I*Z)*(I*Z*I)
            case '-y':
                return -jz*(I*Z*I)*(Z*I*I) -jz*(I*I*Z)*(Z*I*I)
            case '+y':
                return -jz*(Z*I*I)*(I*Z*I) -jz*(Z*I*I)*(I*I*Z)


if __name__=='__main__':
    dims = {}
    dims['phys'] = 8
    dims['bond'] = 3
    dims['chi'] = 9

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
            'projectors': 'half-system',
        },
    }

    ipeps = Ipeps(ipeps_config)
    ipeps.set_model(KagomeIsingModel, {'jz':1.0/4.,'hx':0.0/2.})

    hx_list = [0.1, 0.5, 0.8, 1.0, 1.4, 4.0]
    hx_list = np.arange(0.1, 2.0, 0.1)

    dtau = 0.1
    steps = 10
    ipeps.evolve(dtau, steps=steps)

    for hx in hx_list:
        ipeps.set_model_params(hx=hx/2.)
        dtau = 0.01
        steps = 50
        for _ in range(10):
            print(f"------ hx={hx} ------")
            ipeps.evolve(dtau, steps=steps)

        print(f"------ hx={hx} ------")
        dtau = 0.005
        steps = 200
        ipeps.evolve(dtau, steps=steps)
