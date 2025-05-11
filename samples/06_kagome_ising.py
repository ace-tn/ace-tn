from acetn.ipeps import Ipeps
from acetn.model import Model
from acetn.model.pauli_matrix import pauli_matrices
import numpy as np

class KagomeIsingModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def one_site_observables(self, site):
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        mx_A = X*I*I
        mx_B = I*X*I
        mx_C = I*I*X
        mz_A = Z*I*I
        mz_B = I*Z*I
        mz_C = I*I*Z
        observables = {
            "mag_x(A)": mx_A,
            "mag_x(B)": mx_B,
            "mag_x(C)": mx_C,
            "mag_z(A)": mz_A,
            "mag_z(B)": mz_B,
            "mag_z(C)": mz_C,
            "mx": (mx_A + mx_B + mx_C)/3.,
            "mz": (mz_A + mz_B + mz_C)/3.,
        }
        return observables

    def one_site_hamiltonian(self, site):
        jz = self.params.get('jz')
        hx = self.params.get('hx')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return -jz*(I*Z*Z + Z*Z*I) \
                -hx*(X*I*I + I*X*I + I*I*X)

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

def main(config):
    ipeps = Ipeps(config)
    ipeps.set_model(KagomeIsingModel, {'jz':1.0/4.,'hx':0.0/2.})

    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.measure()

    hx_list = np.arange(0.1, 2.0, 0.1)
    for hx in hx_list:
        ipeps.set_model_params(hx=hx/2.)
        for _ in range(10):
            ipeps.evolve(dtau=0.01, steps=50)
            ipeps.measure()

if __name__=='__main__':
    dims = {}
    dims['phys'] = 8
    dims['bond'] = 3
    dims['chi'] = 9

    config = {
        'dtype': "float64",
        'device': "cpu",
        'TN':{
            'dims': dims,
            'nx': 2,
            'ny': 2,
        },
    }

    main(config)
