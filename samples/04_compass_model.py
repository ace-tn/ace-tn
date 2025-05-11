from acetn.ipeps import Ipeps
from acetn.model import Model
from acetn.model.pauli_matrix import pauli_matrices

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
            phi = X*X
        elif self.bond_direction(bond) in ["+y","-y"]:
            phi = -Z*Z
        observables["phi"] = phi
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

def main(config):
    ipeps = Ipeps(config)
    ipeps.set_model(CompassModel, {'jz':-1.0/4.,'jx':-1.0/4.,'hz':1.0/2.,'hx':1.0/2.})

    ipeps.evolve(dtau=0.1, steps=50)
    ipeps.measure()

    for _ in range(10):
        ipeps.evolve(dtau=0.01, steps=100)
        ipeps.measure()

if __name__=='__main__':
    dims = {}
    dims['phys'] = 2
    dims['bond'] = 2
    dims['chi'] = 10

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
