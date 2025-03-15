from model.model import Model
from model.pauli_matrix import pauli_matrices

class IsingModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def one_site_observables(self, site):
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        observables = {"sx": X, "sz": Z}
        return observables

    def one_site_hamiltonian(self, site):
        hx = self.params.get('hx')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return -hx*X

    def two_site_hamiltonian(self, bond):
        jz = self.params.get('jz')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return -jz*Z*Z
