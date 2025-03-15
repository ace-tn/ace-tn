from ..model import Model
from ..pauli_matrix import pauli_matrices

class HeisenbergModel(Model):
    def __init__(self, config):
        super().__init__(config)

    def initial_site_state(self, site):
        """Initialize in the Neel state"""
        xi,yi = site
        return [1.,0.] if (xi+yi)%2==0 else [0.,1.]

    def one_site_observables(self, site):
        """Measure staggered magnetization"""
        xi,yi = site
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        if (xi + yi) % 2 == 0:
            return {"sx": X, "sz": Z}
        else:
            return {"sx": -X, "sz": -Z}

    def two_site_hamiltonian(self, bond):
        r"""$H_{ij} = \vec{S}_i\cdot\vec{S}_j$"""
        J = self.params.get('J')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return 0.25*J*(X*X + Y*Y + Z*Z)
