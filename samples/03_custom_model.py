from ipeps import Ipeps
from model import Model
from model.pauli_matrix import pauli_matrices
import toml
import numpy as np

if __name__=='__main__':
    # initialize an ipeps instance
    ipeps_config = toml.load("./input/03_custom_model.toml")
    ipeps = Ipeps(ipeps_config)

    # define a custom model class
    class CustomModel(Model):
        def __init__(self, config):
            super().__init__(config)

        # define one-site observables to be measured
        def one_site_observables(self, site):
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            observables = {"sx": X, "sz": Z}
            return observables

        # two_site_hamiltonian method must be specified
        def two_site_hamiltonian(self, bond):
            Jz = self.params.get('Jz')
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            return -Jz*Z*Z

        # optionally, specify a one_site_hamiltonian
        def one_site_hamiltonian(self, site):
            hx = self.params.get('hx')
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            return -hx*X

    # allow the ipeps to access the custom model implementation by registering the model
    ipeps.set_model(CustomModel, {'Jz':1.0,'hx':2.5})

    # start imaginary-time evolution with a few large time steps
    dtau = 0.1
    steps = 10
    ipeps.evolve(dtau, steps=steps)

    # reduce the time step and increase steps for more accurate results
    dtau = 0.01
    steps = 500
    for hx in np.arange(2.5, 3.5, 0.1):
        # model params can be updated before the next evolution
        ipeps.set_model_params(hx=hx)
        ipeps.evolve(dtau, steps=steps)
