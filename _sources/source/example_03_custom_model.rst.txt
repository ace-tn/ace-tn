Building a Custom Model Hamiltonian
===================================

We show how to build a custom Hamiltonian using the Ising model as an example.

Define the Custom Model Class
-----------------------------

To define a custom Hamiltonian model, you need to create a new class that inherits from the `Model` class. This class must implement a method defining the two-site Hamiltonian. Optionally, you can also define a one-site Hamiltonian term, one-site observables, and two-site observables.

Here is how to define a custom model class:

.. code-block:: python

    class CustomModel(Model):
        def __init__(self, config):
            super().__init__(config)

The `__init__` method initializes the custom model by calling the constructor of the `Model` class. It takes in the configuration (`config`), which contains the model config.

Define One-Site Observables
----------------------------

The `one_site_observables` method defines the observables to be measured at each site. For this example, we define the Pauli matrices X and Z as the observables. We provide the Pauli matrices to help with setting up spin-1/2 models, but it is not required to use them.

.. code-block:: python

    def one_site_observables(self, site):
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        observables = {"sx": X, "sz": Z}
        return observables

In this method:

- The `pauli_matrices` function is used to retrieve the Pauli matrices for the given data type (`dtype`) and device (`device`).
- The `observables` dictionary contains the spin operators used to calculate one-site measurements.

Define the Two-Site Hamiltonian
-------------------------------

The `two_site_hamiltonian` method defines the interaction between two neighboring sites. In this case, we define a \( Z \)-type interaction between the sites, with an interaction strength parameter `Jz`.

.. code-block:: python

    def two_site_hamiltonian(self, bond):
        Jz = self.params.get('Jz')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return -Jz*Z*Z

In this method:

- The `bond` parameter contains the sites and bond direction (e.g. [site_1, site_2, k]). Although it is not explicitly used here, the `bond` parameter is useful for defining bond-dependent hamiltonians.
- The interaction between the two sites is given by -Jz*Z*Z, where Z is the Pauli-Z matrix, and `Jz` is the interaction strength.

Define the One-Site Hamiltonian
-------------------------------

You can define a one-site Hamiltonian term using the `one_site_hamiltonian` method. In this example, we define a term that represents an external magnetic field in the \( x \)-direction, controlled by the parameter `hx`.

.. code-block:: python

    def one_site_hamiltonian(self, site):
        hx = self.params.get('hx')
        X,Y,Z,I = pauli_matrices(self.dtype, self.device)
        return -hx*X

In this method:

- The `hx` parameter controls the strength of the magnetic field along the x-axis.
- The term -hx*X represents the interaction of each spin with the external magnetic field along the x-direction.

Setting the Custom Model Class
------------------------------

To use the custom model in your iPEPS simulation, you need to register the model with the iPEPS instance. This allows iPEPS to use your custom Hamiltonian during the evolution steps.

Here is how to initialize an iPEPS instance and use the custom model:

.. code-block:: python

    ipeps_config = toml.load("./input/03_custom_model.toml")
    ipeps = Ipeps(ipeps_config)

    # Set up the custom model and parameters
    ipeps.set_model(CustomModel, {'Jz': 1.0, 'hx': 2.5})

The `set_model` function registers the custom model, where `CustomModel` is the model class we defined earlier, and the parameters `Jz=1.0` and `hx=2.5` are provided as the initial values for the model parameters.

Start Imaginary-Time Evolution
-------------------------------

Once the model is set up, you can start the imaginary-time evolution to compute the ground state.

.. code-block:: python

    ipeps.evolve(dtau=0.1, steps=10)

Sweeping through the Phase Transition
-------------------------------------

The TF-Ising model exhibits a second-order phase transition near `hx=3`. We can calculate the ground state across the transition efficiently by slowly incrementing the field after sufficient imaginary-time evolution steps:

.. code-block:: python

    for hx in np.arange(2.5, 3.5, 0.1):
        ipeps.set_model_params(hx=hx)
        ipeps.evolve(dtau=0.01, steps=500)

In this example:

- The time step `dtau` is reduced to `0.01` for finer updates.
- The number of steps is increased to `500` for more accurate results.
- The external field parameter `hx` is varied between `2.5` and `3.5` in increments of `0.1`.

Full Code Example
-----------------

Here is a complete Python script that demonstrates how to use the custom Hamiltonian model with iPEPS:

.. code-block:: python

    from acetn.ipeps import Ipeps
    from acetn.model import Model
    from acetn.model.pauli_matrix import pauli_matrices
    import toml
    import numpy as np

    class CustomModel(Model):
        def __init__(self, config):
            super().__init__(config)

        def one_site_observables(self, site):
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            observables = {"sx": X, "sz": Z}
            return observables

        def two_site_hamiltonian(self, bond):
            Jz = self.params.get('Jz')
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            return -Jz*Z*Z

        def one_site_hamiltonian(self, site):
            hx = self.params.get('hx')
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            return -hx*X

    if __name__ == '__main__':
        # Initialize the iPEPS
        ipeps_config = toml.load("./input/03_custom_model.toml")
        ipeps = Ipeps(ipeps_config)

        # Set the custom model
        ipeps.set_model(CustomModel, {'Jz': 1.0, 'hx': 2.5})

        # Parameter sweep through the 2nd-order transition
        ipeps.evolve(dtau=0.1, steps=10)
        for hx in np.arange(2.5, 3.5, 0.1):
            ipeps.set_model_params(hx=hx)
            ipeps.evolve(dtau=0.01, steps=500)
