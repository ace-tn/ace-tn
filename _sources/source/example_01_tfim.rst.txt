Transverse-Field Ising Model
============================

The transverse-field Ising model (TFIM) is a simple model exhibiting a quantum phase transition and a common test subject for iPEPS simulations. The model is defined by the Hamiltonian:

.. math::

    H = -J\sum_{\langle i j \rangle} S_i^z S_j^z - h_x \sum_{i} S_i^x

Config File Setup
-----------------

To simulate the ground state of this model, first create the config file `input.toml` or any filename that you prefer containing:

.. code-block:: toml

    dtype = "float64"
    device = "cpu"

    [TN]
    nx = 2
    ny = 2

    [TN.dims]
    phys = 2
    bond = 4
    chi = 16

    [model]
    name = "ising"

    [model.params]
    jz = 1.0
    hx = 0.1

The first two parameters set the data type and which device to store the tensors. The `TN` entries specify the tensor-network unit cell and dimensions, and the `model` entries specify the name of the predefined model to use. Predefined models such as "ising" and "heisenberg" are supplied mainly for testing and benchmarking purposes.

Initialize an iPEPS
-------------------

Given the input config, an iPEPS can be constructed by:

.. code-block:: python

    from acetn.ipeps import Ipeps
    import toml

    config = toml.load('input.toml')
    ipeps = Ipeps(config)

The iPEPS site tensors are initialized in the ferromagnetic configuration for the Ising model with some noise. This is already close to the ground state, but not exactly. The ground state is typically found after a few hundred steps of imaginary-time evolution.

Evolve to the Ground State
--------------------------

We can evolve the iPEPS by:

.. code-block:: python

    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.measure()

The iPEPS should now be close to the ground state.

Additional Evolution Steps
--------------------------

For improved accuracy, the iPEPS can be evolved for more steps:

.. code-block:: python

    for _ in range(5):
        ipeps.evolve(dtau=0.01, steps=100)
        ipeps.measure()

The evolution steps here use smaller `dtau` values for finer updates, iterating multiple times to refine the approximation of the ground state.

Modify Model Parameters
-----------------------

The iPEPS can be evolved again with different model parameters. For example, to use the converged iPEPS to determine the ground state at a different field value (e.g., `hx=0.2`), this can be done by:

.. code-block:: python

    ipeps.set_model_params(hx=0.2)
    ipeps.evolve(dtau=0.01, steps=100)

Full Code Example
-----------------

Here is a complete code example that initializes an iPEPS, evolves it to the ground state, and measures the observables using `hx=0.1` and `hx=0.2`:

.. code-block:: python

    from acetn.ipeps import Ipeps

    def main(config):
        # Initialize iPEPS
        ipeps = Ipeps(config)

        # Determine ground state for hx=0.1 and measure observables
        ipeps.evolve(dtau=0.01, steps=100)
        measurements = ipeps.measure()

        # Determine ground state for hx=0.2 and measure observables
        ipeps.set_model_params(hx=0.2)
        ipeps.evolve(dtau=0.01, steps=100)
        measurements = ipeps.measure()

    if __name__=='__main__':
        config = {
            'dtype': "float64",
            'device': "cpu",
            'TN':{
                'nx': 2,
                'ny': 2,
                'dims': {'phys':2, 'bond':2, 'chi':20},
            },
            'model':{
                'name': 'ising',
                'params':{
                    'jz': 1.0,
                    'hx': 0.1,
                },
            },
        }

        main(config)
