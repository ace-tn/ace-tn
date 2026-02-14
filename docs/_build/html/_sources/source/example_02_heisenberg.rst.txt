Heisenberg Model
================

The Heisenberg model is a well-known quantum spin model that describes interactions between spins in a lattice. In the case of the Heisenberg spin-1/2 model, the Hamiltonian is given by:

.. math::

    H = -J \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j

The ground-state energy is expected to be -0.669437 from precise quantum Monte Carlo calculations.

Config File Setup
-----------------

To calculate the ground-state iPEPS, create a config file (`02_heisenberg.toml`) containing:

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

    [ctmrg]
    steps = 10
    projectors = "half-system"

    [model]
    name = "heisenberg"

    [model.params]
    J = 1.0

The configuration file specifies the data type (`dtype`) and device (`device`), the tensor network (`TN`) properties, and the model parameters.

- The `TN` section specifies the dimensions of the tensor network. The values `nx` and `ny` define the size of the unit cell, and `dims` specifies the physical dimension (`phys`), bond dimension (`bond`), and the maximum corner bond dimension (`chi`).
- The `ctmrg` section specifies the number of steps for the CTMRG algorithm and the type of projectors used for the algorithm.
- The `model` section defines the model to be used, in this case, the Heisenberg model, and its corresponding parameter (`J`).

Initialize an iPEPS
-------------------

Given the input config, an iPEPS can be constructed by:

.. code-block:: python

    from acetn.ipeps import Ipeps
    import toml

    config = toml.load('./input/02_heisenberg.toml')
    ipeps = Ipeps(config)

The iPEPS site tensors are initialized in the Neel-ordered configuration for the Heisenberg model.

Evolve to the Ground State
--------------------------

We can evolve the iPEPS by:

.. code-block:: python

    ipeps.evolve(dtau=0.1, steps=10)
    ipeps.measure()

Using a large imaginary-time step `dtau=0.1` will bring the system closer to the ground state faster than a smaller step size (e.g. `dtau=0.01`).

Additional Evolution Steps
--------------------------

To refine the approximation, we can continue evolving the system using smaller time steps:

.. code-block:: python

    for _ in range(5):
        ipeps.evolve(dtau=0.01, steps=100)
        ipeps.measure()

It is a good practice to start with a larger time step at first and then decrease the time step in further iterations to attain better accuracy.

Full Code Example
-----------------

Here is a complete Python script that initializes the iPEPS, evolves it to the ground state, and measures the observables:

.. code-block:: python

    from acetn.ipeps import Ipeps
    import toml

    def main(config):
        ipeps = Ipeps(config)

        # Initial evolution step
        ipeps.evolve(dtau=0.1, steps=10)
        ipeps.measure()

        # Additional evolution steps
        for _ in range(5):
            ipeps.evolve(dtau=0.01, steps=100)
            ipeps.measure()

    if __name__ == '__main__':
        # Set config options in the input file: "02_heisenberg.toml"
        config = toml.load("./input/02_heisenberg.toml")
        main(config)

In this script:
- The `main` function takes the config file as an argument, initializes the iPEPS object, and runs the evolution steps.
- The initial evolution step uses a larger `dtau` to quickly approach the ground state, followed by finer updates in the loop with smaller `dtau` values.

