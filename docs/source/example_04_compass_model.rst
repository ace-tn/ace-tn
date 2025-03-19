Quantum Compass Model
=====================

The Quantum Compass model serves as an insightful example of a Hamiltonian used to explore quantum phase transitions and has subextensive degeneracy in zero field. Its Hamiltonian is defined as:

.. math::

    H = -\sum_{\vec{r}} \left(J_x S^x_{\vec{r}}S^x_{\vec{r}+\vec{e}_x} + J_z S^z_{\vec{r}}S^z_{\vec{r}+\vec{e}_z} + h_x S^x_{\vec{r}} + h_z S^z_{\vec{r}} \right)

The `two_site_hamiltonian` method defines the bond interactions along the x- and z-directions, using Pauli matrices for spin interactions. The `Model` method `bond_direction` maps each bond to a direction in the set {"+x", "-x", "+y", "-y"} to help with specifying bond-dependent Hamiltonians.

Defining a Custom Model
-----------------------

To define a custom model, import the `Model` class, create a subclass, and implement the `two_site_hamiltonian` and `one_site_hamiltonian` methods. These methods specify bond interactions and single-site terms, respectively. In the example below, we define the Quantum Compass model's Hamiltonian in the `two_site_hamiltonian` method.

Here is an example of how to define the model:

.. code-block:: python

    from acetn.model import Model
    from acetn.model.pauli_matrix import pauli_matrices

    class CompassModel(Model):
        def __init__(self, config):
            super().__init__(config)

        def two_site_hamiltonian(self, bond):
            jx = self.params.get("jx")
            jz = self.params.get("jz")
            X,Y,Z,I = pauli_matrices(self.dtype, self.device)
            if self.bond_direction(bond) in ["+x", "-x"]:
                return -jx*X*X
            elif self.bond_direction(bond) in ["+y", "-y"]:
                return -jz*Z*Z

        def one_site_hamiltonian(self, site):
            hx = self.params.get("hx")
            hz = self.params.get("hz")
            X, Y, Z, I = pauli_matrices(self.dtype, self.device)
            return -hx*X - hz*Z

Defining Observables
--------------------

In addition to the Hamiltonian, we need to define the observables that will be measured during the simulation. For the Quantum Compass model, an important observable is the order parameter that characterizes the spin orientations along the bond directions. The order parameter phi is defined as:

.. math::

    \phi = \sum_{\vec{r}} \left\langle \sigma^x_{\vec{r}} \sigma^x_{\vec{r} + \vec{e}_x} - \sigma^z_{\vec{r}} \sigma^z_{\vec{r} + \vec{e}_z} \right\rangle

This order parameter is negative when the spins are oriented along the ±z-directions and positive when they are oriented along the ±x-directions. We can define this observable in the `two_site_observables` method of the custom model class.

In addition to \phi, we may also want to measure the correlation between the Pauli matrices X and Z along different bonds. This is a component of the vector chirality, denoted \chi, and defined as:

.. math::

    \chi = \sum_{\langle i j \rangle} \left\langle \sigma^x_{\vec{r}_i} \sigma^z_{\vec{r}_j} - \sigma^z_{\vec{r}_i} \sigma^x_{\vec{r}_j} \right\rangle

This characterizes the ordered state when `Jx = Jz = hx = hz`, which can be derived exactly.

We can define both \phi and \chi in the `two_site_observables` method as shown below:

.. code-block:: python

    def two_site_observables(self, bond):
        observables = {}
        X, Y, Z, I = pauli_matrices(self.dtype, self.device)
        if self.bond_direction(bond) in ["+x", "-x"]:
            observables["phi"] = X*X
        elif self.bond_direction(bond) in ["+y", "-y"]:
            observables["phi"] = -Z*Z
        observables["chi"] = X*Z - Z*X
        return observables

Setting up the iPEPS Simulation
-------------------------------

Once the custom model is defined, the iPEPS instance can be configured to run the simulation. The iPEPS configuration specifies physical and bond dimensions. The model parameters are provided for the Hamiltonian.

Here is an example of how to configure the iPEPS simulation:

.. code-block:: python

    config = {
        "dtype": "float64",
        "device": "cpu",
        "TN": {
            "dims": {"phys": 2, "bond": 2, "chi": 10},
            "nx": 2,
            "ny": 2
        },
    }
    ipeps = Ipeps(config)
    ipeps.set_model(CompassModel, {"jx": -1.0 / 4., "jz": -1.0 / 4., "hz": 1.0 / 2., "hx": 1.0 / 2.})

Note: The QCM has subextensive ground-state degeneracy in zero field. You may want to increase the lattice size parameters `nx` and/or `ny` to explore the properties of the ground states.

Running the Imaginary-Time Evolution
------------------------------------

The iPEPS simulation can be used to evolve the system to obtain its ground state. The evolution is performed by iterating through a series of time steps with gradually smaller time intervals.

Here is how to perform the imaginary-time evolution:

.. code-block:: python

    dtau = 0.1
    steps = 50
    ipeps.evolve(dtau, steps=steps)

    dtau = 0.01
    steps = 100
    for _ in range(5):
        ipeps.evolve(dtau, steps=steps)

In this step:
- We start by evolving the system with a larger time step (`dtau = 0.1`) for 50 steps.
- The time step is then reduced to `dtau = 0.01`, and 100 steps are performed to refine the accuracy.

Full Code Example
-----------------

Below is the complete Python script demonstrating the Quantum Compass model with iPEPS:

.. code-block:: python

    from acetn.ipeps import Ipeps
    from acetn.model import Model
    from acetn.model.pauli_matrix import pauli_matrices

    class CompassModel(Model):
        def __init__(self, config):
            super().__init__(config)

        def two_site_hamiltonian(self, bond):
            jx = self.params.get("jx")
            jz = self.params.get("jz")
            X, Y, Z, I = pauli_matrices(self.dtype, self.device)
            if self.bond_direction(bond) in ["+x", "-x"]:
                return -0.25 * jx * X * X
            elif self.bond_direction(bond) in ["+y", "-y"]:
                return -0.25 * jz * Z * Z

        def two_site_observables(self, bond):
            observables = {}
            X, Y, Z, I = pauli_matrices(self.dtype, self.device)
            if self.bond_direction(bond) in ["+x", "-x"]:
                observables["phi"] = X * X
            elif self.bond_direction(bond) in ["+y", "-y"]:
                observables["phi"] = -Z * Z
            return observables

    if __name__ == "__main__":
        config = {
            "dtype": "float64",
            "device": "cpu",
            "TN": {"dims": {"phys": 2, "bond": 2, "chi": 10}, "nx": 2, "ny": 2},
        }

        ipeps = Ipeps(config)
        ipeps.set_model(CompassModel, {"jx": -1.0 / 4., "jz": -1.0 / 4., "hz": 1.0 / 2., "hx": 1.0 / 2.})

        dtau = 0.1
        steps = 50
        ipeps.evolve(dtau, steps=steps)

        dtau = 0.01
        steps = 100
        for _ in range(5):
            ipeps.evolve(dtau, steps=steps)

