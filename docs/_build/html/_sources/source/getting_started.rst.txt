Getting Started
===============

Installation
------------

To get started, first install the project using pip

.. code-block:: bash

   pip install acetn

If Ace-TN is already installed, upgrade to the latest version with

.. code-block:: bash

   pip install --upgrade acetn

Alternatively, install from GitHub with

.. code-block:: bash

   pip install git+https://github.com/ace-tn/ace-tn

Quickstart
----------

After installation, we are ready to build an iPEPS. If running in CPU mode, we recommend specifying 
the number of threads used by setting one of the following environment variables, depending on your 
hardware setup, before getting started. For example:

.. code-block:: bash

    export MKL_NUM_THREADS=4
    export OMP_NUM_THREADS=4

Create a basic **input.toml** file to specify the iPEPS simulation config

.. code-block:: bash

    dtype = "float64"
    device = "cuda"
    
    [TN]
    nx = 2
    ny = 2
    
    [TN.dims]
    phys = 2
    bond = 4
    chi = 16
    
    [model]
    name = "heisenberg"
    
    [model.params]
    J = 1.0

This input can then be used to instantiate an iPEPS on the GPU with FP64 data type, 
physical dimension d=2, bond dimension D=4, boundary bond dimension chi=16, and 
AFM Heisenberg as the target model. The iPEPS is then instantiated with

.. code-block:: python

    from acetn.ipeps import Ipeps
    import toml

    config = toml.load("input.toml")
    ipeps = Ipeps(config)

The basic workflow of an iPEPS ground-state including imaginary-time evolution and measurement can be executed with

.. code-block:: python

    ipeps.evolve(dtau=0.01, steps=500)
    measurements = ipeps.measure()

For a complete reference of all input options, see :doc:`input_settings`. For detailed examples demonstrating usage including custom model construction, see the :doc:`examples` section.
