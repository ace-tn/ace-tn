================
The ipeps Module
================

An iPEPS (infinite projected entangled pair states) is represented as a tensor network. This network is composed of individual site tensors, each representing a physical site in a lattice. The system is organized into three hierarchical layers:

1. **Ipeps**: The top-level simulation object. It extends TensorNetwork to include simulation-specific procedures (evolution, renormalization, measurement) and distributed computing support.
2. **TensorNetwork**: Manages the overall lattice. It is responsible for initializing, saving, loading, and connecting the network by assigning a SiteTensor to every lattice site.
3. **SiteTensor**: The basic building block of the network. Each SiteTensor contains a primary tensor (A) along with its associated corner (C) and edge (E) tensors.

In the sections that follow, we describe each layer in detail.

--------------------------------------------
Ipeps Class: The Top-Level Simulation Object
--------------------------------------------

The ``Ipeps`` class encapsulates the entire iPEPS simulation. It builds upon the TensorNetwork class by incorporating simulation-specific configurations and distributed computing features. Key responsibilities include:

- **Configuration:** Creating an ``IpepsConfig`` from a provided configuration dictionary.
- **Distributed Setup:** Initializing and finalizing the distributed computing environment via the ``setup_distributed()`` and ``finalize_distributed()`` functions.
- **Inheritance:** Calling the parent TensorNetwork constructor to build the underlying tensor network.

Example:

.. code-block:: python

   # Example configuration dictionary for iPEPS simulation.
   config = {
       'dtype': torch.float64,
       'device': torch.device('cpu'),
       'TN': {  # TensorNetwork-specific parameters.
           'nx': 4,
           'ny': 4,
           'dims': {'phys': 3, 'bond': 2, 'chi': 4},
       }
   }

   # Create an iPEPS tensor network.
   ipeps = Ipeps(config)

The distributed environment is set up within the Ipeps constructor and is finalized automatically when the object is deleted.

-----------------------------------------
TensorNetwork Class: Managing the Lattice
-----------------------------------------

The ``TensorNetwork`` class constructs and manages the iPEPS tensor network over a lattice. Each lattice site is represented by a ``SiteTensor``. This class is responsible for:

- **Network Initialization:** Creating a SiteTensor for each site based on the configuration.
- **Lattice Connectivity:** Building site and bond lists that define the network's structure.
- **Persistence:** Saving and loading the network state.
- **Copying:** Duplicating an existing network.

Example:

.. code-block:: python

   # Example configuration for the TensorNetwork.
   class TNConfig:
       nx = 4
       ny = 4
       dims = {'phys': 3, 'bond': 2, 'chi': 4}

   config = TNConfig()

   # Create a new TensorNetwork.
   tensor_network = TensorNetwork(tensor_network=None, config=config, dtype=torch.float64, device=torch.device("cpu"))

   # Retrieve the tensor at site (0, 0).
   site_tensor = tensor_network[(0, 0)]
   print("Site tensor at (0,0):")
   print(site_tensor['A'])

The TensorNetwork class also provides methods to copy, save, and load the network state, as well as to build lists of sites and bonds.

------------------------------------
SiteTensor Class: The Building Block
------------------------------------

The ``SiteTensor`` class represents an individual site within the tensor network. It encapsulates:
- **A (Site Tensor):** The primary tensor holding the site's data.
- **C (Corner Tensors):** A list of tensors representing the corners.
- **E (Edge Tensors):** A list of tensors representing the edges.

Initialization requires a dimensions dictionary that includes:
- **phys:** Physical dimension.
- **bond:** Bond dimension.
- **chi:** Auxiliary dimension for the corner and edge tensors.

The constructor can also take an initial state for the site tensor. It is important that the length of the ``site_state`` list matches the physical dimension (``phys``).

Example:

.. code-block:: python

   # Define the dimensions for the tensor network.
   dims = {'phys': 3, 'bond': 2, 'chi': 4}

   # Define an initial state for the site tensor (length must equal 'phys').
   site_state = [1.0, 0.0, 0.0]

   # Create a new SiteTensor object.
   site_tensor = SiteTensor(dims, site_state=site_state, dtype=torch.float64, device=torch.device("cpu"))

   # Display the initialized site tensor.
   print("Initialized Site Tensor (A):")
   print(site_tensor['A'])

Additionally, the SiteTensor class overloads indexing to easily access and modify:
- **'A'**: The site tensor.
- **'C'**: The corner tensors.
- **'E'**: The edge tensors.

----------
Conclusion
----------

This top-down overview has demonstrated the hierarchical structure of our iPEPS simulation framework:

- **Ipeps:** The top-level simulation object that extends TensorNetwork to incorporate simulation-specific and distributed computing features.
- **TensorNetwork:** The backbone that organizes the lattice by composing individual SiteTensors and managing network state.
- **SiteTensor:** The fundamental unit representing a single lattice site, complete with associated corner and edge tensors.

Together, these classes provide a framework for constructing, simulating, and managing iPEPS tensor networks.
