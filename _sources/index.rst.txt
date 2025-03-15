.. ctmrg documentation master file, created by
   sphinx-quickstart on Fri Jan 10 14:11:02 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Ace-TN Documentation
===================================


.. .. sidebar::
..    :class: sphinxsidebar

..    .. image:: source/png/screenshot4.png
..       :alt: An image in a sidebar
..       :width: 175px

Ace-TN is a framework for simulating infinite projected entangled-pair states (iPEPS) with a focus on
GPU acceleration. We follow a simple pythonic design philosophy with PyTorch tensors as the primary 
tensor-network building blocks, allowing users to quickly setup and manage iPEPS calculations on GPUs.

Features
--------

- Fast iPEPS ground-state calculations using GPUs.
- Flexible interface for custom model hamiltonian design.
- Modular and extensible design enabling quick integration of new iPEPS algorithms in a multi-GPU setting.

Code Example
-------------

The basic workflow of an iPEPS ground-state calculation with Ace-TN can be executed with

.. code-block:: python

    from acetn import Ipeps
    ipeps = Ipeps(config)
    ipeps.evolve(dtau=0.01, steps=500)
    measurements = ipeps.measure()

Please see the `User Guide <source/user_guide.html>`_ for more info on usage and getting started.

Citation
--------
If you found Ace-TN useful for your research, please cite this work with

.. toctree::
   :maxdepth: 1
   :hidden:

   source/user_guide
   source/modules
