.. _input-settings:

Input Settings
==============

Ace-TN uses TOML configuration files to specify iPEPS simulation parameters. The configuration is passed to the ``Ipeps`` constructor after loading with the ``toml`` package:

.. code-block:: python

   from acetn.ipeps import Ipeps
   import toml

   config = toml.load("input.toml")
   ipeps = Ipeps(config)

Only the ``[TN]`` section is strictly required; all other sections are optional and will use defaults if omitted.

Configuration Template
----------------------

The following template includes all available options with their default values. Copy and adapt it for your simulation:

.. literalinclude:: input_template.toml
   :language: toml
   :caption: input_template.toml

Option Reference
----------------

Top-level
~~~~~~~~~

- **dtype** (str): Data type for tensors. Use ``"float32"`` or ``"float64"``. Default: ``"float64"``.
- **device** (str): Compute device. Use ``"cpu"`` or ``"cuda"``. Default: ``"cpu"``. Falls back to CPU if CUDA is requested but unavailable.

Tensor network ``[TN]``
~~~~~~~~~~~~~~~~~~~~~~~~~

- **nx** (int): Unit cell size in the x direction. Default: ``2``.
- **ny** (int): Unit cell size in the y direction. Default: ``2``.
- **TN.dims.phys** (int): Physical dimension (e.g. 2 for spin-1/2). Default: ``2``.
- **TN.dims.bond** (int): Bond dimension D of the iPEPS. Default: ``2``.
- **TN.dims.chi** (int): Maximum boundary bond dimension chi for corner transfer matrices. Default: ``20``.

CTMRG ``[ctmrg]``
~~~~~~~~~~~~~~~~~

- **steps** (int): Number of CTMRG iterations per renormalization. Default: ``40``.
- **projectors** (str): Projector type. ``"half-system"`` or ``"full-system"``. Default: ``"half-system"``.
- **svd_type** (str): SVD algorithm for boundary compression. ``"rsvd"`` (randomized) or ``"full-rank"``. Default: ``"rsvd"``.
- **svd_cutoff** (float): Singular value cutoff. Default: ``1e-12``.
- **rsvd_niter** (int): Power iterations for randomized SVD. Default: ``2``.
- **rsvd_oversampling** (int): Oversampling factor for randomized SVD. Default: ``2``.
- **disable_progressbar** (bool): Suppress progress bar during CTMRG. Default: ``false``. Automatically set in non-interactive SLURM jobs.

Evolution ``[evolution]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **backend** (str): Tensor contraction backend. ``"torch"`` or ``"cutensor"`` (requires GPU build with cuTENSOR). Default: ``"torch"``.
- **update_type** (str): Evolution update scheme. ``"full"`` or ``"simple"``. Default: ``"full"``.
- **use_gauge_fix** (bool): Apply gauge fix for numerical stability. Default: ``true``.
- **gauge_fix_atol** (float): Absolute tolerance for gauge fix. Default: ``1e-12``.
- **positive_approx_cutoff** (float): Cutoff for positive definiteness in tensor operations. Default: ``1e-12``.
- **als_niter** (int): Maximum iterations for the ALS solver (full update). Default: ``100``.
- **als_tol** (float): Convergence tolerance for the ALS solver. Default: ``1e-15``.
- **als_method** (str): Linear solver method. ``"cholesky"`` or ``"pinv"``. Default: ``"cholesky"``.
- **als_epsilon** (float): Regularization for the ALS solver. Default: ``1e-12``.
- **disable_progressbar** (bool): Suppress progress bar during evolution. Default: ``false``.

Model ``[model]``
~~~~~~~~~~~~~~~~~

- **name** (str): Model identifier. Built-in models: ``"heisenberg"``, ``"ising"``. Use ``"custom"`` when registering via :meth:`Ipeps.set_model() <acetn.ipeps.Ipeps.set_model>`.
- **model.params** (table): Model-specific parameters. Keys and values depend on the model.

  - **Heisenberg model**: ``J`` (exchange coupling, default 1.0)
  - **Ising model**: ``jz`` (nearest-neighbor coupling), ``hx`` (transverse field)

For custom models, the model and its parameters are set programmatically with :meth:`set_model() <acetn.ipeps.Ipeps.set_model>`; see :doc:`example_03_custom_model` for details.
