Full Update
===========

The full update (FU) finds the ground state of a local Hamiltonian by performing
imaginary-time evolution on the iPEPS, using the full CTMRG environment to optimise
the updated tensors [Jordan2008]_, [Corboz2016]_.


Imaginary-Time Evolution
------------------------

A two-body gate :math:`g_{ij} = e^{-\delta\tau\, h_{ij}}` (with time step :math:`\delta\tau`)
is applied to each nearest-neighbour pair. Starting from two neighbouring site tensors
:math:`A_i` and :math:`A_j`:

.. image:: png/fu_two_sites.png
   :align: center
   :width: 400px


QR Decomposition
----------------

Each site tensor is factorised with a QR decomposition:

.. math::

   A_i = A^Q_i \; a^R_i, \qquad A_j = A^Q_j \; a^R_j

This splits each site tensor into an isometric part :math:`A^Q` and a smaller reduced tensor
:math:`a^R`. The resulting decomposition of the two-site system is:

.. image:: png/fu_qr_decomposition.png
   :align: center
   :width: 550px


Gate Application
----------------

The gate :math:`g_{ij}` is contracted with the reduced tensors :math:`a^R_i` and
:math:`a^R_j` through their physical indices:

.. math::

   \Theta_{s'_i s'_j} = \sum_{s_i s_j} g^{s'_i s'_j}_{s_i s_j} \; a^R_i(s_i) \; a^R_j(s_j)

.. image:: png/fu_gate_application.png
   :align: center
   :width: 350px


Norm Tensor Construction
------------------------

The gate-updated tensor :math:`\Theta` has an enlarged bond between sites :math:`i` and
:math:`j` that must be compressed back to bond dimension :math:`D`. In the full update this
compression is performed with respect to the full CTMRG environment. The first step is to
construct the **norm tensor** :math:`N_{ij}`.

:math:`N_{ij}` is formed by contracting the converged boundary tensors
(:math:`C^k`, :math:`E^k`) with the isometric parts :math:`A^Q_i` and :math:`A^Q_j` (and
their conjugates) over all indices except the shared bond between sites :math:`i` and
:math:`j`:

.. image:: png/fu_norm_tensor.png
   :align: center
   :width: 550px

The resulting tensor :math:`N_{ij}` is rank-4 with indices :math:`(y, x, Y, X)`, where the
lowercase indices correspond to the ket layer and the uppercase indices to the bra layer.
It encodes how the environment weights the bond between the two sites.

**Positive approximation.** Because the environment tensors are approximate, :math:`N_{ij}`
may not be exactly positive semi-definite. When reshaped to a matrix
:math:`N \in \mathbb{R}^{D^2 \times D^2}`, its eigendecomposition :math:`N = Z \Lambda Z^\dagger`
may contain small negative eigenvalues. These are regularised by shifting the spectrum:

.. math::

   N \;\leftarrow\; N + 2 \max(\epsilon,\, |\lambda_{\min}|)\, \mathbb{1}

where :math:`\lambda_{\min}` is the smallest eigenvalue. The positive-definite square root
:math:`\tilde{N}` (satisfying :math:`\tilde{N}^\dagger \tilde{N} = N`) is then computed from
the corrected eigenvalues.

**Gauge fixing.** The conditioning of the subsequent ALS can be improved by a gauge
transformation [Phien2015]_. QR decompositions of :math:`\tilde{N}` along the :math:`y` and
:math:`x` legs yield triangular factors :math:`R_y` and :math:`R_x`. Their pseudo-inverses
are applied to transform :math:`\tilde{N}` into a better-conditioned form:

.. math::

   \tilde{N}_{zvw} \;\leftarrow\; R_x^{-1} \; \tilde{N} \; R_y^{-1}

and the gate-updated tensor :math:`\Theta` is correspondingly transformed using
:math:`R_x` and :math:`R_y`. After the ALS solve, the inverse transformations are applied to
recover the original gauge.


ALS Solution
------------

With the norm tensor :math:`N_{ij}` (and optionally gauge-fixed :math:`\Theta`) in hand, the
compression is cast as a least-squares problem:

.. math::

   \min_{a^R_i,\, a^R_j} \left\| \Theta - a^R_i \, a^R_j \right\|_{\mathcal{N}}^2

where the norm is defined by :math:`N_{ij}`. Expanding the squared norm gives a cost function:

.. math::

   \mathcal{C} = \langle a^R_i a^R_j | N_{ij} | a^R_i a^R_j \rangle
               - 2\,\mathrm{Re}\, \langle a^R_i a^R_j | N_{ij} | \Theta \rangle

This is solved by **alternating least squares** (ALS): fix one reduced tensor and solve a
linear system for the other, then alternate.

**Fixing** :math:`a^R_j` **and solving for** :math:`a^R_i`. Contracting :math:`N_{ij}` with
:math:`a^R_j` and :math:`\overline{a^R_j}` over the indices belonging to site :math:`j`
produces the **effective norm** :math:`R_i`. Contracting :math:`N_{ij}\,\Theta` with
:math:`\overline{a^R_j}` in the same way produces the **source tensor** :math:`S_i`.
The optimal :math:`a^R_i` then satisfies the linear system:

.. math::

   R_i \; a^R_i = S_i

The same procedure applies when fixing :math:`a^R_i` and solving for :math:`a^R_j`.

**Regularisation.** The effective norm :math:`R_i` is symmetrised and regularised before
solving:

.. math::

   R_i \;\to\; \tfrac{1}{2}(R_i + R_i^\dagger) + \epsilon \, \mathbb{1}

The system is then solved via Cholesky decomposition when :math:`R_i` is well-conditioned,
or via the pseudo-inverse otherwise.

**Convergence.** The ALS iteration alternates between solving for :math:`a^R_i` and
:math:`a^R_j` until the relative change in the cost function falls below a tolerance
:math:`\delta`:

.. math::

   \frac{|\mathcal{C}^{(n)} - \mathcal{C}^{(n-1)}|}{|\mathcal{C}^{(n-1)}|} < \delta


Reassembly
----------

After ALS convergence, a final QR-SVD step balances the bond distribution between the two
reduced tensors. Each tensor is QR-decomposed along its bond leg, and an SVD of the product
of the two R-factors determines the optimal split. The full site tensors are then reassembled:

.. math::

   A'_i = A^Q_i \; a'^R_i, \qquad A'_j = A^Q_j \; a'^R_j


Fast Full Update
----------------

The fast full update (FFU) variant [Phien2015]_ exploits the QR structure to reduce the size
of the effective norm tensor. Instead of forming :math:`N_{ij}` with the full environment and
then contracting with :math:`A^Q`, the isometric parts are absorbed into the boundary tensors
first, yielding a smaller effective norm whose dimensions depend on the QR bond dimensions
rather than the environment bond dimension :math:`\chi`. This makes each ALS iteration
substantially faster while maintaining the same accuracy as the standard full update.


References
----------

.. [Corboz2016] P. Corboz, *Variational optimization with infinite projected entangled-pair states*, Phys. Rev. B **94**, 035133 (2016).

.. [Phien2015] H. N. Phien, J. A. Bengua, H. D. Tuan, P. Corboz, and R. Orús, *Infinite projected entangled pair states algorithm improved: Fast full update and gauge fixing*, Phys. Rev. B **92**, 035142 (2015).
