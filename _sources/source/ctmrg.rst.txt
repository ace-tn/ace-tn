CTMRG
=====

The corner transfer matrix renormalization group (CTMRG) algorithm iteratively computes the
boundary tensors :math:`C^k` and :math:`E^k` that approximate the infinite extension of
the double-layer tensor network [Nishino1996]_, [Baxter1978]_, [Orus2009]_.


Directional Moves
-----------------

The boundary tensors are computed by iteratively absorbing rows or columns of the
double-layer tensor network. Each absorption is called a *directional move*. An up-move at
row :math:`y` inserts the :math:`y+1` row of double-layer tensors and absorbs it into the top
boundary. The following diagram shows the tensors involved in a single up-move:

.. image:: png/ctmrg_up_move.png
   :align: center
   :width: 450px

The top boundary tensors :math:`C^1, E^1, C^2` and the inserted row :math:`a'` are
contracted together, and projectors :math:`P_1, P_2` are applied to truncate the growing
boundary bonds back to dimension :math:`\chi`:

.. image:: png/ctmrg_absorption.png
   :align: center
   :width: 550px

The updated boundary tensors are:

.. math::

   C^{\prime 1}_{(x,y)} &= \mathrm{contract}\!\left(C^1_{(x,y+1)},\; E^4_{(x,y+1)},\; P^1_x\right) \\
   E^{\prime 1}_{(x,y)} &= \mathrm{contract}\!\left(P^2_x,\; E^1_{(x,y+1)},\; a_{(x,y+1)},\; P^1_{x-1}\right) \\
   C^{\prime 2}_{(x,y)} &= \mathrm{contract}\!\left(C^2_{(x,y+1)},\; E^2_{(x,y+1)},\; P^2_{x-1}\right)

Down, left, and right moves are defined analogously. For a unit cell of size
:math:`N_x \times N_y`, a complete absorption from one direction requires :math:`N_y` (or
:math:`N_x`) consecutive moves.


Projector Calculation
---------------------

Each absorption step increases the boundary bond dimension by a factor of :math:`D^2`. To
maintain a maximum boundary bond dimension :math:`\chi`, the updated boundary tensors must be
projected onto a lower-dimensional subspace. The projectors are computed from a truncated SVD
of the contracted environment [Orus2009]_, [Corboz2014]_.

**1. Build quarter tensors**

Four quarter tensors :math:`Q_k` for :math:`k = 1,2,3,4` are formed by contracting corner,
edge, and site tensors at each quadrant of the environment:

.. image:: png/ctmrg_quarter_tensors.png
   :align: center
   :width: 500px

**2. Form half-system tensors**

Contract pairs of quarter tensors vertically to form the left and right halves of the
environment:

.. math::

   R_1 = Q_1 \cdot Q_4, \quad R_2 = Q_2 \cdot Q_3

**3. SVD and truncation**

The full environment is formed by contracting :math:`R_1` and :math:`R_2` along the bonds
to be projected (indicated by the dashed line):

.. image:: png/ctmrg_projector.png
   :align: center
   :width: 450px

Decompose the contraction :math:`R_1 \cdot R_2` via SVD along the bonds to be projected:

.. math::

   R_1 \cdot R_2 = U \, \Sigma \, V^\dagger

Truncate to the largest :math:`\chi` singular values.

**4. Form projectors**

The projectors are constructed from the truncated singular vectors and values:

.. math::

   P_1 = R_1 \cdot U^\dagger \cdot \Sigma^{-1/2}, \quad
   P_2 = R_2 \cdot V \cdot \Sigma^{-1/2}

These projectors approximate a decomposition of the identity and are used in the boundary
tensor absorption contractions above. Alternatively, *half-system* projectors may be formed
from only one of :math:`R_1` or :math:`R_2`, discarding correlations in the other half of the
network [Corboz2014]_.


Convergence
-----------

CTMRG is iterated — performing directional moves in all four directions — until the boundary
tensors reach a fixed point with respect to absorbing a full unit cell. The converged boundary
tensors are then used for measurement of observables and, in the case of the full update, for
constructing the tensor environment used in the ALS optimisation.


References
----------

.. [Baxter1978] R. J. Baxter, *Variational approximations for square lattice models in statistical mechanics*, J. Stat. Phys. **19**, 461 (1978).

.. [Corboz2014] P. Corboz, T. M. Rice, and M. Troyer, *Competing states in the t-J model: Uniform d-wave state versus stripe state*, Phys. Rev. Lett. **113**, 046402 (2014).
