Simple Update
=============

The simple update (SU) provides a computationally efficient alternative to the full update for
performing imaginary-time evolution on the iPEPS [Jiang2008]_. It replaces the full CTMRG
environment with a mean-field approximation encoded in diagonal *lambda tensors* living on the
bonds of the tensor network.


Gamma--Lambda Decomposition
---------------------------

In the simple update, each site tensor is decomposed into a *gamma* tensor :math:`\Gamma_i`
(the on-site component) and diagonal *lambda* tensors :math:`\lambda^{(b)}` on each bond
:math:`b` adjacent to site :math:`i`. This decomposition is equivalent to the Vidal canonical
form [Vidal2007]_:

.. math::

   A_i = \Gamma_i \prod_{b \sim i} \sqrt{\lambda^{(b)}}

The lambda tensors approximate the Schmidt-like weights of the bonds. The full network of
site tensors and lambda tensors looks like:

.. image:: png/su_lambda_network.png
   :align: center
   :width: 450px


Bond Update Procedure
---------------------

Each bond update follows these steps:

**1. Absorb surrounding lambdas**

Starting from the two sites :math:`A_i, A_j` connected by bond :math:`(i,j)`, absorb
(multiply in) the square roots of the lambda tensors on all *surrounding* bonds (those not
being updated). The lambda on the connecting bond :math:`(i,j)` itself is not absorbed:

.. image:: png/su_absorb_before.png
   :align: center
   :width: 450px

After absorption, the result is a pair of tensors with bare external legs:

.. image:: png/su_absorb_after.png
   :align: center
   :width: 400px

**2. QR decomposition**

Each absorbed tensor is factorised using a QR decomposition to extract a smaller reduced
tensor:

.. math::

   A_i \to A^Q_i \; a^R_i, \quad A_j \to A^Q_j \; a^R_j

**3. Gate application**

The two-body gate :math:`g_{ij} = e^{-\delta\tau\, h_{ij}}` is applied to the reduced
tensors through their physical indices:

.. math::

   \Theta = g_{ij} \; a^R_i \; a^R_j

.. image:: png/su_gate_svd.png
   :align: center
   :width: 350px

**4. SVD and truncation**

The resulting tensor :math:`\Theta` is decomposed via SVD:

.. math::

   \Theta = U \, \Sigma \, V^\dagger

.. image:: png/su_svd_result.png
   :align: center
   :width: 400px

The singular-value matrix :math:`\Sigma` is truncated to the :math:`D` largest values and
normalised. The updated lambda tensor on the bond is:

.. math::

   \lambda'^{(i,j)} = \frac{\Sigma_{\le D}}{\| \Sigma_{\le D} \|}

**5. Reassemble site tensors**

The truncated :math:`U` and :math:`V^\dagger` are contracted back with the isometric parts:

.. math::

   a'^R_i = U_{\le D}, \quad a'^R_j = V^\dagger_{\le D}

.. math::

   A'_i = A^Q_i \; a'^R_i, \quad A'_j = A^Q_j \; a'^R_j

**6. Strip surrounding lambdas**

Finally, the surrounding lambda tensors that were absorbed in step 1 are stripped
(divided out) from the updated site tensors, restoring the Gamma--Lambda decomposition.


Measurements
------------

After the simple update evolution converges, measurements of expectation values require a
converged CTMRG environment. The full site tensors :math:`A_i` (with lambdas absorbed) are
used for this purpose. The simple update only accelerates the imaginary-time evolution; the
measurement step is identical to that used in the full update.


References
----------

.. [Jiang2008] H. C. Jiang, Z. Y. Weng, and T. Xiang, *Accurate determination of tensor network state of quantum lattice models in two dimensions*, Phys. Rev. Lett. **101**, 090603 (2008).

.. [Vidal2007] G. Vidal, *Classical simulation of infinite-size quantum lattice systems in one spatial dimension*, Phys. Rev. Lett. **98**, 070201 (2007).
