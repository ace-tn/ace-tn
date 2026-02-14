import time
import torch
from torch import einsum, sqrt
from torch.linalg import svd
from ..evolution.tensor_update import TensorUpdater


class SimpleUpdater(TensorUpdater):
    """
    Implements the simple update algorithm for iPEPS (Jiang et al. 2008).

    Site tensors are stored directly in the iPEPS (no explicit gamma/lambda
    decomposition). Lambda tensors on each bond serve as mean-field
    approximations of the environment. During each bond update, surrounding
    (non-connecting) lambdas are absorbed before the QR/gate/SVD step and
    stripped afterward. The bond lambda is computed from the SVD singular
    values and distributed equally into both sides.

    Invariant: after each update, the site tensor carries lambda on each
    leg that has been updated, consistent with the absorb/strip cycle.
    """

    LAMBDA_CUTOFF = 1e-12

    def __init__(self, ipeps, gate, config):
        super().__init__(ipeps, gate)
        self.config = config
        self.bD = ipeps.dims['bond']

        # Map each (site, direction) -> bond_key for lambda lookup
        # direction: 0=left, 1=up, 2=right, 3=down
        self.site_bonds = {}
        for site in ipeps.site_list:
            self.site_bonds[tuple(site)] = {}

        for bond in ipeps.bond_list:
            s1, s2, k = bond
            s1t, s2t = tuple(s1), tuple(s2)
            key = (s1t, s2t)
            self.site_bonds[s1t][k] = key
            self.site_bonds[s2t][(k + 2) % 4] = key

        # Initialize all lambda tensors to ones (trivial environment)
        self.lambda_tensors = {}
        for bond in ipeps.bond_list:
            s1, s2, k = bond
            key = (tuple(s1), tuple(s2))
            self.lambda_tensors[key] = torch.ones(
                self.bD, dtype=ipeps.dtype, device=ipeps.device
            )

    def _get_lambda(self, site, direction):
        """Get the lambda tensor for a site along a given direction."""
        return self.lambda_tensors[self.site_bonds[tuple(site)][direction]]

    def _absorb_surrounding(self, a, site, exclude_dir):
        """Absorb lambda on all legs EXCEPT the connecting leg."""
        site = tuple(site)
        weights = []
        for d in range(4):
            if d == exclude_dir:
                weights.append(torch.ones_like(self._get_lambda(site, d)))
            else:
                weights.append(self._get_lambda(site, d))
        return einsum("l,u,r,d,lurdp->lurdp",
                       weights[0], weights[1], weights[2], weights[3], a)

    def _strip_surrounding(self, a, site, exclude_dir):
        """Strip lambda from all legs EXCEPT the connecting leg."""
        site = tuple(site)
        weights = []
        for d in range(4):
            if d == exclude_dir:
                weights.append(torch.ones_like(self._get_lambda(site, d)))
            else:
                lam = self._get_lambda(site, d).clamp(min=self.LAMBDA_CUTOFF)
                weights.append(1.0 / lam)
        return einsum("l,u,r,d,lurdp->lurdp",
                       weights[0], weights[1], weights[2], weights[3], a)

    def tensor_update(self, a1, a2, bond):
        """Not used -- simple update overrides bond_update directly."""
        raise NotImplementedError("SimpleUpdater overrides bond_update")

    def bond_update(self, bond):
        """
        Performs a single simple update step for a bond.

        1. Absorb surrounding lambdas (NOT the connecting leg) in the original frame
        2. Permute to rotated frame, QR decompose
        3. Contract reduced tensors and apply gate
        4. SVD truncate; compute lambda_c = sqrt(S / ||S||)
        5. Multiply lambda_c into U and Vh
        6. Recompose and un-permute
        7. Strip surrounding lambdas in the original frame
        8. Apply one-site gate if needed
        9. Update bond lambda and write Tn back to iPEPS

        Returns:
            tuple: (ctm_time, upd_time) where ctm_time is always 0.
        """
        s1, s2, k = bond
        s1t, s2t = tuple(s1), tuple(s2)
        start = time.time()

        # Get site tensors
        a1 = self.ipeps[s1]['A'].clone()
        a2 = self.ipeps[s2]['A'].clone()

        # Absorb surrounding lambdas (not the connecting leg)
        a1 = self._absorb_surrounding(a1, s1t, exclude_dir=k)
        a2 = self._absorb_surrounding(a2, s2t, exclude_dir=(k + 2) % 4)

        # Permute to rotated frame
        a1, a2 = self.permute_bond_tensors(a1, a2, k)

        # QR decompose
        a1q, a1r, a2q, a2r = self.decompose_site_tensors(a1, a2)

        # Contract reduced tensors and apply gate
        nD, bD, pD = a1r.shape
        a12 = einsum("yup,xuq->ypxq", a1r, a2r)
        gate = self.gate[bond]
        a12g = einsum("ypxq,pqrs->yrxs", a12, gate)

        # SVD and truncate
        a12g = a12g.reshape(nD * pD, nD * pD)
        U, S, Vh = svd(a12g)
        S = S[:bD]

        # Compute lambda_c = sqrt(S / ||S||_2)
        s_norm = S.norm()
        lambda_c = sqrt(S / s_norm)

        # Multiply lambda_c into U and Vh
        U = U[:, :bD].reshape(nD, pD, bD)
        Vh = Vh[:bD, :].reshape(bD, nD, pD)
        a1r = einsum("ypa,a->yap", U, lambda_c)
        a2r = einsum("axq,a->xaq", Vh, lambda_c)

        # Recompose full tensors and un-permute
        a1, a2 = self.recompose_site_tensors(a1q, a1r, a2q, a2r)
        a1, a2 = self.permute_bond_tensors(a1, a2, 4 - k)

        # Strip surrounding lambdas (same legs that were absorbed)
        a1 = self._strip_surrounding(a1, s1t, exclude_dir=k)
        a2 = self._strip_surrounding(a2, s2t, exclude_dir=(k + 2) % 4)

        # Apply one-site gate if needed
        if not self.gate.wrap_one_site:
            a1 = einsum("pq,lurdp->lurdq", self.gate[s1], a1)
            a2 = einsum("pq,lurdp->lurdq", self.gate[s2], a2)

        # Update bond lambda and write tensors back to iPEPS
        bond_key = self.site_bonds[s1t][k]
        self.lambda_tensors[bond_key] = lambda_c
        self.ipeps[s1]['A'] = a1
        self.ipeps[s2]['A'] = a2

        upd_time = time.time() - start
        return 0, upd_time
