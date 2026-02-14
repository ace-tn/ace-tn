import time
from torch import einsum, sqrt
from torch.linalg import svd
from ..evolution.tensor_update import TensorUpdater
from ..evolution.lambda_tensors import LambdaTensors


class SimpleUpdater(TensorUpdater):
    """Simple update algorithm for iPEPS (Jiang et al. 2008).

    Lambda tensors on each bond approximate the environment as a mean field.
    Each bond update absorbs surrounding lambdas, applies the gate via
    QR/SVD, then strips the lambdas back. The bond lambda is recomputed
    from the SVD singular values.
    """

    def __init__(self, ipeps, gate, config):
        super().__init__(ipeps, gate)
        self.config = config
        self.bD = ipeps.dims['bond']
        self.lambda_tensors = LambdaTensors(
            ipeps.bond_list, self.bD, ipeps.dtype, ipeps.device
        )

    def tensor_update(self, a1, a2, bond):
        raise NotImplementedError("SimpleUpdater overrides bond_update")

    def bond_update(self, bond):
        """Perform a single simple update step for a bond.

        Returns:
            tuple: (ctm_time, upd_time) where ctm_time is always 0.
        """
        start = time.time()

        a1 = self.ipeps[bond.s1]['A'].clone()
        a2 = self.ipeps[bond.s2]['A'].clone()

        # Absorb surrounding lambdas (not the bond leg)
        a1 = self.lambda_tensors.absorb(a1, bond.s1, exclude_dir=bond.k)
        a2 = self.lambda_tensors.absorb(a2, bond.s2, exclude_dir=(bond.k + 2) % 4)

        a1, a2 = self.permute_bond_tensors(a1, a2, bond.k)
        a1q, a1r, a2q, a2r = self.decompose_site_tensors(a1, a2)

        nD, bD, pD = a1r.shape
        a12 = einsum("yup,xuq->ypxq", a1r, a2r)
        a12g = einsum("ypxq,pqrs->yrxs", a12, self.gate[bond])

        # SVD, truncate, and compute new bond lambda tensor
        U, S, Vh = svd(a12g.reshape(nD * pD, nD * pD))
        S = S[:bD]
        lambda_tensor = sqrt(S / S.norm())

        U = U[:, :bD].reshape(nD, pD, bD)
        Vh = Vh[:bD, :].reshape(bD, nD, pD)
        a1r = einsum("ypa,a->yap", U, lambda_tensor)
        a2r = einsum("axq,a->xaq", Vh, lambda_tensor)

        a1, a2 = self.recompose_site_tensors(a1q, a1r, a2q, a2r)
        a1, a2 = self.permute_bond_tensors(a1, a2, 4 - bond.k)

        # Strip the same surrounding lambdas
        a1 = self.lambda_tensors.strip(a1, bond.s1, exclude_dir=bond.k)
        a2 = self.lambda_tensors.strip(a2, bond.s2, exclude_dir=(bond.k + 2) % 4)

        if not self.gate.wrap_one_site:
            a1 = einsum("pq,lurdp->lurdq", self.gate[bond.s1], a1)
            a2 = einsum("pq,lurdp->lurdq", self.gate[bond.s2], a2)

        self.lambda_tensors[bond] = lambda_tensor
        self.ipeps[bond.s1]['A'] = a1
        self.ipeps[bond.s2]['A'] = a2

        upd_time = time.time() - start
        return 0, upd_time
