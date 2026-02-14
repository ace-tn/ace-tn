from ..utils.benchmarking import record_runtime
from abc import ABC, abstractmethod
from torch import einsum
from torch.linalg import qr

class TensorUpdater(ABC):
    """Abstract base class for iPEPS tensor updates."""

    def __init__(self, ipeps, gate):
        self.ipeps = ipeps
        self.dims = ipeps.dims
        self.gate = gate

    @abstractmethod
    def tensor_update(self):
        pass

    @record_runtime
    def update(self, bond):
        """Retrieve site tensors, permute into the bond frame, apply the
        tensor update, permute back, and optionally apply one-site gates.

        Args:
            bond: The bond to update.

        Returns:
            tuple: Updated site tensors (a1, a2).
        """
        s1,s2,k = bond
        a1 = self.ipeps[s1]['A']
        a2 = self.ipeps[s2]['A']
        a1,a2 = self.permute_bond_tensors(a1, a2, k)
        a1,a2 = self.tensor_update(a1, a2, bond)
        a1,a2 = self.permute_bond_tensors(a1, a2, 4-k)
        if not self.gate.wrap_one_site:
            a1 = einsum("pq,lurdp->lurdq", self.gate[bond.s1], a1)
            a2 = einsum("pq,lurdp->lurdq", self.gate[bond.s2], a2)
        return a1,a2

    def permute_bond_tensors(self, a1, a2, k):
        """Cyclically permute tensor legs so the bond direction aligns with a
        standard orientation."""
        a1 = a1.permute(self.bond_permutation(k))
        a2 = a2.permute(self.bond_permutation(k))
        return a1,a2

    @staticmethod
    def bond_permutation(k):
        """Cyclic index permutation for bond direction ``k``."""
        return [(i+k)%4 for i in range(4)] + [4,]

    @staticmethod
    def decompose_site_tensors(a1, a2):
        """QR-decompose site tensors into environment (Q) and bond-local (R) parts."""
        bD,pD = a1.shape[3:]
        nD = min(bD**3, pD*bD)

        a1_tmp = einsum("lurdp->rdulp", a1).reshape(bD**3, pD*bD)
        a1q,a1r = qr(a1_tmp)
        a1q = a1q.reshape(bD, bD, bD, nD)
        a1r = a1r.reshape(nD, bD, pD)

        a2_tmp = einsum("lurdp->dlurp", a2).reshape(bD**3, pD*bD)
        a2q,a2r = qr(a2_tmp)
        a2q = a2q.reshape(bD, bD, bD, nD)
        a2r = a2r.reshape(nD, bD, pD)

        return a1q, a1r, a2q, a2r

    @staticmethod
    def recompose_site_tensors(a1q, a1r, a2q, a2r):
        """Reconstruct full site tensors from QR components."""
        a1 = einsum('rdux,xlp->lurdp', a1q, a1r)
        a2 = einsum('dlux,xrp->lurdp', a2q, a2r)
        return a1,a2

    def finalize(self):
        """Called after the evolution loop. Override in subclasses if needed."""
        pass
