import torch
from torch import einsum


class LambdaTensors:
    """Mean-field environment weights on bonds for the simple update.

    Stores a diagonal lambda vector for each bond and provides methods to
    absorb and strip these weights into site tensors.
    """

    CUTOFF = 1e-12

    def __init__(self, bond_list, bD, dtype, device):
        self._data = {}
        self._site_bonds = {}

        for bond in bond_list:
            self._data[bond] = torch.ones(bD, dtype=dtype, device=device)
            self._site_bonds.setdefault(bond.s1, {})[bond.k] = bond
            self._site_bonds.setdefault(bond.s2, {})[(bond.k + 2) % 4] = bond

    def __getitem__(self, bond):
        return self._data[bond]

    def __setitem__(self, bond, value):
        self._data[bond] = value

    def get(self, site, direction):
        """Get the lambda vector for a site along a direction (0=l, 1=u, 2=r, 3=d)."""
        return self._data[self._site_bonds[site][direction]]

    def absorb(self, a, site, exclude_dir):
        """Multiply lambda into all legs of a site tensor except ``exclude_dir``."""
        weights = [
            self.get(site, d) if d != exclude_dir
            else torch.ones_like(self.get(site, d))
            for d in range(4)
        ]
        return einsum("l,u,r,d,lurdp->lurdp", *weights, a)

    def strip(self, a, site, exclude_dir):
        """Divide out lambda from all legs of a site tensor except ``exclude_dir``."""
        weights = [
            1.0 / self.get(site, d).clamp(min=self.CUTOFF) if d != exclude_dir
            else torch.ones_like(self.get(site, d))
            for d in range(4)
        ]
        return einsum("l,u,r,d,lurdp->lurdp", *weights, a)
