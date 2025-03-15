import torch
from torch import einsum, conj
from torch.linalg import qr,eigh,svd
from evolution.tensor_update import TensorUpdater
from evolution.als_solver import ALSSolver

class FullUpdater(TensorUpdater):
    """
    A concrete subclass of TensorUpdater that implements the full tensor update algorithm. 
    This class performs tensor decompositions, updates the reduced tensors, applies a gate 
    operation, and recomposes the tensors.
    """

    def __init__(self, ipeps, gate, config):
        """
        Initializes the FullUpdater with the provided iPEPS tensor network and gate.

        Parameters:
        -----------
        ipeps : object
            An instance of the iPEPS tensor network, containing the tensor data for the update process.

        gate : object
            The gate operation or transformation that will be applied during the tensor update.
        """
        super().__init__(ipeps, gate)
        self.config                 = config
        self.use_gauge_fix          = config.use_gauge_fix
        self.gauge_fix_atol         = config.gauge_fix_atol
        self.positive_approx_cutoff = config.positive_approx_cutoff

    def tensor_update(self, a1, a2, bond):
        """
        Performs the full tensor update process for the given bond. This involves tensor 
        decomposition, norm tensor computation, reduced tensor update, and tensor recomposition.

        Parameters:
        -----------
        a1 : Tensor
            The first tensor at the site being updated.

        a2 : Tensor
            The second tensor at the site being updated.

        bond : tuple
            The bond that connects the two tensors (indices and position).

        Returns:
        --------
        a1, a2 : Tensor
            The updated tensors after performing the tensor update operation.
        """
        pD = self.dims["phys"]
        bD = self.dims["bond"]

        a1q,a1r,a2q,a2r,nD = self.decompose_site_tensors(a1, a2, bD, pD)
        n12 = build_norm_tensor(self.ipeps, bond, a1q, a2q)

        gate = self.gate[bond]
        a1r,a2r = self.update_reduced_tensors(a1r, a2r, n12, gate, pD, bD, nD)

        a1,a2 = self.recompose_site_tensors(a1q,a1r,a2q,a2r)

        return a1,a2

    def update_reduced_tensors(self, a1r, a2r, n12, gate, pD, bD, nD):
        """
        Updates the reduced tensors by applying the gate operation and norm tensor, 
        and optionally applying gauge fixing.

        Parameters:
        -----------
        a1r : Tensor
            The reduced tensor for the first site.

        a2r : Tensor
            The reduced tensor for the second site.

        n12 : Tensor
            The norm tensor that represents the bond interaction between the two sites.

        gate : Tensor
            The gate operation to be applied during the update.

        pD : int
            The physical dimension of the tensors.

        bD : int
            The bond dimension of the tensors.

        nD : int
            The dimension of the norm tensor.

        Returns:
        --------
        a1r, a2r : Tensor
            The updated reduced tensors for the first and second sites.
        """
        a12g = einsum("yup,xuq->ypxq", a1r, a2r)
        a12g = einsum("ypxq,pqrs->yxrs", a12g, gate)

        nz = positive_approx(n12, nD, cutoff=self.positive_approx_cutoff)
        if self.use_gauge_fix:
            n12, a12g, nzxr_inv, nzyr_inv = gauge_fix(nz, a12g, nD, atol=self.gauge_fix_atol)
        else:
            n12 = einsum("xyz,XYz->xyXY", nz, conj(nz))

        als_solver = ALSSolver(n12, a12g, bD, pD, nD, self.config)
        a1r,a2r = als_solver.solve()

        if self.use_gauge_fix:
            a1r = einsum("yz,zup->yup", nzyr_inv, a1r)
            a2r = einsum("xw,wvq->xvq", nzxr_inv, a2r)

        return a1r/a1r.norm(), a2r/a2r.norm()

    @staticmethod
    def decompose_site_tensors(a1, a2, bD, pD):
        """
        Decomposes the site tensors `a1` and `a2` into their core and reduced parts using QR decomposition.

        Parameters:
        -----------
        a1 : Tensor
            The first tensor at the site being decomposed.

        a2 : Tensor
            The second tensor at the site being decomposed.

        bD : int
            The bond dimension of the tensors.

        pD : int
            The physical dimension of the tensors.

        Returns:
        --------
        a1q : Tensor
            The core part of the first tensor after decomposition.

        a1r : Tensor
            The reduced part of the first tensor after decomposition.

        a2q : Tensor
            The core part of the second tensor after decomposition.

        a2r : Tensor
            The reduced part of the second tensor after decomposition.

        nD : int
            The dimension of the norm tensor
        """
        nD = min(bD**3, pD*bD)

        a1_tmp = einsum("lurdp->rdulp", a1).reshape(bD**3, pD*bD)
        a1q,a1r = qr(a1_tmp)
        a1q = a1q.reshape(bD, bD, bD, nD)
        a1r = a1r.reshape(nD, bD, pD)

        a2_tmp = einsum("lurdp->dlurp", a2).reshape(bD**3, pD*bD)
        a2q,a2r = qr(a2_tmp)
        a2q = a2q.reshape(bD, bD, bD, nD)
        a2r = a2r.reshape(nD, bD, pD)

        return a1q, a1r, a2q, a2r, nD

    @staticmethod
    def recompose_site_tensors(a1q, a1r, a2q, a2r):
        """
        Reconstructs the full site tensors from the decomposed core and reduced components.

        Parameters:
        -----------
        a1q : Tensor
            The core part of the first tensor.

        a1r : Tensor
            The reduced part of the first tensor.

        a2q : Tensor
            The core part of the second tensor.

        a2r : Tensor
            The reduced part of the second tensor.

        Returns:
        --------
        a1 : Tensor
            The reconstructed tensor for the first site.

        a2 : Tensor
            The reconstructed tensor for the second site.
        """
        a1 = torch.einsum('rdux,xlp->lurdp', a1q, a1r)
        a2 = torch.einsum('dlux,xrp->lurdp', a2q, a2r)
        return a1,a2

def build_norm_tensor(ipeps, bond, a1q, a2q):
    """
    Builds the norm tensor for a given bond in the iPEPS network. The norm tensor is a combination 
    of tensors from the iPEPS network and the decomposed tensors for the two sites. This tensor 
    is used to calculate the bond interactions and tensor updates in the iPEPS algorithm.

    Parameters:
    -----------
    ipeps : object
        An instance of the iPEPS tensor network, which contains the tensors that define the network.

    bond : tuple
        A tuple representing the bond that connects two tensors in the network. Typically, this consists 
        of two site indices and the bond index (s1, s2, k).

    a1q : Tensor
        The core part of the first site tensor after decomposition.

    a2q : Tensor
        The core part of the second site tensor after decomposition.

    Returns:
    --------
    n12 : Tensor
        The resulting norm tensor, which represents the interaction between the two tensors at the given bond.
    """
    s1,s2,k = bond

    # build right half
    c12 = ipeps[s1]['C'][(k+1)%4]
    e12 = ipeps[s1]['E'][(k+1)%4]
    e11 = ipeps[s1]['E'][(k+0)%4]
    c13 = ipeps[s1]['C'][(k+2)%4]
    e13 = ipeps[s1]['E'][(k+2)%4]

    tmp = einsum("ab,bcrR->acrR", c12, e12)
    tmp = einsum("acrR,eauU->crReuU", tmp, e11)
    tmp = einsum("crReuU,RDUY->creuDY", tmp, conj(a1q))
    tmp = einsum("creuDY,rduy->ceDYdy", tmp, a1q)
    n1_tmp = einsum("ab,bfdD->afdD", c13, e13)
    n1_tmp = einsum("afdD,aeDYdy->feYy", n1_tmp, tmp)

    # build left half
    c21 = ipeps[s2]['C'][(k+0)%4]
    e21 = ipeps[s2]['E'][(k+0)%4]
    e24 = ipeps[s2]['E'][(k+3)%4]
    c24 = ipeps[s2]['C'][(k+3)%4]
    e23 = ipeps[s2]['E'][(k+2)%4]

    tmp = einsum("ab,bcuU->acuU", c21, e21)
    tmp = einsum("acuU,ealL->cuUelL", tmp, e24)
    tmp = einsum("cuUelL,DLUX->cuelXD", tmp, conj(a2q))
    tmp = einsum("cuelXD,dlux->ceXDxd", tmp, a2q)
    n2_tmp = einsum("ab,fadD->bfdD", c24, e23)
    n2_tmp = einsum("bfdD,cbXDxd->fcXx", n2_tmp, tmp)

    # contract right-left
    n12 = einsum("fcYy,fcXx->yxYX", n1_tmp, n2_tmp)
    return n12

def positive_approx(n12, nD, cutoff=1e-12):
    """
    Computes a positive approximation of the norm tensor by adjusting its singular values. This process 
    helps to ensure that the norm tensor has non-negative eigenvalues while minimizing numerical errors.

    Parameters:
    -----------
    n12 : Tensor
        The norm tensor to be approximated.

    nD : int
        The dimension of the norm tensor

    cutoff : float, optional (default: 1e-12)
        A threshold value for determining which singular values to retain in the approximation.

    Returns:
    --------
    nz : Tensor
        The updated norm tensor after applying the positive approximation.
    """
    n12 = 0.5*(n12 + einsum("yxYX->YXyx", conj(n12)))
    nw,nz = eigh(n12.reshape(nD**2, nD**2))
    nw_max = torch.abs(nw[-1])
    nw = torch.where(nw / nw_max > cutoff, torch.sqrt(nw), 0.)
    nz = einsum("yxz,z->yxz", nz.reshape(nD, nD, nD**2), nw)
    return nz

def gauge_fix(nz, a12g, nD, atol=1e-12):
    """
    Applies a gauge fixing procedure to the norm tensor and the reduced tensors. This involves 
    performing a Singular Value Decomposition (SVD) on parts of the norm tensor and its inverse 
    to correct for gauge freedom in the tensor network. It also updates the gauge tensor `a12g` 
    accordingly.

    Parameters:
    -----------
    nz : Tensor
        The positive-approximation norm tensor used to adjust the values during gauge fixing.

    a12g : Tensor
        The reduced tensor representing the bond interaction that will be updated during the gauge fixing.

    bD : int
        The bond dimension of the tensors involved.

    pD : int
        The physical dimension of the tensors involved.

    nD : int
        The dimension of the norm tensor

    cutoff : float, optional (default: 1e-12)
        A threshold value for determining which singular values to retain during the SVD.

    Returns:
    --------
    n12 : Tensor
        The updated norm tensor after applying gauge fixing.

    a12g : Tensor
        The updated reduced tensor after applying gauge fixing.

    nzxr_inv : Tensor
        The inverse of the first part of the norm tensor after gauge fixing.

    nzyr_inv : Tensor
        The inverse of the second part of the norm tensor after gauge fixing.
    """
    _,nzyr = qr(einsum("yxz->zxy", nz).reshape(nD**3, nD))
    _,nzxr = qr(einsum("yxz->zyx", nz).reshape(nD**3, nD))

    nzyr_inv = torch.linalg.pinv(nzyr, atol=atol)
    nzxr_inv = torch.linalg.pinv(nzxr, atol=atol)

    nz = einsum("yxz,xw->yzw", nz, nzxr_inv)
    nz = einsum("yzw,yv->zvw", nz, nzyr_inv)
    n12 = einsum("zvw,zVW->vwVW", nz, conj(nz))
    n12 = 0.5*(n12 + einsum("vwVW->VWvw", conj(n12)))

    a12g = einsum("zx,yxpq->yzpq", nzxr, a12g)
    a12g = einsum("wy,yzpq->wzpq", nzyr, a12g)

    return n12, a12g, nzxr_inv, nzyr_inv
