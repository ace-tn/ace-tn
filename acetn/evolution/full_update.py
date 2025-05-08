import torch
from math import ceil
from torch import einsum, conj
from torch.linalg import qr,eigh,svd,pinv
from ..evolution.tensor_update import TensorUpdater
from ..evolution.als_solver import ALSSolver

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
        self.backend                = config.backend
        self.setup_backend()

    def setup_backend(self):
        if self.backend == "cutensor":
            self.load_cutensor_library()
        elif self.backend != "torch":
            raise NotImplementedError(f"Backend {self.backend} not supported.")
        self.build_norm_tensor = torch_build_norm_tensor

    def load_cutensor_library(self):
        try:
            from pathlib import Path
            cutensor_lib_path = Path(__file__).resolve().parent / "../../csrc/evolution/als_contractions.so"
            cutensor_lib_path = cutensor_lib_path.resolve(strict=True)
            torch.ops.load_library(cutensor_lib_path)
            cusolver_lib_path = Path(__file__).resolve().parent / "../../csrc/evolution/cholesky_solve.so"
            cusolver_lib_path = cusolver_lib_path.resolve(strict=True)
            torch.ops.load_library(cusolver_lib_path)
        except:
            self.backend = "torch"

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
        a1q,a1r,a2q,a2r = self.decompose_site_tensors(a1, a2)
        n12 = self.build_norm_tensor(self.ipeps, bond, a1q, a2q)

        gate = self.gate[bond]
        a1r,a2r = self.update_reduced_tensors(a1r, a2r, n12, gate)

        a1,a2 = self.recompose_site_tensors(a1q,a1r,a2q,a2r)

        return a1/a1.norm(),a2/a2.norm()

    def update_reduced_tensors(self, a1r, a2r, n12, gate):
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

        Returns:
        --------
        a1r, a2r : Tensor
            The updated reduced tensors for the first and second sites.
        """
        a12g = einsum("yup,xuq->yxpq", a1r, a2r)
        a12g = einsum("yxpq,pqrs->yxrs", a12g, gate)

        n12, a12g = self.precondition_norm_tensor(n12, a12g)
        a1r,a2r = ALSSolver(n12, a12g, a1r.shape, self.config).solve()
        a1r,a2r = self.finalize_reduced_tensors(a1r, a2r)
        return a1r,a2r

    def precondition_norm_tensor(self, n12, a12g):
        """
        Applies positive approximation and (optionally) gauge fixing to the norm tensor,
        improving conditioning for ALS. The gate-tensor product is also modified when
        gauge fixing is used.

        Parameters:
        -----------
        n12 : Tensor
            The norm tensor.

        a12g : Tensor
            The contracted a1r-a2r-gate.

        Returns:
        --------
        n12, a12g : Tensor
            The updated tensors.
        """
        nz = positive_approx(n12, cutoff=self.positive_approx_cutoff)
        if self.use_gauge_fix:
            n12, a12g, self.nzxr_inv, self.nzyr_inv = gauge_fix(nz, a12g, atol=self.gauge_fix_atol)
        else:
            n12 = einsum("xyz,XYz->xyXY", nz, conj(nz))
        return n12, a12g

    def finalize_reduced_tensors(self, a1r, a2r):
        """
        Finalizes the tensor update using the method shown in Fig.12(b) 
        of arxiv.org/abs/1405.3259.

        Parameters:
        -----------
        a1r : Tensor
            The reduced tensor for the first site.

        a2r : Tensor
            The reduced tensor for the second site.

        Returns:
        --------
        a1r, a2r : Tensor
            The finalized reduced tensors.
        """
        if self.use_gauge_fix:
            a1r = einsum("yz,zup->yup", self.nzyr_inv, a1r)
            a2r = einsum("xw,wvq->xvq", self.nzxr_inv, a2r)

        nD,bD,pD = a1r.shape
        q1,r1 = qr(einsum("yup->ypu", a1r).reshape(nD*pD,bD))
        q2,r2 = qr(einsum("xvq->xqv", a2r).reshape(nD*pD,bD))

        U,s,Vh = svd(einsum("au,bu->ab", r1, r2))
        s = torch.sqrt(s[:bD] / s.norm())

        r1 = einsum('ab,b->ab', U[:, :bD],  s)
        r2 = einsum('ba,b->ba', Vh[:bD, :], s)

        q1 = q1.reshape(nD,pD,bD)
        q2 = q2.reshape(nD,pD,bD)

        a1r = einsum("ypa,au->yup", q1, r1)
        a2r = einsum("xqb,vb->xvq", q2, r2)

        return a1r,a2r

    @staticmethod
    def decompose_site_tensors(a1, a2):
        """
        Decomposes the site tensors `a1` and `a2` into their core and reduced parts using QR decomposition.

        Parameters:
        -----------
        a1 : Tensor
            The first tensor at the site being decomposed.

        a2 : Tensor
            The second tensor at the site being decomposed.

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
        """
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
        a1 = einsum('rdux,xlp->lurdp', a1q, a1r)
        a2 = einsum('dlux,xrp->lurdp', a2q, a2r)
        return a1,a2

def torch_build_norm_tensor(ipeps, bond, a1q, a2q):
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
    s1, s2, k = bond

    # right half
    c12 = ipeps[s1]['C'][(k+1)%4]
    e12 = ipeps[s1]['E'][(k+1)%4]
    e11 = ipeps[s1]['E'][(k+0)%4]
    c13 = ipeps[s1]['C'][(k+2)%4]
    e13 = ipeps[s1]['E'][(k+2)%4]

    # left half
    c21 = ipeps[s2]['C'][(k+0)%4]
    e21 = ipeps[s2]['E'][(k+0)%4]
    e24 = ipeps[s2]['E'][(k+3)%4]
    c24 = ipeps[s2]['C'][(k+3)%4]
    e23 = ipeps[s2]['E'][(k+2)%4]

    return build_norm_tensor_core(c12, e12, e11, c13, e13, a1q,
                                  c21, e21, e24, c24, e23, a2q)

def build_norm_tensor_core(c12, e12, e11, c13, e13, a1q,
                           c21, e21, e24, c24, e23, a2q):
    # build right half
    tmp = einsum("ab,bcrR->acrR", c12, e12)
    tmp = einsum("acrR,eauU->crReuU", tmp, e11)
    tmp = einsum("crReuU,RDUY->creuDY", tmp, conj(a1q))
    tmp = einsum("creuDY,rduy->ceDYdy", tmp, a1q)
    n1_tmp = einsum("ab,bfdD->afdD", c13, e13)
    n1_tmp = einsum("afdD,aeDYdy->feYy", n1_tmp, tmp)

    # build left half
    tmp = einsum("ab,bcuU->acuU", c21, e21)
    tmp = einsum("acuU,ealL->cuUelL", tmp, e24)
    tmp = einsum("cuUelL,DLUX->cuelXD", tmp, conj(a2q))
    tmp = einsum("cuelXD,dlux->ceXDxd", tmp, a2q)
    n2_tmp = einsum("ab,fadD->bfdD", c24, e23)
    n2_tmp = einsum("bfdD,cbXDxd->fcXx", n2_tmp, tmp)

    return einsum("fcYy,fcXx->yxYX", n1_tmp, n2_tmp)


def build_norm_tensor_core_blocked(c12, e12, e11, c13, e13, a1q,
                                   c21, e21, e24, c24, e23, a2q):
    nD = a1q.shape[-1]
    BC = max(8, ceil(nD/4))
    block_range = range(0, nD, BC)
    block_indices = [(b00, b10, min(nD, b00+BC), min(nD, b10+BC))
                     for b00 in block_range for b10 in block_range]

    tmp_r1 = einsum("ab,bcrR->acrR", c12, e12)
    tmp_r1 = einsum("acrR,eauU->crReuU", tmp_r1, e11)
    tmp_r2 = einsum("ab,bfdD->afdD", c13, e13)

    tmp_l1 = einsum("ab,bcuU->acuU", c21, e21)
    tmp_l1 = einsum("acuU,ealL->cuUelL", tmp_l1, e24)
    tmp_l2 = einsum("ab,fadD->bfdD", c24, e23)

    n12 = torch.empty(nD,nD,nD,nD, dtype=a1q.dtype, device=a1q.device)
    for b00,b10,b01,b11 in block_indices:
        # build right half
        tmp_r3 = einsum("crReuU,RDUY->creuDY", tmp_r1, conj(a1q[:,:,:,b00:b01]))
        tmp_r3 = einsum("creuDY,rduy->ceDYdy", tmp_r3, a1q)
        n1_tmp = einsum("afdD,aeDYdy->feYy", tmp_r2, tmp_r3)

        # build left half
        tmp_l3 = einsum("cuUelL,DLUX->cuelXD", tmp_l1, conj(a2q[:,:,:,b10:b11]))
        tmp_l3 = einsum("cuelXD,dlux->ceXDxd", tmp_l3, a2q)
        n2_tmp = einsum("bfdD,cbXDxd->fcXx", tmp_l2, tmp_l3)

        n12[:,:,b00:b01,b10:b11] = einsum("fcYy,fcXx->yxYX", n1_tmp, n2_tmp)
    return n12


def positive_approx(n12, cutoff=1e-12):
    """
    Computes a positive approximation of the norm tensor. Eigenvalues are ensured to be positive
    by dynamic regularization and the condition number is greatly reduced. 

    Parameters:
    -----------
    n12 : Tensor
        The norm tensor to be approximated.

    cutoff : float, optional (default: 1e-12)
        A threshold value for determining the lowest eigenvalues to retain in the approximation.

    Returns:
    --------
    nz : Tensor
        Square root of the updated norm tensor after applying the positive approximation.
    """
    nD = n12.shape[0]
    N = n12.reshape(nD**2, nD**2)
    try:
        nw,nz = eigh(N)
    except torch._C._LinAlgError:
        nz,nw,_ = svd(N)
        nw = torch.flip(nw, dims=[-1])
        nz = torch.flip(nz, dims=[-1])
    while nw[0] < cutoff:
        N += 2*max(cutoff, abs(nw[0]))*torch.eye(nD**2, dtype=N.dtype, device=N.device)
        try:
            nw,nz = eigh(N)
        except torch._C._LinAlgError:
            nz,nw,_ = svd(N)
            nw = torch.flip(nw, dims=[-1])
            nz = torch.flip(nz, dims=[-1])
    nz = nz.reshape(nD, nD, nD**2)*torch.sqrt(nw)
    return nz


def gauge_fix(nz, a12g, atol=1e-12):
    """
    Applies a gauge fixing procedure to the norm tensor square root and the reduced tensors.

    Parameters:
    -----------
    nz : Tensor
        The norm tensor square root used to adjust the values during gauge fixing.

    a12g : Tensor
        The reduced tensor representing the bond interaction that will be updated during the gauge fixing.

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
    nD = a12g.shape[0]
    _,nzyr = qr(einsum("yxz->zxy", nz).reshape(nD**3, nD))
    _,nzxr = qr(einsum("yxz->zyx", nz).reshape(nD**3, nD))

    nzyr_inv = pinv(nzyr, atol=atol)
    nzxr_inv = pinv(nzxr, atol=atol)

    nz = einsum("yxz,xw->yzw", nz, nzxr_inv)
    nz = einsum("yzw,yv->zvw", nz, nzyr_inv)
    n12 = einsum("zvw,zVW->vwVW", nz, conj(nz))

    a12g = einsum("zx,yxpq->yzpq", nzxr, a12g)
    a12g = einsum("wy,yzpq->wzpq", nzyr, a12g)

    return n12, a12g, nzxr_inv, nzyr_inv
