import torch
from torch import einsum, conj, sqrt

class ProjectorCalculator:
    """Computes projectors for iPEPS tensor networks using full or half-system contractions."""
    
    def __init__(self, config):
        """
        Args:
            config (object): Configuration object with projector and SVD settings.
        """
        self.projectors        = config.projectors
        self.svd_type          = config.svd_type
        self.svd_cutoff        = config.svd_cutoff
        self.rsvd_niter        = config.rsvd_niter
        self.rsvd_oversampling = config.rsvd_oversampling
        self.set_calculate()

    def set_calculate(self):
        """
        Sets the `calculate` method based on the projector type.
        
        Raises:
            ValueError: If projector type is invalid.
        """
        if self.projectors is None or self.projectors == "full-system":
            self.calculate = self.calculate_full_system
        elif self.projectors == "half-system":
            self.calculate = self.calculate_half_system
        else:
            raise ValueError(f"Invalid ctmrg projector type: {self.projectors} provided.")

    @staticmethod
    def make_quarter_tensor(site_tensor, k):
        """
        Builds a quarter tensor from site tensors and environments.

        Args:
            site_tensor (object): iPEPS site tensor with bonds and environments.
            k (int): Bond direction index.

        Returns:
            tuple[Tensor, tuple]: Reshaped quarter tensor and its original shape.
        """
        ak  = site_tensor.bond_permute(k)
        ck  = site_tensor['C'][(0+k)%4]
        ek1 = site_tensor['E'][(3+k)%4]
        ek2 = site_tensor['E'][(0+k)%4]

        qk = einsum("ab,bcuU->acuU", ck, ek2)
        qk = einsum("acuU,ealL->cuUelL", qk, ek1)
        qk = einsum("cuUelL,LURDP->cuelRDP", qk, conj(ak))
        qk = einsum("lurdp,cuelRDp->crRedD", ak, qk)

        qD = qk.shape
        qk = qk.reshape(qD[0]*qD[1]*qD[2], qD[3]*qD[4]*qD[5])
        qk/qk.abs().max()
        return qk, qD

    def contract_half_system(self, ipeps, sites, k):
        """
        Contracts four sites to form half-system intermediate tensors.

        Args:
            ipeps (object): iPEPS tensor network.
            sites (list[int]): Four site indices.
            k (int): Bond direction index.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Q1, Q4, and the half contraction result.
        """
        s1, s4 = sites[0], sites[3]
        Q1, q1D = self.make_quarter_tensor(ipeps[s1], k)
        Q4, q4D = self.make_quarter_tensor(ipeps[s4], k+3)
        R = Q1 @ Q4

        Q1 = Q1.view(Q1.shape[0], *q1D[3:])
        Q4 = Q4.view(*q4D[:3], Q4.shape[1])
        R = R / R.abs().max()
        return Q1, Q4, R

    def contract_full_system(self, ipeps, sites, k):
        """
        Contracts four sites to form full-system intermediate tensors.

        Args:
            ipeps (object): iPEPS tensor network.
            sites (list[int]): Four site indices.
            k (int): Bond direction index.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Half-system contractions R1, R2, and their product F.
        """
        s1, s2, s3, s4 = sites
        Q1, q1D = self.make_quarter_tensor(ipeps[s1], k)
        Q2, _   = self.make_quarter_tensor(ipeps[s2], k+1)
        R1 = Q2 @ Q1

        Q3, _   = self.make_quarter_tensor(ipeps[s3], k+2)
        Q4, q4D = self.make_quarter_tensor(ipeps[s4], k+3)
        R2 = Q4 @ Q3

        R1 = R1 / R1.abs().max()
        R2 = R2 / R2.abs().max()
        F = R1 @ R2

        R1 = R1.view(R1.shape[0], *q1D[3:])
        R2 = R2.view(*q4D[:3], R2.shape[1])
        F = F / F.abs().max()
        return R1, R2, F

    def calculate_projectors(self, R1, R2, F, chi):
        """
        Computes projectors via SVD of the intermediate contraction.

        Args:
            R1 (Tensor): Half-system contraction part 1.
            R2 (Tensor): Half-system contraction part 2.
            F (Tensor): Full-system contraction.
            chi (int): Max bond dimension.

        Returns:
            tuple[Tensor, Tensor]: Left and right projectors.
        """
        U, s, V = self.svd(F, chi)
        s = s / s[0]
        cD_new = min(chi, (s > self.svd_cutoff).sum())

        U = U[:, :cD_new] * (1. / sqrt(s[:cD_new]))
        V = V[:, :cD_new] * (1. / sqrt(s[:cD_new]))

        proj1 = einsum("xedD,xz->edDz", R1, conj(U))
        proj2 = einsum("cuUy,yz->cuUz", R2, V)
        return proj1, proj2

    def calculate_half_system(self, ipeps, sites, k):
        """
        Computes projectors using half-system contraction.

        Args:
            ipeps (object): iPEPS tensor network.
            sites (list[int]): Four site indices.
            k (int): Bond direction index.

        Returns:
            tuple[Tensor, Tensor]: Left and right projectors.
        """
        Q1, Q4, R = self.contract_half_system(ipeps, sites, k)
        return self.calculate_projectors(Q1, Q4, R, ipeps.dims["chi"])

    def calculate_full_system(self, ipeps, sites, k):
        """
        Computes projectors using full-system contraction.

        Args:
            ipeps (object): iPEPS tensor network.
            sites (list[int]): Four site indices.
            k (int): Bond direction index.

        Returns:
            tuple[Tensor, Tensor]: Left and right projectors.
        """
        R1, R2, F = self.contract_full_system(ipeps, sites, k)
        return self.calculate_projectors(R1, R2, F, ipeps.dims["chi"])

    def svd(self, A, cD):
        """
        Performs singular value decomposition (SVD), full-rank or RSVD on matrix A.

        Args:
            A (Tensor): Input matrix.
            cD (int): Target dimension (used for RSVD).

        Returns:
            tuple[Tensor, Tensor, Tensor]: SVD result (U, S, V).
        """
        if self.svd_type == "full-rank":
            U,s,Vh = torch.linalg.svd(A)
            V = Vh.mH
        elif self.svd_type == "rsvd":
            q = cD + self.rsvd_oversampling
            U,s,V = torch.svd_lowrank(A, q=q, niter=self.rsvd_niter)
        return U,s,V
