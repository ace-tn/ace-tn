import torch
from torch import einsum,conj

class ProjectorCalculator:
    """
    A class to compute projectors for the iPEPS tensor network.

    This class provides methods for calculating half-system and full-system projectors
    based on the given tensor network configuration.
    """
    def __init__(self, config):
        """
        Initializes the ProjectorCalculator with the given configuration.

        Parameters:
        -----------
        config : dict
            A dictionary containing the configuration for the projector calculation.
        """
        self.projectors        = config.projectors
        self.svd_type          = config.svd_type
        self.svd_cutoff        = config.svd_cutoff
        self.rsvd_niter        = config.rsvd_niter
        self.rsvd_oversampling = config.rsvd_oversampling
        self.set_calculate()

    def set_calculate(self):
        """
        Sets the appropriate method for projector calculation based on the value of the `projectors` attribute.
        
        The method is set to either `calculate_full_system` or `calculate_half_system`, or raises an error 
        if an invalid projector type is provided.
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
        Constructs a "quarter" tensor from the site tensor, applying bond permutations and contractions.

        This function creates a tensor that represents a quarter of the full tensor network by contracting
        the corresponding tensors from the iPEPS network.

        Parameters:
        -----------
        site_tensor : Tensor
            The site tensor from the iPEPS network.
        k : int
            The bond index to be permuted and contracted.

        Returns:
        --------
        Tensor
            The quarter tensor obtained after applying the bond permutations and contractions.
        """
        ak  = site_tensor.bond_permute(k)
        ck  = site_tensor['C'][(0+k)%4]
        ek1 = site_tensor['E'][(3+k)%4]
        ek2 = site_tensor['E'][(0+k)%4]

        qk = einsum("ab,bcuU->acuU", ck, ek2)
        qk = einsum("acuU,ealL->cuUelL", qk, ek1)
        qk = einsum("cuUelL,LURDP->cuelRDP", qk, conj(ak))
        qk = einsum("lurdp,cuelRDp->rRcdDe", ak, qk)
        return qk/qk.norm()

    def calculate_half_system(self, ipeps, sites, k):
        """
        Calculates the half-system projectors for the iPEPS tensor network.

        This method computes the projectors for the half-system by contracting tensors from two sites 
        and performing a singular value decomposition (SVD) to obtain the projector.

        Parameters:
        -----------
        ipeps : object
            The iPEPS tensor network.
        sites : list
            A list of two site indices in the iPEPS network for the half-system.
        k : int
            The bond index around which the contraction is performed.

        Returns:
        --------
        tuple
            A tuple containing the two projectors for the half-system.
        """
        s1 = sites[0]
        s4 = sites[3]

        q1 = self.make_quarter_tensor(ipeps[s1], k)
        q4 = self.make_quarter_tensor(ipeps[s4], k+3)
        r1 = einsum("rRcdDf,dDfsSe->rRcsSe", q1, q4)
        r1 = r1/r1.norm()

        rD = r1.shape
        r1 = r1.reshape(rD[0]*rD[1]*rD[2], rD[3]*rD[4]*rD[5])

        cD = ipeps.dims['chi']
        ur1,sr1,vr1 = self.svd(r1, cD)
        cD_new = min(cD, sum(sr1/sr1[0] > self.svd_cutoff))

        ur1 = ur1[:,:cD_new]*(1./torch.sqrt(sr1[:cD_new]))
        vr1 = vr1[:,:cD_new]*(1./torch.sqrt(sr1[:cD_new]))

        ur1 = ur1.reshape(rD[0],rD[1],rD[2],cD_new)
        vr1 = vr1.reshape(rD[3],rD[4],rD[5],cD_new)

        proj1_i = einsum("rRcdDe,rRcx->edDx", q1, conj(ur1))
        proj2_i = einsum("uUcrRe,rRex->cuUx", q4, vr1)

        return proj1_i, proj2_i

    def calculate_full_system(self, ipeps, sites, k):
        """
        Calculates the full-system projectors for the iPEPS tensor network.

        This method computes the projectors for the full-system by contracting tensors from four sites 
        and performing a singular value decomposition (SVD) to obtain the projectors.

        Parameters:
        -----------
        ipeps : object
            The iPEPS tensor network.
        sites : list
            A list of four site indices in the iPEPS network for the full-system.
        k : int
            The bond index around which the contraction is performed.

        Returns:
        --------
        tuple
            A tuple containing the two projectors for the full-system.
        """
        s1,s2,s3,s4 = sites

        q1 = self.make_quarter_tensor(ipeps[s1], k)
        q2 = self.make_quarter_tensor(ipeps[s2], k+1)
        r1 = einsum("xXclLf,lLfdDe->xXcdDe", q2, q1)

        q3 = self.make_quarter_tensor(ipeps[s3], k+2)
        q4 = self.make_quarter_tensor(ipeps[s4], k+3)
        r2 = einsum("uUclLf,lLfyYe->uUcyYe", q4, q3)

        f0 = einsum("xXcdDf,dDfyYe->xXcyYe", r1, r2)
        f0 = f0/f0.norm()

        fD = f0.shape
        f0 = f0.reshape(fD[0]*fD[1]*fD[2], fD[3]*fD[4]*fD[5])

        cD = ipeps.dims['chi']
        uf0,sf0,vf0 = self.svd(f0, cD)
        cD_new = min(cD, sum(sf0/sf0[0] > self.svd_cutoff))

        uf0 = uf0[:,:cD_new]*(1./torch.sqrt(sf0[:cD_new]))
        vf0 = vf0[:,:cD_new]*(1./torch.sqrt(sf0[:cD_new]))

        uf0 = uf0.reshape(fD[0],fD[1],fD[2],cD_new)
        vf0 = vf0.reshape(fD[3],fD[4],fD[5],cD_new)

        proj1_i = einsum("xXcdDe,xXcz->edDz", r1, conj(uf0))
        proj2_i = einsum("uUcyYe,yYez->cuUz", r2, vf0)

        return proj1_i, proj2_i

    def svd(self, A, cD):
        """
        Performs Singular Value Decomposition (SVD) on the given matrix `A`.

        This method supports both full-rank SVD and randomized SVD (rSVD), depending on the `svd_type`.

        Parameters:
        -----------
        A : Tensor
            The matrix to be decomposed.
        cD : int
            The dimension parameter used for the randomized SVD.

        Returns:
        --------
        tuple
            A tuple containing the left singular vectors (`u`), singular values (`s`), 
            and right singular vectors (`v`) from the decomposition.
        """
        if self.svd_type == "full-rank":
            u,s,v = torch.linalg.svd(A)
            v = v.mH
        elif self.svd_type == "rsvd":
            niter = self.rsvd_niter
            q = cD + self.rsvd_oversampling
            u,s,v = torch.svd_lowrank(A, q=q, niter=niter)
        return u,s,v
