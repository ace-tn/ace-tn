import torch
from torch import einsum, conj
from torch.linalg import pinv, cholesky, solve_triangular, svd

class ALSSolver:
    """
    Alternating Least Squares (ALS) solver for the reduced tensor update.
    """
    def __init__(self, n12, a12g, bD, pD, nD, config):
        """
        Initializes the ALSSolver class with the given norm tensor and gate-tensor product a12g to be approximated.

        Parameters:
        -----------
        n12 : Tensor
            The norm tensor.

        a12g : Tensor
            The approximated gate-tensor product.

        bD : int
            The bond dimension of the tensors.

        pD : int
            The physical dimension of the tensors.

        nD : int
            The dimension of the norm tensor.
        """
        self.niter = config.als_niter
        self.tol = config.als_tol
        self.method = config.als_method
        self.epsilon = config.als_epsilon
        self.backend = config.backend
        self.pD = pD
        self.bD = bD
        self.nD = nD
        self.n12 = n12
        self.a12g = a12g
        self.setup_backend()

    def setup_backend(self):
        if self.backend == "torch":
            self.build_s1 = torch_build_s1
            self.build_s2 = torch_build_s2
            self.build_r1 = torch_build_r1
            self.build_r2 = torch_build_r2
            self.calculate_cost = torch_calculate_cost
        elif self.backend == "cutensor":
            torch.ops.load_library("../als_contractions.so")
            self.build_s1 = torch.ops.cutensor_ext.build_s1
            self.build_s2 = torch.ops.cutensor_ext.build_s2
            self.build_r1 = torch.ops.cutensor_ext.build_r1
            self.build_r2 = torch.ops.cutensor_ext.build_r2
            self.calculate_cost = torch.ops.cutensor_ext.calculate_cost
            self.n12 = self.n12.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)
            self.a12g = self.a12g.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)
        else:
            raise NotImplementedError(f"Backend {self.backend} is not supported.")

    def solve(self):
        """
        Solves for the updated tensors by alternating between solving for `a1r` and `a2r`.

        Returns:
        --------
        tuple
            A tuple containing the updated reduced tensors (`a1r`, `a2r`).
        """
        pD = self.pD
        bD = self.bD
        nD = self.nD
        a1r,a2r,n12g = self.initialize_tensors(bD, pD, nD)
        a1r = a1r.permute(2, 1, 0).contiguous().permute(2, 1, 0)
        a2r = a2r.permute(2, 1, 0).contiguous().permute(2, 1, 0)
        n12g = n12g.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)
        d1 = self.calculate_cost(a1r, a2r, self.a12g, self.n12).abs()
        for i in range(self.niter):
            a1r = self.solve_a1r(n12g, a2r, bD, pD, nD).permute(2, 1, 0).contiguous().permute(2, 1, 0)
            a2r = self.solve_a2r(n12g, a1r, bD, pD, nD).permute(2, 1, 0).contiguous().permute(2, 1, 0)
            d2 = self.calculate_cost(a1r, a2r, self.a12g, self.n12)
            error = abs(d2 - d1)/d1.abs()
            if error < self.tol and i > 1:
                return a1r, a2r
            d1 = d2
        return a1r, a2r

    def initialize_tensors(self, bD, pD, nD):
        a1r,a2r = self.initialize_reduced_tensors(self.a12g, bD, pD, nD)
        n12g = einsum("yxYX,yxpq->YXpq", self.n12, self.a12g)
        return a1r,a2r,n12g

    @staticmethod
    def initialize_reduced_tensors(a12g, bD, pD, nD):
        """
        Initializes the reduced tensors `a1r` and `a2r` from the gate-contracted tensor `a12g`.

        The initial guess for the updated a1r and a2r is determined from a truncated SVD of a12g.

        Parameters:
        -----------
        a12g : Tensor
            The gate-tensor product.

        bD : int
            The bond dimension of the tensors.

        pD : int
            The physical dimension of the tensors.

        nD : int
            The dimension of the norm tensor.

        Returns:
        --------
        tuple
            A tuple containing the initialized reduced tensors `a1r` and `a2r`.
        """
        a12g_tmp = einsum("yxpq->ypxq", a12g).reshape(nD*pD, nD*pD)
        U,S,Vh = svd(a12g_tmp)
        V = Vh.mH
        S = torch.sqrt(S[:bD]/S[0])
        U = U[:,:bD].reshape(nD, pD, bD)
        V = V[:,:bD].reshape(nD, pD, bD)
        a1r = einsum("ypu,u->yup", U, S)
        a2r = einsum("xqv,v->xvq", V, S)
        return a1r,a2r

    def solve_a1r(self, n12g, a2r, bD, pD, nD):
        """
        Solves for the reduced tensor `a1r` in the ALS optimization process.

        This method forms a system of linear equations and solves for `a1r` given the fixed tensor `a2r`.

        Parameters:
        -----------
        n12g : Tensor
            The modified norm tensor used in the ALS optimization.

        a2r : Tensor
            The fixed reduced tensor `a2r`.

        bD : int
            The bond dimension of the tensors.

        pD : int
            The physical dimension of the tensors.

        nD : int
            The dimension of the norm tensor.

        Returns:
        --------
        Tensor
            The solved reduced tensor `a1r`.
        """
        rD = nD*bD
        S = self.build_s1(n12g, a2r).reshape(pD,bD,nD).permute(2,1,0).reshape(rD,pD)
        R = self.build_r1(self.n12, a2r).reshape(bD,nD,bD,nD).permute(3,2,1,0).reshape(rD,rD)
        return self.solve_ar(R, S, bD, pD, nD)

    def solve_a2r(self, n12g, a1r, bD, pD, nD):
        """
        Solves for the reduced tensor `a2r` in the ALS optimization process.

        This method forms a system of linear equations and solves for `a2r` given the fixed tensor `a1r`.

        Parameters:
        -----------
        n12g : Tensor
            The modified norm tensor used in the ALS optimization.
        
        a1r : Tensor
            The fixed reduced tensor `a1r`.
        
        bD : int
            The bond dimension of the tensors.
        
        pD : int
            The physical dimension of the tensors.

        Returns:
        --------
        Tensor
            The solved reduced tensor `a2r`.
        """
        rD = nD*bD
        S = self.build_s2(n12g, a1r).reshape(pD,bD,nD).permute(2,1,0).reshape(rD,pD)
        R = self.build_r2(self.n12, a1r).reshape(bD,nD,bD,nD).permute(3,2,1,0).reshape(rD,rD)
        return self.solve_ar(R, S, bD, pD, nD)

    def solve_ar(self, R, S, bD, pD, nD):
        """
        Solves the linear system for the reduced tensor using either Cholesky decomposition or pseudoinverse.

        Depending on the chosen method, this function either solves the system using Cholesky decomposition or
        computes the pseudoinverse of the matrix `R` to solve the linear system.

        Parameters:
        -----------
        R : Tensor
            The matrix representing the linear system to be solved.

        S : Tensor
            The right-hand side vector of the linear system.

        bD : int
            The bond dimension of the tensors.

        pD : int
            The physical dimension of the tensors.

        Returns:
        --------
        Tensor
            The solved reduced tensor `ar` reshaped to the appropriate dimensions.
        """
        R = 0.5*(R + R.mH)
        match self.method:
            case "cholesky":
                try:
                    R += self.epsilon*torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
                    L = cholesky(R)
                    Y = solve_triangular(L, S, upper=False)
                    ar = solve_triangular(L.mH, Y, upper=True)
                except:
                    ar = torch.linalg.solve(R, S)
            case "pinv":
                R_inv = pinv(R, hermitian=True, rcond=self.epsilon)
                ar = R_inv @ S
        return ar.reshape(nD,bD,pD)

def torch_calculate_cost(a1r, a2r, a12g, n12):
    a12n = einsum("yup,xuq->yxpq", a1r, a2r)

    d2 = einsum("yxYX,yxpq->YXpq", n12, a12n)
    d2 = einsum("YXpq,YXpq->", d2, a12n)

    d3 = einsum("yxYX,yxpq->YXpq", n12, a12g)
    d3 = einsum("YXpq,YXpq->", d3, a12n)

    return d2.real - 2*d3.real

def torch_build_s1(n12g, a2r):
    return torch.einsum('YXpQ,XUQ->YUp', n12g, a2r)

def torch_build_s2(n12g, a1r):
    return torch.einsum('YXPq,YVP->XVq', n12g, a1r)

def torch_build_r1(n12, a2r):
    R = torch.einsum('yxYX,xuq->yYXuq', n12, a2r)
    return torch.einsum('yYXuQ,XUQ->YUyu', R, a2r)

def torch_build_r2(n12, a1r):
    R = torch.einsum('yxYX,yvp->xYXvp', n12, a1r)
    return torch.einsum('xYXvP,YVP->XVxv', R, a1r)
