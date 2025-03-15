from torch import einsum,conj
from .projectors import ProjectorCalculator
from utils.distributed import all_gather_tensor

class DirectionalMover:
    """
    Class for performing directional moves involved in iPEPS boundary renormalization.

    This class implements tensor renormalization moves updating iPEPS boundary tensors 
    in both left-right and up-down directions. It supports both distributed and 
    non-distributed computations.
    """
    def __init__(self, config):
        """
        Initializes the DirectionalMover instance with the configuration.

        Args:
            config (dict): Configuration dictionary containing 'projectors' and other settings.
        """
        projector_calculator = ProjectorCalculator(config)
        self.calculate_projectors = projector_calculator.calculate

    def left_move(self, ipeps, xi):
        """
        Performs the left boundary move for a given column index xi.

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            xi (int): The x-coordinate index of the boundary to be updated.
        """
        proj1 = {}
        proj2 = {}
        for yi in range(ipeps.ny):
            proj1[yi], proj2[yi] = self.calculate_left_projectors(ipeps, xi, yi)
        for yi in range(ipeps.ny):
            xj = (xi+1) % ipeps.nx
            yj = (yi+1) % ipeps.ny
            s1 = (xi,yi)
            s2 = (xj,yi)
            self.renormalize_boundary(ipeps, proj1, proj2, s1, s2, yi, yj, k=0)

    def up_move(self, ipeps, yi):
        """
        Performs the up boundary move for a given row index yi.

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            yi (int): The y-coordinate index of the boundary to be updated.
        """
        proj1 = {}
        proj2 = {}
        for xi in range(ipeps.nx):
            proj1[xi], proj2[xi] = self.calculate_up_projectors(ipeps, xi, yi)
        for xi in range(ipeps.nx):
            xj = (xi+1) % ipeps.nx
            yj = (yi-1+ipeps.ny) % ipeps.ny
            s1 = (xi,yi)
            s2 = (xi,yj)
            self.renormalize_boundary(ipeps, proj1, proj2, s1, s2, xi, xj, k=1)

    def right_move(self, ipeps, xi):
        """
        Performs the right boundary move for a given column index xi.

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            xi (int): The x-coordinate index of the boundary to be updated.
        """
        proj1 = {}
        proj2 = {}
        for yi in range(ipeps.ny):
            proj1[yi], proj2[yi] = self.calculate_right_projectors(ipeps, xi, yi)
        for yi in range(ipeps.ny):
            xj = (xi-1+ipeps.nx) % ipeps.nx
            yj = (yi-1+ipeps.ny) % ipeps.ny
            s1 = (xi,yi)
            s2 = (xj,yi)
            self.renormalize_boundary(ipeps, proj1, proj2, s1, s2, yi, yj, k=2)

    def down_move(self, ipeps, yi):
        """
        Performs the down boundary move for a given row index yi.

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            yi (int): The y-coordinate index of the boundary to be updated.
        """
        proj1 = {}
        proj2 = {}
        for xi in range(ipeps.nx):
            proj1[xi], proj2[xi] = self.calculate_down_projectors(ipeps, xi, yi)
        for xi in range(ipeps.nx):
            xj = (xi-1+ipeps.nx) % ipeps.nx
            yj = (yi+1) % ipeps.ny
            s1 = (xi,yi)
            s2 = (xi,yj)
            self.renormalize_boundary(ipeps, proj1, proj2, s1, s2, xi, xj, k=3)

    def calculate_left_projectors(self, ipeps, xi, yi):
        """
        Calculates the projectors for the left boundary at position (xi, yi).

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            xi (int): The x-coordinate index of the boundary.
            yi (int): The y-coordinate index of the boundary.

        Returns:
            tuple: The calculated left projectors.
        """
        xj = (xi+1) % ipeps.nx
        yj = (yi-1+ipeps.ny) % ipeps.ny
        s1 = (xi,yi)
        s2 = (xj,yi)
        s3 = (xj,yj)
        s4 = (xi,yj)
        sites = [s1,s2,s3,s4]
        return self.calculate_projectors(ipeps, sites, k=0)

    def calculate_right_projectors(self, ipeps, xi, yi):
        """
        Calculates the projectors for the right boundary at position (xi, yi).

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            xi (int): The x-coordinate index of the boundary.
            yi (int): The y-coordinate index of the boundary.

        Returns:
            tuple: The calculated right projectors.
        """
        xj = (xi-1+ipeps.nx) % ipeps.nx
        yj = (yi+1) % ipeps.ny
        s1 = (xi,yi)
        s2 = (xj,yi)
        s3 = (xj,yj)
        s4 = (xi,yj)
        sites = [s1,s2,s3,s4]
        return self.calculate_projectors(ipeps, sites, k=2)

    def calculate_up_projectors(self, ipeps, xi, yi):
        """
        Calculates the projectors for the up boundary at position (xi, yi).

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            xi (int): The x-coordinate index of the boundary.
            yi (int): The y-coordinate index of the boundary.

        Returns:
            tuple: The calculated up projectors.
        """
        xj = (xi-1+ipeps.nx) % ipeps.nx
        yj = (yi-1+ipeps.ny) % ipeps.ny
        s1 = (xi,yi)
        s2 = (xi,yj)
        s3 = (xj,yj)
        s4 = (xj,yi)
        sites = [s1,s2,s3,s4]
        return self.calculate_projectors(ipeps, sites, k=1)

    def calculate_down_projectors(self, ipeps, xi, yi):
        """
        Calculates the projectors for the down boundary at position (xi, yi).

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            xi (int): The x-coordinate index of the boundary.
            yi (int): The y-coordinate index of the boundary.

        Returns:
            tuple: The calculated down projectors.
        """
        xj = (xi+1) % ipeps.nx
        yj = (yi+1) % ipeps.ny
        s1 = (xi,yi)
        s2 = (xi,yj)
        s3 = (xj,yj)
        s4 = (xj,yi)
        sites = [s1,s2,s3,s4]
        return self.calculate_projectors(ipeps, sites, k=3)

    def left_right_move_dist(self, ipeps, x1, x2):
        """
        Performs the distributed left-right boundary move for columns x1 and x2.

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            x1 (int): The x-coordinate index of the left boundary.
            x2 (int): The x-coordinate index of the right boundary.
        """
        rank, ws = ipeps.rank, ipeps.world_size
        proj1l, proj2l = {}, {}
        proj1r, proj2r = {}, {}
        for y_shift in range(0, 2*ipeps.ny, ws):
            if rank < ws//2:
                yi = rank + y_shift//ws
                if yi < ipeps.ny:
                    proj1_i, proj2_i = self.calculate_left_projectors(ipeps, x1, yi)
            elif rank >= ws//2:
                yi = rank - ws//2 + y_shift//ws
                if yi < ipeps.ny:
                    proj1_i, proj2_i = self.calculate_right_projectors(ipeps, x2, yi)

            proj1_all = all_gather_tensor(proj1_i, rank, ws)
            proj2_all = all_gather_tensor(proj2_i, rank, ws)

            for yi in range(ws//2):
                proj1l[yi + y_shift//ws] = proj1_all[yi].to(rank)
                proj2l[yi + y_shift//ws] = proj2_all[yi].to(rank)
                proj1r[yi + y_shift//ws] = proj1_all[yi + ws//2].to(rank)
                proj2r[yi + y_shift//ws] = proj2_all[yi + ws//2].to(rank)

        for yi in range(ipeps.ny):
            xj = (x1+1) % ipeps.nx
            yj = (yi+1) % ipeps.ny
            s1 = (x1,yi)
            s2 = (xj,yi)
            self.renormalize_boundary(ipeps, proj1l, proj2l, s1, s2, yi, yj, k=0)

        for yi in range(ipeps.ny):
            xj = (x2-1+ipeps.nx) % ipeps.nx
            yj = (yi-1+ipeps.ny) % ipeps.ny
            s1 = (x2,yi)
            s2 = (xj,yi)
            self.renormalize_boundary(ipeps, proj1r, proj2r, s1, s2, yi, yj, k=2)

    def up_down_move_dist(self, ipeps, y1, y2):
        """
        Performs the distributed up-down boundary move for rows y1 and y2.

        Args:
            ipeps (IPEPS): The iPEPS object representing the tensor network.
            y1 (int): The y-coordinate index of the up boundary.
            y2 (int): The y-coordinate index of the down boundary.
        """
        rank, ws = ipeps.rank, ipeps.world_size
        proj1u, proj2u = {}, {}
        proj1d, proj2d = {}, {}
        for x_shift in range(0, 2*ipeps.nx, ws):
            if rank < ws//2:
                xi = rank + x_shift//ws
                if xi < ipeps.nx:
                    proj1_i, proj2_i = self.calculate_up_projectors(ipeps, xi, y1)
            elif rank >= ws//2:
                xi = rank - ws//2 + x_shift//ws
                if xi < ipeps.nx:
                    proj1_i, proj2_i = self.calculate_down_projectors(ipeps, xi, y2)

            proj1_all = all_gather_tensor(proj1_i, rank, ws)
            proj2_all = all_gather_tensor(proj2_i, rank, ws)

            for xi in range(ws//2):
                proj1u[xi + x_shift//ws] = proj1_all[xi].to(rank)
                proj2u[xi + x_shift//ws] = proj2_all[xi].to(rank)
                proj1d[xi + x_shift//ws] = proj1_all[xi + ws//2].to(rank)
                proj2d[xi + x_shift//ws] = proj2_all[xi + ws//2].to(rank)

        for xi in range(ipeps.nx):
            xj = (xi+1) % ipeps.nx
            yj = (y1-1+ipeps.ny) % ipeps.ny
            s1 = (xi,y1)
            s2 = (xi,yj)
            self.renormalize_boundary(ipeps, proj1u, proj2u, s1, s2, xi, xj, k=1)

        for xi in range(ipeps.nx):
            xj = (xi-1+ipeps.nx) % ipeps.nx
            yj = (y2+1) % ipeps.ny
            s1 = (xi,y2)
            s2 = (xi,yj)
            self.renormalize_boundary(ipeps, proj1d, proj2d, s1, s2, xi, xj, k=3)

    def renormalize_boundary(self, ipeps, proj1, proj2, s1, s2, i, j, k):
        """
        Renormalizes the boundary tensors of the iPEPS by applying projectors.

        This method updates the boundary tensors of the iPEPS by performing tensor renormalization 
        on the `C` and `E` tensors at the given boundary sites. The updates are based on the projectors 
        provided and the direction specified by the parameter `k`.

        Args:
            ipeps (IPEPS): The iPEPS object containing the tensor network.
            proj1 (dict): Projectors for the first boundary.
            proj2 (dict): Projectors for the second boundary.
            s1 (tuple): Coordinates of the first boundary tensor.
            s2 (tuple): Coordinates of the second boundary tensor.
            i (int): Index for the first boundary (used to index into proj1 and proj2).
            j (int): Index for the second boundary (used to index into proj1 and proj2).
            k (int): The boundary direction (0 for left, 1 for up, 2 for right, 3 for down).

        This method modifies the `C` and `E` tensors at the boundary sites `s2` based on the projectors.
        """
        ci = ipeps[s1]['C'][(3+k)%4]
        ei = ipeps[s1]['E'][(2+k)%4]
        ipeps[s2]['C'][(3+k)%4] = self.renormalize_cj1(ci, ei, proj1[i])

        ci = ipeps[s1]['C'][k]
        ei = ipeps[s1]['E'][k]
        ipeps[s2]['C'][k] = self.renormalize_cj2(ci, ei, proj2[j])

        ai = ipeps[s1].bond_permute(k)
        ei = ipeps[s1]['E'][(3+k)%4]
        ipeps[s2]['E'][(3+k)%4] = self.renormalize_ej(ei, ai, proj2[i], proj1[j])

    @staticmethod
    def renormalize_cj1(ci, ei, proj):
        """
        Renormalizes the `C` tensor for the first boundary based on the projectors.

        This method performs a series of tensor contractions to renormalize the `C` tensor at the first boundary 
        site using the provided `E` tensor and projector.

        Args:
            ci (Tensor): The `C` tensor of the first boundary site.
            ei (Tensor): The `E` tensor of the first boundary site.
            proj (Tensor): The projector for the boundary.

        Returns:
            Tensor: The renormalized `C` tensor after contraction and normalization.
        """
        cj = einsum("ablL,bc->alLc", ei, ci)
        cj = einsum("alLc,clLx->ax", cj, proj)
        return cj/cj.norm()

    @staticmethod
    def renormalize_cj2(ci, ei, proj):
        """
        Renormalizes the `C` tensor for the second boundary based on the projectors.

        This method performs a series of tensor contractions to renormalize the `C` tensor at the second boundary 
        site using the provided `E` tensor and projector.

        Args:
            ci (Tensor): The `C` tensor of the first boundary site.
            ei (Tensor): The `E` tensor of the first boundary site.
            proj (Tensor): The projector for the boundary.

        Returns:
            Tensor: The renormalized `C` tensor after contraction and normalization.
        """
        cj = einsum("ab,bcrR->acrR", ci, ei)
        cj = einsum("arRx,acrR->xc", proj, cj)
        return cj/cj.norm()

    @staticmethod
    def renormalize_ej(ei, ai, proj2, proj1):
        """
        Renormalizes the `E` tensor for the second boundary based on the projectors.

        This method performs a series of tensor contractions to renormalize the `E` tensor at the second boundary 
        site using the provided `ai` tensor and the projectors for both boundaries.

        Args:
            ei (Tensor): The `E` tensor of the first boundary site.
            ai (Tensor): The `A` tensor at the first boundary site, used for contraction.
            proj2 (Tensor): The projector for the second boundary.
            proj1 (Tensor): The projector for the first boundary.

        Returns:
            Tensor: The renormalized `E` tensor after contraction and normalization.
        """ 
        ej = einsum("ablL,buUx->alLuUx", ei, proj1)
        ej = einsum("LURDP,alLuUx->RDPalux", conj(ai), ej)
        ej = einsum("lurdp,RDpalux->rdRDax", ai, ej)
        ej = einsum("rdRDax,adDy->yxrR", ej, proj2)
        return ej/ej.norm()
