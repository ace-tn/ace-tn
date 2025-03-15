import torch
import torch.distributed as dist
from .full_update import FullUpdater
from renormalization.ctmrg import DirectionalMover
from utils.benchmarking import record_runtime

class FastFullUpdater(FullUpdater):
    """
    A subclass of FullUpdater that combines full updates with directional moves using both 
    distributed and non-distributed Corner Transfer Matrix (CTM) updates.

    Attributes:
        mover (DirectionalMover): An instance of the DirectionalMover class responsible for 
            performing directional moves updating the iPEPS boundary tensors following the
            tensor update.
    """
    def __init__(self, ipeps, gate, config):
        """
        Initializes a FastFullUpdater instance.

        Args:
            ipeps (Ipeps): The iPEPS object representing the tensor network to be updated.
            gate (Gate): The gate used for tensor updates.
            config (dict): Configuration settings for the updater.
        """
        super().__init__(ipeps, gate, config)
        self.mover = DirectionalMover(ipeps.config.ctmrg)

    def bond_update(self, bond):
        """
        Updates the tensors for a given bond in the iPEPS network.

        Depending on whether the iPEPS network is distributed or not, this method either broadcasts 
        tensors across ranks or updates them locally. Afterward, it absorbs the updated bond either 
        in a distributed or non-distributed manner.

        Args:
            bond (list): A list representing a bond in the iPEPS network. It contains the site
                indices of two adjacent tensors and an integer (k) that indicates the direction 
                of the update.

        Returns:
            tuple: A tuple containing two values:
                - ctm_time (float): The time spent on the CTM update for the bond.
                - upd_time (float): The time spent on the full update for the bond.
        """
        s1,s2,_ = bond
        if self.ipeps.rank == 0:
            a1,a2, upd_time = self.update(bond)
        else:
            upd_time = 0
            a1 = torch.empty_like(self.ipeps[s1]['A'])
            a2 = torch.empty_like(self.ipeps[s2]['A'])

        if self.ipeps.is_distributed:
            a1 = a1.contiguous()
            dist.broadcast(a1, src=0)
            a2 = a2.contiguous()
            dist.broadcast(a2, src=0)

        self.ipeps[s1]['A'] = a1
        self.ipeps[s2]['A'] = a2

        if self.ipeps.is_distributed:
            ctm_time = self.absorb_bond_dist(bond)
        else:
            ctm_time = self.absorb_bond(bond)

        return ctm_time, upd_time

    @record_runtime
    def absorb_bond(self, bond):
        """
        Absorbs a bond in the iPEPS network using non-distributed updates.

        The method performs directional moves, updating the boundary tensors in the direction specified 
        by the bond direction index (k). The boundary tensors are updated using local (non-distributed) 
        moves, and the bond is absorbed by the directional moves.

        Args:
            bond (list): A list representing a bond in the iPEPS network. It contains the 
                indices of two adjacent tensors and an integer (k) that specifies the direction 
                of the move.
        
        Returns:
            None
        """
        s1,s2,k = bond
        match k:
            case 0:
                self.mover.right_move(self.ipeps, s1[0])
                self.mover.left_move(self.ipeps, s2[0])
            case 1:
                self.mover.down_move(self.ipeps, s1[1])
                self.mover.up_move(self.ipeps, s2[1])
            case 2:
                self.mover.left_move(self.ipeps, s1[0])
                self.mover.right_move(self.ipeps, s2[0])
            case 3:
                self.mover.up_move(self.ipeps, s1[1])
                self.mover.down_move(self.ipeps, s2[1])

    @record_runtime
    def absorb_bond_dist(self, bond):
        """
        Absorbs a bond in the iPEPS network using distributed updates.

        The method performs directional moves, updating the boundary tensors in the direction specified 
        by the bond direction index (k). The boundary tensors are updated using distributed moves, 
        and the bond is absorbed by the directional moves.

        Args:
            bond (list): A list representing a bond in the iPEPS network. It contains the 
                indices of two adjacent tensors and an integer (k) that specifies the direction 
                of the move.

        Returns:
            None
        """
        s1,s2,k = bond
        match k:
            case 0:
                self.mover.left_right_move_dist(self.ipeps, s2[0], s1[0])
            case 1:
                self.mover.up_down_move_dist(self.ipeps, s2[1], s1[1])
            case 2:
                self.mover.left_right_move_dist(self.ipeps, s1[0], s2[0])
            case 3:
                self.mover.up_down_move_dist(self.ipeps, s1[1], s2[1])
