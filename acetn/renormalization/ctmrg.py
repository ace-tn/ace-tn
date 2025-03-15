from .directional_mover import DirectionalMover
from tqdm import tqdm

def ctmrg(ipeps, config):
    """
    Runs the CTMRG algorithm for a specified number of steps.

    Depending on whether the iPEPS is distributed or not, the algorithm performs 
    boundary tensor updates in left-right and up-down directions.

    Args:
        ipeps (Ipeps): The iPEPS object representing the tensor network.
        config (dict): The CTMRG Configuration dictionary.
    """
    mover = DirectionalMover(config)
    disable_progressbar = config.disable_progressbar or ipeps.rank!=0
    pbar = tqdm(desc="Renormalizing boundary tensors", unit=" sweeps", disable=disable_progressbar)
    for _ in range(config.steps):
        pbar.update(1)
        if ipeps.is_distributed:
            for xi in range(ipeps.nx):
                mover.left_right_move_dist(ipeps, xi, (ipeps.nx - xi+1) % ipeps.nx)
            for yi in range(ipeps.ny):
                mover.up_down_move_dist(ipeps, (ipeps.ny - yi+1) % ipeps.ny, yi)
        else:
            for xi in range(ipeps.nx):
                mover.left_move(ipeps, xi)
                mover.right_move(ipeps, (ipeps.nx - xi+1) % ipeps.nx)
            for yi in range(ipeps.ny):
                mover.up_move(ipeps, (ipeps.ny - yi+1) % ipeps.ny)
                mover.down_move(ipeps, yi)
