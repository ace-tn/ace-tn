import time
from tqdm import tqdm
from .gate import Gate
from .fast_full_update import FastFullUpdater
from .simple_update import SimpleUpdater
from ..utils.logger import logger
import logging


def create_tensor_updater(ipeps, gate, config):
    """Create the appropriate tensor updater based on ``config.update_type``."""
    match config.update_type:
        case "full":
            return FastFullUpdater(ipeps, gate, config)
        case "simple":
            return SimpleUpdater(ipeps, gate, config)
        case _:
            raise ValueError(f"Unknown update_type: {config.update_type}")


def evolve(ipeps, dtau, steps, model, config):
    """Evolve the iPEPS tensor network in imaginary time.

    Args:
        ipeps: The iPEPS tensor network.
        dtau: Imaginary-time step size.
        steps: Number of evolution steps.
        model: Model that defines the Hamiltonian and gates.
        config: Evolution configuration.
    """
    gate = Gate(model, dtau, ipeps.bond_list, ipeps.site_list)
    tensor_updater = create_tensor_updater(ipeps, gate, config)
    is_debug_mode = logger.isEnabledFor(logging.DEBUG)
    upd_time = 0
    ctm_time = 0
    loop_start = time.time()
    iter_start = time.time()
    disable_pbar = config.disable_progressbar or ipeps.rank!=0
    for i in tqdm(range(1,steps+1), desc=f"Updating tensors", disable=disable_pbar):
        for bond in ipeps.bond_list:
            runtimes = tensor_updater.bond_update(bond)
            upd_time += runtimes[0]
            ctm_time += runtimes[1]
        ipeps.bond_list.reverse()
        if is_debug_mode and i % 10 == 0:
            iter_end = time.time()
            ctm_time /= len(ipeps.bond_list)
            upd_time /= len(ipeps.bond_list)
            logger.debug(f"Iteration {i}/{steps} completed using dtau={float(dtau)}")
            logger.debug(f"time per iteration: {(iter_end - iter_start)/10:.6f} seconds")
            logger.debug(f"full update: {upd_time/10:.6f} seconds")
            logger.debug(f"ctm update: {ctm_time/10:.6f} seconds")
            ctm_time = 0
            upd_time = 0
            iter_start = time.time()
    tensor_updater.finalize()
    loop_end = time.time()
    tqdm.write(f"Finished imaginary-time evolution")
    tqdm.write(f"Total runtime: {(loop_end - loop_start):.6f} seconds")
