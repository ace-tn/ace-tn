import torch
import torch.distributed as dist
import os
import builtins
from tqdm import tqdm
from torch.distributed.elastic.multiprocessing.errors import record


@record
def setup_distributed(device):
    """
    Initializes the distributed environment and device configuration.
    
    Args:
        device (torch.device): The device to use (e.g. 'cpu' or 'cuda'). 
    """
    rank = 0
    world_size = 1

    is_distributed = "RANK" in os.environ and \
                     "WORLD_SIZE" in os.environ and \
                     int(os.environ["WORLD_SIZE"]) > 1
    if is_distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        setup_distributed_print(rank)
        device_count = get_device_count(device)
        if device_count < world_size:
            raise RuntimeError(
                f"{world_size} devices requested but only {device_count} available")
        device = setup_process_group(rank, world_size, device)

    return rank, world_size, device, is_distributed


@record
def get_device_count(device):
    """
    Calculate device count for distributed setup on supported devices.
    """
    if device == torch.device("cpu"):
        return os.cpu_count()
    elif device == torch.device("cuda"):
        return torch.cuda.device_count()
    else:
        raise NotImplementedError(
            f"Distributed calculations are not supported " +
            f"for {device.type} device type.")


def setup_process_group(rank, world_size, device):
    """
    Initializes the process group for multi-device execution.

    Sets the backend and assigns a device to each rank.
    """
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        assert rank == dist.get_rank()
        assert world_size == dist.get_world_size()
        if device != torch.device("cpu"):
            device = rank
            torch.cuda.set_device(device)
    return device


def setup_distributed_print(rank):
    """
    Redirects print statements to only show from rank 0 in distributed execution.
    """
    if rank != 0:
        builtins.print = lambda *args, **kwargs: None
        tqdm.write = lambda *args, **kwargs: None


def finalize_distributed():
    """
    Finalizes the distributed environment and cleans up resources.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def all_gather_tensor(tensor, rank, ws):
    """
    Gathers tensors from all workers in a distributed setting.

    Args:
        tensor (Tensor): The tensor to be gathered.
        rank (int): The rank of the current worker.
        ws (int): The total number of workers in the distributed setup.

    Returns:
        list: A list of gathered tensors from all workers.
    """
    # gather shapes
    tensor_shape = torch.tensor(list(tensor.shape), dtype=int, device=tensor.device)
    tensor_all_shapes = [torch.empty(4, dtype=int, device=tensor.device) for _ in range(ws)]
    dist.all_gather(tensor_all_shapes, tensor_shape)
    # gather tensors
    tensor_all = []
    for i in range(ws):
        tensor_buf_shape = list(tensor_all_shapes[i].to(rank))
        tensor_buf = torch.empty(tensor_buf_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_all.append(tensor_buf)
    dist.all_gather(tensor_all, tensor)
    return tensor_all