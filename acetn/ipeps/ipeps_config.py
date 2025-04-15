import torch
import os
from dataclasses import dataclass, field
from typing import Dict, Union, Literal
from tqdm import tqdm

@dataclass
class TNConfig:
    nx: int = 2
    ny: int = 2
    dims: Dict[str, int] = field(
        default_factory=lambda: {'phys': 2, 'bond': 2, 'chi': 20})
    initial_site_states: Literal["ferro", "neel"] = None

@dataclass
class CTMRGConfig:
    steps: int = 40
    projectors: Literal["half-system", "full-system"] = "half-system"
    svd_type: Literal["rsvd", "full-rank"] = "rsvd"
    svd_cutoff: float = 1e-12
    rsvd_niter: int = 2
    rsvd_oversampling: int = 2
    disable_progressbar: bool = False

@dataclass
class EvolutionConfig:
    update_type: str = "full"
    use_gauge_fix: bool = True
    gauge_fix_atol: float = 1e-12
    positive_approx_cutoff: float = 1e-12
    als_niter: int = 100
    als_tol: int = 1e-12
    als_method: Literal["cholesky", "pinv"] = "cholesky"
    als_epsilon: float = 1e-12
    disable_progressbar: bool = False

@dataclass
class ModelConfig:
    dtype: torch.dtype
    device: torch.device
    dim: int
    name: str = None
    params: Dict[str, int] = None

@dataclass
class IpepsConfig:
    """Configuration class for iPEPS setup."""
    TN: Dict
    dtype: Union[str, torch.dtype] = "float64"
    device: Union[str, torch.device] = "cpu"
    ctmrg: Dict = field(default_factory=lambda: {})
    evolution: Dict = field(default_factory=lambda: {})
    model: Dict = field(default_factory=lambda: {})

    def __post_init__(self):
        self.set_dtype(self.dtype)
        self.set_device(self.device)
        self.TN = TNConfig(**self.TN)
        self.ctmrg = CTMRGConfig(**self.ctmrg)
        self.evolution = EvolutionConfig(**self.evolution)
        self.model = ModelConfig(dtype=self.dtype,
                           device=self.device,
                           dim=self.TN.dims['phys'],
                           **self.model)
        # disable tqdm progress bar in SLURM environment
        self.disable_pbar_if_slurm()

    def disable_pbar_if_slurm(self):
        if 'SLURM_JOB_ID' in os.environ:
            # enable if interactive job
            job_name = os.getenv('SLURM_JOB_NAME', '')
            job_partition = os.getenv('SLURM_JOB_PARTITION', '')
            if 'interactive' in job_name or job_partition == 'interactive':
                disable_pbar = False
            else:
                disable_pbar = True
            self.ctmrg.disable_progressbar = disable_pbar
            self.evolution.disable_progressbar = disable_pbar

    def set_dtype(self, dtype):
        """Set the dtype to either the provided value or a default."""
        self.dtype = torch.float64 if dtype is None else dtype
        if isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)

    def set_device(self, device):
        """Set the device to either the provided value or a default."""
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        # Use cpu if device=cuda is set in config but cuda is not available
        if self.device == torch.device("cuda") and not torch.cuda.is_available():
            tqdm.write("Torch not compiled with CUDA enabled")
            tqdm.write("Using device='cpu'")
            self.device = torch.device("cpu")
