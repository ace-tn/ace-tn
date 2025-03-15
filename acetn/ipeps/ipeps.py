import torch
from tqdm import tqdm
from .ipeps_config import IpepsConfig
from .tensor_network import TensorNetwork
from ..renormalization.ctmrg import ctmrg
from ..evolution.evolve import evolve
from ..measurement.measure import measure
from ..model.model_factory import model_factory
from ..utils.distributed import *
from ..utils.logger import *

class Ipeps(TensorNetwork):
    """
    Infinite Projected Entangled Pair States (iPEPS) tensor network class.

    Handles initialization, evolution, renormalization, and measurement 
    procedures for iPEPS simulations, including distributed setup.

    Attributes:
        config (IpepsConfig): Configuration for iPEPS simulation.
        dtype (torch.dtype): Data type for tensors.
        device (torch.device): Device (CPU/GPU) for computation.
        is_distributed (bool): Flag for distributed computation.
        rank (int): Process rank in distributed setting.
        world_size (int): Total number of processes in distributed setting.
    """
    def __init__(self, config, ipeps=None):
        """
        Initialize the iPEPS tensor network.

        Args:
            config (dict): Dictionary of configuration parameters.
            ipeps (optional): Initial iPEPS tensor network.
        """
        self.config = IpepsConfig(**config)
        self.dtype = self.config.dtype
        self.device = self.config.device
        self.setup_distributed()
        log_initial_message(self.device, config)
        super().__init__(ipeps, self.config.TN, self.dtype, self.device)
        self.site_states_initialized = False

    def __del__(self):
        """
        Finalize the distributed environment upon object deletion.
        """
        if self.is_distributed:
            finalize_distributed()

    def setup_distributed(self):
        """
        Set up distributed environment if available.
        """
        self.is_distributed = False
        self.rank, self.world_size, self.device, self.is_distributed \
            = setup_distributed(self.device)

    def set_chi(self, chi):
        """
        Set the boundary bond dimension chi.

        Args:
            chi (int): Boundary bond dimension.
        """
        self.dims['chi'] = chi
        tqdm.write(f"Boundary bond dimension set: chi={chi}")

    def set_model(self, model_cls, params, name=None):
        """
        Register and set a model for simulation.

        Args:
            model_cls (type): Model class to register.
            params (dict): Parameters for the model.
            name (str, optional): Name for the custom model.
        """
        name = "custom" if name is None else name.lower()
        model_factory.register(name, model_cls)
        self.config.model.name = name
        self.config.model.params = params

    def set_model_params(self, **kwargs):
        """
        Update model parameters.

        Args:
            `**kwargs`: Key-value pairs of model parameters to update.
        """
        for key,val in kwargs.items():
            self.config.model.params[key] = val
            tqdm.write(f"Model parameter set: {key}={val}")

    def renormalize(self):
        """
        Perform CTMRG renormalization on the current iPEPS state.
        """
        ctmrg(self, self.config.ctmrg)

    @torch.inference_mode()
    def evolve(self, dtau, steps, model=None):
        """
        Apply imaginary time evolution to the iPEPS state.

        Args:
            dtau (float): Time step for evolution.
            steps (int): Number of evolution steps.
            model (optional): Custom model instance to use.
        """
        model = model_factory.create(self.config.model) if model is None else model
        if not self.site_states_initialized:
            self.initialize_site_tensors(model.initial_site_state)
        log_evolve_start_message(dtau, self.dims, model)
        self.renormalize()
        evolve(self, dtau, steps, model, self.config.evolution)
        self.renormalize()

    def measure(self, model=None):
        """
        Perform measurements on the current iPEPS state.

        Args:
            model (optional): Custom model instance to use.

        Returns:
            dict: Measurement results.
        """
        model = model_factory.create(self.config.model) if model is None else model
        return measure(self, model)
