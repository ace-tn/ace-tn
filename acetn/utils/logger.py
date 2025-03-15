import logging
import os
import toml
import torch
from tqdm import tqdm

class Logger:
    def __init__(self, name: str, log_level=logging.INFO, log_file="debug.log"):
        self.logger = logging.getLogger(name)

        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        log_level = os.getenv('ACETN_LOG_LEVEL', 'INFO').upper()
        self.logger.setLevel(log_levels.get(log_level))

        self.log_file = log_file if log_level == logging.DEBUG else None
        self._configure_handlers()

    def _configure_handlers(self):
        log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
        formatter = logging.Formatter(log_format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        #if self.log_file:
        #    file_handler = logging.FileHandler(self.log_file)
        #    file_handler.setLevel(logging.DEBUG)
        #    file_handler.setFormatter(formatter)
        #    self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger


logger = Logger(__name__, log_level=logging.INFO).get_logger()


def log_initial_message(device, config):
    """
    Print startup information and configuration.
    
    Args:
        config (dict): Original configuration dictionary.
    """
    tqdm.write("Starting iPEPS calculation with config:")
    tqdm.write("-"*40)
    tqdm.write(toml.dumps(config) + "-"*40)
    log_device_info(device)
    tqdm.write("Initializing iPEPS...")


def log_device_info(device):
    """
    Print information about the computation device.
    """
    if device == torch.device("cpu"):
        tqdm.write(f"Running in CPU mode using {torch.get_num_threads()} threads")


def log_evolve_start_message(dtau, dims, model):
    """
    Print evolution startup information.

    Args:
        dtau: Imaginary-time step used in evolution.
        dims: Dictionary containing iPEPS dimensions.
        model: Model instance being used in evolution.
    """
    tqdm.write(f"Start imaginary-time evolution with dtau={float(dtau)}...")
    tqdm.write("Bond dimensions:")
    tqdm.write(f"(D, chi): ({dims['bond']}, {dims['chi']})")
    tqdm.write("Model parameters:")
    tqdm.write('\n'.join(f"{key}: {value}" for key, value in model.params.items()))
