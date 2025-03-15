from .ipeps import Ipeps
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
torch.set_grad_enabled(False)
