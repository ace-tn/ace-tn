import torch
from pathlib import Path

lib = list(Path(__file__).parent.glob("_C*.so"))
assert len(lib) == 1
torch.ops.load_library(str(lib[0]))
