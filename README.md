# Ace-TN: GPU-Accelerated iPEPS Simulation
## About
Ace-TN was designed to benchmark and explore GPU-based acceleration of common algorithms used to simulate infinite projected entangled-pair states (iPEPS) with the corner-transfer matrix renormalization group (CTMRG) method. We provide a framework that can run an end-to-end iPEPS ground-state calculation.

See our preprint for more details at [https://arxiv.org/abs/2503.13900](https://arxiv.org/abs/2503.13900).

## Getting Started
### Installation
To install Ace-TN, use:
```bash
pip install acetn
```
### Quickstart
After installation, a simple example calculation can be run by using `python3 script.py` for the following `script.py`:
```bash
from acetn.ipeps import Ipeps

# Set simulation options in the config
config = {
    "TN":{
        "nx": 2,
        "ny": 2,
        "dims": {"phys": 2, "bond": 2, "chi": 20},
    },
    "model":{
        "name": "heisenberg",
        "params": {"J": 1.0},
    },
}

# Initialize an iPEPS
ipeps = Ipeps(config)

# Perform imaginary-time evolution steps
ipeps.evolve(dtau=0.01, steps=100)

# Measure observables
measurements = ipeps.measure()
```
For more details on usage, see the project [documentation](https://ace-tn.github.io/ace-tn/) or try some of the example scripts in the `samples` directory.

## Optional cuTENSOR Acceleration

Ace-TN includes an optional C++ extension that uses NVIDIA's cuTENSOR and cuSOLVER libraries for accelerated tensor contractions in the ALS solver.

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (with cuSOLVER)
- [cuTENSOR library](https://developer.nvidia.com/cutensor) installed (e.g., `libcutensor.so` in `/usr/local/lib` or `$CUDA_HOME/lib64`)
- PyTorch with CUDA support

### Building with cuTENSOR

To build and install with the cuTENSOR extension:

```bash
ACETN_BUILD_CUTENSOR=1 pip install .
```

### Usage

Once built, set the backend in your evolution config:

```python
config = {
    "evolution": {
        "backend": "cutensor",  # Use cuTENSOR backend (default: "torch")
    },
    # ... other config options
}
```

If the cuTENSOR extension is not available, the library will automatically fall back to the PyTorch backend.

## Multi-GPU Usage
Run a script `script.py` in multi-GPU mode using `N` processes and `N` GPUs with
```
torchrun --nproc_per_node=N script.py
```

## Citing
If you found Ace-TN useful for your research, please cite:
```
@misc{AceTN_2025,
      title={Ace-TN: GPU-Accelerated Corner-Transfer-Matrix Renormalization of Infinite Projected Entangled-Pair States}, 
      author={Addison D. S. Richards and Erik S. Sørensen},
      year={2025},
      eprint={2503.13900},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el},
      url={https://arxiv.org/abs/2503.13900}, 
}
```

## License
Ace-TN is licensed under the Apache-2.0 License. See the `LICENSE` file for more details.
