# Ace-TN: Simple and Efficient iPEPS Simulation
## About
Ace-TN is an efficient and easy-to-use library for simulating infinite projected entangled-pair states (iPEPS) using the corner-transfer matrix renormalization group (CTMRG) method.

- **GPU Acceleration**: Supports GPU computations, offering a significant performance boost for larger and more complex iPEPS simulations.
- **Multi-GPU Support**: Can utilize multiple GPUs, allowing users to scale calculations to the largest possible system sizes.
- **Flexible and Easy to Use**: Easy and straightforward API, providing easy management of tensors and flexible model Hamiltonian construction.

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

## Multi-GPU Usage
Run a script `script.py` in multi-GPU mode using `N` processes and `N` GPUs with
```
torchrun --nproc_per_node=N script.py
```

## License
Ace-TN is licensed under the Apache-2.0 License. See the `LICENSE` file for more details.
