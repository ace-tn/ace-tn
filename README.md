# Ace-TN
Library for efficient simulation of infinite projected entangled-pair states (iPEPS) based on the corner-transfer matrix renormalization group (CTMRG) method with GPU acceleration. To install, use
```bash
pip install -e .
```
See the project [documentation](https://ace-tn.github.io/ace-tn/) or examples in the `samples` directory for usage.

Run a script `script.py` in multi-GPU mode using `N` processes with
```
torchrun --nproc_per_node=N script.py
```
