# Ace-TN
Corner-transfer matrix renormalizization of infinite projected entangled-pair states with GPU acceleration. To install, use
```bash
pip install -e .
```
See the examples in the `samples` directory for usage.

Run a script `script.py` in multi-GPU mode using `N` processes with
```
torchrun --nproc_per_node=N script.py
```
