"""
Setup script for acetn with optional cuTENSOR extension.

Usage:
    pip install .                    # Pure Python only
    ACETN_BUILD_CUTENSOR=1 pip install .  # With cuTENSOR extension
"""

import os
import warnings
from pathlib import Path
from setuptools import setup

BUILD_CUTENSOR = os.environ.get("ACETN_BUILD_CUTENSOR", "0") == "1"
ROOT_DIR = Path(__file__).parent.resolve()

_ext_modules = None
_cmdclass = None


def get_extensions():
    """Get extension modules and cmdclass, with caching."""
    global _ext_modules, _cmdclass

    if _ext_modules is not None:
        return _ext_modules, _cmdclass

    _ext_modules = []
    _cmdclass = {}

    if not BUILD_CUTENSOR:
        return _ext_modules, _cmdclass

    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
    except ImportError:
        warnings.warn("PyTorch not found, skipping cuTENSOR extension")
        return _ext_modules, _cmdclass

    if CUDA_HOME is None:
        warnings.warn("CUDA_HOME not set, skipping cuTENSOR extension")
        return _ext_modules, _cmdclass

    if not torch.cuda.is_available():
        warnings.warn("CUDA not available, skipping cuTENSOR extension")
        return _ext_modules, _cmdclass

    # Check for cuTENSOR
    cutensor_lib_dirs = [
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        os.path.join(CUDA_HOME, "lib64"),
    ]
    cutensor_found = any(
        os.path.exists(os.path.join(d, "libcutensor.so")) for d in cutensor_lib_dirs
    )

    if not cutensor_found:
        warnings.warn("cuTENSOR library not found, skipping cuTENSOR extension")
        return _ext_modules, _cmdclass

    print("Building cuTENSOR extension...")

    _ext_modules = [
        CUDAExtension(
            name="acetn.evolution._extensions._C_cutensor",
            sources=[
                "csrc/evolution/als_solve.cpp",
                "csrc/evolution/als_tensors.cpp",
                "csrc/linalg/contraction.cpp",
            ],
            include_dirs=[
                str(ROOT_DIR / "csrc"),
                str(ROOT_DIR / "csrc" / "linalg"),
                str(ROOT_DIR / "csrc" / "evolution"),
            ],
            libraries=["cutensor", "cusolver"],
            define_macros=[("_GLIBCXX_USE_CXX11_ABI", "0")],
            extra_compile_args={
                "cxx": ["-O3", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": ["-O3"],
            },
        )
    ]
    _cmdclass = {"build_ext": BuildExtension}

    return _ext_modules, _cmdclass


ext_modules, cmdclass = get_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
