"""
Extension loading module for compiled C++ extensions.

This module automatically loads all compiled C++ extensions (shared objects)
found in this directory when imported.
"""
import torch
from pathlib import Path
import glob
import warnings


def load_extensions():
    """
    Load all compiled C++ extensions (shared objects) in this directory.
    
    Extensions are loaded into torch.ops for use by the Python code.
    Failures to load are logged as warnings but do not prevent the module
    from being imported.
    """
    ext_dir = Path(__file__).parent
    so_files = glob.glob(str(ext_dir / "_C*.so"))
    
    if not so_files:
        # No extensions found - this is okay for development
        return
    
    for so_file in so_files:
        try:
            torch.ops.load_library(so_file)
        except Exception as e:
            warnings.warn(
                f"Failed to load extension {Path(so_file).name}: {e}",
                RuntimeWarning
            )


# Automatically load extensions when this module is imported
load_extensions()

