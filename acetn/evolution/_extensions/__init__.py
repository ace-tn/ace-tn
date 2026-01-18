"""
Extension loading module for compiled C++ extensions.

This module automatically loads the cuTENSOR extension if available.
The extension provides GPU-accelerated tensor contractions via cuTENSOR.
"""
import warnings

CUTENSOR_AVAILABLE = False


def load_cutensor_extension():
    """
    Load the cuTENSOR C++ extension if available.
    
    Returns True if the extension was loaded successfully, False otherwise.
    """
    global CUTENSOR_AVAILABLE

    try:
        # Import torch first to set up library paths for libc10.so etc.
        import torch  # noqa: F401
        # Try importing the compiled extension module
        from acetn.evolution._extensions import _C_cutensor  # noqa: F401
        CUTENSOR_AVAILABLE = True
    except ImportError:
        # Extension not built - this is fine, fall back to pure PyTorch
        pass
    except Exception as e:
        warnings.warn(
            f"Failed to load cuTENSOR extension: {e}",
            RuntimeWarning
        )

    return CUTENSOR_AVAILABLE


# Automatically load extension when this module is imported
load_cutensor_extension()

