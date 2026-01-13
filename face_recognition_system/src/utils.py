import os
import sys
from . import config

def setup_directories():
    """Creates necessary directories if they don't exist."""
    os.makedirs(config.RAW_IMG_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

def check_dependencies():
    """Checks for libraries that might crash the app later."""
    try:
        import scipy
    except ImportError:
        print("[ERROR] 'scipy' is missing. Please install it: pip install scipy")
        sys.exit(1)