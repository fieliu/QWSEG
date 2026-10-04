"""Import the repository-level, versioned RGB-T augmentation library."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[5])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rgbt_c.torch_adapter import RGBTDegrader  # noqa: E402,F401
