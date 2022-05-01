from pathlib import Path
import sys

dir = Path(__file__)
module_root = str(dir.parent)
if module_root not in sys.path:
    sys.path.append(module_root)