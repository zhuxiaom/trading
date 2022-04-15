from pathlib import Path
import sys

dir = Path(__file__)
if dir not in sys.path:
    sys.path.append(dir.parent.parent)

