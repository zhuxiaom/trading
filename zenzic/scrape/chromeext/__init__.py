from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
from pathlib import Path

def uBlockOrigin():
    folder = os.path.dirname(Path(__file__).absolute())
    files = [f for f in os.listdir(folder) if f.startswith("uBlock-Origin") and f.endswith(".crx")]
    return os.path.join(folder, files[0])
