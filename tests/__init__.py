"""Make the modules in scripts/ importable by the tests.

Run from the repository root: python -m unittest discover -s tests -t .
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
