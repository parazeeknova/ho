"""Make the sibling version_manager importable for tests in this directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
