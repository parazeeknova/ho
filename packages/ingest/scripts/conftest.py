"""Make the sibling scripts/ modules importable for tests.

``scripts/`` has no package ``__init__.py``, so pytest's rootdir-insertion does
not put the scripts dirs on ``sys.path``. Insert them explicitly so
``import grill_persona`` / ``import ux`` resolve (grill_persona lives in
packages/autofill/scripts). Tests live alongside the scripts in this dir.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]  # packages/ingest/scripts -> repo root
for _p in (
    _REPO,
    Path(__file__).resolve().parent.parent,
    _REPO / "packages" / "autofill" / "scripts",
):
    sys.path.insert(0, str(_p))
