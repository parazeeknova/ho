"""Make the sibling scripts/ modules importable for tests.

``scripts/`` has no package ``__init__.py`` and the tests live one level down
in ``scripts/tests/``, so pytest's rootdir-insertion no longer puts the scripts
dirs on ``sys.path``. Insert them explicitly so ``import grill_persona`` /
``import ux`` resolve (grill_persona moved to packages/autofill/scripts).
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
for _p in (
    _REPO,
    Path(__file__).resolve().parent.parent,
    _REPO / "packages" / "autofill" / "scripts",
):
    sys.path.insert(0, str(_p))
