"""Make the sibling scripts/ modules importable for tests.

``scripts/`` has no package ``__init__.py`` and the tests live one level down
in ``scripts/tests/``, so pytest's rootdir-insertion no longer puts ``scripts/``
on ``sys.path``. Insert it explicitly so ``import grill_persona`` / ``import ux``
resolve.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
