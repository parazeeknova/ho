"""ho — job matching pipeline.

By default runs the radar v2 pipeline. Set LEGACY_PIPELINE=true to use
the legacy orchestrator instead.
"""

import os


def run() -> None:
    if os.environ.get("LEGACY_PIPELINE", "").lower() in ("1", "true", "yes"):
        from src.pipeline.orchestrator import run as _legacy_run

        _legacy_run()
    else:
        from src.radar.orchestrator import run as _radar_run

        _radar_run()


if __name__ == "__main__":
    run()
