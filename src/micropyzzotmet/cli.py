from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "micropyzzotmet currently supports Python 3.10-3.12. "
            "Please create an environment with Python 3.12 or 3.11."
        )

    from .main_micromet import run_micropezzomet

    parser = argparse.ArgumentParser(
        prog="micropyzzotmet",
        description="Run the MicroPyzzotMet downscaling workflow from a JSON config file.",
    )
    parser.add_argument(
        "config",
        help="Path to the JSON configuration file",
    )

    args = parser.parse_args()

    start = time.time()
    run_micropezzomet(args.config)
    elapsed = time.time() - start

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"Run completed in {mins} min {secs} s.")


if __name__ == "__main__":
    main()