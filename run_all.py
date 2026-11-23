"""Convenience entry point: `python run_all.py` == `python -m mindforecast.cli run`."""

from mindforecast.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run"]))
