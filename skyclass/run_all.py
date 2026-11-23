"""Convenience entry point: `python run_all.py` == `python -m skyclass.cli run`."""

from skyclass.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run"]))
