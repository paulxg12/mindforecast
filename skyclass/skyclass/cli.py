"""Command line interface: run the full SkyClass study."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_observations
from .features import add_features
from .models import evaluate_both_regimes, redshift_only_baseline
from .report import (
    plot_class_profile,
    plot_confusion,
    plot_redshift_by_class,
    plot_regime_comparison,
    write_html,
)


def run(
    test_size: float = 0.2, seed: int = 0, out_dir: Path = Path("artifacts")
) -> dict[str, object]:
    """Train both regimes, write figures plus an HTML report, and print a summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = load_observations()
    featured = add_features(objects)

    results = evaluate_both_regimes(featured, test_size=test_size, seed=seed)
    redshift_only = redshift_only_baseline(featured, test_size=test_size, seed=seed)

    figures = {
        "regime": plot_regime_comparison(results, out_dir / "regime_comparison.png"),
        "confusion_photometric": plot_confusion(
            results["photometric (no redshift)"]["confusion"],
            results["photometric (no redshift)"]["labels"],
            "Photometric only",
            out_dir / "confusion_photometric.png",
        ),
        "confusion_spectroscopic": plot_confusion(
            results["spectroscopic (+ redshift)"]["confusion"],
            results["spectroscopic (+ redshift)"]["labels"],
            "Photometric + redshift",
            out_dir / "confusion_spectroscopic.png",
        ),
        "redshift": plot_redshift_by_class(objects, out_dir / "redshift_by_class.png"),
        "class_profile": plot_class_profile(featured, out_dir / "class_colours.png"),
    }

    report = write_html(
        out_dir / "report.html",
        results,
        redshift_only,
        figures,
        class_counts=objects["class"].value_counts(),
        n_objects=int(len(objects)),
        test_size=test_size,
        seed=seed,
    )

    for regime, result in results.items():
        slug = "photometric" if regime.startswith("photometric") else "spectroscopic"
        result["metrics"].to_csv(out_dir / f"metrics_{slug}.csv", index=False)
        result["per_class"].to_csv(out_dir / f"per_class_{slug}.csv", index=False)

    photo = results["photometric (no redshift)"]["metrics"]
    spec = results["spectroscopic (+ redshift)"]["metrics"]

    print(f"objects: {len(objects):,}   features: photometric {len(results['photometric (no redshift)']['features'])}, "
          f"spectroscopic {len(results['spectroscopic (+ redshift)']['features'])}")
    print()
    print("photometric only")
    print(photo.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("with redshift")
    print(spec.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"redshift-only reference: macro F1 {redshift_only['macro_f1']:.4f}")
    print(f"report -> {report}")

    return {
        "results": results,
        "redshift_only": redshift_only,
        "artifacts": out_dir,
        "report": report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="skyclass", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="train, evaluate and write the report")
    run_cmd.add_argument("--test-size", type=float, default=0.2, help="held-out fraction")
    run_cmd.add_argument("--seed", type=int, default=0, help="split seed")
    run_cmd.add_argument("--out", type=Path, default=Path("artifacts"), help="output directory")

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 2

    run(test_size=args.test_size, seed=args.seed, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
