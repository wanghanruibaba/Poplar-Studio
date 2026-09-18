#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot measured-versus-reconstructed leaf area for four reconstruction methods."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from plot_style import apply_figure_style
apply_figure_style()
import numpy as np


METHODS = {
    "fixed_poisson": {
        "label": "Fixed Poisson",
        "area_col": "fixed_poisson_area_cm2",
        "color": "#4C78A8",
    },
    "adaptive_poisson": {
        "label": "Adaptive Poisson",
        "area_col": "adaptive_poisson_area_cm2",
        "color": "#F28E2B",
    },
    "ball_pivoting": {
        "label": "Ball pivoting",
        "area_col": "ball_pivoting_area_cm2",
        "color": "#59A14F",
    },
    "pca": {
        "label": "Boundary-constrained (this study)",
        "area_col": "pca_area_cm2",
        "color": "#B07AA1",
    },
}


def configure_style() -> None:
    """Match the typography and axes used by the original figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def finite_array(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = []
    for row_number, row in enumerate(rows, start=2):
        if column not in row:
            raise KeyError(f"Missing CSV column: {column}")
        try:
            value = float(row[column])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid numeric value in {column}, CSV row {row_number}: "
                f"{row[column]!r}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite value in {column}, CSV row {row_number}: {value}"
            )
        values.append(value)
    return np.asarray(values, dtype=float)


def calculate_statistics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    """Return agreement metrics, OLS fit, and regression residuals.

    ``agreement_r2`` intentionally retains the definition from the original
    script: 1 - SSE(prediction-reference) / SST(reference).  It is different
    from ``regression_r2``, which describes the fitted OLS line.
    """
    raw_error = prediction - reference
    agreement_sse = float(np.sum(raw_error**2))
    reference_sst = float(np.sum((reference - reference.mean()) ** 2))
    agreement_r2 = 1.0 - agreement_sse / reference_sst
    rmse = float(np.sqrt(np.mean(raw_error**2)))

    slope, intercept = np.polyfit(reference, prediction, 1)
    fitted = slope * reference + intercept
    regression_residual = prediction - fitted
    regression_sse = float(np.sum(regression_residual**2))
    prediction_sst = float(np.sum((prediction - prediction.mean()) ** 2))
    regression_r2 = 1.0 - regression_sse / prediction_sst

    return {
        "n": int(len(reference)),
        "agreement_r2": float(agreement_r2),
        "regression_r2": float(regression_r2),
        "slope": float(slope),
        "intercept_cm2": float(intercept),
        "rmse_cm2": rmse,
        "rrmse_percent": float(rmse / reference.mean() * 100.0),
        "bias_cm2": float(raw_error.mean()),
        "fitted": fitted,
        "regression_residual": regression_residual,
        "raw_error": raw_error,
        "residual_sd_cm2": float(np.std(regression_residual, ddof=1)),
    }


def signed_equation(slope: float, intercept: float) -> str:
    sign = "+" if intercept >= 0 else "−"
    return f"y = {slope:.3f}x {sign} {abs(intercept):.2f}"


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def draw_scatter(
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
    statistics: dict[str, dict],
    output_dir: Path,
) -> None:
    all_values = [reference, *predictions.values()]
    lower = max(0.0, min(float(values.min()) for values in all_values) * 0.95)
    upper = max(float(values.max()) for values in all_values) * 1.05
    grid = np.linspace(lower, upper, 200)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.35), sharex=True, sharey=True)
    for panel, (axis, (method, config)) in enumerate(zip(axes.flat, METHODS.items())):
        prediction = predictions[method]
        item = statistics[method]
        axis.scatter(
            reference,
            prediction,
            s=15,
            alpha=0.62,
            color=config["color"],
            edgecolor="white",
            linewidth=0.25,
            rasterized=True,
        )
        axis.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            color="#555555",
            linewidth=0.9,
        )
        axis.plot(
            grid,
            item["slope"] * grid + item["intercept_cm2"],
            color=config["color"],
            linewidth=1.2,
        )
        annotation = (
            f"{signed_equation(item['slope'], item['intercept_cm2'])}\n"
            f"Agreement $R^2$ = {item['agreement_r2']:.3f}\n"
            f"RMSE = {item['rmse_cm2']:.2f} cm$^2$\n"
            f"rRMSE = {item['rrmse_percent']:.1f}%\n"
            f"Bias = {item['bias_cm2']:.2f} cm$^2$"
        )
        axis.text(
            0.04,
            0.96,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=6.9,
            linespacing=1.25,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.2},
        )
        axis.set_title(config["label"], loc="left", fontsize=9, fontweight="bold")
        axis.text(
            -0.13,
            1.04,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#E6E6E6", linewidth=0.5, alpha=0.7)
        axis.tick_params(axis="both", which="major", labelsize=9)

    for axis in axes[-1, :]:
        axis.set_xlabel("Measured leaf area (cm$^2$)", fontsize=10)
    for axis in axes[:, 0]:
        axis.set_ylabel("Reconstructed area (cm$^2$)", fontsize=10)
    fig.suptitle(
        "Leaf-area agreement across reconstruction methods",
        fontsize=10.5,
        y=0.995,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.095, top=0.92,
                        wspace=0.13, hspace=0.15)
    save_figure(fig, output_dir / "four_method_area_scatter_with_equations")
    plt.close(fig)








def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the four-method leaf-area scatter plot with OLS equations and agreement metrics."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=script_dir.parent / "data" / "benchmark" / "sampled_four_method_leaf_areas.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir.parent / "results" / "area")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_csv)
    if len(rows) != 240:
        raise ValueError(f"Expected 240 paired leaves, found {len(rows)}")

    reference = finite_array(rows, "measured_area_cm2")
    predictions = {
        method: finite_array(rows, config["area_col"])
        for method, config in METHODS.items()
    }
    statistics = {
        method: calculate_statistics(reference, prediction)
        for method, prediction in predictions.items()
    }

    configure_style()
    draw_scatter(reference, predictions, statistics, output_dir)

    for method, config in METHODS.items():
        item = statistics[method]
        print(
            f"{config['label']}: {signed_equation(item['slope'], item['intercept_cm2'])}; "
            f"agreement R2={item['agreement_r2']:.3f}; "
            f"regression R2={item['regression_r2']:.3f}"
        )
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
